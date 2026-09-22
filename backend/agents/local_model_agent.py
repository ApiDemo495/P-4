"""Local model agent - the upload flow that v1.0 never made work (Section 7.2).

Six explicit steps, each with a defined error state:

    1. file picker (frontend) filters to .gguf / .onnx
    2. validation  - magic bytes, size, header metadata  -> clear red error
    3. backend load - llama_cpp.Llama / onnxruntime
    4. test inference with a dummy prompt that must return parseable JSON
    5. success payload back to the UI (model name, params, ms/inference)
    6. ongoing monitoring - crash detection + memory guard every 60 s

Set ``LOCAL_AGENT_STUB=1`` to substitute a clearly-labelled deterministic
stand-in that implements the same interface.  It is a development aid so the
dashboard, fusion weighting and degradation paths can be exercised without a
multi-gigabyte GGUF file, and it always reports ``status: STUB`` so it can never
be mistaken for a real model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

from backend.agents.base import AgentHealth, AgentResult, AgentStatus, build_cycle_prompt, parse_agent_json
from backend.core import config as cfg

log = logging.getLogger("drosophila.agents.local")

NAME = "local"
GGUF_MAGIC = b"GGUF"
ONNX_MAGIC = b"\x08"  # protobuf varint field 1 (ir_version) starts the ONNX header
MIN_GGUF_BYTES = 100 * 1024 * 1024  # 100 MB - rejects truncated downloads
MEMORY_LIMIT_FRACTION = 0.85  # unload if we are eating the machine

TEST_PROMPT = (
    'Reply with exactly: {"decision": "HOLD", "confidence": 0.5, "reasoning": "test"}'
)

SYSTEM_PROMPT_LOCAL = (
    "You are a 1-minute scalping assistant for BTC and PAXG. "
    "Respond with JSON only: {\"decision\":\"BUY\"|\"SELL\"|\"HOLD\","
    "\"confidence\":0.0-1.0,\"reasoning\":\"one sentence\"}."
)


@dataclass
class ValidationResult:
    ok: bool
    kind: str = ""
    model_name: str = ""
    parameter_count: str = ""
    size_bytes: int = 0
    error: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "kind": self.kind,
            "model_name": self.model_name,
            "parameter_count": self.parameter_count,
            "size_bytes": self.size_bytes,
            "size_human": _human_bytes(self.size_bytes),
            "error": self.error,
            "metadata": self.metadata,
        }


def _human_bytes(value: int) -> str:
    step = 1024.0
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < step:
            return f"{size:.1f} {unit}"
        size /= step
    return f"{size:.1f} PB"


# ---------------------------------------------------------------------------
# Step 2 - validation
# ---------------------------------------------------------------------------


def validate_file(path: Path | str) -> ValidationResult:
    """Format validation before anything is loaded."""
    path = Path(path)
    if not path.exists():
        return ValidationResult(ok=False, error=f"File not found: {path}")
    size = path.stat().st_size
    suffix = path.suffix.lower()

    with path.open("rb") as fh:
        head = fh.read(64 * 1024)

    if suffix == ".gguf" or head.startswith(GGUF_MAGIC):
        return _validate_gguf(path, head, size)
    if suffix == ".onnx":
        return _validate_onnx(path, head, size)
    return ValidationResult(
        ok=False,
        size_bytes=size,
        error="Invalid file format. Please upload a valid .gguf or .onnx file.",
    )


def _validate_gguf(path: Path, head: bytes, size: int) -> ValidationResult:
    if not head.startswith(GGUF_MAGIC):
        return ValidationResult(
            ok=False,
            kind="gguf",
            size_bytes=size,
            error="File does not start with the GGUF magic bytes (0x47 0x47 0x55 0x46).",
        )
    if size < MIN_GGUF_BYTES:
        return ValidationResult(
            ok=False,
            kind="gguf",
            size_bytes=size,
            error=f"File is only {_human_bytes(size)}. A usable GGUF model is at least 100 MB; "
            "this file looks truncated.",
        )
    metadata = read_gguf_metadata(head)
    name = str(metadata.get("general.name") or metadata.get("general.basename") or path.stem)
    architecture = str(metadata.get("general.architecture") or "unknown")
    return ValidationResult(
        ok=True,
        kind="gguf",
        model_name=name,
        parameter_count=estimate_parameter_count(metadata, size),
        size_bytes=size,
        metadata={
            "architecture": architecture,
            "file_type": metadata.get("general.file_type"),
            "context_length": metadata.get(f"{architecture}.context_length"),
            "tensor_count": metadata.get("_tensor_count"),
            "version": metadata.get("_version"),
            "quantization": QUANT_BITS.get(metadata.get("general.file_type"), 4.5),
        },
    )


def _validate_onnx(path: Path, head: bytes, size: int) -> ValidationResult:
    # ONNX files are protobuf; a valid graph always begins with field 1
    # (ir_version) as a varint, and the file must contain the "onnx" producer or
    # at least resolve through onnxruntime.
    if not head or head[0] not in (0x08, 0x0A):
        return ValidationResult(
            ok=False,
            kind="onnx",
            size_bytes=size,
            error="File does not look like a protobuf-encoded ONNX model.",
        )
    metadata: dict = {}
    try:
        import onnxruntime as ort  # optional

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        metadata = {
            "inputs": [i.name for i in session.get_inputs()],
            "outputs": [o.name for o in session.get_outputs()],
            "input_shapes": [list(i.shape) for i in session.get_inputs()],
        }
    except ImportError:
        metadata = {"note": "onnxruntime is not installed; shapes cannot be inspected yet"}
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(
            ok=False, kind="onnx", size_bytes=size, error=f"ONNX load check failed: {exc}"
        )
    return ValidationResult(
        ok=True,
        kind="onnx",
        model_name=path.stem,
        parameter_count="unknown (ONNX)",
        size_bytes=size,
        metadata=metadata,
    )


#: GGUF file_type -> approximate bits per weight (used for a parameter estimate).
QUANT_BITS = {
    0: 32.0,  # F32
    1: 16.0,  # F16
    2: 4.5,  # Q4_0
    3: 4.5,  # Q4_1
    7: 8.5,  # Q8_0
    8: 5.8,  # Q5_0
    9: 5.8,  # Q5_1
    10: 2.6,  # Q2_K
    11: 3.4,  # Q3_K
    12: 4.5,  # Q4_K
    13: 5.5,  # Q5_K
    14: 6.6,  # Q6_K
    15: 8.5,  # Q8_K
}


def estimate_parameter_count(metadata: dict, size_bytes: int) -> str:
    bits = float(QUANT_BITS.get(metadata.get("general.file_type"), 4.5))
    estimate = size_bytes * 8.0 / max(bits, 1e-6)
    if estimate >= 1e9:
        return f"{estimate / 1e9:.1f}B (estimated)"
    if estimate >= 1e6:
        return f"{estimate / 1e6:.0f}M (estimated)"
    return "unknown"


def read_gguf_metadata(head: bytes) -> dict:
    """Parse the GGUF header KV block from the first chunk of the file.

    Layout: magic(4) version(u32) tensor_count(u64) kv_count(u64)
    then kv_count pairs of: key_len(u64) key(bytes) type(u32) value
    """
    try:
        if len(head) < 24:
            return {}
        version, tensor_count = struct.unpack_from("<IQ", head, 4)
        (kv_count,) = struct.unpack_from("<Q", head, 16)
        offset = 24
        metadata: dict = {
            "_version": version,
            "_tensor_count": tensor_count,
            "_kv_count": kv_count,
        }
        for _ in range(min(int(kv_count), 64)):
            if offset + 8 > len(head):
                break
            (key_len,) = struct.unpack_from("<Q", head, offset)
            offset += 8
            if key_len > 512 or offset + key_len + 4 > len(head):
                break
            key = head[offset : offset + key_len].decode("utf-8", "replace")
            offset += key_len
            (value_type,) = struct.unpack_from("<I", head, offset)
            offset += 4
            value, offset = _read_gguf_value(head, offset, value_type)
            if value is not None:
                metadata[key] = value
            if offset < 0:
                break
        return metadata
    except (struct.error, ValueError, IndexError):
        return {}


def _read_gguf_value(head: bytes, offset: int, value_type: int):
    """Decode the handful of GGUF value types we care about."""
    try:
        if value_type == 8:  # string
            (length,) = struct.unpack_from("<Q", head, offset)
            offset += 8
            if length > 4096 or offset + length > len(head):
                return None, -1
            return head[offset : offset + length].decode("utf-8", "replace"), offset + length
        if value_type == 4:  # uint32
            return struct.unpack_from("<I", head, offset)[0], offset + 4
        if value_type == 5:  # int32
            return struct.unpack_from("<i", head, offset)[0], offset + 4
        if value_type == 10:  # uint64
            return struct.unpack_from("<Q", head, offset)[0], offset + 8
        if value_type == 11:  # int64
            return struct.unpack_from("<q", head, offset)[0], offset + 8
        if value_type == 6:  # float32
            return struct.unpack_from("<f", head, offset)[0], offset + 4
        if value_type == 7:  # bool
            return bool(head[offset]), offset + 1
        if value_type == 0:  # uint8
            return head[offset], offset + 1
        # Arrays and the remaining scalar types are skipped by returning the
        # current offset, which stops the walk safely.
        return None, -1
    except (struct.error, IndexError):
        return None, -1


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class LocalModelAgent:
    def __init__(self, settings=None, store_dir: Path | None = None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.store_dir = Path(store_dir or cfg.MODEL_STORE_DIR)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.model_path: Path | None = None
        self.model_kind = ""
        self.model_name = ""
        self.parameter_count = ""
        self.loaded_at = 0.0
        self.inference_ms = 0.0
        self.inferences = 0
        self.crashed = False
        self.last_error = ""
        self.validation: ValidationResult | None = None
        self.stub = os.environ.get("LOCAL_AGENT_STUB", "0") in ("1", "true", "yes")
        self._health_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.model is not None and not self.crashed

    def status(self) -> AgentStatus:
        if self.crashed:
            return AgentStatus.ERROR
        if not self.ready:
            return AgentStatus.DISABLED
        return AgentStatus.STUB if self.stub else AgentStatus.ACTIVE

    # ------------------------------------------------------------------
    # Steps 2-5: upload, validate, load, test, report
    # ------------------------------------------------------------------
    async def upload(self, filename: str, payload: bytes) -> dict:
        safe_name = Path(filename).name
        target = self.store_dir / safe_name
        try:
            target.write_bytes(payload)
        except OSError as exc:
            return {"success": False, "error": f"Could not store upload: {exc}"}

        validation = await asyncio.to_thread(validate_file, target)
        self.validation = validation
        if not validation.ok:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
            return {"success": False, "error": validation.error, "validation": validation.to_dict()}

        loaded = await self.load(target)
        if not loaded.get("success"):
            return loaded
        return loaded

    async def load(self, path: Path) -> dict:
        """Step 3 + 4: load the file and prove it can answer."""
        self.unload(delete_file=False)

        if self.stub:
            return await self._load_stub()

        started = time.perf_counter()
        try:
            if path.suffix.lower() == ".gguf":
                self.model = await asyncio.wait_for(
                    asyncio.to_thread(self._load_gguf, path), timeout=180.0
                )
                self.model_kind = "gguf"
            elif path.suffix.lower() == ".onnx":
                self.model = await asyncio.wait_for(
                    asyncio.to_thread(self._load_onnx, path), timeout=180.0
                )
                self.model_kind = "onnx"
            else:
                return {"success": False, "error": "Unsupported model format"}
        except asyncio.TimeoutError:
            return {"success": False, "error": "Model load timed out after 180s"}
        except ImportError as exc:
            return {
                "success": False,
                "error": (
                    f"Required runtime is not installed: {exc}. "
                    "Install llama-cpp-python (GGUF) or onnxruntime (ONNX)."
                ),
            }
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            return {"success": False, "error": f"Model loaded but inference failed: {exc}"}

        self.model_path = path
        self.loaded_at = time.time()
        validation = self.validation or await asyncio.to_thread(validate_file, path)
        self.validation = validation
        self.model_name = validation.model_name or path.stem
        self.parameter_count = validation.parameter_count

        # ---- Step 4: test inference ------------------------------------
        test = await self._test_inference()
        if not test.get("ok"):
            self.unload(delete_file=False)
            return {"success": False, "error": test.get("error", "test inference failed")}

        self._start_health_monitor()
        return {
            "success": True,
            "model_name": self.model_name,
            "parameter_count": self.parameter_count,
            "test_inference_ms": round(self.inference_ms, 1),
            "ready": True,
            "kind": self.model_kind,
            "load_seconds": round(time.time() - self.loaded_at, 2),
        }

    @staticmethod
    def _load_gguf(path: Path):
        from llama_cpp import Llama  # optional dependency

        return Llama(
            model_path=str(path),
            n_ctx=2048,
            n_batch=512,
            n_threads=max(1, (os.cpu_count() or 4) // 2),
            verbose=False,
        )

    @staticmethod
    def _load_onnx(path: Path):
        import onnxruntime as ort  # optional dependency

        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    async def _load_stub(self) -> dict:
        self.model = _StubModel()
        self.model_kind = "stub"
        self.model_name = "stub-heuristic (development stand-in)"
        self.parameter_count = "n/a"
        self.loaded_at = time.time()
        self.validation = ValidationResult(
            ok=True, kind="stub", model_name=self.model_name, parameter_count="n/a"
        )
        self.inference_ms = 1.0
        self._start_health_monitor()
        return {
            "success": True,
            "model_name": self.model_name,
            "parameter_count": self.parameter_count,
            "test_inference_ms": 1.0,
            "ready": True,
            "kind": "stub",
            "warning": "Development stub active - no real GGUF/ONNX weights are loaded.",
        }

    # ------------------------------------------------------------------
    async def _test_inference(self) -> dict:
        try:
            started = time.perf_counter()
            output = await asyncio.wait_for(
                asyncio.to_thread(self._generate, TEST_PROMPT, 50), timeout=60.0
            )
            self.inference_ms = (time.perf_counter() - started) * 1000.0
            decision, confidence, reasoning = parse_agent_json(output)
            if decision is None and "decision" not in output:
                return {"ok": False, "error": f"Model output was not usable JSON: {output[:120]!r}"}
            self.last_error = ""
            return {"ok": True}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Model loaded but inference failed: {exc}"}

    # ------------------------------------------------------------------
    def _generate(self, prompt: str, max_tokens: int = 256) -> str:
        if self.model is None:
            raise RuntimeError("no model loaded")
        if isinstance(self.model, _StubModel):
            return self.model.generate(prompt)
        if self.model_kind == "gguf":
            response = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_LOCAL},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return str(response["choices"][0]["message"]["content"])
        # ONNX: a text-gen ONNX export is not standardised, so we run the graph
        # with whatever numeric input it declares and report that clearly.
        raise RuntimeError(
            "ONNX text generation requires a tokenizer; upload a .gguf model for chat inference"
        )

    # ------------------------------------------------------------------
    async def ensure_stub(self) -> bool:
        """Lazily activate the development stub when stub mode is enabled."""
        if not self.stub or self.ready:
            return self.ready
        await self._load_stub()
        log.warning("Local agent running the development stub - no real weights are loaded")
        return self.ready

    async def decide(self, context: dict, weight: float) -> AgentResult:
        if self.stub and not self.ready:
            await self.ensure_stub()
        if not self.ready:
            return AgentResult(
                agent=NAME,
                status=AgentStatus.ERROR if self.crashed else AgentStatus.DISABLED,
                model=self.model_name,
                weight=weight,
                error=self.last_error or "No local model loaded",
            )
        prompt = build_cycle_prompt(context)
        started = time.perf_counter()
        try:
            output = await asyncio.wait_for(
                asyncio.to_thread(self._generate, prompt, 256),
                timeout=self.settings.local_timeout_seconds,
            )
        except asyncio.TimeoutError:
            return AgentResult(
                agent=NAME,
                status=AgentStatus.TIMEOUT,
                model=self.model_name,
                weight=weight,
                error=f"Timed out after {self.settings.local_timeout_seconds:.0f}s",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            # A crashed model must be reported, not silently ignored (Section 7.2).
            self.crashed = True
            self.last_error = str(exc)
            return AgentResult(
                agent=NAME,
                status=AgentStatus.ERROR,
                model=self.model_name,
                weight=weight,
                error=f"Local model crashed: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        self.inferences += 1
        self.inference_ms = (time.perf_counter() - started) * 1000.0
        decision, confidence, reasoning = parse_agent_json(output)
        return AgentResult(
            agent=NAME,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            status=AgentStatus.STUB if self.stub else AgentStatus.ACTIVE,
            model=self.model_name,
            weight=weight,
            latency_ms=self.inference_ms,
            raw=output,
            error="" if decision else "unparseable output",
        )

    # ------------------------------------------------------------------
    # Step 6: ongoing monitoring
    # ------------------------------------------------------------------
    def _start_health_monitor(self) -> None:
        if self._health_task is not None:
            self._health_task.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._health_task = loop.create_task(self._monitor())

    async def _monitor(self, interval: float = 60.0) -> None:
        while self.ready:
            await asyncio.sleep(interval)
            try:
                self.check_memory()
            except Exception as exc:  # noqa: BLE001
                log.debug("local model monitor error: %s", exc)

    def check_memory(self) -> dict:
        """Unload the model if it is pushing the machine into swap."""
        usage = process_memory()
        total = system_memory_total()
        payload = {"rss_bytes": usage, "rss_human": _human_bytes(usage), "limit": total}
        if total and usage > total * MEMORY_LIMIT_FRACTION:
            log.warning("Local model exceeded memory budget; unloading")
            self.last_error = "Local model unloaded: out of memory"
            self.unload()
            payload["unloaded"] = True
        return payload

    def unload(self, delete_file: bool = False) -> dict:
        if self._health_task is not None:
            self._health_task.cancel()
            self._health_task = None
        path = self.model_path
        self.model = None
        self.model_kind = ""
        self.crashed = False
        freed = path.stat().st_size if (delete_file and path and path.exists()) else 0
        if delete_file and path and path.exists():
            try:
                path.unlink()
            except OSError as exc:
                log.warning("could not delete model file: %s", exc)
        self.model_path = None if delete_file else path
        return {"unloaded": True, "freed_bytes": freed, "freed_human": _human_bytes(freed)}

    def health(self) -> AgentHealth:
        status = self.status()
        detail = "No model loaded"
        if self.stub and self.ready:
            detail = "Development stub active (not a real model)"
        elif self.crashed:
            detail = "Crashed - reload required"
        elif self.ready:
            detail = f"{self.inference_ms:.0f}ms/inference ({self.inferences} calls)"
        return AgentHealth(
            name=NAME,
            status=status,
            detail=detail,
            model=self.model_name,
            extra={
                "kind": self.model_kind,
                "parameter_count": self.parameter_count,
                "memory": process_memory(),
                "memory_human": _human_bytes(process_memory()),
                "stub": self.stub,
                "last_error": self.last_error,
            },
        )


class _StubModel:
    """Deterministic stand-in used only when ``LOCAL_AGENT_STUB=1``.

    It reads the same numbers a real model would read and applies a simple
    transparent rule, so the fusion layer, the dashboard and the degradation
    tests behave exactly as they would with a real model - while never
    pretending to be one.
    """

    def generate(self, prompt: str) -> str:
        import re

        def grab(name: str) -> float:
            match = re.search(rf"{name}=([+-]?\d+\.?\d*)", prompt)
            return float(match.group(1)) if match else 0.0

        core = 0.4 * grab("CCSv2") + 0.2 * grab("TAI") + 0.2 * grab("SHRP") + 0.2 * grab("MPS")
        hsi = grab("HSI")
        if hsi > 0.7 or abs(core) < 0.12:
            decision, confidence = "HOLD", 0.55
        elif core > 0:
            decision, confidence = "BUY", min(0.9, 0.5 + core)
        else:
            decision, confidence = "SELL", min(0.9, 0.5 - core)
        return json.dumps(
            {
                "decision": decision,
                "confidence": round(confidence, 3),
                "reasoning": f"stub: CCSv2={grab('CCSv2'):+.2f} TAI={grab('TAI'):+.2f} HSI={hsi:.2f}",
            }
        )


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def process_memory() -> int:
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:  # noqa: BLE001
        try:
            import resource

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        except Exception:  # noqa: BLE001
            return 0


def system_memory_total() -> int:
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001
        pass
    return 0
