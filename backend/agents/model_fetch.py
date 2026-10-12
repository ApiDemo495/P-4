"""Server-side model download (Round AU).

A 1 GB GGUF pushed through the browser dies in the Codespaces port-forwarding
proxy (it has body-size and time limits that return 5xx long before uvicorn
sees the end of the file).  So the engine fetches the file itself: paste a
Hugging Face URL (or ``hf://org/repo/file.gguf``), the download streams to
``backend/models/<name>.part`` with live progress, is renamed when complete,
validated by magic bytes and loaded into the requested slot.  One download at
a time; a second request while one is running is refused with the progress
of the first.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import httpx

from backend.agents.local_model_agent import _human_bytes

_HF_SHORT = re.compile(r"^hf://([^/]+)/([^/]+)/(.+)$")
_HF_BLOB = re.compile(r"^(https://huggingface\.co/[^/]+/[^/]+)/blob/([^/]+)/(.+)$")
_ALLOWED_SUFFIX = (".gguf", ".onnx")


def normalise_url(raw: str) -> str:
    """Accept the forms people actually paste and return a direct file URL.

    * ``hf://Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf``
    * ``https://huggingface.co/<org>/<repo>/blob/main/<file>`` (the page URL)
    * ``https://huggingface.co/<org>/<repo>/resolve/main/<file>`` (direct)
    * any other https URL ending in .gguf / .onnx
    """
    url = (raw or "").strip()
    m = _HF_SHORT.match(url)
    if m:
        url = f"https://huggingface.co/{m.group(1)}/{m.group(2)}/resolve/main/{m.group(3)}"
    m = _HF_BLOB.match(url)
    if m:
        url = f"{m.group(1)}/resolve/{m.group(2)}/{m.group(3)}"
    if not url.startswith(("https://", "http://")):
        raise ValueError("Paste a full https:// URL or hf://org/repo/file.gguf")
    name = Path(url.split("?", 1)[0]).name
    if not name.lower().endswith(_ALLOWED_SUFFIX):
        raise ValueError("The URL must point at a .gguf or .onnx file (on Hugging Face use the 'resolve' link of the file, or paste the file page URL)")
    return url


class ModelFetcher:
    """One background download at a time, with a progress dict for the UI."""

    def __init__(self, pool, store_dir: Path) -> None:
        self.pool = pool
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.task: asyncio.Task | None = None
        self.state: dict = {"active": False, "phase": "idle"}

    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.task is not None and not self.task.done()

    def status(self) -> dict:
        return dict(self.state)

    def files_on_disk(self) -> list[dict]:
        """Complete model files already in ``backend/models`` (drag-and-drop
        into the VS Code explorer lands here and bypasses the proxy)."""
        out = []
        for path in sorted(self.store_dir.iterdir()) if self.store_dir.exists() else []:
            if path.is_file() and path.suffix.lower() in _ALLOWED_SUFFIX:
                st = path.stat()
                out.append({"file": path.name, "size_bytes": st.st_size, "size_human": _human_bytes(st.st_size),
                            "modified": st.st_mtime})
        return out

    def start(self, raw_url: str, slot: int) -> dict:
        if self.active:
            return {"started": False, "error": "a download is already running", "fetch": self.status()}
        url = normalise_url(raw_url)
        name = Path(url.split("?", 1)[0]).name
        self.state = {
            "active": True, "phase": "connecting", "url": url, "file": name, "slot": int(slot),
            "bytes": 0, "total": None, "percent": None, "speed_bps": 0.0, "eta_seconds": None,
            "started_at": time.time(), "error": "", "result": None,
        }
        self.task = asyncio.create_task(self._run(url, name, int(slot)))
        return {"started": True, "fetch": self.status()}

    def cancel(self) -> dict:
        if self.active:
            self.task.cancel()
            self.state.update({"active": False, "phase": "cancelled"})
        return self.status()

    # ------------------------------------------------------------------
    async def _run(self, url: str, name: str, slot: int) -> None:
        target = self.store_dir / name
        part = self.store_dir / (name + ".part")
        try:
            # Resume if a partial file from an interrupted attempt is there.
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
            async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
                async with client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code == 416:            # range beyond the file: start over
                        offset = 0
                        part.unlink(missing_ok=True)
                        raise _Restart()
                    if resp.status_code >= 400:
                        raise RuntimeError(f"HTTP {resp.status_code} from {resp.url.host}"
                                           + (" - gated model: accept the licence on Hugging Face and use a public mirror, or download it yourself into backend/models/"
                                              if resp.status_code in (401, 403) else ""))
                    if resp.status_code != 206:
                        offset = 0
                    total = resp.headers.get("content-length")
                    total_bytes = (int(total) + offset) if total else None
                    self.state.update({"phase": "downloading", "total": total_bytes, "bytes": offset})
                    done = offset
                    t0, b0 = time.time(), offset
                    mode = "ab" if offset else "wb"
                    with part.open(mode) as handle:
                        async for chunk in resp.aiter_bytes(1 << 20):
                            handle.write(chunk)
                            done += len(chunk)
                            now = time.time()
                            if now - t0 >= 0.5:
                                speed = (done - b0) / (now - t0)
                                self.state.update({
                                    "bytes": done,
                                    "percent": round(100.0 * done / total_bytes, 1) if total_bytes else None,
                                    "speed_bps": speed,
                                    "eta_seconds": round((total_bytes - done) / speed) if (total_bytes and speed > 0) else None,
                                })
                                t0, b0 = now, done
                    self.state.update({"bytes": done, "percent": 100.0 if total_bytes else None})
            part.replace(target)
            self.state.update({"phase": "validating"})
            result = await self.pool.load_path(target, slot)
            self.state.update({"phase": "done" if result.get("success") else "load_failed",
                               "result": result, "error": "" if result.get("success") else result.get("error", "")})
        except _Restart:
            self.state.update({"active": False})
            self.task = asyncio.create_task(self._run(url, name, slot))
            return
        except asyncio.CancelledError:
            self.state.update({"phase": "cancelled", "error": "cancelled (the .part file is kept so the next attempt resumes)"})
        except Exception as exc:  # noqa: BLE001
            self.state.update({"phase": "failed", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            self.state["active"] = False


class _Restart(Exception):
    pass
