"""Round BQ - the jevos decision agent (feder-cr/jev), the primary AI agent.

`jev <https://github.com/feder-cr/jev>`_ is Federico Elia's open-source
CPU decision model: one native binary serving ``POST /v1/systemone`` that
reads a *state* once and answers yes/no (``noul``), multiple-choice and
score questions **with probabilities read from the option logits** - no
text is generated, so there is nothing to parse and nothing to hallucinate,
and a short request answers in tens of milliseconds on a laptop CPU.

That shape fits the Signal Lock Protocol better than a chat model: the
window state (every formula, the book lean, the news, the brain consensus)
is sent once, several rule-questions are asked about it in the same call,
and each answer is a calibrated probability that lands long before the
8-second lock deadline.

What this module does
---------------------
* ``JevInstaller`` - downloads the ``jevos-v5`` release for this platform
  (the ~27 MB binary archive and the ~630 MB OpenVINO INT8 model) from
  GitHub Releases into ``backend/models/jev/`` with resume, **verifies both
  against the release's SHA256SUMS.txt** (Round AV: nothing runs before it
  is validated) and unpacks them.  Automatic (zero-command rule), opt out
  with ``JEV_AUTOINSTALL=0``.
* ``JevServer`` - owns the ``jev serve`` child process (127.0.0.1 only,
  restarted if it dies, stopped with the engine).  ``JEV_URL`` points the
  agent at a server you run yourself instead.
* ``JevAgent`` - the orchestrator-facing agent: ``decide`` / ``health`` /
  ``test`` like the Gemini and local agents, so fusion, hindsight, the
  meta-combiner, the lock-weight table and the UI treat it as one more
  graded voter (``agent:jev``).

Honesty notes (Round AM/AS): jevos is a general decision model trained on
rule-and-scenario tasks, not on markets.  Its edge here is that it applies
the stated rules to the stated evidence consistently and fast; whether
that beats the other voters is measured by the hourly hindsight review and
the meta-combiner, not assumed.  The sandbox the app is built in cannot
reach GitHub's release storage - a fresh Codespace can, and the installer
reports every step on the agent row.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import platform
import shutil
import ssl
import sys
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path

import httpx

from backend.agents.base import AgentHealth, AgentResult, AgentStatus
from backend.core import config as cfg

log = logging.getLogger("drosophila.agents.jev")

RELEASE_TAG = "jevos-v5"
RELEASE_BASE = f"https://github.com/feder-cr/jev/releases/download/{RELEASE_TAG}"
MODEL_ARCHIVE = f"{RELEASE_TAG}-openvino-int8.zip"
SUMS_FILE = "SHA256SUMS.txt"
DEFAULT_PORT = 8017
TIMEOUT_S = 9.0          # same budget as the other agents (prefetch lead 10.5 s)
HEALTH_TIMEOUT_S = 2.5
MODEL_FILES = ("model.json", "openvino_model.xml", "openvino_model.bin", "tokenizer.gguf")

#: The formulas each rule-question reads (the ``WINDOW`` set and the groups of
#: the 31 main formulas).  Each tuple: question id, sign (+1 = "yes" means up,
#: -1 = "yes" means down), formulas, the rule sentence (jev's design: put the
#: rule in the question, do the sums in code).
RULES: tuple[tuple[str, int, tuple[str, ...], str], ...] = (
    ("continuation", +1, ("TAI", "MPL", "FIH", "RBB", "AFPR", "MCPE"),
     "Rule: when most of the micro-structure and window formulas TAI, MPL, FIH, RBB, AFPR and MCPE are "
     "positive, the last trade {w} seconds from now is usually above the price now. "
     "Given the state, will the price be higher in {w} seconds?"),
    ("flow", +1, ("DGW", "LCS", "BAR", "VSD"),
     "Rule: positive order-book formulas DGW, LCS, BAR and VSD mean the buyers are lifting offers and the "
     "price drifts up; negative ones mean sellers hit bids. Do the buyers dominate this book?"),
    ("exhaustion", -1, ("RAR", "PRT", "WDR", "ERC", "LAI", "ANA"),
     "Rule: a move that is stretched (RAR, PRT and WDR strongly positive while ERC says choppy) snaps "
     "back down within the window; a stretched fall snaps back up. Is a move down more likely than a move "
     "up in the next {w} seconds?"),
    ("news", +1, ("NIV", "SMD"),
     "Rule: a positive news impact NIV with a bullish latest headline lifts the price inside the window, "
     "a negative one pushes it down; near zero the news is neutral. Does the news push the price up?"),
    ("brain", +1, ("CCSv2", "KCAE", "DSKD"),
     "Rule: the Drosophila mushroom-body consensus CCSv2 above zero is a BUY vote and below zero a SELL "
     "vote; KCAE is how strongly the circuit fired. Is the brain consensus bullish?"),
)

GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("micro-structure", ("TAI", "AFPR", "SED", "VSD")),
    ("order book", ("DGW", "LCS", "BAR")),
    ("hedge", ("HRDD", "SHRP", "GCDV", "HSI")),
    ("regime", ("RSV", "VSS", "ERC")),
    ("temporal", ("MCPE", "MPS", "TWRS")),
    ("kalman", ("DSKD",)),
    ("window", ("PIN", "MPL", "RBB", "RAR", "WDR", "PRT", "FIH", "LAI", "ANA")),
    ("news", ("NIV", "SMD")),
    ("brain", ("KCAE", "CCSv2")),
)


# ---------------------------------------------------------------------------
# paths and platform
# ---------------------------------------------------------------------------
def jev_home() -> Path:
    raw = os.environ.get("JEV_HOME", "").strip()
    return Path(raw) if raw else Path(cfg.MODEL_STORE_DIR) / "jev"


def platform_archive() -> str | None:
    """The release asset for this machine, or None when none is published."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine in ("x86_64", "amd64"):
        return "jev-linux-x64.tar.gz"
    if system == "darwin" and machine in ("arm64", "aarch64"):
        return "jev-macos-arm64.tar.gz"
    if system == "windows" and machine in ("amd64", "x86_64"):
        return "jev-windows-x64.zip"
    return None


def find_binary(home: Path) -> Path | None:
    for name in ("jev", "jev.exe"):
        for candidate in (home / name, home / "jev" / name):
            if candidate.is_file():
                return candidate
    if home.is_dir():
        for candidate in home.rglob("jev*"):
            if candidate.is_file() and candidate.name in ("jev", "jev.exe"):
                return candidate
    return None


def find_model_dir(home: Path) -> Path | None:
    for candidate in (home / "model", home / "jev" / "model"):
        if all((candidate / f).is_file() for f in MODEL_FILES):
            return candidate
    if home.is_dir():
        for meta in home.rglob("model.json"):
            if all((meta.parent / f).is_file() for f in MODEL_FILES):
                return meta.parent
    return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sums(text: str) -> dict[str, str]:
    """``<hash>  <name>`` lines -> {name: hash} (the release's SHA256SUMS.txt)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            out[parts[-1].lstrip("*")] = parts[0].lower()
    return out


# ---------------------------------------------------------------------------
# installer
# ---------------------------------------------------------------------------
STATE_FILE = "install.json"


class _State(dict):
    """The installer's progress dict; every update is mirrored to
    ``home/install.json`` so the engine (another process) can read it."""

    def __init__(self, path: Path | None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.path = path

    def update(self, *args, **kwargs) -> None:  # type: ignore[override]
        super().update(*args, **kwargs)
        self.flush()

    def flush(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(self), default=str))
            tmp.replace(self.path)
        except OSError:
            pass


def read_install_state(home: Path) -> dict:
    try:
        return json.loads((home / STATE_FILE).read_text())
    except (OSError, ValueError):
        return {"active": False, "phase": "idle", "error": "", "asset": "", "percent": None}


class JevInstaller:
    """Fetch + verify + unpack the release into ``home``.

    Round BR: it runs in its **own process** at the lowest CPU/IO priority
    (``nice -n 19 ionice -c 3 python -m backend.agents.jev_agent install``):
    hashing and unpacking 630 MB inside the engine process starved the window
    loop on a 2-core Codespace (countdown lag, late picks).  Progress goes
    through ``home/install.json``.
    """

    def __init__(self, home: Path, persist: bool = True) -> None:
        self.home = home
        self.state: dict = _State(home / STATE_FILE if persist else None,
                                  {"active": False, "phase": "idle", "error": "", "asset": "",
                                   "bytes": 0, "total": None, "percent": None, "pid": os.getpid()})
        self.task: asyncio.Task | None = None

    def start(self) -> None:
        if self.task is None or self.task.done():
            self.state.update({"active": True, "phase": "starting", "error": ""})
            self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        archive = platform_archive()
        try:
            if archive is None:
                raise RuntimeError(f"no jev release for {platform.system()}/{platform.machine()} - build it from source (docs/API_KEYS_AND_MODELS.md)")
            self.home.mkdir(parents=True, exist_ok=True)
            verify: bool | ssl.SSLContext = True
            try:
                sums = await self._fetch_sums(verify)
            except httpx.HTTPError as exc:
                if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
                    raise
                verify = ssl.create_default_context()
                sums = await self._fetch_sums(verify)
            for asset in (archive, MODEL_ARCHIVE):
                target = self.home / asset
                if not (target.exists() and sha256_of(target) == sums.get(asset)):
                    await self._download(asset, target, verify)
                    self.state.update({"phase": "verifying", "asset": asset})
                    digest = await asyncio.to_thread(sha256_of, target)
                    expected = sums.get(asset)
                    if expected and digest != expected:
                        target.unlink(missing_ok=True)
                        raise RuntimeError(f"{asset}: SHA-256 {digest[:12]}… does not match the release's {expected[:12]}… - deleted, will retry")
                self.state.update({"phase": "unpacking", "asset": asset})
                await asyncio.to_thread(self._unpack, target)
            if find_binary(self.home) is None or find_model_dir(self.home) is None:
                raise RuntimeError("archives unpacked but jev binary or model/ folder not found")
            self.state.update({"phase": "done", "error": ""})
        except asyncio.CancelledError:
            self.state.update({"phase": "cancelled"})
        except Exception as exc:  # noqa: BLE001
            self.state.update({"phase": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]})
        finally:
            self.state.update({"active": False})

    async def _fetch_sums(self, verify) -> dict[str, str]:
        self.state.update({"phase": "fetching checksums", "asset": SUMS_FILE})
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0, verify=verify) as client:
            resp = await client.get(f"{RELEASE_BASE}/{SUMS_FILE}")
            resp.raise_for_status()
        sums = parse_sums(resp.text)
        if not sums:
            raise RuntimeError("SHA256SUMS.txt empty or unreadable")
        return sums

    async def _download(self, asset: str, target: Path, verify) -> None:
        part = target.with_suffix(target.suffix + ".part")
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
        self.state.update({"phase": "downloading", "asset": asset, "bytes": offset, "total": None, "percent": None})
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, verify=verify) as client:
            async with client.stream("GET", f"{RELEASE_BASE}/{asset}", headers=headers) as resp:
                if resp.status_code == 416:
                    part.unlink(missing_ok=True)
                    return await self._download(asset, target, verify)
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code} from {resp.url.host} for {asset}")
                if resp.status_code != 206:
                    offset = 0
                total = resp.headers.get("content-length")
                total_bytes = (int(total) + offset) if total else None
                self.state.update({"total": total_bytes})
                done, t0, b0 = offset, time.time(), offset
                with part.open("ab" if offset else "wb") as handle:
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
        part.replace(target)
        self.state.update({"bytes": done, "percent": 100.0 if total_bytes else None})

    def _unpack(self, archive: Path) -> None:
        dest = self.home
        if archive.name.endswith(".tar.gz"):
            with tarfile.open(archive, "r:gz") as tar:
                members = [m for m in tar.getmembers() if not (m.name.startswith("/") or ".." in m.name)]
                tar.extractall(dest, members=members)  # noqa: S202 - filtered above
        elif archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    if info.filename.startswith("/") or ".." in info.filename:
                        continue
                    zf.extract(info, dest)
        binary = find_binary(dest)
        if binary is not None:
            binary.chmod(binary.stat().st_mode | 0o111)


# ---------------------------------------------------------------------------
# server child
# ---------------------------------------------------------------------------
class JevServer:
    def __init__(self, home: Path, port: int, log_path: Path | None = None) -> None:
        self.home = home
        self.port = port
        self.log_path = log_path or (Path(cfg.REPO_ROOT) / "logs" / "jev.log" if hasattr(cfg, "REPO_ROOT") else home / "jev.log")
        self.proc: subprocess.Popen | None = None
        self.started_at = 0.0
        self.launches = 0
        self.last_error = ""

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> bool:
        binary, model_dir = find_binary(self.home), find_model_dir(self.home)
        if binary is None or model_dir is None:
            self.last_error = "binary or model folder missing"
            return False
        if self.alive:
            return True
        cores = os.cpu_count() or 2
        forced = int(os.environ.get("JEV_THREADS", "0") or 0)
        # Round BR: one core always stays with the engine's event loop (the
        # countdown and the lock), and the server runs at nice 10 so a busy
        # model compile can never delay a boundary.
        threads = max(1, min(cores, forced or max(1, cores - 1)))
        cmd = [str(binary), "serve", "--host", "127.0.0.1", "--port", str(self.port),
               "--model-dir", str(model_dir), "--threads", str(threads)]
        if shutil.which("nice"):
            cmd = ["nice", "-n", str(int(os.environ.get("JEV_NICE", "10") or 10))] + cmd
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.log_path.open("ab")
            self.proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         start_new_session=True, cwd=str(binary.parent))
            self.started_at = time.time()
            self.launches += 1
            self.last_error = ""
            log.info("jev serve launched (pid %s, %s threads, port %s)", self.proc.pid, threads, self.port)
            return True
        except OSError as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        except OSError:
            pass
        self.proc = None


# ---------------------------------------------------------------------------
# agent
# ---------------------------------------------------------------------------
def _lean(value: float) -> str:
    if value > 0.5:
        return "strongly up"
    if value > 0.1:
        return "up"
    if value < -0.5:
        return "strongly down"
    if value < -0.1:
        return "down"
    return "flat"


def build_state(context: dict) -> dict:
    """The *state* jev reads once: every formula with its sign spelled out,
    grouped as the UI groups them, plus the tally, the headline and the brain."""
    f = {k: float(v) for k, v in (context.get("formulas") or {}).items() if isinstance(v, (int, float))}
    w = int(round(cfg.window_seconds()))
    groups = {}
    for group, names in GROUPS:
        groups[f"formulas {group}"] = ", ".join(f"{n} {f[n]:+.3f} ({_lean(f[n])})" for n in names if n in f) or "no data"
    directional = [n for n in f if n not in ("PIN", "HSI", "KCAE", "ERC") and abs(f[n]) > 0.05]
    ups = sum(1 for n in directional if f[n] > 0)
    downs = len(directional) - ups
    return {
        "asset": f"{context.get('asset', 'BTC')}/USDT",
        "price": round(float(context.get("price") or 0.0), 2),
        "rule_of_the_window": f"the call covers exactly the next {w} seconds; it wins only if the last traded price is strictly "
                  f"above (up) or strictly below (down) the price now; unchanged is a loss",
        "tally": f"{ups} formulas point up, {downs} point down, out of {len(directional)} that have a direction",
        "tie_risk": f"PIN (probability the window ends unchanged) {f.get('PIN', 0.0):.2f}; hedge stress HSI {f.get('HSI', 0.0):.2f}",
        **groups,
        "latest_headline": str(context.get("headline") or "none"),
        "brain_consensus": f"CCSv2 {f.get('CCSv2', 0.0):+.3f} ({_lean(f.get('CCSv2', 0.0))}), confidence "
                           f"{float(context.get('ccs_confidence') or 0.0):.0%}, status {context.get('brain_status', 'UNKNOWN')}",
        "recent_performance": f"DRG {float(context.get('drg') or 0.0):+.3f}; recent win rate {float(context.get('win_rate') or 0.0):.0%}",
    }


def build_questions() -> dict:
    w = int(round(cfg.window_seconds()))
    qs = {qid: {"type": "noul", "instructions": rule.format(w=w)} for qid, _sign, _names, rule in RULES}
    qs["side"] = {
        "type": "choice",
        "instructions": f"Weigh every group of the state. Where is the last traded price {w} seconds from now?",
        "criteria": {"up": f"strictly above the price now after {w} seconds",
                     "down": f"strictly below (or unchanged, which also loses an up call) after {w} seconds"},
    }
    return qs


def combine(answers: dict) -> tuple[float, dict[str, float]]:
    """P(up) from the answers: half the direct choice, half the mean of the
    rule-questions (each oriented so that 1 = up)."""
    parts: dict[str, float] = {}
    for qid, sign, _names, _rule in RULES:
        ans = answers.get(qid) or {}
        p = ans.get("noul")
        if isinstance(p, (int, float)):
            p = min(1.0, max(0.0, float(p)))
            parts[qid] = p if sign > 0 else 1.0 - p
    side = answers.get("side") or {}
    probs = side.get("probabilities") or {}
    p_side = probs.get("up")
    if isinstance(p_side, (int, float)):
        parts["side"] = min(1.0, max(0.0, float(p_side)))
    if not parts:
        return 0.5, parts
    rules = [v for k, v in parts.items() if k != "side"]
    if "side" in parts and rules:
        p_up = 0.5 * parts["side"] + 0.5 * (sum(rules) / len(rules))
    elif rules:
        p_up = sum(rules) / len(rules)
    else:
        p_up = parts["side"]
    return p_up, parts


class JevAgent:
    name = "jev"

    def __init__(self, settings=None, home: Path | None = None, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.home = Path(home) if home else jev_home()
        self.external_url = os.environ.get("JEV_URL", "").strip().rstrip("/")
        self.port = int(os.environ.get("JEV_PORT", str(DEFAULT_PORT)) or DEFAULT_PORT)
        self.api_key = os.environ.get("JEV_API_KEY", "").strip()
        self.url = self.external_url or f"http://127.0.0.1:{self.port}"
        self.autoinstall = os.environ.get("JEV_AUTOINSTALL", "1").strip().lower() not in ("0", "false", "no")
        self.installer = JevInstaller(self.home, persist=False)   # in-process fallback only
        self._install_proc: subprocess.Popen | None = None
        self.server = JevServer(self.home, self.port)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self.ready = False
        self.model = ""
        self.fingerprint = ""
        self.calls = 0
        self.failures = 0
        self.last_error = ""
        self.last_latency_ms = 0.0
        self.last_parts: dict[str, float] = {}
        self._task: asyncio.Task | None = None
        self._retry_at = 0.0

    # ------------------------------------------------------------------
    @property
    def installed(self) -> bool:
        return find_binary(self.home) is not None and find_model_dir(self.home) is not None

    @property
    def configured(self) -> bool:
        return bool(self.external_url) or self.installed or self.autoinstall

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT_S, connect=2.0),
                                             transport=self._transport, headers=self._headers())
        return self._client

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Background supervisor: install if missing, keep the server up,
        probe /health.  Never blocks the event loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._supervise())

    async def _supervise(self) -> None:
        while True:
            try:
                if not self.external_url:
                    if not self.installed:
                        if self.autoinstall and not self._install_running():
                            phase = self.install_state().get("phase")
                            # a failed download is retried every two minutes, not every tick
                            if phase in (None, "idle", "starting") or (phase in ("failed", "cancelled", "done") and time.time() >= self._retry_at):
                                self._retry_at = time.time() + 120.0
                                self._spawn_installer()
                    elif not self.server.alive:
                        self.server.start()
                await self.probe()
            except Exception as exc:  # noqa: BLE001 - the supervisor must never die
                self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            await asyncio.sleep(15.0 if self.ready else 5.0)

    # -- the installer child ---------------------------------------------
    def _install_running(self) -> bool:
        return self._install_proc is not None and self._install_proc.poll() is None

    def install_state(self) -> dict:
        st = read_install_state(self.home)
        st["active"] = bool(self._install_running())
        return st

    def _spawn_installer(self) -> None:
        """Download/verify/unpack in a separate process at the lowest priority."""
        cmd = [sys.executable, "-m", "backend.agents.jev_agent", "install", "--home", str(self.home)]
        if shutil.which("ionice"):
            cmd = ["ionice", "-c", "3"] + cmd
        if shutil.which("nice"):
            cmd = ["nice", "-n", "19"] + cmd
        try:
            self.home.mkdir(parents=True, exist_ok=True)
            log_path = Path(cfg.REPO_ROOT) / "logs" / "jev_install.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("ab")
            self._install_proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                                  start_new_session=True, cwd=str(cfg.REPO_ROOT))
            _State(self.home / STATE_FILE).update({"active": True, "phase": "starting", "error": "", "pid": self._install_proc.pid})
        except OSError as exc:
            self.last_error = f"installer could not start: {exc}"[:200]
            _State(self.home / STATE_FILE).update({"active": False, "phase": "failed", "error": self.last_error})

    async def shutdown(self) -> None:
        if self._task is not None:
            self._task.cancel()
        if self._install_running():
            try:
                self._install_proc.terminate()
            except OSError:
                pass
        self.server.stop()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def probe(self) -> dict:
        try:
            resp = await self.client().get(f"{self.url}/health", timeout=HEALTH_TIMEOUT_S)
            resp.raise_for_status()
            body = resp.json()
            self.ready = body.get("status") == "ready"
            self.model = str(body.get("model") or "")
            self.fingerprint = str(body.get("fingerprint") or "")[:16]
            return {"ok": self.ready, "body": body}
        except (httpx.HTTPError, ValueError) as exc:
            self.ready = False
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}

    async def test(self) -> dict:
        """``/api/agents/jev/test`` - the same probe, phrased for the settings page."""
        result = await self.probe()
        if result.get("ok"):
            return {"valid": True, "detail": f"{self.model} ready at {self.url} (fingerprint {self.fingerprint}…)", "model": self.model}
        return {"valid": False, "detail": self._why_not_ready(result.get("error", ""))}

    def _why_not_ready(self, error: str = "") -> str:
        st = self.install_state()
        if self.external_url:
            return f"JEV_URL {self.url} not answering: {error or 'no response'}"
        if not self.installed:
            if st.get("active"):
                pct = f" {st['percent']}%" if st.get("percent") is not None else ""
                return f"installing: {st.get('phase')} {st.get('asset', '')}{pct}".strip()
            if st.get("phase") == "failed":
                return f"install failed: {st.get('error')}"
            if not self.autoinstall:
                return "JEV_AUTOINSTALL=0 and no install in backend/models/jev"
            return "not installed yet (download starts with the engine)"
        if not self.server.alive:
            return f"server not running: {self.server.last_error or 'starting'}"
        return f"server starting (compiling the model, ~20-60 s): {error}"

    # ------------------------------------------------------------------
    async def decide(self, context: dict, weight: float) -> AgentResult:
        started = time.perf_counter()
        if not self.configured:
            return AgentResult(agent=self.name, status=AgentStatus.DISABLED, weight=weight, error="not installed")
        if not self.ready:
            status = AgentStatus.LOADING if (self._install_running() or self.server.alive) else AgentStatus.INACTIVE
            return AgentResult(agent=self.name, status=status, weight=weight, error=self._why_not_ready())
        body = {"model": "jev-latest", "state": build_state(context), "questions": build_questions()}
        try:
            resp = await self.client().post(f"{self.url}/v1/systemone", json=body)
            latency = (time.perf_counter() - started) * 1000.0
            self.last_latency_ms = latency
            if resp.status_code == 429:
                return AgentResult(agent=self.name, status=AgentStatus.RATE_LIMITED, weight=weight, latency_ms=latency, error="429")
            if resp.status_code >= 400:
                self.failures += 1
                self.last_error = f"HTTP {resp.status_code}: {resp.text[:160]}"
                return AgentResult(agent=self.name, status=AgentStatus.ERROR, weight=weight, latency_ms=latency, error=self.last_error)
            payload = resp.json()
        except httpx.TimeoutException:
            self.failures += 1
            self.last_error = f"timeout after {TIMEOUT_S:.0f} s"
            return AgentResult(agent=self.name, status=AgentStatus.TIMEOUT, weight=weight,
                               latency_ms=(time.perf_counter() - started) * 1000.0, error=self.last_error)
        except (httpx.HTTPError, ValueError) as exc:
            self.failures += 1
            self.ready = False
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            return AgentResult(agent=self.name, status=AgentStatus.ERROR, weight=weight,
                               latency_ms=(time.perf_counter() - started) * 1000.0, error=self.last_error)

        answers = payload.get("answers") or {}
        p_up, parts = combine(answers)
        self.last_parts = parts
        self.calls += 1
        self.model = str(payload.get("model") or self.model)
        if not parts:
            self.last_error = "no answers in reply"
            return AgentResult(agent=self.name, status=AgentStatus.ERROR, weight=weight, latency_ms=self.last_latency_ms, error=self.last_error)
        # An exact 0.5 has no side: lean with the formula tally so the agent
        # still names one (HOLD is not a vote) - at the lowest confidence.
        if abs(p_up - 0.5) < 1e-9:
            f = context.get("formulas") or {}
            p_up += 1e-6 * (1 if float(f.get("MPL", 0.0) or 0.0) + float(f.get("TAI", 0.0) or 0.0) >= 0 else -1)
        decision = "BUY" if p_up > 0.5 else "SELL"
        confidence = max(0.02, min(0.95, abs(2.0 * p_up - 1.0)))
        detail = " ".join(f"{k}={v:.2f}" for k, v in parts.items())
        reasoning = (f"jevos P(up)={p_up:.3f} from {len(parts)} rule answers [{detail}] "
                     f"in {self.last_latency_ms:.0f} ms; no text generated, probabilities read from the option logits.")
        return AgentResult(agent=self.name, decision=decision, confidence=confidence, reasoning=reasoning[:400],
                           status=AgentStatus.ACTIVE, latency_ms=self.last_latency_ms, model=self.model,
                           weight=weight, raw=json.dumps(answers)[:2000])

    # ------------------------------------------------------------------
    def health(self) -> AgentHealth:
        st = self.install_state()
        extra = {
            "calls": self.calls, "failures": self.failures, "last_error": self.last_error,
            "url": self.url, "installed": self.installed, "fingerprint": self.fingerprint,
            "install": {k: st.get(k) for k in ("phase", "asset", "percent", "error", "eta_seconds")},
            "server_alive": self.server.alive, "server_launches": self.server.launches,
            "last_parts": {k: round(v, 3) for k, v in self.last_parts.items()},
            "latency_ms": round(self.last_latency_ms, 1),
            "release": RELEASE_TAG,
        }
        if not self.configured:
            return AgentHealth(self.name, AgentStatus.DISABLED, "not installed (JEV_AUTOINSTALL=0)", extra=extra)
        if self.ready:
            return AgentHealth(self.name, AgentStatus.ACTIVE,
                               f"{self.model or 'jevos'} · {self.calls} calls · {self.last_latency_ms:.0f} ms", model=self.model, extra=extra)
        if st.get("phase") == "failed" or (self.installed and self.server.last_error):
            return AgentHealth(self.name, AgentStatus.ERROR, self._why_not_ready(), extra=extra)
        return AgentHealth(self.name, AgentStatus.LOADING, self._why_not_ready(), extra=extra)


# ---------------------------------------------------------------------------
# ``python -m backend.agents.jev_agent install --home DIR`` - the low-priority
# installer process the engine spawns (see JevAgent._spawn_installer).
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="jev_agent")
    sub = parser.add_subparsers(dest="cmd")
    inst = sub.add_parser("install", help="download, verify and unpack the jevos release")
    inst.add_argument("--home", default=str(jev_home()))
    args = parser.parse_args(argv)
    if args.cmd != "install":
        parser.print_help()
        return 2
    try:
        os.nice(19)
    except (AttributeError, OSError):
        pass
    installer = JevInstaller(Path(args.home))

    async def run() -> None:
        installer.start()
        await installer.task

    asyncio.run(run())
    print(json.dumps(dict(installer.state)))
    return 0 if installer.state.get("phase") == "done" else 1


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(_main(sys.argv[1:]))
