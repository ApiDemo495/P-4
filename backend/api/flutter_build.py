"""The Flutter web client, supervised by the engine (Round L.1).

The user's rule is *zero commands*: the Flutter SDK download and the web
build must happen on their own.  Until now that job belonged to the
devcontainer hooks (``tools/codespace_autostart.sh`` detaches
``frontend/run_web.sh``), which has two weak points a user cannot see:

* a lifecycle hook's background children can be reaped when the hook ends,
  and a Codespace that sleeps mid-download leaves a dead pid file;
* ``/flutter`` answered a bare 404 JSON, so "the SDK is not downloading" was
  the only thing anyone could say about it.

So the engine now owns it.  On start-up (in a Codespace, or when
``AUTO_FLUTTER=1``) it starts the download in its own session if the build is
missing and nothing is running; ``GET /api/flutter/status`` reports the stage
and the tail of ``flutter-setup.log``; ``POST /api/flutter/build`` (re)starts
it; and ``/flutter`` serves a self-refreshing status page with a *Retry*
button while the build does not exist yet.  Everything the hooks did is still
there - this is the belt to their braces.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from backend.core import config as cfg

log = logging.getLogger("drosophila.flutter")

REPO_ROOT: Path = cfg.REPO_ROOT
BUILD_DIR = REPO_ROOT / "frontend" / "build" / "web"
RUN_WEB = REPO_ROOT / "frontend" / "run_web.sh"
STATE_DIR = REPO_ROOT / ".run"  # never .devcontainer/ - VS Code watches it
PID_FILE = STATE_DIR / "flutter.pid"
LOG_FILE = REPO_ROOT / "flutter-setup.log"
def sdk_dir() -> Path:
    """Where the SDK lives: FLUTTER_HOME, else the path frontend/run_web.sh
    remembered in .run/flutter_home (it moves to ~/flutter-sdk when ~/flutter
    could not be freed), else ~/flutter."""
    env = os.environ.get("FLUTTER_HOME")
    if env:
        return Path(env)
    remembered = cfg.REPO_ROOT / ".run" / "flutter_home"
    try:
        text = remembered.read_text().strip()
        if text:
            return Path(text)
    except OSError:
        pass
    return Path.home() / "flutter"

#: What the log's last progress line means, for the status page.
STAGES = (
    ("4/4", "building the web bundle (2-5 min on first build)"),
    ("3/4", "working out the API base URL"),
    ("2/4", "preparing the project (flutter pub get)"),
    ("precache", "downloading the Dart SDK and the web engine"),
    ("release archive", "downloading the Flutter SDK release archive (~1 GB)"),
    ("cloning the stable Flutter SDK", "downloading the Flutter SDK (~700 MB)"),
    ("resuming the existing clone", "resuming the Flutter SDK download"),
    ("1/4", "locating the Flutter SDK"),
)


def built() -> bool:
    return (BUILD_DIR / "index.html").is_file()


def _pid() -> int | None:
    try:
        value = int(PID_FILE.read_text().strip() or 0)
    except (OSError, ValueError):
        return None
    return value or None


def running() -> bool:
    pid = _pid()
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A reused pid could belong to anything; check it is still our script.
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return True
    return "run_web.sh" in cmdline or "flutter" in cmdline


def log_tail(lines: int = 25) -> list[str]:
    try:
        text = LOG_FILE.read_text(errors="replace")
    except OSError:
        return []
    out = [re.sub(r"\x1b\[[0-9;]*m", "", line).rstrip() for line in text.splitlines()]
    return [line for line in out if line.strip()][-lines:]


def _stage(tail: list[str]) -> str:
    for line in reversed(tail):
        for needle, label in STAGES:
            if needle in line:
                return label
    return "not started" if not tail else "starting"


def _last_error(tail: list[str]) -> str:
    for line in reversed(tail):
        if "✘" in line or "failed" in line.lower() or "could not" in line.lower():
            return line.strip()
    return ""


def wanted() -> bool:
    """Should the engine start the build on its own?

    In a Codespace the answer is yes unless ``AUTO_FLUTTER=0``; anywhere else
    only when ``AUTO_FLUTTER=1`` is set explicitly (tests and sandboxes must
    never start a 700 MB download by accident).
    """
    flag = os.environ.get("AUTO_FLUTTER", "").strip()
    if flag == "0":
        return False
    if flag == "1":
        return True
    return bool(os.environ.get("CODESPACE_NAME"))


def status() -> dict:
    tail = log_tail()
    is_built = built()
    is_running = running()
    try:
        log_age = round(time.time() - LOG_FILE.stat().st_mtime, 1)
    except OSError:
        log_age = None
    return {
        "built": is_built,
        "running": is_running,
        "wanted": wanted(),
        "stage": "built - served at /flutter" if is_built else (_stage(tail) if is_running else
                                                                ("stopped: " + (_last_error(tail) or "not running")
                                                                 if tail else "not started")),
        "sdk_present": (sdk_dir() / "bin" / "flutter").exists(),
        "sdk_dir": str(sdk_dir()),
        "install_route": "release archive first (one resumable file), git clone only as fallback",
        "build_dir": str(BUILD_DIR),
        "pid": _pid() if is_running else None,
        "log": str(LOG_FILE),
        "log_age_seconds": log_age,
        "log_tail": tail,
        "last_error": _last_error(tail) if not is_running and not is_built else "",
        "retry": "POST /api/flutter/build  (or the Retry button on /flutter)",
        "manual": "INSTALL_FLUTTER=1 bash frontend/run_web.sh",
    }


def start(force: bool = False) -> dict:
    """Start the download + build detached in its own session.

    ``setsid`` puts it in a new session so no parent's clean-up (a devcontainer
    hook finishing, the engine restarting) can take it down half-way.
    """
    if built() and not force:
        return {"started": False, "reason": "already built", **status()}
    if running():
        return {"started": False, "reason": "already running", **status()}
    if not RUN_WEB.is_file():
        return {"started": False, "reason": f"{RUN_WEB} is missing", **status()}
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "INSTALL_FLUTTER": "1",
        "AUTO_FLUTTER": "1",
        "PORT": str(cfg.SETTINGS.port),
        "FLUTTER_SUPPRESS_ANALYTICS": "true",
        "CI": "true",
    })
    log_handle = open(LOG_FILE, "ab")  # noqa: SIM115 - handed to the child
    try:
        log_handle.write(f"\n==> engine (re)started the Flutter setup at {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
        log_handle.flush()
        # Lowest priority: the Dart compiler must never starve the engine that
        # launched it (the panel froze under a running countdown when it did).
        command = ["bash", str(RUN_WEB)]
        if shutil.which("nice"):
            command = ["nice", "-n", "19"] + command
        proc = subprocess.Popen(  # noqa: S603 - our own script
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_handle.close()
    PID_FILE.write_text(str(proc.pid))
    log.info("Flutter SDK download + web build started (pid %d) - log: %s", proc.pid, LOG_FILE)
    return {"started": True, "pid": proc.pid, **status()}


def ensure_started() -> dict | None:
    """Called once at engine start-up: begin the build if it is wanted and
    neither present nor in progress.  Never raises."""
    try:
        if not wanted() or built() or running():
            return None
        return start()
    except Exception as exc:  # noqa: BLE001 - optional feature, never fatal
        log.warning("could not start the Flutter setup: %s", exc)
        return None


def status_page(port: int) -> str:
    """The page served at /flutter while there is no build yet."""
    s = status()
    tail = "\n".join(s["log_tail"]) or "(no log yet - the download has not started)"
    badge = ("building" if s["running"] else ("stopped" if s["log_tail"] else "not started"))
    colour = "#f5c451" if s["running"] else "#ff6b6b"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Flutter client - building</title>
<meta http-equiv="refresh" content="6">
<style>
 body{{background:#0b0f19;color:#dbe6ff;font:14px/1.5 system-ui,sans-serif;margin:0;padding:32px}}
 .card{{max-width:860px;margin:0 auto;background:#111827;border:1px solid #1f2a44;border-radius:12px;padding:22px 26px}}
 h1{{font-size:20px;margin:0 0 6px}} .badge{{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:700;color:#0b0f19;background:{colour}}}
 pre{{background:#0b0f19;border:1px solid #1f2a44;border-radius:8px;padding:12px;font-size:12px;overflow:auto;max-height:340px;white-space:pre-wrap}}
 .muted{{color:#8a9bbd}} a{{color:#7cc4ff}} button{{background:#2563eb;color:#fff;border:0;border-radius:8px;padding:8px 14px;font-weight:700;cursor:pointer}}
 code{{background:#0b0f19;padding:1px 6px;border-radius:4px}}
</style></head><body><div class="card">
<h1>Flutter client &nbsp;<span class="badge">{badge}</span></h1>
<p class="muted">This page refreshes itself every 6 s and switches to the app when the build lands.
The web dashboard at <a href="/">/</a> has every feature meanwhile.</p>
<p><b>stage:</b> {s["stage"]}<br>
<b>SDK:</b> {"present" if s["sdk_present"] else "not downloaded yet"} ({s["sdk_dir"]})<br>
<b>log:</b> <code>{s["log"]}</code>{f' · last write {s["log_age_seconds"]} s ago' if s["log_age_seconds"] is not None else ""}</p>
{f'<p style="color:#ff6b6b"><b>last error:</b> {s["last_error"]}</p>' if s["last_error"] else ""}
<form method="post" action="/api/flutter/build?redirect=1"><button type="submit">{"Restart the download / build" if s["running"] else "Start the download + build now"}</button>
<span class="muted"> &nbsp;no terminal needed - the engine runs <code>frontend/run_web.sh</code> in the background</span></form>
<h3 style="margin-bottom:6px">flutter-setup.log (tail)</h3>
<pre>{tail}</pre>
<p class="muted">By hand, if you prefer: <code>INSTALL_FLUTTER=1 bash frontend/run_web.sh</code> &nbsp;·&nbsp; status: <code>GET /api/flutter/status</code></p>
</div></body></html>"""
