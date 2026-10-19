"""Self-update from the browser, and automatically in a Codespace.

The user's rule is *zero commands* and *never `git pull`* (a pull that asks
questions is how a Codespace ends up detached from its branch).  So the engine
runs ``tools/self_update.sh`` for them:

* ``GET  /api/update/status``  -> is this checkout behind its own branch?
* ``POST /api/update/apply``   -> fetch, fast-forward, re-provision if
  requirements changed, restart the engine (detached, in its own session -
  the restart takes the engine down for a few seconds and the dashboard
  reconnects by itself).
* In a Codespace (or ``AUTO_UPDATE=1``) a background task checks every
  ``AUTO_UPDATE_MINUTES`` (default 10) and applies a clean fast-forward on its
  own.  ``AUTO_UPDATE=0`` turns that off; the endpoints stay.

The script never checks out, never switches branches, never touches ``main``
and refuses anything that is not a fast-forward.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time

from backend.core import config as cfg

log = logging.getLogger("drosophila.update")

SCRIPT = cfg.REPO_ROOT / "tools" / "self_update.sh"
LOG_FILE = cfg.REPO_ROOT / ".run" / "update.log"

_last: dict = {"checked_at": 0.0, "result": None}
_CACHE_SECONDS = 60.0


def auto_enabled() -> bool:
    flag = os.environ.get("AUTO_UPDATE", "").strip()
    if flag == "0":
        return False
    if flag == "1":
        return True
    return bool(os.environ.get("CODESPACE_NAME"))


def interval_seconds() -> float:
    try:
        minutes = float(os.environ.get("AUTO_UPDATE_MINUTES", "10"))
    except ValueError:
        minutes = 10.0
    return max(1.0, minutes) * 60.0


async def check(force: bool = False) -> dict:
    """Run ``self_update.sh --check`` (cached for a minute)."""
    now = time.time()
    if not force and _last["result"] is not None and now - _last["checked_at"] < _CACHE_SECONDS:
        return _last["result"]
    if not SCRIPT.is_file():
        return {"ok": False, "reason": "tools/self_update.sh missing", "behind": 0}
    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", str(SCRIPT), "--check",
            cwd=str(cfg.REPO_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PORT": str(cfg.SETTINGS.port)},
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=120)
        line = out.decode(errors="replace").strip().splitlines()
        result = json.loads(line[-1]) if line else {"ok": False, "reason": "no output"}
        if err and not result.get("ok", True):
            result["stderr"] = err.decode(errors="replace")[-300:]
    except asyncio.TimeoutError:
        result = {"ok": False, "reason": "fetch timed out", "behind": 0}
    except Exception as exc:  # noqa: BLE001 - never let a diagnostic break the API
        result = {"ok": False, "reason": str(exc), "behind": 0}
    result["auto"] = auto_enabled()
    result["auto_interval_seconds"] = interval_seconds()
    result["checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    result["log"] = str(LOG_FILE)
    _last.update(checked_at=now, result=result)
    return result


def apply() -> dict:
    """Start ``self_update.sh --apply`` detached; it restarts the engine itself."""
    if not SCRIPT.is_file():
        return {"started": False, "reason": "tools/self_update.sh missing"}
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(  # noqa: S603 - our own script
        ["bash", str(SCRIPT), "--apply"],
        cwd=str(cfg.REPO_ROOT),
        env={**os.environ, "PORT": str(cfg.SETTINGS.port)},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    _last["result"] = None  # force a fresh check afterwards
    log.info("self-update started (pid %d) - log: %s", proc.pid, LOG_FILE)
    return {"started": True, "pid": proc.pid, "log": str(LOG_FILE),
            "note": "the engine restarts in a few seconds if there was something to apply"}


def log_tail(lines: int = 20) -> list[str]:
    try:
        return [ln.rstrip() for ln in LOG_FILE.read_text(errors="replace").splitlines() if ln.strip()][-lines:]
    except OSError:
        return []


async def auto_loop() -> None:
    """Background task: check every few minutes, fast-forward when clean."""
    await asyncio.sleep(90)  # let the engine settle first
    while True:
        try:
            result = await check(force=True)
            if result.get("ok") and int(result.get("behind") or 0) > 0 and result.get("can_fast_forward"):
                log.info("self-update: %s commit(s) behind origin/%s - applying",
                         result["behind"], result.get("branch"))
                apply()
                return  # the engine is about to be restarted with the new code
        except Exception as exc:  # noqa: BLE001
            log.debug("auto-update check failed: %s", exc)
        await asyncio.sleep(interval_seconds())
