"""What the zero-command Codespace start actually did (Round S).

``tools/codespace_autostart.sh`` journals every step to ``.run/logs/``; this
module turns those files into one answer for ``GET /api/system/autostart`` so
a failed self-start or self-download is diagnosable from the app.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from backend.core import config as cfg

RUN_DIR = cfg.REPO_ROOT / ".run"
LOG_DIR = RUN_DIR / "logs"
JOURNAL = LOG_DIR / "autostart.journal"
STAMP = RUN_DIR / ".provisioned"
FLUTTER_LOG = cfg.REPO_ROOT / "flutter-setup.log"
PIP_LOG = Path("/tmp/pip-install.log")
SERVER_LOG = cfg.REPO_ROOT / "server.log"


def _tail(path: Path, lines: int) -> list[str]:
    try:
        return [ln.rstrip() for ln in path.read_text(errors="replace").splitlines()][-lines:]
    except OSError:
        return []


def _file(path: Path, lines: int) -> dict:
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "bytes": stat.st_size,
            "age_seconds": round(max(0.0, time.time() - stat.st_mtime), 1),
            "tail": _tail(path, lines),
        }
    except OSError:
        return {"path": str(path), "exists": False, "bytes": 0, "age_seconds": None, "tail": []}


def _journal_rows(lines: int) -> list[dict]:
    rows = []
    for raw in _tail(JOURNAL, lines):
        parts = raw.split(None, 3)
        if len(parts) == 4:
            rows.append({"at": parts[0], "level": parts[1], "hook": parts[2], "message": parts[3]})
    return rows


def status(lines: int = 60) -> dict:
    journal = _journal_rows(lines)
    fails = [r for r in journal if r["level"] == "fail"]
    warns = [r for r in journal if r["level"] == "warn"]
    setup_logs = sorted(LOG_DIR.glob("setup-*.log")) if LOG_DIR.exists() else []
    provisioned = STAMP.exists()
    in_codespace = bool(os.environ.get("CODESPACE_NAME"))
    if not journal and not provisioned:
        verdict = "no autostart has run here (not a Codespace, or hooks never fired)"
    elif fails:
        verdict = "a self-start step failed - see the journal"
    elif not provisioned:
        verdict = "provisioning has not finished (pip download in progress or interrupted)"
    else:
        verdict = "provisioned; the engine is answering (you are reading this from it)"
    return {
        "verdict": verdict,
        "healthy": provisioned and not fails,
        "in_codespace": in_codespace,
        "codespace": os.environ.get("CODESPACE_NAME") or None,
        "provisioned": provisioned,
        "requirements_fingerprint": (_tail(STAMP, 1) or [""])[0][:12],
        "hooks": {
            "postCreateCommand": "bash tools/codespace_autostart.sh --provision",
            "postStartCommand": "bash tools/codespace_autostart.sh --start",
            "postAttachCommand": "bash tools/codespace_autostart.sh --attach",
        },
        "failures": fails,
        "warnings": warns,
        "journal": journal,
        "logs": {
            "setup_passes": [_file(p, lines) for p in setup_logs],
            "pip": _file(PIP_LOG, lines),
            "flutter": _file(FLUTTER_LOG, min(lines, 30)),
            "server": _file(SERVER_LOG, min(lines, 30)),
        },
        "retry": {
            "automatic": "every container start and every editor attach re-runs the hooks; "
            "a changed requirements.txt or a missing .venv re-provisions by itself",
            "manual": "bash tools/codespace_autostart.sh --attach",
        },
    }
