"""Round AW - the StateVault: everything the engine has *learned* survives a
restart.

The user's requirement, verbatim: the app must "permanently run and save
history and really learn", and a reconnect "must never lose history".  Until
this round the engine's memory lived in the process (``Store`` falls back to
an in-memory dict when there is no Redis - and there is none in a Codespace),
so every restart - a Codespace resume, a self-update, a crash - started the
competition ledger, the boundary picker, the dopamine brain, the outcome
buffer and the accuracy figures from zero.

The vault writes one pickle per named item to ``.run/vault/`` (git-ignored,
on the Codespace disk, so it survives stop/resume and self-updates) with an
atomic rename, and reads them back on ``start()``.  Items are *adopted* - the
saved object's state is copied into the live object (``__dict__`` update) so
every reference the rest of the engine already holds stays valid.

A vault entry is refused when it was written by a different schema version
or is older than ``max_age`` (a 3-day-old streak is not a live streak).
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("drosophila.vault")

#: bump when the pickled layout of any persisted object changes incompatibly
SCHEMA = 1


class StateVault:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.last_saved_at: float = 0.0
        self.last_restored: dict[str, float] = {}
        self.errors: dict[str, str] = {}

    # -- storage ---------------------------------------------------------
    def _path(self, name: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        return self.root / f"{safe}.pkl"

    def save(self, name: str, payload: Any) -> bool:
        """Pickle ``payload`` under ``name`` atomically.  Never raises."""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            blob = pickle.dumps({"schema": SCHEMA, "saved_at": time.time(), "payload": payload}, protocol=4)
            tmp = self._path(name).with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                fh.write(blob)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._path(name))
            self.last_saved_at = time.time()
            self.errors.pop(name, None)
            return True
        except Exception as exc:  # noqa: BLE001 - persistence must never take the engine down
            self.errors[name] = f"save: {exc}"
            log.debug("vault save(%s) failed: %s", name, exc)
            return False

    def load(self, name: str, max_age: float | None = None) -> Any | None:
        """The saved payload, or ``None`` when absent, stale or unreadable."""
        path = self._path(name)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as fh:
                record = pickle.load(fh)
            if not isinstance(record, dict) or record.get("schema") != SCHEMA:
                self.errors[name] = "load: schema mismatch"
                return None
            age = time.time() - float(record.get("saved_at", 0.0))
            if max_age is not None and age > max_age:
                self.errors[name] = f"load: {age / 3600:.1f} h old, ignored"
                return None
            self.last_restored[name] = float(record.get("saved_at", 0.0))
            return record.get("payload")
        except Exception as exc:  # noqa: BLE001
            self.errors[name] = f"load: {exc}"
            log.debug("vault load(%s) failed: %s", name, exc)
            return None

    # -- adoption --------------------------------------------------------
    @staticmethod
    def adopt(target: Any, saved: Any) -> bool:
        """Copy ``saved``'s state into ``target`` in place (same class only)."""
        if saved is None or type(saved) is not type(target):
            return False
        if hasattr(target, "__dict__") and hasattr(saved, "__dict__"):
            target.__dict__.update(saved.__dict__)
            return True
        return False

    def restore_into(self, name: str, target: Any, max_age: float | None = None) -> bool:
        return self.adopt(target, self.load(name, max_age=max_age))

    def to_dict(self) -> dict:
        files = sorted(p.name for p in self.root.glob("*.pkl")) if self.root.exists() else []
        return {
            "path": str(self.root),
            "items": files,
            "last_saved_at": self.last_saved_at,
            "restored": {k: v for k, v in self.last_restored.items()},
            "errors": dict(self.errors),
        }
