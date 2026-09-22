"""Shared application state for the API layer.

A single ``CycleManager`` instance is created by the FastAPI lifespan and kept
here so every router can reach it without import cycles.
"""

from __future__ import annotations

import logging

from backend.agents.local_model_agent import LocalModelAgent
from backend.core.cycle_manager import CycleManager

log = logging.getLogger("drosophila.api")

_manager: CycleManager | None = None


def set_manager(manager: CycleManager) -> None:
    global _manager
    _manager = manager


def get_manager() -> CycleManager:
    if _manager is None:  # pragma: no cover - only before startup completes
        raise RuntimeError("Cycle manager is not running yet")
    return _manager


def manager_or_none() -> CycleManager | None:
    return _manager


def local_agent() -> LocalModelAgent:
    return get_manager().agents.local
