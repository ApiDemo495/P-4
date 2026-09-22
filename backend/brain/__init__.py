"""Drosophila mushroom-body brain module.

Submodules are imported lazily so that ``import backend.brain`` stays cheap and
free of import cycles:

    backend.brain.brain                 -> Brain (the manager object)
    backend.brain.startup_verification  -> the mandatory 5-step sequence
    backend.brain.graph_convolution     -> 3-layer propagation
    backend.brain.spectral_cluster      -> 2000 KCs -> 50 clusters
    backend.brain.health_check          -> 5-minute monitoring
    backend.brain.fallback/*.csv        -> always-committed safety net
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "Brain",
    "graph_convolution",
    "health_check",
    "matrix_builder",
    "query_circuits",
    "spectral_cluster",
    "startup_verification",
]

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    from backend.brain.brain import Brain


def __getattr__(name: str):
    """PEP 562 lazy submodule access."""
    if name == "Brain":
        from backend.brain.brain import Brain

        return Brain
    if name in __all__:
        import importlib

        return importlib.import_module(f"backend.brain.{name}")
    raise AttributeError(f"module 'backend.brain' has no attribute {name!r}")
