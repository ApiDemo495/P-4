"""The brain manager: one object that always owns a usable circuit.

The rest of the application never talks to neuPrint, never touches the CSV and
never has to know which one is in play.  It asks ``brain.conv`` for a
convolution and ``brain.status()`` for the dashboard.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np

from backend.brain import graph_convolution as gc
from backend.brain import health_check, startup_verification
from backend.brain.matrix_builder import checksum, stats
from backend.brain.startup_verification import BrainStatus
from backend.core import config as cfg

log = logging.getLogger("drosophila.brain")


class Brain:
    """Owns the 80x80 mushroom-body adjacency matrix and its convolution."""

    def __init__(self, settings=None, cache=None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.cache = cache
        self.matrix: np.ndarray | None = None
        self.conv: gc.GraphConvolution | None = None
        self.status: BrainStatus = BrainStatus.DEAD
        self.message: str = "not started"
        self.dataset: str = ""
        self.steps: list[dict] = []
        self.cluster_info: dict = {}
        self.loaded_at: float = 0.0
        self.verification: startup_verification.VerificationResult | None = None
        self.last_health: health_check.BrainHealth | None = None
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> BrainStatus:
        result = await startup_verification.run_verification(self.settings, self.cache)
        self._apply(result)
        if self.status is BrainStatus.LIVE_CONNECTED:
            log.info(
                "Drosophila brain: LIVE connection to neuPrint %s", result.dataset or "hemibrain"
            )
        elif self.status is BrainStatus.LIVE_FLYWIRE:
            log.info("Drosophila brain: LIVE connection to FlyWire (%s)", result.dataset)
        elif self.status is BrainStatus.CACHED:
            log.info("Drosophila brain: loaded cached matrix (checksum %s)", checksum(result.matrix))
        elif self.status is BrainStatus.FALLBACK_CSV:
            log.warning("Drosophila brain: using pre-computed fallback matrix")

        await self.refresh_health(ping=False)
        self._tasks.append(
            asyncio.create_task(self._health_loop(), name="brain-health")
        )
        self._tasks.append(
            asyncio.create_task(self._refresh_loop(), name="brain-refresh")
        )
        return self.status

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def _apply(self, result: startup_verification.VerificationResult) -> None:
        self.matrix = result.matrix
        self.conv = gc.GraphConvolution(result.matrix)
        self.status = result.status
        self.message = result.message
        self.dataset = result.dataset
        self.steps = result.steps
        self.cluster_info = result.cluster_info
        self.loaded_at = time.time()
        self.verification = result

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    async def refresh_health(self, ping: bool = True) -> health_check.BrainHealth:
        self.last_health = await health_check.check_brain(
            self.matrix,
            self.conv,
            source=self.status.value,
            server=self.settings.neuprint_server,
            ping=ping,
        )
        return self.last_health

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.brain_health_interval_seconds)
            try:
                health = await self.refresh_health(ping=True)
                if not health.healthy:
                    log.warning("brain health: %s (%s)", health.status, health.message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("brain health loop error: %s", exc)

    async def _refresh_loop(self) -> None:
        """Re-query neuPrint every 24 hours (Section 6.4)."""
        while True:
            await asyncio.sleep(self.settings.brain_cache_ttl_seconds)
            try:
                if self.status in (BrainStatus.FALLBACK_CSV, BrainStatus.CACHED):
                    log.info("Attempting scheduled brain matrix refresh")
                    await self.reconnect()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("brain refresh failed: %s", exc)

    async def reconnect(self) -> BrainStatus:
        """Force a fresh verification ('Force Reconnect' in Settings)."""
        result = await startup_verification.run_verification(self.settings, self.cache)
        self._apply(result)
        await self.refresh_health(ping=False)
        return self.status

    def load_matrix(self, matrix: np.ndarray, source: str = "MANUAL") -> None:
        """Install a matrix directly (used by the upload/inspect endpoints)."""
        if matrix.shape != (gc.N_NODES, gc.N_NODES):
            raise ValueError(f"expected {(gc.N_NODES, gc.N_NODES)}, got {matrix.shape}")
        self.matrix = matrix
        self.conv = gc.GraphConvolution(matrix)
        self.status = BrainStatus.CACHED
        self.message = f"manually loaded ({source})"
        self.loaded_at = time.time()

    def save_to(self, path: Path) -> Path:
        from backend.brain import matrix_builder as mb

        if self.matrix is None:
            raise RuntimeError("no matrix loaded")
        return mb.save_csv(self.matrix, path)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def is_live(self) -> bool:
        return self.status in (BrainStatus.LIVE_CONNECTED, BrainStatus.LIVE_FLYWIRE)

    @property
    def gain(self) -> float:
        return self.conv.gain if self.conv else 0.0

    def status_dict(self) -> dict:
        health = self.last_health.to_dict() if self.last_health else None
        payload = {
            "status": self.status.value,
            "message": self.message,
            "dataset": self.dataset,
            "loaded_at": self.loaded_at,
            "is_live": self.is_live,
            "steps": self.steps,
            "cluster_info": self.cluster_info,
            "health": health,
            "gain": round(self.gain, 4),
            "node_layout": {
                "pn": [0, 20],
                "kenyon_cells": [gc.KC_START, gc.KC_END],
                "dans": [gc.PAM, gc.OA],
                "mbons": [gc.MBON_APPROACH, gc.MBON_CONFIDENCE],
                "lateral_horn": [gc.LH_APPROACH, gc.LH_NEUTRAL],
            },
            "pn_names": list(gc.PN_NAMES),
        }
        if self.matrix is not None:
            payload["matrix"] = stats(self.matrix)
        return payload

    def matrix_payload(self, limit: int | None = None) -> dict:
        from backend.brain import matrix_builder as mb

        if self.matrix is None:
            return {"edges": [], "stats": {}}
        return {
            "edges": mb.to_edge_list(self.matrix, limit=limit),
            "stats": stats(self.matrix),
            "node_types": mb.NODE_TYPES,
        }
