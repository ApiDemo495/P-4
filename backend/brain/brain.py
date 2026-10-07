"""The brain manager: one object that always owns a usable circuit.

The rest of the application never talks to neuPrint, never touches the CSV and
never has to know which one is in play.  It asks ``brain.conv`` for a
convolution and ``brain.status()`` for the dashboard.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from backend.brain import connectome_data, graph_convolution as gc
from backend.brain import health_check, startup_verification, whole_brain
from backend.brain.matrix_builder import checksum, stats
from backend.brain.startup_verification import BrainStatus
from backend.core import config as cfg

log = logging.getLogger("drosophila.brain")


class Brain:
    """Owns the brain: the FlyWire v783 connectome (mushroom body per pass,
    whole brain every 2 s) and, until it is loaded, the 80x80 fallback."""

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
        # --- Round AQ: the real brain -------------------------------------
        # The 80x80 matrix above is now only the last-resort fallback.  Once
        # the FlyWire v783 connectome is on disk the mushroom body (8 353
        # neurons) runs on every formula pass and the whole brain (138 639
        # neurons, 15.09 M connections) on a worker thread every 2 s.
        self.data = connectome_data.ConnectomeData()
        self.mb: whole_brain.ConnectomeGraph | None = None
        self.whole: whole_brain.ConnectomeGraph | None = None
        self.whole_readout: whole_brain.Readout | None = None
        self.whole_error: str = ""
        self.whole_passes: int = 0
        self.mb_passes: int = 0
        self.mb_last_us: int = 0
        self.last_input: tuple | None = None   # (vector, drg, hsi) of the latest formula pass
        self._whole_busy = False
        self.connectome_loaded_at: float = 0.0
        self.connectome_error: str = ""

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
        self._tasks.append(
            asyncio.create_task(self._connectome_loop(), name="brain-connectome")
        )
        return self.status

    # ------------------------------------------------------------------
    # Round AQ: the real connectome
    # ------------------------------------------------------------------
    @property
    def connectome_ready(self) -> bool:
        return self.mb is not None

    def _load_graphs(self) -> None:
        """Worker thread: make sure the data is there, then load both graphs."""
        if not self.data.ensure():
            return
        mb = whole_brain.load_mushroom_body(self.data.data_dir)
        whole = whole_brain.load_whole_brain(self.data.data_dir)
        self.mb, self.whole = mb, whole
        self.connectome_loaded_at = time.time()

    async def _connectome_loop(self) -> None:
        """Load (downloading first if needed, retrying on failure), then run
        the whole brain on the latest formula pass every 2 s."""
        backoff = 30.0
        while self.mb is None:
            try:
                await asyncio.to_thread(self._load_graphs)
            except Exception as exc:  # noqa: BLE001
                self.connectome_error = f"{type(exc).__name__}: {exc}"[:300]
                log.warning("connectome load failed: %s", self.connectome_error)
            if self.mb is not None:
                self.status = BrainStatus.LIVE_FLYWIRE
                self.dataset = connectome_data.DATASET
                self.message = (f"{connectome_data.DATASET}: {self.whole.n:,} neurons, "
                                f"{self.whole.connections:,} connections ({self.whole.synapses_total:,} synapses); "
                                f"mushroom body {self.mb.n:,} neurons on every pass, whole brain every 2 s")
                log.info("Drosophila brain: %s", self.message)
                break
            if not self.data.enabled:
                return
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 600.0)
        while True:
            await asyncio.sleep(2.0)
            if self.whole is None or self.last_input is None or self._whole_busy:
                continue
            vector, drg, hsi = self.last_input
            self._whole_busy = True
            try:
                self.whole_readout = await asyncio.to_thread(self.whole.propagate, vector, drg, hsi, True)
                self.whole_passes += 1
                self.whole_error = ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.whole_error = f"{type(exc).__name__}: {exc}"[:300]
            finally:
                self._whole_busy = False

    def mushroom_body_pass(self, vector, drg: float, hsi: float):
        """The per-pass circuit for CCSv2 / KCAE: the real mushroom body when
        loaded, else None (the caller falls back to the 80x80 convolution)."""
        if self.mb is None:
            return None
        self.last_input = (np.asarray(vector, dtype=np.float32).copy(), float(drg), float(hsi))
        readout = self.mb.propagate(vector, drg=drg, hsi=hsi, detail=True)
        self.mb_passes += 1
        self.mb_last_us = readout.elapsed_us
        return readout

    def whole_brain_fresh(self, max_age: float = 10.0) -> whole_brain.Readout | None:
        r = self.whole_readout
        if r is None or time.time() - r.computed_at > max_age:
            return None
        return r

    def connectome_payload(self) -> dict:
        out = {
            "data": self.data.to_dict(),
            "loaded": self.connectome_ready,
            "loaded_at": self.connectome_loaded_at,
            "error": self.connectome_error,
            "fallback_in_use": not self.connectome_ready,
            "fallback": "80x80 mushroom-body stand-in (used only until the connectome is loaded)",
        }
        if self.mb is not None:
            out["mushroom_body"] = {**self.mb.populations(), "passes": self.mb_passes, "last_pass_us": self.mb_last_us,
                                    "hops": self.mb.hops, "name": self.mb.name}
        if self.whole is not None:
            pops = self.whole.populations()
            pops.pop("formula_groups", None)
            out["whole_brain"] = {**pops, "passes": self.whole_passes, "hops": self.whole.hops, "name": self.whole.name,
                                  "error": self.whole_error, "busy": self._whole_busy,
                                  "readout": self.whole_readout.to_dict() if self.whole_readout else None}
        return out

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def _apply(self, result: startup_verification.VerificationResult) -> None:
        self.matrix = result.matrix
        self.conv = gc.GraphConvolution(result.matrix)
        if self.connectome_ready:
            # the real connectome outranks whatever the 80x80 verification found
            self.steps = result.steps
            self.verification = result
            return
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
        payload["connectome"] = {
            "loaded": self.connectome_ready,
            "phase": self.data.status.phase,
            "percent": self.data.to_dict().get("percent", 0.0),
            "neurons": self.whole.n if self.whole else 0,
            "connections": self.whole.connections if self.whole else 0,
            "synapses": self.whole.synapses_total if self.whole else 0,
            "mb_neurons": self.mb.n if self.mb else 0,
            "whole_passes": self.whole_passes,
            "mb_passes": self.mb_passes,
            "mb_last_us": self.mb_last_us,
            "whole_last_us": self.whole_readout.elapsed_us if self.whole_readout else 0,
            "whole_balance": round(self.whole_readout.balance, 4) if self.whole_readout else None,
        }
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
