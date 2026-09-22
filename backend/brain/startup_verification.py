"""Five-step brain startup verification (Section 6.2).

This is the module that fixes v1.0's biggest failure: the brain was specified
but never actually connected, and when the API was unreachable the module
*silently* returned zeros.  Here every step has a timeout, a logged outcome and
a defined successor, and the final result is guaranteed to be a usable matrix.

    Step 1  neuPrint reachability          (5 s)  -> step 2
    Step 2  neuPrint authentication        (5 s)  -> step 3
    Step 3  query + cluster + build + cache(30 s) -> LIVE_CONNECTED
    Step 4  FlyWire / CAVE fallback        (10 s) -> LIVE_FLYWIRE
    Step 5  committed CSV fallback          --    -> FALLBACK_CSV

The module NEVER raises: the worst case is FALLBACK_CSV, and if even that file is
missing it is regenerated in-process from the deterministic generator.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from backend.brain import matrix_builder as mb
from backend.core import config as cfg

log = logging.getLogger("drosophila.brain.startup")


class BrainStatus(str, Enum):
    LIVE_CONNECTED = "LIVE_CONNECTED"
    LIVE_FLYWIRE = "LIVE_FLYWIRE"
    FALLBACK_CSV = "FALLBACK_CSV"
    CACHED = "CACHED"
    DEAD = "DEAD"


@dataclass
class VerificationResult:
    status: BrainStatus
    matrix: np.ndarray
    message: str
    dataset: str = ""
    steps: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    cache_hit: bool = False
    cluster_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "message": self.message,
            "dataset": self.dataset,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "cache_hit": self.cache_hit,
            "steps": self.steps,
            "cluster_info": self.cluster_info,
        }


async def run_verification(
    settings=None,
    cache=None,
    force_fallback: bool | None = None,
) -> VerificationResult:
    """Execute steps 1-5 and return a guaranteed-usable matrix."""
    settings = settings or cfg.SETTINGS
    started = time.perf_counter()
    steps: list[dict] = []
    force = settings.brain_force_fallback if force_fallback is None else force_fallback

    def record(step: str, ok: bool, detail: str, elapsed: float) -> None:
        steps.append(
            {"step": step, "ok": ok, "detail": detail, "elapsed_ms": round(elapsed * 1000, 1)}
        )

    # -- Step 0 (optimisation, implied by 6.3e/6.4): reuse a fresh cache ----
    if cache is not None and not force:
        t0 = time.perf_counter()
        try:
            cached = await cache.get_pickle("brain:adjacency:80x80")
        except Exception as exc:  # noqa: BLE001
            cached = None
            log.debug("brain cache read failed: %s", exc)
        if isinstance(cached, np.ndarray) and cached.shape == (80, 80):
            record("0-cache", True, "fresh 80x80 matrix from Redis", time.perf_counter() - t0)
            return VerificationResult(
                status=BrainStatus.CACHED,
                matrix=cached,
                message="Loaded cached mushroom-body matrix",
                dataset=settings.neuprint_dataset,
                steps=steps,
                elapsed_seconds=time.perf_counter() - started,
                cache_hit=True,
            )
        record("0-cache", False, "no cached matrix", time.perf_counter() - t0)

    if not force:
        # ---- Step 1: connectivity --------------------------------------
        t0 = time.perf_counter()
        reachable = await _step1_connectivity(settings)
        record("1-connectivity", reachable, "neuPrint version endpoint", time.perf_counter() - t0)
        if reachable:
            # ---- Step 2: authentication --------------------------------
            t0 = time.perf_counter()
            client, auth_error = await _step2_authenticate(settings)
            record("2-auth", client is not None, auth_error, time.perf_counter() - t0)
            if client is not None:
                # ---- Step 3: query, cluster, build, cache --------------
                t0 = time.perf_counter()
                matrix, cluster_info, error = await _step3_query(
                    client, settings, timeout=30.0
                )
                ok = matrix is not None
                record("3-query", ok, error or "hemibrain mushroom body subgraph", time.perf_counter() - t0)
                if ok:
                    await _cache_matrix(cache, matrix)
                    _persist_fallback(matrix)
                    return VerificationResult(
                        status=BrainStatus.LIVE_CONNECTED,
                        matrix=matrix,
                        message=f"LIVE connection to neuPrint {settings.neuprint_dataset}",
                        dataset=settings.neuprint_dataset,
                        steps=steps,
                        elapsed_seconds=time.perf_counter() - started,
                        cluster_info=cluster_info,
                    )

    if not force:
        # ---- Step 4: FlyWire / CAVE ------------------------------------
        t0 = time.perf_counter()
        matrix, error = await _step4_flywire(settings)
        ok = matrix is not None
        record("4-flywire", ok, error or "FlyWire via CAVE", time.perf_counter() - t0)
        if ok:
            await _cache_matrix(cache, matrix)
            _persist_fallback(matrix)
            return VerificationResult(
                status=BrainStatus.LIVE_FLYWIRE,
                matrix=matrix,
                message=f"LIVE connection to FlyWire ({settings.cave_dataset})",
                dataset=settings.cave_dataset,
                steps=steps,
                elapsed_seconds=time.perf_counter() - started,
            )

    # ---- Step 5: committed CSV fallback --------------------------------
    t0 = time.perf_counter()
    try:
        matrix = load_fallback_matrix()
        record("5-fallback", True, str(mb.default_fallback_path()), time.perf_counter() - t0)
        log.warning("Drosophila brain: using pre-computed fallback matrix")
        return VerificationResult(
            status=BrainStatus.FALLBACK_CSV,
            matrix=matrix,
            message="Using pre-computed fallback matrix",
            dataset="fallback",
            steps=steps,
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:  # noqa: BLE001
        record("5-fallback", False, str(exc), time.perf_counter() - t0)
        # Last resort: regenerate in-process.  The brain must never be absent.
        from backend.brain.generate_fallback_matrix import build_matrix

        matrix = build_matrix()
        return VerificationResult(
            status=BrainStatus.FALLBACK_CSV,
            matrix=matrix,
            message="Fallback CSV missing; regenerated deterministically",
            dataset="fallback",
            steps=steps,
            elapsed_seconds=time.perf_counter() - started,
        )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


async def _step1_connectivity(settings) -> bool:
    import httpx

    url = f"{settings.neuprint_server.rstrip('/')}/api/version"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            return response.status_code < 500
    except Exception as exc:  # noqa: BLE001
        log.info("neuPrint unreachable: %s", exc)
        return False


async def _step2_authenticate(settings):
    if not settings.neuprint_token:
        return None, "no NEUPRINT_APPLICATION_CREDENTIALS token configured"
    try:
        from backend.brain.query_circuits import connect_neuprint

        client = await asyncio.wait_for(
            asyncio.to_thread(
                connect_neuprint,
                settings.neuprint_token,
                settings.neuprint_server,
                settings.neuprint_dataset,
            ),
            timeout=5.0,
        )
        return client, "authenticated"
    except asyncio.TimeoutError:
        return None, "authentication timed out after 5s"
    except ImportError:
        return None, "neuprint-python is not installed"
    except Exception as exc:  # noqa: BLE001
        return None, f"authentication failed: {exc}"


async def _step3_query(client, settings, timeout: float = 30.0):
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_build_from_neuprint_sync, client, settings), timeout=timeout
        )
    except asyncio.TimeoutError:
        return None, {}, "query timed out after 30s"
    except Exception as exc:  # noqa: BLE001
        return None, {}, f"query failed: {exc}"


def _build_from_neuprint_sync(client, settings):
    from backend.brain import query_circuits as qc
    from backend.brain import spectral_cluster as sc

    circuit = qc.fetch_circuit(client, settings.neuprint_dataset)
    if circuit.total == 0:
        raise RuntimeError("neuPrint returned no neurons for the mushroom body query")

    kc_ids = [int(n.get("bodyId") or 0) for n in circuit.kenyon_cells if n.get("bodyId")]
    kc_edges, _ = qc.fetch_kc_adjacency(client, kc_ids) if kc_ids else ([], {})
    cluster = sc.cluster_kenyon_cells(kc_ids, kc_edges, sc.N_CLUSTERS)

    indices = qc.select_representatives(circuit, cluster.labels, kc_ids)
    all_ids = [v for v in indices.values() if v]
    edges = qc.fetch_subgraph_edges(client, all_ids)
    matrix = qc.build_matrix_from_neuprint(circuit, cluster.labels, kc_ids, edges)
    return matrix, sc.cluster_summary(cluster)


async def _step4_flywire(settings):
    if not settings.cave_token:
        return None, "no CAVE_TOKEN configured"
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_build_from_flywire_sync, settings), timeout=10.0
        )
    except asyncio.TimeoutError:
        return None, "FlyWire query timed out after 10s"
    except ImportError:
        return None, "caveclient is not installed"
    except Exception as exc:  # noqa: BLE001
        return None, f"FlyWire failed: {exc}"


def _build_from_flywire_sync(settings):
    from backend.brain import query_circuits as qc

    client = qc.connect_cave(settings.cave_server, settings.cave_dataset, settings.cave_token)
    circuit = qc.fetch_flywire_mushroom_body(client)
    if circuit.total == 0:
        raise RuntimeError("CAVE returned no mushroom-body neurons")
    return qc.build_matrix_from_flywire(circuit), {}


# ---------------------------------------------------------------------------
# Fallback handling
# ---------------------------------------------------------------------------


def load_fallback_matrix(path: Path | None = None) -> np.ndarray:
    """Load the committed CSV; regenerate it if it has gone missing."""
    path = Path(path) if path else mb.default_fallback_path()
    if not path.exists():
        from backend.brain.generate_fallback_matrix import build_matrix

        matrix = build_matrix()
        mb.save_csv(matrix, path)
        return matrix
    matrix = mb.load_csv(path)
    if matrix.shape != (80, 80) or not np.any(matrix):
        raise RuntimeError(f"fallback matrix at {path} is empty or malformed")
    return matrix


async def _cache_matrix(cache, matrix: np.ndarray) -> None:
    if cache is None:
        return
    try:
        await cache.set_pickle("brain:adjacency:80x80", matrix, ttl=86_400)
    except Exception as exc:  # noqa: BLE001
        log.debug("could not cache brain matrix: %s", exc)


def _persist_fallback(matrix: np.ndarray) -> None:
    """Section 6.3f - after a successful live query, refresh the fallback CSV."""
    try:
        mb.save_csv(matrix, mb.default_fallback_path())
    except Exception as exc:  # noqa: BLE001
        log.debug("could not refresh fallback CSV: %s", exc)
