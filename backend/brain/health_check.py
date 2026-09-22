"""Brain health monitoring - every 5 minutes (Section 6.6).

Four checks, in order:

1. is an adjacency matrix loaded at all?
2. is it the right shape (80x80)?
3. does a graph convolution of a probe vector produce non-zero output?
4. is neuPrint still reachable (non-blocking, 3 s timeout)?

The result is surfaced verbatim in the Agent Dashboard.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from backend.brain import graph_convolution as gc

log = logging.getLogger("drosophila.brain.health")


@dataclass
class BrainHealth:
    status: str  # HEALTHY | DEAD | CORRUPT | ZERO_OUTPUT
    message: str = ""
    source: str = ""
    matrix_checksum: str = ""
    checked_at: float = field(default_factory=time.time)
    neuprint_live: bool = False
    output_magnitude: float = 0.0

    @property
    def healthy(self) -> bool:
        return self.status == "HEALTHY"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "message": self.message,
            "source": self.source,
            "matrix_checksum": self.matrix_checksum,
            "neuprint_live": self.neuprint_live,
            "output_magnitude": round(self.output_magnitude, 6),
            "checked_at": self.checked_at,
        }


async def neuprint_ping(server: str = "https://neuprint.janelia.org", timeout: float = 3.0) -> bool:
    """Non-blocking reachability probe."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{server.rstrip('/')}/api/version")
            return response.status_code < 500
    except Exception:  # noqa: BLE001
        return False


async def neuprint_test_token(
    token: str,
    server: str = "https://neuprint.janelia.org",
    dataset: str = "hemibrain:v1.2.1",
    timeout: float = 10.0,
) -> dict:
    """Validate a neuPrint token the way the brain module will use it.

    neuPrint accepts either ``x-neuprint-token`` or ``Authorization: Bearer``;
    both are tried so a valid token is never reported as invalid because of a
    header-name difference.  `/api/databaseInfo` requires authentication, so a
    200 means the token really works for this dataset.
    """
    import httpx

    candidate = (token or "").strip()
    if not candidate:
        return {"valid": False, "error": "No token entered"}

    url = f"{server.rstrip('/')}/api/databaseInfo"
    schemes = (
        {"x-neuprint-token": candidate},
        {"Authorization": f"Bearer {candidate}"},
    )
    last_error = ""
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            reachable = False
            for headers in schemes:
                response = await client.get(url, params={"dataset": dataset}, headers=headers)
                reachable = True
                if response.status_code == 200:
                    try:
                        payload = response.json()
                    except Exception:  # noqa: BLE001
                        payload = {}
                    databases = payload.get("dataset") or payload.get("databases") or dataset
                    if isinstance(databases, (list, tuple)):
                        databases = ", ".join(str(d) for d in databases[:3])
                    return {
                        "valid": True,
                        "detail": f"token accepted · dataset {databases}",
                        "reachable": True,
                    }
                if response.status_code in (401, 403):
                    return {
                        "valid": False,
                        "error": f"neuPrint rejected the token (HTTP {response.status_code})",
                        "reachable": True,
                    }
                last_error = f"HTTP {response.status_code} from {url}"
            if reachable:
                return {"valid": False, "error": last_error or "unexpected response", "reachable": True}
            return {"valid": False, "error": "neuPrint did not answer", "reachable": False}
    except Exception as exc:  # noqa: BLE001
        return {
            "valid": False,
            "error": f"could not reach {server} ({exc})",
            "reachable": False,
            "hint": (
                "Some sandboxes and corporate networks block neuprint.janelia.org. "
                "The engine keeps working on the committed 80x80 fallback matrix."
            ),
        }


async def check_brain(
    adjacency: np.ndarray | None,
    conv: gc.GraphConvolution | None,
    source: str = "",
    server: str = "https://neuprint.janelia.org",
    ping: bool = True,
) -> BrainHealth:
    """Run the four checks and classify the brain."""
    if adjacency is None:
        return BrainHealth(status="DEAD", message="No adjacency matrix loaded", source=source)

    if adjacency.shape != (gc.N_NODES, gc.N_NODES):
        return BrainHealth(
            status="CORRUPT",
            message=f"Wrong matrix dimensions: {adjacency.shape}",
            source=source,
        )

    if conv is None:
        try:
            conv = gc.GraphConvolution(adjacency)
        except Exception as exc:  # noqa: BLE001
            return BrainHealth(status="CORRUPT", message=f"Convolution failed: {exc}", source=source)

    rng = np.random.default_rng(11)
    probe = rng.uniform(-1.0, 1.0, size=gc.N_NODES)
    output = graph_convolution(probe, adjacency)
    magnitude = float(np.max(np.abs(output)))
    if magnitude <= 1e-12 or np.all(output == 0):
        return BrainHealth(
            status="ZERO_OUTPUT",
            message="Brain producing all zeros",
            source=source,
            matrix_checksum=_checksum(adjacency),
        )

    live = await neuprint_ping(server) if ping else False
    return BrainHealth(
        status="HEALTHY",
        message="All checks passed",
        source=source or ("LIVE" if live else "CACHED"),
        matrix_checksum=_checksum(adjacency),
        neuprint_live=live,
        output_magnitude=magnitude,
    )


def _checksum(matrix: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(matrix).tobytes()).hexdigest()[:8]


async def health_loop(brain, interval_seconds: float = 300.0) -> None:
    """Background loop started by the application lifespan."""
    while True:
        try:
            await brain.refresh_health()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("brain health check failed: %s", exc)
        await asyncio.sleep(interval_seconds)


def graph_convolution(test_input: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    """Section 6.6 check 3, kept as a module-level helper."""
    return gc.graph_convolution(test_input, adjacency)
