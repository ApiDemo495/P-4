"""Brain health monitoring - every 5 minutes (Section 6.6).

Four checks, in order:

1. is an adjacency matrix loaded at all?
2. is it the right shape (80x80)?
3. does a graph convolution of a probe vector produce non-zero output?
4. is neuPrint still reachable (non-blocking, 3 s timeout)?

The result is surfaced verbatim in the Agent Dashboard.
"""

from __future__ import annotations

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

    Round AK: neuPrint's ``/api/databaseInfo`` route no longer exists (the
    server moved platforms; it answers 404 for any token).  The test now runs a
    one-row Cypher query through ``POST /api/custom/custom`` - the exact call
    the connectome loader makes - with the token as ``Authorization: Bearer``.
    200 with rows ⇒ the token works for this dataset.  401/403 ⇒ rejected.
    If the query route itself is unavailable the public ``/api/dbmeta/datasets``
    is used to confirm the server and the dataset, and the token is reported
    "accepted by the server" only when that call succeeds *with* the header.
    """
    import httpx

    candidate = (token or "").strip()
    if not candidate:
        return {"valid": False, "error": "No token entered"}
    base = server.rstrip("/")
    headers = {"Authorization": f"Bearer {candidate}", "Content-Type": "application/json"}
    query_url = f"{base}/api/custom/custom"
    meta_url = f"{base}/api/dbmeta/datasets"
    body = {"cypher": "MATCH (m:Meta) RETURN m.dataset AS dataset, m.lastDatabaseEdit AS edited LIMIT 1",
            "dataset": dataset}
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.post(query_url, json=body, headers=headers)
            if response.status_code == 200:
                try:
                    data = (response.json() or {}).get("data") or []
                except Exception:  # noqa: BLE001
                    data = []
                found = str(data[0][0]) if data and data[0] else dataset
                return {"valid": True, "reachable": True,
                        "detail": f"token accepted · Cypher answered for dataset {found}"}
            if response.status_code in (401, 403):
                return {"valid": False, "reachable": True,
                        "error": f"neuPrint rejected the token (HTTP {response.status_code}) - copy it again from "
                                 f"{base} → Account → Auth Token (it is a long JWT, not the 64-character Google key)"}
            if response.status_code == 400:
                try:
                    msg = (response.json() or {}).get("error") or response.text
                except Exception:  # noqa: BLE001
                    msg = response.text
                if "dataset" in str(msg).lower():
                    meta = await client.get(meta_url, headers=headers)
                    names = sorted((meta.json() or {}).keys()) if meta.status_code == 200 else []
                    return {"valid": True, "reachable": True,
                            "detail": f"token accepted, but dataset {dataset!r} is unknown - set NEUPRINT_DATASET to one of: "
                                      f"{', '.join(names[:6]) or 'see ' + meta_url}"}
                return {"valid": False, "reachable": True, "error": f"neuPrint answered HTTP 400: {str(msg)[:160]}"}
            # Any other code: confirm server + dataset through the public metadata route.
            meta = await client.get(meta_url, headers=headers)
            if meta.status_code in (401, 403):
                return {"valid": False, "reachable": True, "error": f"neuPrint rejected the token (HTTP {meta.status_code})"}
            if meta.status_code == 200:
                names = list((meta.json() or {}).keys())
                known = dataset in names
                return {"valid": True, "reachable": True,
                        "detail": (f"token accepted by the server · dataset {dataset} {'present' if known else 'NOT found'}"
                                   f" · query route answered HTTP {response.status_code}")}
            return {"valid": False, "reachable": True,
                    "error": f"HTTP {response.status_code} from {query_url} and HTTP {meta.status_code} from {meta_url}"}
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


def graph_convolution(test_input: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    """Section 6.6 check 3, kept as a module-level helper."""
    return gc.graph_convolution(test_input, adjacency)
