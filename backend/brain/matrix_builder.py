"""Build / load / cache the 80x80 mushroom-body adjacency matrix.

Two producers feed the same format:

* ``query_circuits.py`` - live neuPrint or FlyWire (CAVE) queries
* ``generate_fallback_matrix.py`` - the committed CSV that must always exist

and one consumer: ``graph_convolution.GraphConvolution``.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path

import numpy as np

from backend.brain import graph_convolution as gc
from backend.core import config as cfg

log = logging.getLogger("drosophila.brain.matrix")

CSV_HEADER = ("source_idx", "target_idx", "weight", "source_type", "target_type")

#: Human-readable names for every index, written into the CSV for inspection.
NODE_TYPES: list[str] = (
    [f"PN_{name}" for name in gc.PN_NAMES]
    + [f"KC_cluster_{i}" for i in range(gc.N_KC)]
    + ["DAN_PAM", "DAN_PPL1", "OA_VUM"]
    + ["MBON_alpha3", "MBON_gamma5beta2a", "MBON_beta2beta2a", "MBON_alpha2"]
    + ["LH_approach", "LH_avoid", "LH_neutral"]
)


def empty_matrix() -> np.ndarray:
    return np.zeros((gc.N_NODES, gc.N_NODES), dtype=np.float64)


def set_edge(matrix: np.ndarray, src: int, dst: int, weight: float) -> None:
    matrix[src, dst] = float(weight)


def scale_to_unit(matrix: np.ndarray) -> np.ndarray:
    """Divide by max |weight| so every entry lies in [-1, 1]."""
    peak = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    if peak <= 0:
        return matrix
    return matrix / peak


def save_csv(matrix: np.ndarray, path: Path | str) -> Path:
    """Persist the matrix in the sparse edge-list format from Section 6.3f."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for src in range(gc.N_NODES):
            for dst in range(gc.N_NODES):
                weight = float(matrix[src, dst])
                if abs(weight) < 1e-9:
                    continue
                writer.writerow(
                    [src, dst, f"{weight:.6f}", NODE_TYPES[src], NODE_TYPES[dst]]
                )
                rows += 1
    log.info("Saved brain matrix: %s (%d edges)", path, rows)
    return path


def load_csv(path: Path | str) -> np.ndarray:
    """Load an edge-list CSV into a dense 80x80 matrix."""
    path = Path(path)
    matrix = empty_matrix()
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path} is empty")
        # Tolerate the spec's header ordering as well as a headerless file.
        has_header = header and not header[0].strip().lstrip("-").isdigit()
        if not has_header:
            fh.seek(0)
            reader = csv.reader(fh)
        for row in reader:
            if not row or len(row) < 3:
                continue
            try:
                src, dst, weight = int(row[0]), int(row[1]), float(row[2])
            except (ValueError, IndexError):
                continue
            if 0 <= src < gc.N_NODES and 0 <= dst < gc.N_NODES:
                matrix[src, dst] = weight
    return matrix


def checksum(matrix: np.ndarray) -> str:
    """Stable checksum.

    The CSV stores weights with 6 decimals, so hashing raw float64 bytes would
    give a different digest before and after a save/load round-trip.  Rounding
    first makes the checksum identical whether the matrix came from neuPrint,
    from Redis or from the committed file.
    """
    rounded = np.round(np.ascontiguousarray(matrix, dtype=np.float64), 6)
    return hashlib.md5(rounded.tobytes()).hexdigest()[:8]


def stats(matrix: np.ndarray) -> dict:
    non_zero = matrix[matrix != 0]
    return {
        "shape": list(matrix.shape),
        "edges": int(np.count_nonzero(matrix)),
        "positive_edges": int(np.count_nonzero(matrix > 0)),
        "inhibitory_edges": int(np.count_nonzero(matrix < 0)),
        "max_abs_weight": round(float(np.max(np.abs(matrix))) if matrix.size else 0.0, 6),
        "mean_abs_weight": round(float(np.mean(np.abs(non_zero))) if non_zero.size else 0.0, 6),
        "checksum": checksum(matrix),
    }


def to_edge_list(matrix: np.ndarray, limit: int | None = None) -> list[dict]:
    edges: list[dict] = []
    for src in range(gc.N_NODES):
        for dst in range(gc.N_NODES):
            weight = float(matrix[src, dst])
            if abs(weight) < 1e-9:
                continue
            edges.append(
                {
                    "source": src,
                    "target": dst,
                    "weight": round(weight, 5),
                    "source_type": NODE_TYPES[src],
                    "target_type": NODE_TYPES[dst],
                }
            )
            if limit is not None and len(edges) >= limit:
                return edges
    return edges


def default_fallback_path() -> Path:
    return cfg.BRAIN_FALLBACK_DIR / "mb_adjacency_80x80.csv"
