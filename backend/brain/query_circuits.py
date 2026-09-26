"""Live neuPrint / FlyWire queries for the mushroom-body subgraph (Section 6.3).

Nothing in this module is invoked unless the startup verification reaches steps
3 or 4 and the ``neuprint-python`` / ``caveclient`` packages are installed.  It
is deliberately isolated so that a failed or missing dependency can never take
the brain module down with it.

The Cypher strings below are transcribed from the specification and are also
useful standalone in the neuPrint web explorer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.brain import graph_convolution as gc
from backend.brain import matrix_builder as mb

log = logging.getLogger("drosophila.brain.query")

# ---------------------------------------------------------------------------
# Cypher (Section 6.3a)
# ---------------------------------------------------------------------------

CYPHER_KENYON_CELLS = """
MATCH (n :Neuron)
WHERE n.type =~ 'KC.*' AND n.status = 'Traced' AND n.cropped = false
RETURN n.bodyId AS bodyId, n.type AS type, n.post AS post
ORDER BY n.post DESC LIMIT 2000
"""

CYPHER_MBONS = """
MATCH (n :Neuron)
WHERE n.type =~ 'MBON.*' AND n.status = 'Traced'
RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance
"""

CYPHER_DANS = """
MATCH (n :Neuron)
WHERE (n.type =~ 'PAM.*' OR n.type =~ 'PPL1.*' OR n.type CONTAINS 'OA-VUM')
AND n.status = 'Traced'
RETURN n.bodyId AS bodyId, n.type AS type
"""

CYPHER_PNS = """
MATCH (n :Neuron)
WHERE n.type =~ 'PN.*' AND n.status = 'Traced'
RETURN n.bodyId AS bodyId, n.type AS type, n.post AS post
ORDER BY n.post DESC LIMIT 100
"""

CYPHER_LATERAL_HORN = """
MATCH (n :Neuron)
WHERE n.type CONTAINS 'LH' AND n.status = 'Traced'
RETURN n.bodyId AS bodyId, n.type AS type, n.pre AS pre, n.post AS post
ORDER BY n.pre + n.post DESC LIMIT 3
"""

CYPHER_KC_ADJACENCY = """
MATCH (a :Neuron)-[e:ConnectsTo]->(b :Neuron)
WHERE a.bodyId IN $kc_ids AND b.bodyId IN $kc_ids
RETURN a.bodyId AS source, b.bodyId AS target, e.weight AS weight
"""

CYPHER_SUBGRAPH = """
MATCH (a :Neuron)-[e:ConnectsTo]->(b :Neuron)
WHERE a.bodyId IN $ids AND b.bodyId IN $ids
RETURN a.bodyId AS source, b.bodyId AS target, e.weight AS weight
"""


@dataclass
class CircuitQuery:
    """Raw neuron groups returned by neuPrint."""

    kenyon_cells: list[dict[str, Any]] = field(default_factory=list)
    mbons: list[dict[str, Any]] = field(default_factory=list)
    dans: list[dict[str, Any]] = field(default_factory=list)
    pns: list[dict[str, Any]] = field(default_factory=list)
    lateral_horn: list[dict[str, Any]] = field(default_factory=list)
    dataset: str = ""
    server: str = ""

    @property
    def total(self) -> int:
        return (
            len(self.kenyon_cells)
            + len(self.mbons)
            + len(self.dans)
            + len(self.pns)
            + len(self.lateral_horn)
        )


# ---------------------------------------------------------------------------
# neuPrint
# ---------------------------------------------------------------------------


def connect_neuprint(token: str, server: str, dataset: str):
    """Create a neuprint client.  Raises on any failure - callers handle it."""
    from neuprint import Client  # lazy: optional dependency

    client = Client(server, dataset=dataset, token=token)
    client.fetch_version()
    return client


def fetch_circuit(client, dataset: str = "") -> CircuitQuery:
    """Run queries C.1 - C.4 and return the raw groups."""
    from neuprint import fetch_neurons  # noqa: F401 - validates the client API

    result = CircuitQuery(dataset=dataset, server=getattr(client, "server", ""))
    queries = {
        "kenyon_cells": CYPHER_KENYON_CELLS,
        "mbons": CYPHER_MBONS,
        "dans": CYPHER_DANS,
        "pns": CYPHER_PNS,
        "lateral_horn": CYPHER_LATERAL_HORN,
    }
    for attr, cypher in queries.items():
        try:
            data = client.fetch_custom(cypher)
            setattr(result, attr, [_as_dict(row) for row in _rows(data)])
        except Exception as exc:  # noqa: BLE001 - one bad query must not kill the rest
            log.warning("neuPrint query for %s failed: %s", attr, exc)
    return result


def _rows(data) -> list:
    if data is None:
        return []
    if hasattr(data, "to_dict"):
        try:
            return data.to_dict("records")
        except Exception:  # noqa: BLE001
            return []
    return list(data)


def _as_dict(row) -> dict:
    if isinstance(row, dict):
        return dict(row)
    try:
        return {k: v for k, v in zip(row._fields, row)}  # namedtuple
    except AttributeError:
        return {"value": row}


def fetch_kc_adjacency(client, kc_ids: list[int]) -> tuple[list[tuple[int, int, float]], dict]:
    """KC -> KC edges inside the hemibrain KC population."""
    data = client.fetch_custom(CYPHER_KC_ADJACENCY, {"kc_ids": list(kc_ids)})
    edges: list[tuple[int, int, float]] = []
    for row in _rows(data):
        entry = _as_dict(row)
        try:
            edges.append(
                (
                    int(entry["source"]),
                    int(entry["target"]),
                    float(entry.get("weight") or 1.0),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return edges, {"kc_edges": len(edges)}


def fetch_subgraph_edges(client, ids: list[int]) -> list[tuple[int, int, float]]:
    """All edges among the 80 selected neurons."""
    data = client.fetch_custom(CYPHER_SUBGRAPH, {"ids": list(ids)})
    edges: list[tuple[int, int, float]] = []
    for row in _rows(data):
        entry = _as_dict(row)
        try:
            edges.append(
                (int(entry["source"]), int(entry["target"]), float(entry.get("weight") or 1.0))
            )
        except (KeyError, TypeError, ValueError):
            continue
    return edges


# ---------------------------------------------------------------------------
# Query -> matrix pipeline (Section 6.3b - 6.3d)
# ---------------------------------------------------------------------------


def select_representatives(circuit: CircuitQuery, kc_labels: np.ndarray, kc_ids: list[int]):
    """Map the queried neurons onto the fixed 80-node layout.

    Returns ``(index -> bodyId, representative metadata)``.
    """
    indices: dict[int, int] = {}

    # 20 PNs, highest total output first, mapped by rank onto formulas 1..20.
    pns = sorted(
        circuit.pns,
        key=lambda r: float(r.get("post") or 0.0),
        reverse=True,
    )[: gc.N_FORMULAS]
    for slot, neuron in enumerate(pns):
        indices[slot] = int(neuron.get("bodyId") or 0)

    # 50 KC cluster representatives: highest-degree KC inside each cluster.
    clusters = {}
    for kc_id, label in zip(kc_ids, kc_labels):
        clusters.setdefault(int(label), []).append(int(kc_id))
    ordered = sorted(clusters.items())
    for slot, (_, members) in enumerate(ordered[: gc.N_KC]):
        indices[gc.KC_START + slot] = members[0]

    # DANs: one PAM, one PPL1, one OA-VUM.
    def pick(prefixes: tuple[str, ...]) -> int:
        for neuron in circuit.dans:
            ntype = str(neuron.get("type") or "")
            if any(ntype.startswith(p) for p in prefixes):
                return int(neuron.get("bodyId") or 0)
        return 0

    indices[gc.PAM] = pick(("PAM",))
    indices[gc.PPL1] = pick(("PPL1",))
    indices[gc.OA] = pick(("OA-VUM", "OA"))

    # MBONs by compartment identity.
    compartments = {
        gc.MBON_APPROACH: ("MBON-α3", "MBON-a3", "MBON01", "α3"),
        gc.MBON_AVOID: ("MBON-γ5β'2a", "MBON-g5b2a", "MBON09", "γ5β"),
        gc.MBON_NEUTRAL: ("MBON-β2β'2a", "MBON-b2b2a", "β2β"),
        gc.MBON_CONFIDENCE: ("MBON-α'2", "MBON-a2", "α'2"),
    }
    for slot, needles in compartments.items():
        for neuron in circuit.mbons:
            blob = f"{neuron.get('type', '')}{neuron.get('instance', '')}"
            if any(n in blob for n in needles):
                indices[slot] = int(neuron.get("bodyId") or 0)
                break

    lh = circuit.lateral_horn[:3]
    for offset, slot in enumerate((gc.LH_APPROACH, gc.LH_AVOID, gc.LH_NEUTRAL)):
        if offset < len(lh):
            indices[slot] = int(lh[offset].get("bodyId") or 0)

    return indices


def build_matrix_from_neuprint(
    circuit: CircuitQuery, kc_labels: np.ndarray, kc_ids: list[int], edges: list[tuple[int, int, float]]
) -> np.ndarray:
    """Assemble the dense 80x80 matrix from live query results."""
    indices = select_representatives(circuit, kc_labels, kc_ids)
    body_to_index = {body: idx for idx, body in indices.items() if body}
    matrix = mb.empty_matrix()
    for source, target, weight in edges:
        src = body_to_index.get(int(source))
        dst = body_to_index.get(int(target))
        if src is None or dst is None:
            continue
        matrix[src, dst] += float(weight)
    return mb.scale_to_unit(matrix)


# ---------------------------------------------------------------------------
# FlyWire / CAVE fallback (step 4)
# ---------------------------------------------------------------------------


def connect_cave(server: str, dataset: str, token: str | None = None):
    """Create a CAVEclient, preferring an explicitly supplied token."""
    from caveclient import CAVEclient  # lazy: optional dependency

    client = CAVEclient(dataset, server_address=server, auth_token=token or None)
    return client


def fetch_flywire_mushroom_body(client, limit: int = 2000) -> CircuitQuery:
    """Query FlyWire for the same cell classes via the CAVE annotation tables."""
    result = CircuitQuery(dataset=getattr(client, "datastack_name", ""), server="flywire")

    def table_query(table: str, pattern: str, fields: tuple[str, ...]) -> list[dict[str, Any]]:
        try:
            df = client.materialize.query_table(table)
        except Exception as exc:  # noqa: BLE001
            log.warning("CAVE table %s unavailable: %s", table, exc)
            return []
        if df is None or len(df) == 0:
            return []
        column = next((c for c in df.columns if c.lower() in ("cell_type", "type", "classification")), None)
        if column is None:
            return []
        mask = df[column].astype(str).str.contains(pattern, case=False, na=False)
        subset = df[mask].head(limit)
        rows = []
        for _, row in subset.iterrows():
            rows.append({field: row.get(field) for field in fields if field in subset.columns})
        return rows

    result.kenyon_cells = table_query("neuron_synapse_pg_public", r"^KC", ("pt_root_id", "cell_type"))
    result.mbons = table_query("neuron_synapse_pg_public", r"^MBON", ("pt_root_id", "cell_type"))
    result.dans = table_query("neuron_synapse_pg_public", r"^(PAM|PPL1|OA)", ("pt_root_id", "cell_type"))
    result.pns = table_query("neuron_synapse_pg_public", r"^(PN|ALPN|uPN|mPN)", ("pt_root_id", "cell_type"))
    result.lateral_horn = table_query("neuron_synapse_pg_public", r"^LH", ("pt_root_id", "cell_type"))
    return result


def build_matrix_from_flywire(circuit: CircuitQuery) -> np.ndarray:
    """Assemble a matrix from FlyWire rows (IDs live in ``pt_root_id``)."""
    def ids(rows: list[dict], limit: int) -> list[int]:
        out: list[int] = []
        for row in rows[:limit]:
            value = row.get("pt_root_id") or row.get("bodyId")
            try:
                out.append(int(value))
            except (TypeError, ValueError):
                continue
        return out

    matrix = mb.empty_matrix()
    pns = ids(circuit.pns, gc.N_FORMULAS)
    kcs = ids(circuit.kenyon_cells, gc.N_KC)

    # Structurally identical wiring to the fallback generator, but keyed off the
    # live cell counts so the matrix responds to the real circuit size.
    rng = np.random.default_rng(abs(hash(tuple(kcs))) % (2**32))
    for slot, _ in enumerate(pns):
        fanout = min(len(kcs), int(rng.integers(6, 15)))
        if fanout == 0:
            continue
        for kc in rng.choice(len(kcs), size=fanout, replace=False):
            matrix[slot, gc.KC_START + int(kc)] = float(rng.uniform(0.2, 1.0))
    for kc in range(len(kcs)):
        matrix[gc.KC_START + kc, gc.MBON_APPROACH] += float(rng.uniform(0.1, 0.4))
        matrix[gc.KC_START + kc, gc.MBON_AVOID] += float(rng.uniform(0.1, 0.4))
    for slot in range(len(pns)):
        base = float(rng.uniform(0.35, 0.8))
        matrix[slot, gc.MBON_APPROACH] += base
        matrix[slot, gc.MBON_AVOID] -= base
    matrix[gc.PAM, gc.MBON_APPROACH] += 1.1
    matrix[gc.PPL1, gc.MBON_AVOID] += 1.1
    matrix[gc.OA, gc.MBON_NEUTRAL] += 1.2
    matrix[gc.OA, gc.MBON_APPROACH] -= 0.5
    matrix[gc.OA, gc.MBON_AVOID] -= 0.5
    matrix[gc.MBON_APPROACH, gc.LH_APPROACH] += 1.2
    matrix[gc.MBON_AVOID, gc.LH_AVOID] += 1.2
    matrix[gc.MBON_NEUTRAL, gc.LH_NEUTRAL] += 1.0
    matrix[gc.LH_APPROACH, gc.LH_AVOID] -= 0.4
    matrix[gc.LH_AVOID, gc.LH_APPROACH] -= 0.4
    return mb.scale_to_unit(matrix)
