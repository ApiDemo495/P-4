"""Spectral clustering of Kenyon Cells: ~2000 neurons -> 50 clusters (6.3b).

Steps, exactly as the specification lays them out:

    1. build the sparse KC -> KC adjacency W_KC
    2. graph Laplacian L = D - W_KC
    3. first 50 eigenvectors of L (scipy.sparse.linalg.eigsh)
    4. k-means (k = 50) on the eigenvector embedding
    5. per cluster, keep the highest-degree KC as the representative and sum the
       intra-cluster edge weight

The live path needs scipy; the fallback chunking path keeps the pipeline alive
when scipy or the query result is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("drosophila.brain.cluster")

N_CLUSTERS = 50


@dataclass
class ClusterResult:
    labels: np.ndarray
    representatives: list[int]
    intra_cluster_weight: list[float]
    method: str
    explained: float = 0.0


def _fallback_labels(n_items: int, n_clusters: int) -> np.ndarray:
    """Deterministic contiguous chunking when spectral clustering is unavailable."""
    labels = np.zeros(n_items, dtype=np.int32)
    if n_items == 0:
        return labels
    edges = np.linspace(0, n_items, n_clusters + 1).astype(int)
    for cluster in range(n_clusters):
        start, stop = edges[cluster], edges[cluster + 1]
        labels[start:stop] = cluster
    return labels


def cluster_kenyon_cells(
    body_ids: list[int],
    edges: list[tuple[int, int, float]],
    n_clusters: int = N_CLUSTERS,
) -> ClusterResult:
    """Cluster KCs by the spectrum of their connectivity graph."""
    n_items = len(body_ids)
    if n_items == 0:
        return ClusterResult(np.zeros(0, dtype=np.int32), [], [], "empty")

    index_of = {int(body): i for i, body in enumerate(body_ids)}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for source, target, weight in edges:
        i = index_of.get(int(source))
        j = index_of.get(int(target))
        if i is None or j is None or i == j:
            continue
        rows.append(i)
        cols.append(j)
        vals.append(float(weight))

    if n_items < n_clusters * 2 or len(vals) < n_items:
        labels = _fallback_labels(n_items, n_clusters)
        return _finish(body_ids, labels, rows, cols, vals, "chunked")

    try:
        from scipy.cluster.vq import kmeans2
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import eigsh

        w = coo_matrix((vals, (rows, cols)), shape=(n_items, n_items)).tocsr()
        w = w.maximum(w.T)  # symmetrise
        degree = np.asarray(w.sum(axis=1)).ravel()
        lap = coo_matrix(
            (
                np.concatenate([degree, -w.data]),
                (
                    np.concatenate([np.arange(n_items), w.tocoo().row]),
                    np.concatenate([np.arange(n_items), w.tocoo().col]),
                ),
            ),
            shape=(n_items, n_items),
        ).tocsr()

        k = min(n_clusters, n_items - 2)
        values, vectors = eigsh(lap.astype(np.float64), k=k, which="SM")
        order = np.argsort(values)
        vectors = vectors[:, order]
        # Row-normalise the embedding, standard practice before k-means.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embedding = vectors / norms

        rng = np.random.default_rng(7)
        centroids, labels = kmeans2(embedding, k, minit="++", seed=rng, missing="warn")
        labels = np.asarray(labels, dtype=np.int32)
        explained = float(
            np.sum(values[order][:k]) / max(np.sum(degree), 1e-9)
        )
        return _finish(body_ids, labels, rows, cols, vals, "spectral", explained)
    except Exception as exc:  # noqa: BLE001 - clustering is best-effort
        log.warning("Spectral clustering unavailable (%s); using chunked fallback", exc)
        labels = _fallback_labels(n_items, n_clusters)
        return _finish(body_ids, labels, rows, cols, vals, "chunked")


def _finish(
    body_ids: list[int],
    labels: np.ndarray,
    rows: list[int],
    cols: list[int],
    vals: list[float],
    method: str,
    explained: float = 0.0,
) -> ClusterResult:
    """Pick a representative and measure total internal weight per cluster."""
    n_items = len(body_ids)
    degree = np.zeros(n_items, dtype=np.float64)
    for i, j, w in zip(rows, cols, vals):
        degree[i] += abs(w)

    representatives: list[int] = []
    intra: list[float] = []
    unique = sorted(set(int(x) for x in labels))
    for cluster in unique:
        members = np.where(labels == cluster)[0]
        if members.size == 0:
            continue
        best = members[int(np.argmax(degree[members]))]
        representatives.append(int(body_ids[best]))
        member_set = set(int(m) for m in members)
        intra.append(
            float(sum(w for i, j, w in zip(rows, cols, vals) if i in member_set and j in member_set))
        )
    return ClusterResult(labels, representatives, intra, method, explained)


def cluster_summary(result: ClusterResult) -> dict:
    labels = result.labels
    return {
        "method": result.method,
        "clusters": int(len(set(int(x) for x in labels))) if labels.size else 0,
        "representatives": len(result.representatives),
        "total_intra_cluster_weight": round(float(sum(result.intra_cluster_weight)), 4),
        "explained_eigenvalue_mass": round(result.explained, 4),
    }
