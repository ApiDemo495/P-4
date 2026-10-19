"""Helpers every domain module shares."""
from __future__ import annotations

import numpy as np

from backend.genesis import ops
from backend.genesis.features import Frame


def finish(frame: Frame, signal: np.ndarray, raw: np.ndarray, how: str = "tanh", n: int = 55,
           scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Normalise a signal (per the variant's ``norm``) and sanitise both arrays."""
    with np.errstate(all="ignore"):
        sig = ops.normalize(np.asarray(signal, dtype=float), how, n=max(n, 21), scale=scale)
    sig = np.where(np.isfinite(sig), np.clip(sig, -1.0, 1.0), np.nan)
    raw = np.where(np.isfinite(raw), raw, np.nan)
    return sig, raw


def momentum_sign(frame: Frame, k: int = 3) -> np.ndarray:
    """Sign of the last k-candle move - the 'last move' regime formulas follow or fade."""
    return ops.sign(frame.close - ops.shift(frame.close, k))


def path_pair(frame: Frame, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) 2-D path coordinates for a signature / transport kernel."""
    if kind == "price_trades":
        return frame.close, frame.trades
    if kind == "price_depth":
        return frame.close, np.nan_to_num(frame.bid_depth, nan=np.nan)
    if kind == "logret_vol":
        return np.nancumsum(frame.ret), np.nancumsum(frame.vol_ret)
    if kind == "price_other":
        return frame.close, frame.other_close
    return frame.close, frame.volume


def scale_of(frame: Frame, n: int) -> np.ndarray:
    """Local return scale (σ of returns over n), floored."""
    return frame.roll("ret_std", max(n, 5)) + 1e-6


def tail_windows(frame: Frame, n: int, cols: list[np.ndarray], minmax: bool = True):
    """Yield ``(t, W)`` for the last ``frame.eval_tail`` positions, where W is the
    (n, d) point cloud of the trailing window, min-max normalised per column.
    Windows containing NaN are skipped."""
    N = frame.n
    start = tail_start(frame, n)
    M = np.vstack(cols).T  # (N, d)
    for t in range(start, N):
        W = M[t - n + 1: t + 1]
        if np.isnan(W).any():
            continue
        if minmax:
            lo, hi = W.min(axis=0), W.max(axis=0)
            W = (W - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
        yield t, W


def pdist_sq(W: np.ndarray) -> np.ndarray:
    G = W @ W.T
    sq = np.diag(G)
    D = sq[:, None] + sq[None, :] - 2.0 * G
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def components(adj: np.ndarray) -> int:
    """Number of connected components of an undirected adjacency matrix."""
    n = len(adj)
    seen = np.zeros(n, dtype=bool)
    count = 0
    for s in range(n):
        if seen[s]:
            continue
        count += 1
        frontier = np.zeros(n, dtype=bool)
        frontier[s] = True
        seen[s] = True
        while frontier.any():
            reach = adj[frontier].any(axis=0) & ~seen
            seen |= reach
            frontier = reach
    return count


def tail_start(frame: Frame, n: int) -> int:
    return max(n - 1, frame.n - int(frame.eval_tail))


def mst_lengths(D: np.ndarray) -> np.ndarray:
    """Prim's MST edge lengths = death times of the H0 persistence barcode."""
    n = len(D)
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
    best[0] = 0.0
    out = []
    for _ in range(n):
        u = int(np.argmin(np.where(in_tree, np.inf, best)))
        in_tree[u] = True
        if best[u] > 0:
            out.append(best[u])
        best = np.where(in_tree, best, np.minimum(best, D[u]))
    return np.sqrt(np.asarray(out))
