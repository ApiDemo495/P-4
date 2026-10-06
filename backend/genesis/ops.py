"""Vectorised primitives shared by every domain kernel (Layer 1/2 helpers).

All functions take and return 1-D float arrays aligned with the candle axis;
positions where a quantity is undefined (not enough history) are NaN.  Nothing
here loops in Python over the candle axis except the few algorithms that are
inherently sequential (Lempel-Ziv parsing, p-adic valuation), and those are
O(n) with tiny constants.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

EPS = 1e-12
WINDOWS = (3, 5, 8, 13, 21, 34, 55)
NORMS = ("z", "tanh", "rank")


def nan(n: int) -> np.ndarray:
    return np.full(int(n), np.nan)


def shift(x: np.ndarray, k: int = 1) -> np.ndarray:
    out = nan(len(x))
    if k >= 0:
        out[k:] = x[: len(x) - k] if k else x
    else:
        out[:k] = x[-k:]
    return out


def diff(x: np.ndarray, k: int = 1) -> np.ndarray:
    return x - shift(x, k)


def windows(x: np.ndarray, n: int) -> np.ndarray:
    """(len(x), n) view of trailing windows; rows before n-1 are NaN-padded."""
    n = int(max(1, n))
    if len(x) < n:
        return np.full((len(x), n), np.nan)
    w = sliding_window_view(x, n)
    pad = np.full((n - 1, n), np.nan)
    return np.vstack((pad, w))


def rolling(x: np.ndarray, n: int, fn) -> np.ndarray:
    w = windows(x, n)
    with np.errstate(all="ignore"):
        return fn(w, axis=1)


def rmean(x, n):
    return rolling(x, n, np.nanmean)


def rstd(x, n):
    return rolling(x, n, np.nanstd)


def rsum(x, n):
    return rolling(x, n, np.nansum)


def rmax(x, n):
    return rolling(x, n, np.nanmax)


def rmin(x, n):
    return rolling(x, n, np.nanmin)


def rmedian(x, n):
    return rolling(x, n, np.nanmedian)


def rskew(x, n):
    w = windows(x, n)
    with np.errstate(all="ignore"):
        m = np.nanmean(w, axis=1, keepdims=True)
        s = np.nanstd(w, axis=1, keepdims=True) + EPS
        return np.nanmean(((w - m) / s) ** 3, axis=1)


def rkurt(x, n):
    w = windows(x, n)
    with np.errstate(all="ignore"):
        m = np.nanmean(w, axis=1, keepdims=True)
        s = np.nanstd(w, axis=1, keepdims=True) + EPS
        return np.nanmean(((w - m) / s) ** 4, axis=1) - 3.0


def ema(x: np.ndarray, n: int) -> np.ndarray:
    alpha = 2.0 / (n + 1.0)
    out = nan(len(x))
    acc = math.nan
    for i, v in enumerate(x):
        if math.isnan(v):
            out[i] = acc
            continue
        acc = v if math.isnan(acc) else alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def zscore(x: np.ndarray, n: int = 55) -> np.ndarray:
    with np.errstate(all="ignore"):
        return (x - rmean(x, n)) / (rstd(x, n) + EPS)


def rrank(x: np.ndarray, n: int = 55) -> np.ndarray:
    """Percentile rank of the latest value inside its trailing window, in [-1, 1]."""
    w = windows(x, n)
    with np.errstate(all="ignore"):
        last = w[:, -1:]
        r = np.nanmean(np.where(np.isnan(w), np.nan, (w < last).astype(float)), axis=1)
    return 2.0 * r - 1.0


def normalize(x: np.ndarray, how: str, n: int = 55, scale: float = 1.0) -> np.ndarray:
    """z: tanh(z/2) · tanh: tanh(x/scale) · rank: rolling percentile -> [-1, 1]."""
    with np.errstate(all="ignore"):
        if how == "z":
            return np.tanh(zscore(x, n) / 2.0)
        if how == "rank":
            return rrank(x, n)
        return np.tanh(x / (scale + EPS))


def clip1(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0)


def sign(x: np.ndarray) -> np.ndarray:
    return np.sign(np.nan_to_num(x))


def logret(close: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.log(close) - np.log(shift(close, 1))


def safe_log(x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.log(np.where(x > 0, x, np.nan))


def entropy_bits(p: np.ndarray, axis: int = -1) -> np.ndarray:
    with np.errstate(all="ignore"):
        q = np.where(p > 0, p, 1.0)
        return -np.nansum(p * np.log2(q), axis=axis)


def binarize(x: np.ndarray) -> np.ndarray:
    return (np.nan_to_num(x) > 0).astype(np.int8)


def ternarize(x: np.ndarray, band: float) -> np.ndarray:
    y = np.nan_to_num(x)
    return np.where(y > band, 2, np.where(y < -band, 0, 1)).astype(np.int8)


def lz76(seq) -> int:
    """Lempel-Ziv (1976) complexity: number of distinct phrases in the
    left-to-right exhaustive parse (Kaspar-Schuster algorithm)."""
    s = [int(v) for v in seq]
    n = len(s)
    if n == 0:
        return 0
    i, k, l = 0, 1, 1
    c, k_max = 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c


def rolling_lz(sym: np.ndarray, n: int, alphabet: int = 2) -> np.ndarray:
    """Normalised LZ complexity c · log_a(n) / n over trailing windows."""
    out = nan(len(sym))
    if len(sym) < n:
        return out
    norm = math.log(n, alphabet) / n
    for t in range(n - 1, len(sym)):
        out[t] = lz76(sym[t - n + 1: t + 1]) * norm
    return out


def padic_valuation(k: np.ndarray, p: int) -> np.ndarray:
    """v_p(k) for integer arrays (v_p(0) := 0 by convention here)."""
    k = np.abs(np.nan_to_num(k)).astype(np.int64)
    v = np.zeros(len(k), dtype=np.float64)
    live = k > 0
    cur = k.copy()
    while True:
        div = live & (cur % p == 0)
        if not div.any():
            break
        v[div] += 1
        cur[div] //= p
    return v


def hurst_rs(x: np.ndarray, n: int) -> np.ndarray:
    """Rescaled-range Hurst exponent over trailing windows (two sub-scales)."""
    w = windows(x, n)
    out = nan(len(x))
    ok = ~np.isnan(w).any(axis=1)
    if not ok.any():
        return out
    ww = w[ok]
    scales = [max(4, n // 4), max(8, n // 2), n]
    logs_rs, logs_n = [], []
    for m in scales:
        segs = ww[:, -m:]
        dev = segs - segs.mean(axis=1, keepdims=True)
        cum = np.cumsum(dev, axis=1)
        r = cum.max(axis=1) - cum.min(axis=1)
        s = segs.std(axis=1) + EPS
        logs_rs.append(np.log((r / s) + EPS))
        logs_n.append(math.log(m))
    A = np.vstack([np.asarray(logs_n), np.ones(len(scales))]).T
    Y = np.vstack(logs_rs)  # scales x rows
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    out[ok] = np.clip(coef[0], 0.0, 1.0)
    return out


def wasserstein1_cdf(p: np.ndarray, q: np.ndarray, grid: np.ndarray) -> float:
    """W1 between two discrete distributions on the same sorted grid."""
    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)
    cdf_p, cdf_q = np.cumsum(p), np.cumsum(q)
    dx = np.diff(grid, append=grid[-1] + (grid[-1] - grid[-2] if len(grid) > 1 else 1.0))
    return float(np.sum(np.abs(cdf_p - cdf_q) * dx))


def wasserstein1_samples(a: np.ndarray, b: np.ndarray) -> float:
    """1-D W1 between two empirical samples = ∫|F_a - F_b| via sorted quantiles."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if a.size == 0 or b.size == 0:
        return math.nan
    grid = np.linspace(0, 1, 64, endpoint=False) + 0.5 / 64
    return float(np.mean(np.abs(np.quantile(a, grid) - np.quantile(b, grid))))


def grunwald_letnikov(x: np.ndarray, alpha: float, n: int) -> np.ndarray:
    """Fractional derivative of order alpha, GL weights truncated at n."""
    w = np.ones(n)
    for k in range(1, n):
        w[k] = w[k - 1] * (k - 1 - alpha) / k
    W = windows(x, n)[:, ::-1]  # most recent first
    with np.errstate(all="ignore"):
        return np.nansum(W * w, axis=1) * (~np.isnan(W).any(axis=1))
