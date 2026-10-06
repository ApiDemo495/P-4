"""Optimal-transport readings of the order book, computed once per minute.

The book arrives as ``(2, L, 2)`` arrays (bids, asks; price, qty).  Each side
is turned into a probability distribution over *distance from mid in basis
points* on a fixed 1-bp grid (0..GRID_BPS), so two snapshots taken at
different mids are comparable and the 1-D transport problems have closed
forms:

    W_p(μ, ν)^p = ∫₀¹ |F_μ⁻¹(u) − F_ν⁻¹(u)|^p du

Sinkhorn (entropic OT) is run on the same grid with a squared-distance cost.
"""
from __future__ import annotations

import math

import numpy as np

GRID_BPS = 60
GRID = np.arange(GRID_BPS + 1, dtype=np.float64)


def side_pmf(book: np.ndarray, side: int) -> np.ndarray | None:
    bids, asks = book[0], book[1]
    bb, ba = float(bids[0, 0]), float(asks[0, 0])
    if bb <= 0 or ba <= bb:
        return None
    mid = 0.5 * (bb + ba)
    rows = bids if side == 0 else asks
    px, qty = rows[:, 0], rows[:, 1]
    ok = (px > 0) & (qty > 0)
    if not ok.any():
        return None
    dist = np.abs(px[ok] - mid) / mid * 1e4
    idx = np.clip(np.round(dist).astype(int), 0, GRID_BPS)
    pmf = np.bincount(idx, weights=qty[ok], minlength=GRID_BPS + 1).astype(np.float64)
    s = pmf.sum()
    return pmf / s if s > 0 else None


def quantiles(pmf: np.ndarray, u: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(pmf)
    return GRID[np.minimum(np.searchsorted(cdf, u, side="left"), GRID_BPS)]


_U = (np.arange(64) + 0.5) / 64.0


def w_p(p1: np.ndarray, p2: np.ndarray, p: int = 1) -> float:
    q1, q2 = quantiles(p1, _U), quantiles(p2, _U)
    return float(np.mean(np.abs(q1 - q2) ** p) ** (1.0 / p))


def centroid(pmf: np.ndarray) -> float:
    return float(np.dot(pmf, GRID))


def sinkhorn(a: np.ndarray, b: np.ndarray, eps: float = 2.0, iters: int = 60) -> float:
    """Entropic OT cost ⟨P, C⟩ with C_ij = (i − j)², after ``iters`` scalings."""
    C = (GRID[:, None] - GRID[None, :]) ** 2
    K = np.exp(-C / max(eps, 1e-6))
    a = a + 1e-12
    b = b + 1e-12
    u = np.ones_like(a)
    for _ in range(iters):
        v = b / (K.T @ u + 1e-300)
        u = a / (K @ v + 1e-300)
    P = u[:, None] * K * v[None, :]
    return float(np.sum(P * C))


def barycenter(pmfs: list[np.ndarray]) -> np.ndarray:
    """1-D Wasserstein barycenter = average of quantile functions."""
    q = np.mean([quantiles(p, _U) for p in pmfs], axis=0)
    idx = np.clip(np.round(q).astype(int), 0, GRID_BPS)
    out = np.bincount(idx, minlength=GRID_BPS + 1).astype(np.float64)
    return out / out.sum()


def extrapolated_pmf(prev: np.ndarray, now: np.ndarray, t: float = 2.0) -> np.ndarray:
    """McCann displacement interpolation extended past t=1: quantile
    q_t = q_prev + t (q_now − q_prev)."""
    q = quantiles(prev, _U) + t * (quantiles(now, _U) - quantiles(prev, _U))
    idx = np.clip(np.round(q).astype(int), 0, GRID_BPS)
    out = np.bincount(idx, minlength=GRID_BPS + 1).astype(np.float64)
    return out / out.sum()


def minute_readings(first_book: np.ndarray | None, last_book: np.ndarray | None,
                    books_in_minute: list[np.ndarray]) -> dict:
    """All OT columns for one closed minute.  NaN when the book was absent."""
    out = {k: math.nan for k in ("w1_bid", "w1_ask", "w2_bid", "w2_ask", "sink_bid", "sink_ask",
                                 "bid_centroid", "ask_centroid", "bary_bid", "bary_ask", "ot_extrap_imb")}
    if first_book is None or last_book is None:
        return out
    fb, fa = side_pmf(first_book, 0), side_pmf(first_book, 1)
    lb, la = side_pmf(last_book, 0), side_pmf(last_book, 1)
    if fb is None or fa is None or lb is None or la is None:
        return out
    out["w1_bid"], out["w1_ask"] = w_p(fb, lb, 1), w_p(fa, la, 1)
    out["w2_bid"], out["w2_ask"] = w_p(fb, lb, 2), w_p(fa, la, 2)
    out["sink_bid"], out["sink_ask"] = sinkhorn(fb, lb), sinkhorn(fa, la)
    out["bid_centroid"], out["ask_centroid"] = centroid(lb), centroid(la)
    pb = [p for p in (side_pmf(b, 0) for b in books_in_minute[-5:]) if p is not None]
    pa = [p for p in (side_pmf(b, 1) for b in books_in_minute[-5:]) if p is not None]
    if len(pb) >= 2 and len(pa) >= 2:
        out["bary_bid"] = w_p(lb, barycenter(pb), 1)
        out["bary_ask"] = w_p(la, barycenter(pa), 1)
    eb, ea = extrapolated_pmf(fb, lb), extrapolated_pmf(fa, la)
    # extrapolated book: a side whose mass is projected to sit closer to mid is the pressing side
    cb, ca = centroid(eb), centroid(ea)
    out["ot_extrap_imb"] = (ca - cb) / (ca + cb + 1e-9)
    return out


def trade_flow_transport(buy_prices: list[float], sell_prices: list[float], mid: float) -> tuple[float, float]:
    """TFTC: W1 between buy-print and sell-print price distributions (bps of
    mid) and the direction mean(buys) − mean(sells)."""
    if len(buy_prices) < 2 or len(sell_prices) < 2 or mid <= 0:
        return math.nan, math.nan
    b = (np.asarray(buy_prices) / mid - 1.0) * 1e4
    s = (np.asarray(sell_prices) / mid - 1.0) * 1e4
    u = (np.arange(32) + 0.5) / 32.0
    w1 = float(np.mean(np.abs(np.quantile(b, u) - np.quantile(s, u))))
    return w1, float(b.mean() - s.mean())
