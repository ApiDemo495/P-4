"""Shared mathematics for Category I - the formulas designed for the 53 s
contest window (Round AX).

Everything here reads the frozen snapshot only: the tick tape
``[time_ms, price, qty, side]`` (quote prints carry qty = 0 / side = 0) and
the two order-book frames ``book`` / ``book_prev`` of shape (2, L, 2) -
bids then asks, each level ``(price, qty)``.

The contest rule shapes these helpers: a window is won on the *last trade*
at :53 against the last trade at :00, strictly - an unchanged price is a
loss.  So the tick size, the horizon-matched return distribution and the
best quotes are what every formula in this category starts from.
"""

from __future__ import annotations

import numpy as np

from backend.core import config as cfg

#: stride (s) between the overlapping horizon windows laid over the tape
STRIDE_S = 5.0


def window_seconds() -> float:
    return float(cfg.window_seconds())


def trades_only(ticks: np.ndarray) -> np.ndarray:
    """The trade prints (qty > 0); the whole tape when there are none."""
    if ticks is None or ticks.ndim != 2 or ticks.shape[0] == 0:
        return np.zeros((0, 4))
    t = ticks[ticks[:, 2] > 0]
    return t if t.shape[0] else ticks


def tick_size(prices: np.ndarray, bid: float = 0.0, ask: float = 0.0) -> float:
    """The exchange tick, estimated from the tape: a low quantile of the
    non-zero price steps, floored by half the spread, bounded to sanity."""
    if prices is None or prices.size < 2:
        return 0.0
    last = float(prices[-1])
    steps = np.abs(np.diff(prices[-600:]))
    steps = steps[steps > 0]
    tick = float(np.percentile(steps, 20)) if steps.size else last * 1e-5
    if bid > 0 and ask > bid:
        tick = max(tick, (ask - bid) * 0.5)
    return float(min(max(tick, last * 1e-7), last * 1e-3))


def best_quotes(book: np.ndarray | None) -> tuple[float, float, float, float]:
    """(bid, ask, bid_qty, ask_qty) of the top level; zeros when absent."""
    try:
        b = np.asarray(book, dtype=np.float64)
        if b.ndim != 3 or b.shape[0] < 2 or b.shape[1] < 1:
            return 0.0, 0.0, 0.0, 0.0
        bid, bq = float(b[0, 0, 0]), float(b[0, 0, 1])
        ask, aq = float(b[1, 0, 0]), float(b[1, 0, 1])
        if bid <= 0 or ask <= 0 or ask < bid:
            return 0.0, 0.0, 0.0, 0.0
        return bid, ask, max(bq, 0.0), max(aq, 0.0)
    except Exception:  # noqa: BLE001
        return 0.0, 0.0, 0.0, 0.0


def horizon_returns(ticks: np.ndarray, horizon_s: float | None = None,
                    stride_s: float = STRIDE_S) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping ``horizon_s`` windows laid over the trade tape every
    ``stride_s`` seconds: the (open, close) price pairs - exactly what the
    contest measures, many times over the tape.  Returns (opens, closes);
    empty arrays when the tape is shorter than two horizons."""
    horizon_s = float(horizon_s or window_seconds())
    tr = trades_only(ticks)
    if tr.shape[0] < 20:
        return np.zeros(0), np.zeros(0)
    t = tr[:, 0] / 1000.0
    p = tr[:, 1]
    if t[-1] - t[0] < 2.0 * horizon_s:
        return np.zeros(0), np.zeros(0)
    starts = np.arange(t[0], t[-1] - horizon_s, stride_s)
    i0 = np.searchsorted(t, starts, side="right") - 1
    i1 = np.searchsorted(t, starts + horizon_s, side="right") - 1
    keep = (i0 >= 0) & (i1 > i0)
    return p[i0[keep]], p[i1[keep]]


def last_window(ticks: np.ndarray, seconds: float | None = None) -> np.ndarray:
    """The trade prints inside the last ``seconds`` of tape."""
    seconds = float(seconds or window_seconds())
    tr = trades_only(ticks)
    if tr.shape[0] == 0:
        return tr
    cutoff = tr[-1, 0] - seconds * 1000.0
    return tr[tr[:, 0] >= cutoff]


def runs(prices: np.ndarray) -> list[tuple[int, int]]:
    """Monotone runs of the price path as (sign, length-in-moves); flat
    prints extend the current run (they do not break it)."""
    if prices is None or prices.size < 2:
        return []
    d = np.sign(np.diff(prices))
    out: list[tuple[int, int]] = []
    cur, n = 0, 0
    for s in d:
        if s == 0:
            if cur:
                n += 1
            continue
        if s == cur:
            n += 1
        else:
            if cur:
                out.append((int(cur), n))
            cur, n = int(s), 1
    if cur:
        out.append((int(cur), n))
    return out
