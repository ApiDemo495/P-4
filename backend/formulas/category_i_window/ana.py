"""FORMULA 31 - Analogue-Window Outcome (ANA).

A tiny empirical classifier trained on the tape itself.  Every overlapping
70 s window on the tape is labelled by the state at its *open* - the side
of the opening print, whether the preceding 10 s moved up or down, and
whether the window before it closed up - eight cells.  ANA finds the cell
the *current* boundary is in and reports the excess share of analogue
windows that closed strictly up: 2·P(up | cell) − 1, shrunk by the cell's
sample size (n / (n + 8)).  Ties count as neither.

No other formula conditions on the boundary state; ANA is the engine's
"what usually happened from here, on this tape, today".
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import finite, trace
from backend.formulas.category_i_window import _window as W

NAME = "ANA"
CATEGORY = "I"
TITLE = "Analogue-Window Outcome"
BRAIN_NODE = ""
DIRECTIONAL = True
LATENCY_MS = 0.4
DESCRIPTION = "Excess up-share of past windows that opened in the same boundary state as this one."

PRIOR_N = 8.0
LOOKBACK_S = 10.0


class State:
    __slots__ = ("last",)

    def __init__(self) -> None:
        self.last = 0.0

    def to_dict(self) -> dict:
        return {"last": self.last}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last = float(payload.get("last", 0.0))
        return obj


def _cell(side: float, pre: float, prev_close: float) -> int:
    return (1 if side > 0 else 0) | ((1 if pre > 0 else 0) << 1) | ((1 if prev_close > 0 else 0) << 2)


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    ticks = snapshot.ticks(asset)
    tr = W.trades_only(ticks)
    horizon = W.window_seconds()
    if tr.shape[0] < 40:
        return 0.0
    t = tr[:, 0] / 1000.0
    p = tr[:, 1]
    s = tr[:, 3]
    if t[-1] - t[0] < 2.0 * horizon + LOOKBACK_S:
        trace(ctx, "note", 0, "tape shorter than two windows - ANA abstains")
        return 0.0
    starts = np.arange(t[0] + horizon + LOOKBACK_S, t[-1] - horizon, W.STRIDE_S)
    if starts.size < 6:
        return 0.0
    i0 = np.searchsorted(t, starts, side="right") - 1
    i1 = np.searchsorted(t, starts + horizon, side="right") - 1
    ipre = np.searchsorted(t, starts - LOOKBACK_S, side="right") - 1
    iprev = np.searchsorted(t, starts - horizon, side="right") - 1
    keep = (i0 >= 0) & (i1 > i0) & (ipre >= 0) & (iprev >= 0)
    i0, i1, ipre, iprev = i0[keep], i1[keep], ipre[keep], iprev[keep]
    outcome = np.sign(p[i1] - p[i0])
    cells = np.array([_cell(s[a], p[a] - p[b], p[a] - p[c]) for a, b, c in zip(i0, ipre, iprev)])
    # the current boundary state
    now_i = p.size - 1
    pre_i = int(np.searchsorted(t, t[-1] - LOOKBACK_S, side="right") - 1)
    prev_i = int(np.searchsorted(t, t[-1] - horizon, side="right") - 1)
    cur = _cell(s[now_i], p[now_i] - p[max(pre_i, 0)], p[now_i] - p[max(prev_i, 0)])
    mask = cells == cur
    n = int(np.sum(mask))
    ups = int(np.sum(outcome[mask] > 0))
    downs = int(np.sum(outcome[mask] < 0))
    decided = ups + downs
    excess = (2.0 * ups / decided - 1.0) if decided else 0.0
    shrink = n / (n + PRIOR_N)
    value = excess * shrink
    state.last = value
    trace(ctx, "analogue windows", int(cells.size), "labelled windows on the tape")
    trace(ctx, "boundary cell", cur, "entry side | 10 s pre-move | previous window (3 bits)")
    trace(ctx, "same-cell windows", n, f"up {ups} · down {downs} · tie {n - decided}")
    trace(ctx, "excess up-share", excess, "2·P(up | cell) − 1")
    trace(ctx, "shrink", shrink, "n / (n + 8)")
    return finite(value)


DOUBLE_CHECK = "value = excess up-share × shrink"


def double_check(t: dict, asset: str) -> float:
    if "excess up-share" not in t or "shrink" not in t:
        return 0.0
    return float(t["excess up-share"]) * float(t["shrink"])
