"""FORMULA 27 - Window Drift Ratio (WDR).

Momentum measured at *exactly* the contest horizon.  Overlapping 70 s
windows are laid over the tape every 5 s; their returns are the sample the
contest draws from.  WDR is the t-statistic of their recency-weighted mean
(half-life 2 windows) against their dispersion, squashed: positive when the
tape has been closing windows up more often than chance *at this horizon*.

MPS measures persistence over 60 ticks and TAI acceleration over 30; WDR is
the only reading whose sample unit is the window itself.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite, tanh, trace
from backend.formulas.category_i_window import _window as W

NAME = "WDR"
CATEGORY = "I"
TITLE = "Window Drift Ratio"
BRAIN_NODE = ""
DIRECTIONAL = True
LATENCY_MS = 0.2
DESCRIPTION = "t-statistic of the recency-weighted mean return of overlapping 70 s windows."

T_SCALE = 2.0


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


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    opens, closes = W.horizon_returns(snapshot.ticks(asset))
    if opens.size < 6:
        trace(ctx, "note", int(opens.size), "fewer than 6 horizon windows - WDR abstains")
        return 0.0
    rets = closes / opens - 1.0
    n = rets.size
    per_window = W.window_seconds() / W.STRIDE_S
    w = 0.5 ** ((n - 1 - np.arange(n)) / (2.0 * per_window))
    w /= w.sum()
    mean = float(np.sum(w * rets))
    var = float(np.sum(w * (rets - mean) ** 2))
    # effective sample size of the weights, so overlap cannot fake precision
    n_eff = 1.0 / float(np.sum(w ** 2))
    n_indep = max(1.0, n_eff / per_window)
    t_stat = mean / (np.sqrt(var / n_indep) + EPS) if var > 0 else 0.0
    value = tanh(t_stat / T_SCALE)
    state.last = value
    trace(ctx, "horizon windows", n, f"overlapping {W.window_seconds():.0f} s windows, 5 s stride")
    trace(ctx, "weighted mean return", mean * 1e4, "bps per window (half-life 2 windows)")
    trace(ctx, "dispersion", np.sqrt(var) * 1e4, "bps per window")
    trace(ctx, "independent windows", n_indep, "effective sample / overlap")
    trace(ctx, "t-statistic", t_stat, "mean / (sd / sqrt(n_indep))")
    return finite(value)


DOUBLE_CHECK = "value = tanh(t-statistic / 2)"


def double_check(t: dict, asset: str) -> float:
    if "t-statistic" not in t:
        return 0.0
    return tanh(float(t["t-statistic"]) / T_SCALE)
