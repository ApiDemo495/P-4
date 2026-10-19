"""FORMULA 23 - Pinned-Window Probability (PIN).

The contest's silent killer: a window whose last trade at :70 equals the
last trade at :00 is a LOSS whichever side was called.  PIN estimates that
probability *for this tape right now* from two independent readings and
takes the larger:

* empirical - lay overlapping 70 s windows over the tape every 5 s and count
  the share whose close equals its open to the tick;
* parametric - P(|ΔP| < tick/2) under a normal with the tape's own horizon
  sigma: 2·Φ(tick / (2σ₅₃)) − 1, i.e. how many ticks one window-sigma is.

PIN is a *risk*, not a direction (DIRECTIONAL = False): it drives the
boundary picker's tie-break strength and the TIE RISK read-out, and the
fusion sizes conviction down with it.
"""

from __future__ import annotations

import math

import numpy as np

from backend.formulas._util import finite, trace
from backend.formulas.category_i_window import _window as W

NAME = "PIN"
CATEGORY = "I"
TITLE = "Pinned-Window Probability"
BRAIN_NODE = ""
DIRECTIONAL = False
LATENCY_MS = 0.3
DESCRIPTION = "Probability that the window closes exactly where it opened (a tie = a loss)."


class State:
    __slots__ = ("last", "samples")

    def __init__(self) -> None:
        self.last = 0.0
        self.samples = 0

    def to_dict(self) -> dict:
        return {"last": self.last, "samples": self.samples}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last = float(payload.get("last", 0.0))
        obj.samples = int(payload.get("samples", 0))
        return obj


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    ticks = snapshot.ticks(asset)
    tr = W.trades_only(ticks)
    if tr.shape[0] < 20:
        return 0.0
    bid, ask, _, _ = W.best_quotes(snapshot.book(asset))
    tick = W.tick_size(tr[:, 1], bid, ask)
    opens, closes = W.horizon_returns(ticks)
    horizon = W.window_seconds()
    if opens.size >= 6:
        flat_share = float(np.mean(np.abs(closes - opens) < 0.5 * tick))
        sigma = float(np.std(closes / opens - 1.0)) * float(opens[-1])
        basis = int(opens.size)
    else:
        # a tape shorter than two windows: scale the last-window moves by sqrt(time)
        last = W.last_window(ticks)
        span = (last[-1, 0] - last[0, 0]) / 1000.0 if last.shape[0] > 2 else 0.0
        if span <= 1.0:
            return 0.0
        rets = np.diff(np.log(last[:, 1]))
        sigma = float(np.sqrt(np.sum(rets ** 2) * (horizon / span))) * float(last[-1, 1])
        flat_share = 0.0
        basis = 0
    sigma_ticks = sigma / tick if tick > 0 else 0.0
    parametric = max(0.0, 2.0 * _phi(0.5 / sigma_ticks) - 1.0) if sigma_ticks > 1e-9 else 1.0
    pin = max(flat_share, parametric)
    pin = float(min(1.0, max(0.0, pin)))
    state.last, state.samples = pin, basis

    trace(ctx, "tick size", tick, "price units (tape estimate)")
    trace(ctx, "horizon windows", basis, f"overlapping {horizon:.0f} s windows on the tape")
    trace(ctx, "flat share", flat_share, "windows that closed exactly at their open")
    trace(ctx, "window sigma", sigma_ticks, "ticks per window")
    trace(ctx, "parametric tie", parametric, "2·Φ(tick/(2σ)) − 1")
    trace(ctx, "pin probability", pin, "max(flat share, parametric)")
    return finite(pin)


DOUBLE_CHECK = "value = max(flat share, parametric tie), parametric = 2·Φ(0.5 / sigma_ticks) − 1, clipped to [0, 1]"


def double_check(t: dict, asset: str) -> float:
    if "flat share" not in t or "window sigma" not in t:
        return 0.0
    s = float(t["window sigma"])
    par = max(0.0, 2.0 * _phi(0.5 / s) - 1.0) if s > 1e-9 else 1.0
    return min(1.0, max(0.0, max(float(t["flat share"]), par)))
