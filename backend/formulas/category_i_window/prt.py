"""FORMULA 28 - Previous-Window Reversal Tendency (PRT).

Does a window that closed up tend to be followed by one that closes down?
PRT estimates the lag-one autocorrelation ρ₁ of *non-overlapping* 53 s
window returns on the tape and applies it to the window that just ended:
value = ρ₁ × sign(last window return) × min(1, |last| / σ).  Negative ρ₁ on
an up window leans SELL (mean reversion at the horizon), positive leans BUY
(horizon momentum).  Shrunk toward zero by the sample size (ρ̂ ± 1/√n).

WDR asks "which way have windows been closing?"; PRT asks "given how the
last one closed, which way does the next one lean?" - a conditional read.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite, trace
from backend.formulas.category_i_window import _window as W

NAME = "PRT"
CATEGORY = "I"
TITLE = "Previous-Window Reversal Tendency"
BRAIN_NODE = ""
DIRECTIONAL = True
LATENCY_MS = 0.2
DESCRIPTION = "Lag-1 autocorrelation of window returns applied to the window that just closed."


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
    horizon = W.window_seconds()
    opens, closes = W.horizon_returns(snapshot.ticks(asset), stride_s=horizon)
    if opens.size < 5:
        trace(ctx, "note", int(opens.size), "fewer than 5 consecutive windows - PRT abstains")
        return 0.0
    rets = closes / opens - 1.0
    r0, r1 = rets[:-1], rets[1:]
    sd = float(np.std(rets))
    if sd <= EPS:
        return 0.0
    rho = float(np.corrcoef(r0, r1)[0, 1]) if r0.size > 2 else 0.0
    if not np.isfinite(rho):
        rho = 0.0
    n = rets.size
    shrink = max(0.0, 1.0 - 1.0 / np.sqrt(n) / max(abs(rho), EPS)) if rho != 0 else 0.0
    rho_shrunk = rho * shrink
    last = float(rets[-1])
    strength = min(1.0, abs(last) / sd)
    value = rho_shrunk * np.sign(last) * strength
    state.last = value
    trace(ctx, "consecutive windows", n, f"non-overlapping {horizon:.0f} s windows")
    trace(ctx, "lag-1 autocorrelation", rho, "ρ₁ of window returns")
    trace(ctx, "shrunk ρ₁", rho_shrunk, "ρ₁ × max(0, 1 − 1/(√n·|ρ₁|))")
    trace(ctx, "last window", last * 1e4, "bps")
    trace(ctx, "last window strength", np.sign(last) * strength, "sign × min(1, |last| / σ)")
    return finite(float(value))


DOUBLE_CHECK = "value = shrunk ρ₁ × last window strength"


def double_check(t: dict, asset: str) -> float:
    if "shrunk ρ₁" not in t or "last window strength" not in t:
        return 0.0
    return float(t["shrunk ρ₁"]) * float(t["last window strength"])
