"""FORMULA 9 - Safe-Haven Rotation Pulse (SHRP)  [NEW in v2.0].

Correlation tells you whether two assets move together.  SHRP tells you where
the money is actually *going*.  It sums recency-weighted signed dollar flow for
BTC and PAXG and reports the rotation between them:

    R > 0  -> capital leaving BTC for gold ("flight to safety")
    R < 0  -> capital leaving gold for BTC ("risk-on")

Asset-adjusted output:
    BTC:  SHRP = -R
    PAXG: SHRP = +R

Brain mapping: mushroom-body calyx subdivision CA - the context-integration
area that sets the overall approach/avoidance balance.

Latency budget: < 0.15 ms.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, SelfScale, finite, trace

NAME = "SHRP"
CATEGORY = "C"
TITLE = "Safe-Haven Rotation Pulse"
BRAIN_NODE = "MB Calyx CA"
DIRECTIONAL = True
LATENCY_MS = 0.15
DESCRIPTION = "Recency-weighted dollar-flow rotation between BTC (risk) and PAXG (safety)."

FLOW_SECONDS = 60.0


class State:
    __slots__ = ("last_r", "last_flow_btc", "last_flow_paxg", "scale_btc", "scale_paxg")

    def __init__(self) -> None:
        self.last_r = 0.0
        self.last_flow_btc = 0.0
        self.last_flow_paxg = 0.0
        # One typical-flow level per leg: a rotation is measured leg by leg.
        self.scale_btc = SelfScale(decay=0.95, floor=0.0)
        self.scale_paxg = SelfScale(decay=0.95, floor=0.0)

    def to_dict(self) -> dict:
        return {
            "last_r": self.last_r,
            "last_flow_btc": self.last_flow_btc,
            "last_flow_paxg": self.last_flow_paxg,
            "scale_btc": self.scale_btc.to_dict(),
            "scale_paxg": self.scale_paxg.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_r = float(payload.get("last_r", 0.0))
        obj.last_flow_btc = float(payload.get("last_flow_btc", 0.0))
        obj.last_flow_paxg = float(payload.get("last_flow_paxg", 0.0))
        obj.scale_btc = SelfScale.from_dict(payload.get("scale_btc", {}))
        obj.scale_paxg = SelfScale.from_dict(payload.get("scale_paxg", {}))
        return obj


def _flow_imbalance(ticks: np.ndarray, now_ms: float) -> float:
    """Recency-weighted signed flow as a share of that leg's own turnover.

    The raw dollar sum (v2.0.0) cannot tell a rotation from a busy minute: it was
    divided by its own magnitude EMA, so two dead legs read +/-0.6.  Turning it
    into net-over-turnover - how one-sided the flow is, not how large - keeps the
    meaning ("where is the money going") and goes to zero when a leg prints buys
    and sells in equal size.

        imbalance = sum(w * vol * side * price) / sum(w * vol * price)

    """
    if ticks is None or ticks.size == 0:
        return 0.0
    cutoff = now_ms - FLOW_SECONDS * 1000.0
    window = ticks[ticks[:, 0] >= cutoff]
    if window.shape[0] < 2:
        window = ticks[-min(ticks.shape[0], 10) :]
    if window.shape[0] < 1:
        return 0.0
    t = window.shape[0]
    w = (np.arange(1, t + 1, dtype=np.float64) / t) ** 2
    notional = np.abs(window[:, 2] * window[:, 1])
    net = window[:, 2] * window[:, 3] * window[:, 1]
    turnover = float(np.sum(w * notional))
    if turnover <= EPS:
        return 0.0
    return float(max(-1.0, min(1.0, float(np.sum(w * net)) / turnover)))


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    now_ms = snapshot.timestamp * 1000.0
    f_b = _flow_imbalance(snapshot.ticks("BTC"), now_ms)
    f_g = _flow_imbalance(snapshot.ticks("PAXG"), now_ms)
    state.last_flow_btc, state.last_flow_paxg = f_b, f_g
    state.scale_btc.update(abs(f_b))
    state.scale_paxg.update(abs(f_g))

    # Both legs are already on the same scale (a share of their own turnover), so
    # the rotation is just the difference: gold bid and crypto offered is a
    # rotation into gold, and two balanced books cancel.
    rotation = 0.5 * (f_g - f_b)
    r = max(-1.0, min(1.0, rotation))
    state.last_r = r
    trace(ctx, "BTC flow imbalance", f_b, "net / turnover over the flow window")
    trace(ctx, "PAXG flow imbalance", f_g, "net / turnover over the flow window")
    trace(ctx, "rotation", r, "1.0 = a full rotation into one leg")
    return finite(-r if asset.upper() == "BTC" else r)
