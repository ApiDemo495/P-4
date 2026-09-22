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

from backend.formulas._util import EPS, finite

NAME = "SHRP"
CATEGORY = "C"
TITLE = "Safe-Haven Rotation Pulse"
BRAIN_NODE = "MB Calyx CA"
DIRECTIONAL = True
LATENCY_MS = 0.15
DESCRIPTION = "Recency-weighted dollar-flow rotation between BTC (risk) and PAXG (safety)."

FLOW_SECONDS = 60.0


class State:
    __slots__ = ("last_r", "last_flow_btc", "last_flow_paxg")

    def __init__(self) -> None:
        self.last_r = 0.0
        self.last_flow_btc = 0.0
        self.last_flow_paxg = 0.0

    def to_dict(self) -> dict:
        return {
            "last_r": self.last_r,
            "last_flow_btc": self.last_flow_btc,
            "last_flow_paxg": self.last_flow_paxg,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_r = float(payload.get("last_r", 0.0))
        obj.last_flow_btc = float(payload.get("last_flow_btc", 0.0))
        obj.last_flow_paxg = float(payload.get("last_flow_paxg", 0.0))
        return obj


def _dollar_flow(ticks: np.ndarray, now_ms: float) -> float:
    """Sum of recency-weighted (i/T)^2 * volume * side * price over the window."""
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
    return float(np.sum(w * window[:, 2] * window[:, 3] * window[:, 1]))


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    now_ms = snapshot.timestamp * 1000.0
    f_b = _dollar_flow(snapshot.ticks("BTC"), now_ms)
    f_g = _dollar_flow(snapshot.ticks("PAXG"), now_ms)
    state.last_flow_btc, state.last_flow_paxg = f_b, f_g

    r = (f_g - f_b) / (abs(f_g) + abs(f_b) + EPS)
    state.last_r = r
    r = max(-1.0, min(1.0, r))
    return finite(-r if asset.upper() == "BTC" else r)
