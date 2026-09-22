"""FORMULA 17 - Time-Weighted Return Skewness (TWRS).

Third moment of the tick-return distribution, weighted toward the recent window.
Positive skew means an occasional fat right tail (big up-prints) while negative
skew means the opposite.  Standard skewness averages the whole hour equally,
which is useless at a 60-second horizon - the character of the last 10 ticks is
what matters.

Brain mapping: ORN class Or85a - an asymmetric response-profile receptor.

Latency budget: < 0.1 ms.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite, tanh

NAME = "TWRS"
CATEGORY = "E"
TITLE = "Time-Weighted Return Skewness"
BRAIN_NODE = "ORN Or85a"
DIRECTIONAL = True
LATENCY_MS = 0.1
DESCRIPTION = "Recency-weighted third moment of tick returns (tail asymmetry)."

WINDOW = 60
POWER = 2.0


class State:
    __slots__ = ("last_skew", "last_m2")

    def __init__(self) -> None:
        self.last_skew = 0.0
        self.last_m2 = 0.0

    def to_dict(self) -> dict:
        return {"last_skew": self.last_skew, "last_m2": self.last_m2}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_skew = float(payload.get("last_skew", 0.0))
        obj.last_m2 = float(payload.get("last_m2", 0.0))
        return obj


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    prices = snapshot.prices(asset)
    if prices.size < 12:
        return 0.0
    window = prices[-min(WINDOW, prices.size) :]
    returns = np.diff(np.log(np.maximum(window, EPS)))
    n = returns.size
    if n < 8:
        return 0.0

    weights = (np.arange(1, n + 1, dtype=np.float64) / n) ** POWER
    w_sum = float(np.sum(weights)) + EPS

    mean = float(np.sum(weights * returns) / w_sum)
    centered = returns - mean
    m2 = float(np.sum(weights * centered**2) / w_sum)
    m3 = float(np.sum(weights * centered**3) / w_sum)
    state.last_m2 = m2

    skew = m3 / (m2**1.5 + EPS)
    state.last_skew = skew
    return finite(tanh(skew))
