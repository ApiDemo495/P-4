"""FORMULA 10 - Gold-Crypto Divergence Velocity (GCDV)  [NEW in v2.0].

Not correlation (static), not correlation drift (slow) - the *speed* at which
the BTC and PAXG price paths are separating.  Both series are normalised to
their first grid value, differenced, and the average of the last 10 first-
differences is the divergence velocity.  A rapidly widening gap means the market
has genuine directional conviction rather than noise.

    BTC:  GCDV as-is   (BTC outpacing gold = bullish for BTC)
    PAXG: GCDV negated

Brain mapping: T4/T5 lobula plate motion detectors - the fly's speed-of-flow
circuits, applied to the flow of relative price.

Latency budget: < 0.1 ms.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite, tanh

NAME = "GCDV"
CATEGORY = "C"
TITLE = "Gold-Crypto Divergence Velocity"
BRAIN_NODE = "T4/T5 motion"
DIRECTIONAL = True
LATENCY_MS = 0.1
DESCRIPTION = "Speed of divergence between normalised BTC and PAXG price paths."

VELOCITY_POINTS = 10
GAIN = 100.0


class State:
    __slots__ = ("last_divergence", "last_velocity")

    def __init__(self) -> None:
        self.last_divergence = 0.0
        self.last_velocity = 0.0

    def to_dict(self) -> dict:
        return {"last_divergence": self.last_divergence, "last_velocity": self.last_velocity}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_divergence = float(payload.get("last_divergence", 0.0))
        obj.last_velocity = float(payload.get("last_velocity", 0.0))
        return obj


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    synced = snapshot.synced
    if synced is None or not synced.valid:
        return 0.0

    p_b = synced.btc_prices
    p_g = synced.paxg_prices
    if p_b.size < 3 or p_g.size < 3:
        return 0.0

    base_b = p_b[0] if p_b[0] > EPS else EPS
    base_g = p_g[0] if p_g[0] > EPS else EPS
    d = (p_b / base_b) - (p_g / base_g)

    k = min(VELOCITY_POINTS, d.size - 1)
    if k <= 0:
        return 0.0
    velocity = float(np.mean(np.diff(d)[-k:]))
    state.last_divergence = float(d[-1])
    state.last_velocity = velocity

    score = tanh(GAIN * velocity)
    return finite(score if asset.upper() == "BTC" else -score)
