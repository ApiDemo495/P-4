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

from backend.formulas._util import EPS, SelfScale, finite, tanh, trace

NAME = "GCDV"
CATEGORY = "C"
TITLE = "Gold-Crypto Divergence Velocity"
BRAIN_NODE = "T4/T5 motion"
DIRECTIONAL = True
LATENCY_MS = 0.1
DESCRIPTION = "Speed of divergence between normalised BTC and PAXG price paths."

VELOCITY_POINTS = 10
GAIN = 1.0
#: A divergence speed of 1e-4 per second is 6 bps per minute of separation
#: between BTC and PAXG: that is a rotation worth trading.  The floor has to sit
#: above the noise floor of *two independent* legs - measured at ~3e-5 per
#: second on the balanced tape - or a dead pair reads as a strong divergence.
MEANINGFUL_VELOCITY = 2e-4


class State:
    __slots__ = ("last_divergence", "last_velocity", "velocity_scale")

    def __init__(self) -> None:
        self.last_divergence = 0.0
        self.last_velocity = 0.0
        self.velocity_scale = SelfScale(decay=0.95, floor=MEANINGFUL_VELOCITY)

    def to_dict(self) -> dict:
        return {
            "last_divergence": self.last_divergence,
            "last_velocity": self.last_velocity,
            "velocity_scale": self.velocity_scale.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_divergence = float(payload.get("last_divergence", 0.0))
        obj.last_velocity = float(payload.get("last_velocity", 0.0))
        obj.velocity_scale = SelfScale.from_dict(payload.get("velocity_scale", {}))
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
    diffs = np.diff(d)
    # Two estimates of the same quantity: the instantaneous speed over the last
    # 10 grid points, and the average speed across the whole grid.  Ten diffs of
    # a 60-point grid carry a whole basis point of standard error, which is why
    # a dead pair read +/-0.4 in v2.0.0.  Blending them keeps a real separation
    # while cancelling the endpoint noise.
    velocity = float(np.mean(diffs[-k:]))
    grid_drift = float((d[-1] - d[0]) / max(1.0, float(d.size - 1)))
    combined = 0.5 * (velocity + grid_drift)
    denom = max(MEANINGFUL_VELOCITY, 0.5 * state.velocity_scale.denominator(GAIN))
    state.velocity_scale.update(abs(combined))
    state.last_divergence = float(d[-1])
    state.last_velocity = combined

    trace(ctx, "normalised BTC path", float(p_b[-1] / base_b), "relative to the first grid value")
    trace(ctx, "normalised PAXG path", float(p_g[-1] / base_g), "relative to the first grid value")
    trace(ctx, "divergence d", float(d[-1]), "BTC_norm - PAXG_norm")
    trace(ctx, "velocity (mean of last 10 diffs)", velocity, "per second")
    trace(ctx, "grid drift (whole window)", grid_drift, "per second")
    trace(ctx, "blended velocity", combined, "per second")
    trace(ctx, "meaningful velocity floor", denom, "per second (2 bps / minute)")
    trace(ctx, "velocity / scale", combined / (denom + EPS), "1.0 = a meaningful divergence speed")

    score = tanh(combined / (denom + EPS))
    return finite(score if asset.upper() == "BTC" else -score)
