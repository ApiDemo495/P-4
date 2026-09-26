"""FORMULA 4 - Volume Shock Detector (VSD).

Candle-volume indicators tell you about the minute that already closed.  VSD
works on individual tick volumes: it compares the median of the last 20 ticks
with the median of the preceding 100 and scales by a **median-absolute-
deviation** sigma, which is immune to the single 40-BTC print that would blow
up a mean/stddev estimator.

Brain mapping: mechanosensory bristle neurons (class III/IV) - the fly's shock
detectors, wired straight into the escape reflex.

Latency budget: < 0.3 ms (two medians over <= 120 ticks).
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import finite, mad, tanh, trace

NAME = "VSD"
CATEGORY = "A"
TITLE = "Volume Shock Detector"
BRAIN_NODE = "Mechano class III/IV"
DIRECTIONAL = True
LATENCY_MS = 0.3
DESCRIPTION = "Robust (median/MAD) volume spike detection with directional sign."

TOTAL_TICKS = 120
Z_DIVISOR = 3.0
EPS = 1e-12

#: Materiality gate: a median-volume difference smaller than this fraction of
#: the baseline median is not a shock, however many robust sigmas it is worth.
#: Without it a quiet tape (tiny MAD) reported +-0.38 out of pure noise.
MIN_SHOCK_RATIO = 0.15


class State:
    __slots__ = ("last_z", "last_baseline")

    def __init__(self) -> None:
        self.last_z = 0.0
        self.last_baseline = 0.0

    def to_dict(self) -> dict:
        return {"last_z": self.last_z, "last_baseline": self.last_baseline}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_z = float(payload.get("last_z", 0.0))
        obj.last_baseline = float(payload.get("last_baseline", 0.0))
        return obj


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    ticks = snapshot.ticks(asset)
    base_window = int(params.get("vsd_baseline_window", 100))
    recent_window = int(params.get("vsd_recent_window", 20))
    need = base_window + recent_window

    if ticks.shape[0] < need:
        # PAXG often does not deliver 120 ticks per minute.  Section 5.2 allows
        # interpolation for the >60-tick formulas; we do it in the snapshot
        # layer, so here we simply run on whatever we have.
        if ticks.shape[0] < 20:
            return 0.0
        split = max(5, int(ticks.shape[0] * 0.8))
    else:
        ticks = ticks[-need:]
        split = base_window

    volumes = ticks[:, 2]
    sides = ticks[:, 3]

    baseline = volumes[:split]
    recent = volumes[split:]
    if recent.size == 0:
        return 0.0

    mu = float(np.median(baseline))
    sigma = mad(baseline)
    state.last_baseline = mu

    recent_median = float(np.median(recent))
    z = (recent_median - mu) / (sigma + EPS)
    state.last_z = z

    ratio = recent_median / (mu + EPS)
    if abs(ratio - 1.0) < MIN_SHOCK_RATIO:
        trace(ctx, "baseline median vol", mu, "contracts")
        trace(ctx, "recent median vol", recent_median, "contracts")
        trace(ctx, "ratio recent / baseline", ratio, "below the materiality gate - no shock")
        return 0.0

    prices = ticks[:, 1]
    anchor = prices[max(0, split - 1)]
    price_sign = float(np.sign(prices[-1] - anchor))
    if price_sign == 0.0:
        # A volume spike with no price move at all carries no direction: saying
        # "up" or "down" here is exactly the "random numbers" failure mode.
        trace(ctx, "baseline median vol", mu, "contracts")
        trace(ctx, "recent median vol", recent_median, "contracts")
        trace(ctx, "MAD sigma", sigma, "contracts")
        trace(ctx, "z", z, "robust sigmas")
        trace(ctx, "price sign", 0.0, "price did not move - no shock direction")
        return 0.0

    trace(ctx, "baseline median vol", mu, "contracts")
    trace(ctx, "recent median vol", recent_median, "contracts")
    trace(ctx, "MAD sigma", sigma, "contracts")
    trace(ctx, "ratio recent / baseline", ratio, "materiality gate is 0.15")
    trace(ctx, "z (volume shock)", z, "robust sigmas")
    trace(ctx, "price move over the shock", float(prices[-1] - anchor), "price units")
    trace(ctx, "price sign", price_sign, "+1 up / -1 down")
    net_sides = float(np.sum(sides[split:]))
    trace(ctx, "net aggressor flow", net_sides, "contracts (context only)")

    return finite(tanh(z / Z_DIVISOR) * price_sign)
