"""FORMULA 3 - Spread Elasticity Detector (SED).

A market-maker intention detector.  If the spread *widens right after a buy
trade*, makers are pulling their asks - they expect price to keep going up.
If it tightens after a buy, they are happy to sell into it.  SED turns the
signed response of the spread to trade flow into a directional read.

Brain mapping: GRN class Gr5a (sugar sensor) - is the market offering you a
sweet deal, or pulling the plate away?

Latency budget: < 0.1 ms over <= 30 ticks.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, SelfScale, finite, ols_slope, tanh, trace

NAME = "SED"
CATEGORY = "A"
TITLE = "Spread Elasticity Detector"
BRAIN_NODE = "GRN Gr5a"
DIRECTIONAL = True
LATENCY_MS = 0.1
DESCRIPTION = "Signed spread response to aggressive flow - market-maker intention."

WINDOW_TICKS = 30
GAIN = 1.0
#: A spread move of 5 % of the average spread per typical unit of flow is
#: already a strong elasticity; the self-scaling term raises the bar further in
#: markets that are structurally jumpy.
MEANINGFUL_EFFECT = 0.05


class State:
    __slots__ = ("last_spread_avg", "last_beta", "effect_scale")

    def __init__(self) -> None:
        self.last_spread_avg = 0.0
        self.last_beta = 0.0
        self.effect_scale = SelfScale(decay=0.97, floor=MEANINGFUL_EFFECT)

    def to_dict(self) -> dict:
        return {
            "last_spread_avg": self.last_spread_avg,
            "last_beta": self.last_beta,
            "effect_scale": self.effect_scale.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_spread_avg = float(payload.get("last_spread_avg", 0.0))
        obj.last_beta = float(payload.get("last_beta", 0.0))
        obj.effect_scale = SelfScale.from_dict(payload.get("effect_scale", {}))
        return obj


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    ticks = snapshot.ticks(asset)
    spreads = snapshot.spread_history(asset)
    if ticks.shape[0] < 5 or spreads.shape[0] < 3:
        return 0.0

    window = min(WINDOW_TICKS, ticks.shape[0])
    ticks = ticks[-window:]
    if ticks.shape[0] < 8:
        return 0.0

    # L2 spreads are interpolated onto the tick timestamps (Section 3.1 F3).
    spread_series = np.interp(ticks[:, 0], spreads[:, 0], spreads[:, 1])
    spread_avg = float(np.mean(spread_series))
    state.last_spread_avg = spread_avg
    trace(ctx, "mean spread", spread_avg, "price units")

    # Signed aggressor volume per tick step, aligned with the spread changes.
    volumes = ticks[1:, 2]
    sides = ticks[1:, 3]
    signed_flow = volumes * sides
    delta_spread = np.diff(spread_series)
    if signed_flow.size < 6 or float(np.std(signed_flow)) <= EPS:
        trace(ctx, "beta (dspread/dflow)", 0.0, "skipped: flow has no dispersion")
        return 0.0

    # OLS slope of d(spread) on the *signed* flow: "how much does the spread
    # widen after a buy print, and tighten after a sell print?".  That signed
    # response *is* the maker intention - it needs no second sign multiply
    # (which is how the flat tape ended up pinned at -1.000).
    beta = ols_slope(signed_flow, delta_spread)
    state.last_beta = beta
    trace(ctx, "beta (dspread/dflow)", beta, "spread per unit flow")

    typical_flow = float(np.mean(np.abs(signed_flow)))
    # Dimensionless: the spread change the typical trade implies, in units of
    # the average spread.  Without this the raw beta is ~1e-9 and every read is
    # 0.000 (the original bug), which is what "random numbers" looked like.
    if spread_avg <= EPS:
        return 0.0
    effect = beta * typical_flow / spread_avg
    denom = state.effect_scale.denominator(1.0)
    state.effect_scale.update(abs(effect))
    trace(ctx, "typical |flow|", typical_flow, "contracts")
    trace(ctx, "relative effect", effect, "fraction of the average spread")
    trace(ctx, "scale used", denom, "fraction of the average spread")
    trace(ctx, "net signed flow", float(np.sum(signed_flow)), "contracts (+ = net buyers)")
    return finite(tanh(GAIN * effect / (denom + EPS)))
