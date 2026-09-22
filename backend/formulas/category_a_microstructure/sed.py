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

from backend.formulas._util import EPS, finite, tanh

NAME = "SED"
CATEGORY = "A"
TITLE = "Spread Elasticity Detector"
BRAIN_NODE = "GRN Gr5a"
DIRECTIONAL = True
LATENCY_MS = 0.1
DESCRIPTION = "Signed spread response to aggressive flow - market-maker intention."

WINDOW_TICKS = 30
RECENT_POINTS = 10
GAIN = 10.0


class State:
    __slots__ = ("last_spread_avg",)

    def __init__(self) -> None:
        self.last_spread_avg = 0.0

    def to_dict(self) -> dict:
        return {"last_spread_avg": self.last_spread_avg}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_spread_avg = float(payload.get("last_spread_avg", 0.0))
        return obj


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    ticks = snapshot.ticks(asset)
    spreads = snapshot.spread_history(asset)
    if ticks.shape[0] < 5 or spreads.shape[0] < 3:
        return 0.0

    window = min(WINDOW_TICKS, ticks.shape[0])
    ticks = ticks[-window:]

    # L2 spreads are interpolated onto the tick timestamps (Section 3.1 F3).
    spread_series = np.interp(ticks[:, 0], spreads[:, 0], spreads[:, 1])
    delta_spread = np.diff(spread_series)
    signed = delta_spread * ticks[1:, 3]  # sign of the trade that preceded the change

    k = min(RECENT_POINTS, signed.size)
    if k == 0:
        return 0.0
    recent = float(np.sum(signed[-k:]))
    spread_avg = float(np.mean(spread_series))
    state.last_spread_avg = spread_avg

    raw = GAIN * recent / (spread_avg + EPS)
    return finite(tanh(raw))
