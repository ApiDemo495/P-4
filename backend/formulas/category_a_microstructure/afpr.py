"""FORMULA 2 - Aggressive Flow Pressure Ratio (AFPR).

Power-law recency weighting of buy vs. sell volume.  A simple buy/sell ratio
treats the last 2 seconds and the last 30 seconds as equally important; AFPR
gives the most recent tick a weight of 1.0 while a tick halfway back gets
(1/2)^power.  In scalping the tail matters ~10x more, which is exactly what the
cubic (BTC) / quadratic (PAXG) weights encode.

Brain mapping: ORN class Or42b - broadly tuned, sets the background bias of the
antennal lobe rather than triggering an urgent reflex.

Latency budget: < 0.1 ms (single pass over <= 60 ticks).
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite

NAME = "AFPR"
CATEGORY = "A"
TITLE = "Aggressive Flow Pressure Ratio"
BRAIN_NODE = "ORN Or42b"
DIRECTIONAL = True
LATENCY_MS = 0.1
DESCRIPTION = "Recency-weighted (power-law) aggressive buy vs. sell volume imbalance."

WINDOW_TICKS = 60


class State:
    __slots__ = ("last_ratio",)

    def __init__(self) -> None:
        self.last_ratio = 0.0

    def to_dict(self) -> dict:
        return {"last_ratio": self.last_ratio}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_ratio = float(payload.get("last_ratio", 0.0))
        return obj


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    ticks = snapshot.ticks(asset)
    if ticks.shape[0] < 5:
        return 0.0

    window = min(WINDOW_TICKS, ticks.shape[0])
    ticks = ticks[-window:]
    volumes = ticks[:, 2]
    sides = ticks[:, 3]

    t = ticks.shape[0]
    weights = (np.arange(1, t + 1, dtype=np.float64) / t) ** float(params.get("afpr_power", 3.0))

    wv = weights * volumes
    buy = float(np.sum(wv[sides > 0]))
    sell = float(np.sum(wv[sides < 0]))

    ratio = (buy - sell) / (buy + sell + EPS)
    state.last_ratio = ratio
    return finite(max(-1.0, min(1.0, ratio)))
