"""FORMULA 30 - Large-Aggressor Imbalance (LAI).

Who is hitting with *size*.  Over the last window the trades above the
tape's own 80th size percentile are kept; LAI is their signed notional
share, (Σ buy − Σ sell) / Σ all, times the share of total notional those
large prints represent.  Small prints are the crowd; the top quintile is
whoever is actually moving inventory - and inside 53 s inventory moves
decide the close.

AFPR is a recency-weighted ratio over *every* print; LAI deliberately throws
80 % of them away and reads the rest unweighted by time.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite, trace
from backend.formulas.category_i_window import _window as W

NAME = "LAI"
CATEGORY = "I"
TITLE = "Large-Aggressor Imbalance"
BRAIN_NODE = ""
DIRECTIONAL = True
LATENCY_MS = 0.1
DESCRIPTION = "Signed notional share of the top-quintile prints of the last window."

QUANTILE = 80.0


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
    last = W.last_window(snapshot.ticks(asset))
    if last.shape[0] < 10:
        return 0.0
    sides = last[:, 3]
    if not np.any(sides != 0):
        trace(ctx, "note", 0, "no aggressor sides on the tape - LAI abstains")
        return 0.0
    notional = last[:, 1] * last[:, 2]
    cut = float(np.percentile(notional, QUANTILE))
    big = notional >= cut
    if not np.any(big):
        return 0.0
    buy = float(np.sum(notional[big & (sides > 0)]))
    sell = float(np.sum(notional[big & (sides < 0)]))
    share = (buy + sell) / (float(np.sum(notional)) + EPS)
    imbalance = (buy - sell) / (buy + sell + EPS)
    value = imbalance * share
    state.last = value
    trace(ctx, "prints in window", int(last.shape[0]), f"last {W.window_seconds():.0f} s")
    trace(ctx, "size cut", cut, "80th percentile notional")
    trace(ctx, "large buy notional", buy, "quote units")
    trace(ctx, "large sell notional", sell, "quote units")
    trace(ctx, "large share", share, "of all notional")
    trace(ctx, "imbalance", imbalance, "(buy − sell) / (buy + sell)")
    return finite(value)


DOUBLE_CHECK = "value = imbalance × large share"


def double_check(t: dict, asset: str) -> float:
    if "imbalance" not in t or "large share" not in t:
        return 0.0
    return float(t["imbalance"]) * float(t["large share"])
