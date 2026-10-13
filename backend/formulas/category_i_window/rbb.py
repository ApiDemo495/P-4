"""FORMULA 25 - Roll Bounce Bias (RBB).

The contest enters at the last *trade*.  A trade that printed at the ask
(a buyer's print) sits half a spread above the mid; if nothing happens the
next print is as likely at the bid - which reads as "down".  RBB measures
that mechanical bias: −side_of_entry × (half spread / tick) × (1 − |drift|
share), so it is strongest on a wide, quiet spread and vanishes when the tape
is actually trending (then the bounce is swamped).  Roll (1984) is the
estimator of the effective spread from the negative lag-1 autocovariance of
price changes; RBB uses it to size the bounce when the book is missing.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite, tanh, trace
from backend.formulas.category_i_window import _window as W

NAME = "RBB"
CATEGORY = "I"
TITLE = "Roll Bounce Bias"
BRAIN_NODE = ""
DIRECTIONAL = True
LATENCY_MS = 0.1
DESCRIPTION = "Bid-ask bounce against the side the entry trade printed on (Roll 1984)."


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
    tr = W.trades_only(snapshot.ticks(asset))
    if tr.shape[0] < 10:
        return 0.0
    side = float(tr[-1, 3])
    if side == 0.0:
        trace(ctx, "note", 0, "entry print carries no aggressor side - RBB abstains")
        return 0.0
    bid, ask, _, _ = W.best_quotes(snapshot.book(asset))
    tick = W.tick_size(tr[:, 1], bid, ask)
    if bid > 0:
        half_spread = 0.5 * (ask - bid)
        basis = "book"
    else:
        d = np.diff(tr[-300:, 1])
        cov = float(np.mean(d[1:] * d[:-1])) if d.size > 3 else 0.0
        half_spread = float(np.sqrt(-cov)) if cov < 0 else 0.0
        basis = "Roll estimator"
    half_ticks = half_spread / max(tick, EPS)
    last = W.last_window(snapshot.ticks(asset))
    drift = abs(float(last[-1, 1] - last[0, 1])) if last.shape[0] > 1 else 0.0
    path = float(np.sum(np.abs(np.diff(last[:, 1])))) if last.shape[0] > 1 else 0.0
    drift_share = drift / (path + EPS)
    value = -side * tanh(half_ticks) * (1.0 - drift_share)
    state.last = value
    trace(ctx, "entry side", side, "+1 printed at the ask, −1 at the bid")
    trace(ctx, "half spread", half_ticks, f"ticks ({basis})")
    trace(ctx, "drift share", drift_share, "|net move| / path length of the last window")
    trace(ctx, "bounce", value, "−side × tanh(half spread) × (1 − drift share)")
    return finite(value)


DOUBLE_CHECK = "value = −entry side × tanh(half spread ticks) × (1 − drift share)"


def double_check(t: dict, asset: str) -> float:
    if "entry side" not in t or "half spread" not in t:
        return 0.0
    return -float(t["entry side"]) * tanh(float(t["half spread"])) * (1.0 - float(t["drift share"]))
