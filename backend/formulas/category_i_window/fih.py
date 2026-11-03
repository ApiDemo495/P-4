"""FORMULA 29 - Flow Imbalance at Horizon (FIH).

Order-flow imbalance at the *book* level (Cont, Kukanov & Stoikov 2014),
not the trade level: between the previous book frame and this one,
  e = [Δ bid size (or +new size if the bid price rose, −old size if it fell)]
    − [Δ ask size (or −new size if the ask price fell, +old size if it rose)]
summed over the top 5 levels, each side divided by its own depth - the dimensionless pressure that linearly predicts the next price
change in that paper.  Squashed with tanh at 0.5 depth-units.

AFPR counts who *hit*; FIH counts who *showed up and left* on the quotes,
which moves the price without a single trade - exactly the mechanism that
breaks a pinned window.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, SelfScale, finite, tanh, trace

NAME = "FIH"
CATEGORY = "I"
TITLE = "Flow Imbalance at Horizon"
BRAIN_NODE = ""
DIRECTIONAL = True
LATENCY_MS = 0.1
DESCRIPTION = "Book-level order-flow imbalance (Cont-Kukanov-Stoikov) over the top 5 levels, per unit depth."

LEVELS = 5
#: floor of the self-calibrating scale: a tenth of the top-5 depth is always
#: a material imbalance; above that the tape's own typical |OFI| sets the unit
SCALE_FLOOR = 0.1


class State:
    __slots__ = ("last", "scale")

    def __init__(self) -> None:
        self.last = 0.0
        self.scale = SelfScale(decay=0.97, floor=SCALE_FLOOR)

    def to_dict(self) -> dict:
        return {"last": self.last, "scale": self.scale.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last = float(payload.get("last", 0.0))
        if payload.get("scale"):
            obj.scale = SelfScale.from_dict(payload["scale"], floor=SCALE_FLOOR)
        return obj


def _side(prev: np.ndarray, cur: np.ndarray, is_bid: bool) -> float:
    """CKS contribution of one side over the levels present in both frames."""
    n = min(prev.shape[0], cur.shape[0], LEVELS)
    e = 0.0
    for i in range(n):
        p0, q0 = float(prev[i, 0]), float(prev[i, 1])
        p1, q1 = float(cur[i, 0]), float(cur[i, 1])
        if p0 <= 0 or p1 <= 0:
            continue
        if is_bid:
            e += q1 if p1 > p0 else (-q0 if p1 < p0 else q1 - q0)
        else:
            # an ask that moves UP was lifted (buy pressure, +q_old); one that
            # moves DOWN was placed inside the spread (sell pressure, −q_new)
            e += q0 if p1 > p0 else (-q1 if p1 < p0 else -(q1 - q0))
    return e


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    try:
        cur = np.asarray(snapshot.book(asset), dtype=np.float64)
        prev = np.asarray(snapshot.book_prev(asset), dtype=np.float64)
    except Exception:  # noqa: BLE001
        return 0.0
    if cur.ndim != 3 or prev.ndim != 3 or cur.shape[1] < 1 or prev.shape[1] < 1:
        trace(ctx, "note", 0, "no two book frames - FIH abstains")
        return 0.0
    e_bid = _side(prev[0], cur[0], True)
    e_ask = _side(prev[1], cur[1], False)
    # each side is normalised by its own depth, so a book that thins (or
    # thickens) on both sides at once reads as no imbalance
    depth_bid = (float(np.sum(cur[0, :LEVELS, 1])) + float(np.sum(prev[0, :LEVELS, 1]))) / 2.0
    depth_ask = (float(np.sum(cur[1, :LEVELS, 1])) + float(np.sum(prev[1, :LEVELS, 1]))) / 2.0
    if depth_bid <= EPS or depth_ask <= EPS:
        return 0.0
    depth = 0.5 * (depth_bid + depth_ask)
    ofi = 0.5 * (e_bid / depth_bid + e_ask / depth_ask)
    scale = state.scale.denominator(2.0)
    state.scale.update(ofi)
    value = tanh(ofi / scale)
    state.last = value
    trace(ctx, "bid-side flow", e_bid, "size added to (−removed from) the top 5 bids")
    trace(ctx, "ask-side flow", e_ask, "asks lifted / pulled (+) vs placed (−) on the top 5 levels")
    trace(ctx, "top-5 depth", depth, f"bid {depth_bid:.3f} · ask {depth_ask:.3f} (mean of the two frames)")
    trace(ctx, "OFI", ofi, "½(bid flow / bid depth + ask flow / ask depth)")
    trace(ctx, "scale", scale, "max(0.1, 2 × typical |OFI|)")
    return finite(value)


DOUBLE_CHECK = "value = tanh(OFI / scale), scale = max(0.1, 2 × typical |OFI|)"


def double_check(t: dict, asset: str) -> float:
    if "OFI" not in t or "scale" not in t:
        return 0.0
    return tanh(float(t["OFI"]) / max(float(t["scale"]), EPS))
