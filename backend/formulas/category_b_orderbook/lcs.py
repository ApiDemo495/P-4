"""FORMULA 6 - Liquidity Cliff Score (LCS).

Finds the worst level-to-level liquidity drop-off on each side of the book.  A
deep ask-side cliff means there is nothing above the offer to stop a buyer, so a
modest push sends price through empty space; a deep bid-side cliff means the
floor is missing.  LCS reports the asymmetry, which is what actually predicts
direction.

Sign convention (as resolved in the spec):
    LCS = tanh( (|C_ask| - |C_bid|) / (|C_ask| + |C_bid| + eps) )
    ask cliff deeper  -> numerator > 0 -> bullish

Brain mapping: GRN class Gr66a (bitter / aversive) - spotting the level where
price could fall off a cliff.

Latency budget: < 0.05 ms.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite, tanh, trace

NAME = "LCS"
CATEGORY = "B"
TITLE = "Liquidity Cliff Score"
BRAIN_NODE = "GRN Gr66a"
DIRECTIONAL = True
LATENCY_MS = 0.05
DESCRIPTION = "Asymmetry of the deepest liquidity drop-off between bid and ask ladders."


class State:
    __slots__ = ("last_c_ask", "last_c_bid")

    def __init__(self) -> None:
        self.last_c_ask = 0.0
        self.last_c_bid = 0.0

    def to_dict(self) -> dict:
        return {"last_c_ask": self.last_c_ask, "last_c_bid": self.last_c_bid}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_c_ask = float(payload.get("last_c_ask", 0.0))
        obj.last_c_bid = float(payload.get("last_c_bid", 0.0))
        return obj


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    book = snapshot.book(asset)
    if book is None or book.shape != (2, 20, 2):
        return 0.0

    ask_qty = book[1, :, 1]
    bid_qty = book[0, :, 1]
    if ask_qty.size < 3 or bid_qty.size < 3 or not ask_qty.any() or not bid_qty.any():
        return 0.0

    # A "cliff" is the worst level-to-level drop in resting size as you walk the
    # ladder outward, measured against that side's top-of-book size.  Both sides
    # use the same sign convention, so a symmetric book reads exactly 0 - the
    # old code took abs() of values whose sign had already been discarded and
    # reported +0.55 for two identical ladders.
    scale_ask = float(ask_qty[0]) + EPS
    scale_bid = float(bid_qty[0]) + EPS
    w_ask = max(0.0, float(np.max(-np.diff(ask_qty))) / scale_ask)
    w_bid = max(0.0, float(np.max(-np.diff(bid_qty))) / scale_bid)
    state.last_c_ask, state.last_c_bid = -w_ask * scale_ask, -w_bid * scale_bid
    trace(ctx, "ask cliff (worst drop / top size)", w_ask, "fraction of top-of-book")
    trace(ctx, "bid cliff (worst drop / top size)", w_bid, "fraction of top-of-book")

    if w_ask <= 0.0 and w_bid <= 0.0:
        trace(ctx, "asymmetry", 0.0, "no cliff on either side")
        return 0.0

    raw = (w_ask - w_bid) / (w_ask + w_bid + EPS)
    trace(ctx, "asymmetry", raw, "positive = the ask ladder is the broken one")
    return finite(tanh(raw))
