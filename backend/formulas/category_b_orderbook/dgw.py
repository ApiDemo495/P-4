"""FORMULA 5 - Depth Gravity Well (DGW).

Price is attracted to clusters of resting orders.  DGW locates the "centre of
mass" of each side of the book - weighting each level by the *square* of its
size, so a 100 BTC wall counts 100x more than a 1 BTC order rather than 100x/100
- and reports which side's mass sits closer to the mid.  Large bids near the
mid and asks far away means gravity pulls up.

Brain mapping: proprioceptive chordotonal neurons (femoral) - the fly's sense of
which way is down.

Latency budget: < 0.05 ms (40 multiply-adds).
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite

NAME = "DGW"
CATEGORY = "B"
TITLE = "Depth Gravity Well"
BRAIN_NODE = "Chordotonal (femoral)"
DIRECTIONAL = True
LATENCY_MS = 0.05
DESCRIPTION = "Volume-squared centre of mass of each book side vs. the mid price."


class State:
    __slots__ = ("last_gravity", "last_p_bid", "last_p_ask")

    def __init__(self) -> None:
        self.last_gravity = 0.0
        self.last_p_bid = 0.0
        self.last_p_ask = 0.0

    def to_dict(self) -> dict:
        return {
            "last_gravity": self.last_gravity,
            "last_p_bid": self.last_p_bid,
            "last_p_ask": self.last_p_ask,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_gravity = float(payload.get("last_gravity", 0.0))
        obj.last_p_bid = float(payload.get("last_p_bid", 0.0))
        obj.last_p_ask = float(payload.get("last_p_ask", 0.0))
        return obj


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    book = snapshot.book(asset)
    if book is None or book.shape != (2, 20, 2):
        return 0.0

    bid_prices, bid_qty = book[0, :, 0], book[0, :, 1]
    ask_prices, ask_qty = book[1, :, 0], book[1, :, 1]

    valid_bids = (bid_prices > 0) & (bid_qty > 0)
    valid_asks = (ask_prices > 0) & (ask_qty > 0)
    if not valid_bids.any() or not valid_asks.any():
        return 0.0

    m = (float(bid_prices[valid_bids][0]) + float(ask_prices[valid_asks][0])) / 2.0

    w_bid = bid_qty[valid_bids] ** 2
    w_ask = ask_qty[valid_asks] ** 2
    p_bid = float(np.sum(bid_prices[valid_bids] * w_bid) / (np.sum(w_bid) + EPS))
    p_ask = float(np.sum(ask_prices[valid_asks] * w_ask) / (np.sum(w_ask) + EPS))
    state.last_p_bid, state.last_p_ask = p_bid, p_ask

    span = p_ask - p_bid
    if not np.isfinite(span) or abs(span) < EPS:
        return 0.0

    gravity = ((m - p_bid) - (p_ask - m)) / span
    state.last_gravity = gravity
    return finite(max(-1.0, min(1.0, gravity)))
