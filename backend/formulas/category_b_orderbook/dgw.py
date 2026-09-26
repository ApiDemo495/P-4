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

from backend.formulas._util import EPS, finite, tanh, trace, tanh, trace

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

    # Both distances are positive.  The size-weighted centre of mass sits FAR
    # from the mid on the side whose liquidity is stretched away from the touch,
    # and the specification reads the asymmetry as
    #     gravity = (g_bid - mid) - (mid - g_ask) = distance_ask - distance_bid
    # so positive means the ask ladder is the emptier one -> upward pull.
    distance_bid = m - p_bid
    distance_ask = p_ask - m
    if m <= EPS or not np.isfinite(distance_bid) or not np.isfinite(distance_ask):
        return 0.0

    # Normalise by the mid, not by the distance between the two centres of mass:
    # on a healthy book that distance is about one spread wide, and dividing by
    # it turned rounding noise into +-1 spikes every cycle.
    relative = (distance_ask - distance_bid) / m
    state.last_gravity = relative
    trace(ctx, "mid", m, "price")
    trace(ctx, "bid centre of mass", p_bid, "price")
    trace(ctx, "ask centre of mass", p_ask, "price")
    trace(ctx, "bid distance from mid", distance_bid, "price units")
    trace(ctx, "ask distance from mid", distance_ask, "price units")
    trace(ctx, "relative gravity", relative, "fraction of mid")

    # 2 bps of asymmetry is a clear pull; the value is symmetric and saturating.
    gravity = 1e4 * relative / 2.0
    trace(ctx, "gravity (bps-equivalent)", gravity, "2 bps saturates")
    return finite(tanh(gravity))
