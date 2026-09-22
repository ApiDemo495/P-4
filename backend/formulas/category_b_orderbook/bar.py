"""FORMULA 7 - Book Absorption Rate (BAR).

Compares the current L2 snapshot with the one taken 15 seconds earlier.  If
resting bids shrank, sellers are eating through them (bearish).  If resting asks
shrank, buyers are eating through them (bullish).  Only the top 10 levels are
used - that is the zone an aggressor can actually reach inside a minute.

Brain mapping: ORN class Or22a (ethyl butyrate - the fly's "fruit" detector),
used for tracking how fast food is being consumed.

Latency budget: < 0.05 ms.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite

NAME = "BAR"
CATEGORY = "B"
TITLE = "Book Absorption Rate"
BRAIN_NODE = "ORN Or22a"
DIRECTIONAL = True
LATENCY_MS = 0.05
DESCRIPTION = "Net consumption of resting top-10 bid vs. ask liquidity between snapshots."

ACTIVE_LEVELS = 10


class State:
    __slots__ = ("last_a_bid", "last_a_ask")

    def __init__(self) -> None:
        self.last_a_bid = 0.0
        self.last_a_ask = 0.0

    def to_dict(self) -> dict:
        return {"last_a_bid": self.last_a_bid, "last_a_ask": self.last_a_ask}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_a_bid = float(payload.get("last_a_bid", 0.0))
        obj.last_a_ask = float(payload.get("last_a_ask", 0.0))
        return obj


def _side_total(book: np.ndarray, side: int, levels: int) -> float:
    if book is None or book.shape != (2, 20, 2):
        return 0.0
    return float(np.sum(book[side, :levels, 1]))


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    now = snapshot.book(asset)
    prev = snapshot.book_prev(asset)
    if now is None or prev is None:
        return 0.0
    if not np.any(prev) or not np.any(now):
        return 0.0

    a_bid = max(0.0, _side_total(prev, 0, ACTIVE_LEVELS) - _side_total(now, 0, ACTIVE_LEVELS))
    a_ask = max(0.0, _side_total(prev, 1, ACTIVE_LEVELS) - _side_total(now, 1, ACTIVE_LEVELS))
    state.last_a_bid, state.last_a_ask = a_bid, a_ask

    if a_bid <= 0.0 and a_ask <= 0.0:
        return 0.0

    raw = (a_ask - a_bid) / (a_ask + a_bid + EPS)
    return finite(max(-1.0, min(1.0, raw)))
