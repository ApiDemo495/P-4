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

from backend.core import config as cfg

from backend.formulas._util import EPS, SelfScale, finite, tanh, trace

NAME = "BAR"
CATEGORY = "B"
TITLE = "Book Absorption Rate"
BRAIN_NODE = "ORN Or22a"
DIRECTIONAL = True
LATENCY_MS = 0.05
DESCRIPTION = "Net consumption of resting top-10 bid vs. ask liquidity between snapshots."

ACTIVE_LEVELS = 10
#: 5 % of a side's top-10 depth consumed between snapshots is a full vote.
FULL_VOTE_FRACTION = 0.05
#: Below half a percent on both sides nothing was absorbed - it is noise.
MIN_FRACTION = 0.005


class State:
    __slots__ = ("last_a_bid", "last_a_ask", "asym_scale")

    def __init__(self) -> None:
        self.last_a_bid = 0.0
        self.last_a_ask = 0.0
        # Typical |asymmetry| of this book: a venue whose snapshots routinely
        # churn 40 % of the depth must not read every churn as a full vote.
        self.asym_scale = SelfScale(decay=0.97, floor=FULL_VOTE_FRACTION)

    def to_dict(self) -> dict:
        return {"last_a_bid": self.last_a_bid, "last_a_ask": self.last_a_ask, "asym_scale": self.asym_scale.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_a_bid = float(payload.get("last_a_bid", 0.0))
        obj.last_a_ask = float(payload.get("last_a_ask", 0.0))
        obj.asym_scale = SelfScale.from_dict(payload.get("asym_scale", {})) if payload.get("asym_scale") else obj.asym_scale
        return obj


def _side_total(book: np.ndarray, side: int, levels: int) -> float:
    if book is None or book.shape != (2, cfg.L2_DEPTH_LEVELS, 2):
        return 0.0
    return float(np.sum(book[side, :levels, 1]))


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    now = snapshot.book(asset)
    prev = snapshot.book_prev(asset)
    if now is None or prev is None:
        return 0.0
    if not np.any(prev) or not np.any(now):
        return 0.0

    prev_bid, prev_ask = _side_total(prev, 0, ACTIVE_LEVELS), _side_total(prev, 1, ACTIVE_LEVELS)
    a_bid = max(0.0, prev_bid - _side_total(now, 0, ACTIVE_LEVELS))
    a_ask = max(0.0, prev_ask - _side_total(now, 1, ACTIVE_LEVELS))
    state.last_a_bid, state.last_a_ask = a_bid, a_ask

    # Round AJ: absorption is measured as the *fraction* of each side's
    # resting depth that disappeared, and the asymmetry of those fractions is
    # scaled by FULL_VOTE_FRACTION.  The old (ask − bid)/(ask + bid) printed
    # −1.000 when 0.01 BTC left the bid and nothing left the ask - a rounding
    # event read as "very strong downward pressure".
    f_bid = a_bid / (prev_bid + EPS)
    f_ask = a_ask / (prev_ask + EPS)
    trace(ctx, "bid depth absorbed", a_bid, f"contracts gone from the top {ACTIVE_LEVELS} bid levels")
    trace(ctx, "ask depth absorbed", a_ask, f"contracts gone from the top {ACTIVE_LEVELS} ask levels")
    trace(ctx, "bid fraction absorbed", f_bid, "share of resting bid depth consumed")
    trace(ctx, "ask fraction absorbed", f_ask, "share of resting ask depth consumed")
    if max(f_bid, f_ask) < MIN_FRACTION:
        return 0.0

    asym = f_ask - f_bid
    # Divisor = the larger of 5 % and this book's own typical |asymmetry|
    # (EMA): a full vote means "more one-sided than this venue usually is".
    scale = state.asym_scale.denominator(2.0)
    state.asym_scale.update(abs(asym))
    raw = asym / scale
    trace(ctx, "absorption asymmetry", asym, "ask fraction − bid fraction")
    trace(ctx, "asymmetry scale", scale, f"max({FULL_VOTE_FRACTION}, 2 × typical |asymmetry|)")
    return finite(tanh(raw))


DOUBLE_CHECK = "value = tanh(asymmetry / scale), scale = max(0.05, 2 x typical |asymmetry|); 0 when both fractions < 0.5 %"


def double_check(t: dict, asset: str) -> float:
    """Independent re-derivation of the output from the traced intermediates."""
    if "absorption asymmetry" not in t or "asymmetry scale" not in t:
        return 0.0
    f_bid, f_ask = float(t.get("bid fraction absorbed", 0.0)), float(t.get("ask fraction absorbed", 0.0))
    if max(f_bid, f_ask) < MIN_FRACTION:
        return 0.0
    return tanh(float(t["absorption asymmetry"]) / (float(t["asymmetry scale"]) + EPS))
