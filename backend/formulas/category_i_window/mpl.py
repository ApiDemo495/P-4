"""FORMULA 24 - Microprice Lean (MPL).

Where the *next* trade is more likely to print, in ticks, from the Stoikov
microprice: μ = (bid·Q_ask + ask·Q_bid) / (Q_bid + Q_ask).  The quote with
the thinner queue is the one that gets consumed first, so μ leans toward it.
Signed by (μ − last trade) / tick and squashed; the score is scaled by the
top-of-book depth imbalance so a 1-lot queue cannot claim a strong lean.

Different from DGW (depth *gaps*), LCS (ladder *continuity*) and BAR (depth
*absorbed* between frames): MPL uses only the two best queues and the last
trade, and reads in ticks - the unit the tie is decided in.
"""

from __future__ import annotations

from backend.formulas._util import EPS, finite, tanh, trace
from backend.formulas.category_i_window import _window as W

NAME = "MPL"
CATEGORY = "I"
TITLE = "Microprice Lean"
BRAIN_NODE = ""
DIRECTIONAL = True
LATENCY_MS = 0.05
DESCRIPTION = "Stoikov microprice minus the last trade, in ticks, weighted by queue imbalance."

LEAN_SCALE_TICKS = 0.5


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
    if tr.shape[0] < 2:
        return 0.0
    bid, ask, bq, aq = W.best_quotes(snapshot.book(asset))
    if bid <= 0 or bq + aq <= 0:
        trace(ctx, "note", 0, "no top-of-book - MPL abstains")
        return 0.0
    last = float(tr[-1, 1])
    tick = W.tick_size(tr[:, 1], bid, ask)
    micro = (bid * aq + ask * bq) / (bq + aq + EPS)
    lean_ticks = (micro - last) / max(tick, EPS)
    imbalance = (bq - aq) / (bq + aq + EPS)
    # the lean is only credible when the imbalance that produced it is real
    credibility = min(1.0, abs(imbalance) / 0.2)
    value = tanh(lean_ticks / LEAN_SCALE_TICKS) * credibility
    state.last = value
    trace(ctx, "best bid / ask", f"{bid:.2f} / {ask:.2f}", f"queues {bq:.3f} / {aq:.3f}")
    trace(ctx, "microprice", micro, "(bid·Q_ask + ask·Q_bid) / (Q_bid + Q_ask)")
    trace(ctx, "last trade", last, "the contest's reference price")
    trace(ctx, "lean", lean_ticks, "ticks from last trade to microprice")
    trace(ctx, "queue imbalance", imbalance, "(Q_bid − Q_ask) / (Q_bid + Q_ask)")
    trace(ctx, "credibility", credibility, "min(1, |imbalance| / 0.2)")
    return finite(value)


DOUBLE_CHECK = "value = tanh(lean_ticks / 0.5) × min(1, |queue imbalance| / 0.2)"


def double_check(t: dict, asset: str) -> float:
    if "lean" not in t or "queue imbalance" not in t:
        return 0.0
    return tanh(float(t["lean"]) / LEAN_SCALE_TICKS) * min(1.0, abs(float(t["queue imbalance"])) / 0.2)
