"""FORMULA 19 - News Impact Velocity (NIV)  [NEW in v2.0].

Turns the last five headlines into one number.  Each item contributes
``sentiment x credibility x exp(-age/300)``: a Tier-1 wire story two minutes old
dominates a Tier-4 blog post from twenty minutes ago, and anything older than
five minutes carries under 37% weight.

    NIV = SUM(s_j * c_j * exp(-dt_j/300)) / SUM(c_j * exp(-dt_j/300) + eps)

Already bounded in [-1, +1] by construction, because every s_j is.

Brain mapping: Johnston's organ (antennal mechanosensory) - environmental
awareness beyond direct price action.

Latency budget: < 0.01 ms (five items).
"""

from __future__ import annotations

import math

from backend.formulas._util import EPS, finite

NAME = "NIV"
CATEGORY = "G"
TITLE = "News Impact Velocity"
BRAIN_NODE = "Johnston's organ"
DIRECTIONAL = True
LATENCY_MS = 0.01
DESCRIPTION = "Credibility- and age-weighted sentiment of the five latest headlines."

ITEMS = 5
DECAY_SECONDS = 300.0


class State:
    __slots__ = ("last_niv", "contributing_items")

    def __init__(self) -> None:
        self.last_niv = 0.0
        self.contributing_items = 0

    def to_dict(self) -> dict:
        return {"last_niv": self.last_niv, "contributing_items": self.contributing_items}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_niv = float(payload.get("last_niv", 0.0))
        obj.contributing_items = int(payload.get("contributing_items", 0))
        return obj


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    items = snapshot.news_items or ()
    if not items:
        state.last_niv = 0.0
        state.contributing_items = 0
        return 0.0

    now = snapshot.timestamp
    num = 0.0
    den = 0.0
    used = 0
    for item in list(items)[:ITEMS]:
        age = max(0.0, now - float(item.published_at))
        weight = float(item.credibility) * math.exp(-age / DECAY_SECONDS)
        if weight <= 0.0:
            continue
        num += float(item.sentiment) * weight
        den += weight
        used += 1

    state.contributing_items = used
    if den <= EPS:
        state.last_niv = 0.0
        return 0.0

    niv = num / den
    state.last_niv = niv
    return finite(max(-1.0, min(1.0, niv)))
