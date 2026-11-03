"""FORMULA 19 - News Impact Velocity (NIV)  [NEW in v2.0].

Turns the last five headlines into one number.  Each item contributes
``sentiment x credibility x exp(-age/300)``: a Tier-1 wire story two minutes old
dominates a Tier-4 blog post from twenty minutes ago, and anything older than
five minutes carries under 37% weight.

    NIV = SUM(s_j * c_j * exp(-dt_j/300)) / (SUM(c_j * exp(-dt_j/300)) + W0)

``W0`` (= 0.5, the *resting evidence*) replaces the specification's ``eps``
(Deviation 19-A): with a bare ``eps`` the decay cancels between numerator and
denominator, so a single bullish wire from twenty minutes ago kept reading
+0.72 forever and the brain's strongest input never moved.  With ``W0`` a
stale headline decays towards zero, and five fresh credible headlines that
agree still read ~90 % of their sentiment.

Still bounded in [-1, +1], because every s_j is and W0 > 0.

Brain mapping: Johnston's organ (antennal mechanosensory) - environmental
awareness beyond direct price action.

Latency budget: < 0.01 ms (five items).
"""

from __future__ import annotations

import math

from backend.formulas._util import EPS, finite, trace

NAME = "NIV"
CATEGORY = "G"
TITLE = "News Impact Velocity"
BRAIN_NODE = "Johnston's organ"
DIRECTIONAL = True
LATENCY_MS = 0.01
DESCRIPTION = "Credibility- and age-weighted sentiment of the five latest headlines."

ITEMS = 5
DECAY_SECONDS = 300.0
#: Resting evidence weight - see the module docstring (Deviation 19-A).
RESTING_WEIGHT = 0.5


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
        weight = float(item.credibility) * float(getattr(item, "relevance", 1.0) or 1.0) * math.exp(-age / DECAY_SECONDS)
        if weight <= 0.0:
            continue
        num += float(item.sentiment) * weight
        den += weight
        used += 1

    state.contributing_items = used
    trace(ctx, "headlines weighted", used, f"of {min(len(items), ITEMS)} considered")
    trace(ctx, "weighted sentiment sum", num, "sentiment x credibility x e^(-age/tau)")
    trace(ctx, "weight sum", den, "credibility x e^(-age/tau)")
    trace(ctx, "resting weight W0", RESTING_WEIGHT, "stale news decays to zero")
    if den <= EPS:
        state.last_niv = 0.0
        return 0.0

    niv = num / (den + RESTING_WEIGHT)
    state.last_niv = niv
    return finite(max(-1.0, min(1.0, niv)))


DOUBLE_CHECK = "value = clip(weighted sentiment sum / (weight sum + W0), -1, 1)"


def double_check(t: dict, asset: str) -> float:
    """Independent re-derivation of the output from the traced intermediates."""
    if "weight sum" not in t or float(t["weight sum"]) <= EPS:
        return 0.0
    w0 = float(t.get("resting weight W0", RESTING_WEIGHT))
    return max(-1.0, min(1.0, float(t["weighted sentiment sum"]) / (float(t["weight sum"]) + w0)))
