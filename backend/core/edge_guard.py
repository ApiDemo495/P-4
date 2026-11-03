"""Round AM - the live edge guard.

A record that is *significantly worse than a coin flip* is not noise, it is
information: the tape has been doing the opposite of what the engine says.
Hedge-fund desks treat a sign-flipped model as a model with an edge and a
wrong sign.  The guard measures the engine's **raw** side (what the recipe
said before any inversion) against the decided windows (flat windows do not
count) and, when the Wilson upper confidence bound of the raw hit rate is
below one half, publishes the opposite side until the raw hit rate has
recovered to one half.

Everything is measured on the raw side, so the guard cannot oscillate: a
successful inversion keeps the raw rate low (the raw side keeps losing) and
the guard stays on; it only releases when the raw side itself is right again.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MIN_DECIDED = 12       # decided windows before the guard may act
LOOKBACK = 30          # raw outcomes considered
Z = 1.96               # 95 % Wilson interval
RELEASE_AT = 0.50      # raw hit rate (point estimate) that switches the guard off


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class GuardState:
    inverted: bool = False
    raw_hit_rate: float | None = None
    decided: int = 0
    upper: float | None = None
    lower: float | None = None
    note: str = ""

    def as_dict(self) -> dict:
        return {"inverted": self.inverted, "raw_hit_rate": self.raw_hit_rate, "decided": self.decided,
                "upper": self.upper, "lower": self.lower, "note": self.note,
                "min_decided": MIN_DECIDED, "lookback": LOOKBACK}


def assess(raw_outcomes, currently_inverted: bool) -> GuardState:
    """``raw_outcomes``: +1 / -1 per decided window, measured on the engine's
    raw side (oldest first).  Returns the guard's state for the next window."""
    rows = [int(x) for x in (raw_outcomes or []) if x]
    rows = rows[-LOOKBACK:]
    n = len(rows)
    if n < MIN_DECIDED:
        return GuardState(inverted=False, decided=n,
                          note=f"edge guard arming: {n}/{MIN_DECIDED} decided windows")
    k = sum(1 for x in rows if x > 0)
    p = k / n
    lo, hi = wilson(k, n)
    if currently_inverted:
        inverted = p < RELEASE_AT
        note = (f"INVERTED: the raw recipe side has hit {p:.0%} of the last {n} decided windows "
                f"(95% CI {lo:.0%}–{hi:.0%}); the published side is the opposite until the raw rate is back to 50%"
                if inverted else
                f"edge guard released: raw side back to {p:.0%} of {n}")
    else:
        inverted = hi < 0.5
        note = (f"INVERTED: the raw recipe side hit only {p:.0%} of the last {n} decided windows "
                f"(95% CI {lo:.0%}–{hi:.0%}, upper bound below 50%) - publishing the opposite side"
                if inverted else
                f"edge guard watching: raw side {p:.0%} of {n} (95% CI {lo:.0%}–{hi:.0%})")
    return GuardState(inverted=inverted, raw_hit_rate=round(p, 4), decided=n,
                      upper=round(hi, 4), lower=round(lo, 4), note=note)
