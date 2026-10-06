"""FORMULA 17 - Time-Weighted Return Skewness (TWRS).

Third moment of the tick-return distribution, weighted toward the recent window.
Positive skew means an occasional fat right tail (big up-prints) while negative
skew means the opposite.  Standard skewness averages the whole hour equally,
which is useless at a 60-second horizon - the character of the last 10 ticks is
what matters.

Brain mapping: ORN class Or85a - an asymmetric response-profile receptor.

Latency budget: < 0.1 ms.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from backend.formulas._util import EPS, finite, tanh, trace

NAME = "TWRS"
CATEGORY = "E"
TITLE = "Time-Weighted Return Skewness"
BRAIN_NODE = "ORN Or85a"
DIRECTIONAL = True
LATENCY_MS = 0.1
DESCRIPTION = "Recency-weighted third moment of tick returns (tail asymmetry)."

WINDOW = 60
#: Recency weighting is linear (i/n).  A *square* weight (v2.0.0) put 80 % of
#: the mass on the last 700 ticks, so the tails the third moment is made of
#: were discounted away and a skew that was visibly +1.3 on the tape read
#: +0.007.
POWER = 1.0
#: Rolling horizon of tick returns the skew is estimated on.  A third moment
#: needs the tails: 60 ticks (one window) contains roughly one tail observation,
#: which is why the old per-window skew looked like a random number generator.
HORIZON = 3000
WINSOR_SIGMA = 4.0
SKEW_SCALE = 1.5
GATE_T_LOW, GATE_T_HIGH = 2.0, 5.0
#: The engine hands the formula one window per cycle, so the state has to stitch
#: the windows into a single continuous return series.  v2.0.0 appended only the
#: trailing 60 prices per window: the deque filled with *disjoint* snippets and
#: the reported third moment had nothing to do with the tape's tails.
TICKS_PER_WINDOW = 600


class State:
    __slots__ = ("last_skew", "last_m2", "returns", "last_price", "last_ts")

    def __init__(self) -> None:
        self.last_skew = 0.0
        self.last_m2 = 0.0
        self.returns: deque[float] = deque(maxlen=HORIZON)
        self.last_price = 0.0  # closes the gap between two windows
        self.last_ts = 0.0  # guards against re-processing the same snapshot

    def to_dict(self) -> dict:
        return {
            "last_skew": self.last_skew,
            "last_m2": self.last_m2,
            "returns": list(self.returns),
            "last_price": self.last_price,
            "last_ts": self.last_ts,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_skew = float(payload.get("last_skew", 0.0))
        obj.last_m2 = float(payload.get("last_m2", 0.0))
        obj.last_price = float(payload.get("last_price", 0.0))
        obj.last_ts = float(payload.get("last_ts", 0.0))
        for value in payload.get("returns", []):
            obj.returns.append(float(value))
        return obj


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    prices = np.asarray(snapshot.prices(asset), dtype=np.float64)
    if prices.size < 12:
        return 0.0

    # Formulas only ever see the current snapshot, so the horizon lives in the
    # state; the recency weighting below still emphasises the newest returns.
    ticks = snapshot.ticks(asset)
    last_ts = float(ticks[-1, 0]) if ticks is not None and ticks.size else 0.0
    fresh_n = 0
    if last_ts > state.last_ts:
        fresh = np.diff(np.log(np.maximum(prices, EPS)))
        if fresh.size and state.last_price > 0.0:
            # The link return makes the series continuous across windows.
            link = float(np.log(max(prices[0], EPS) / max(state.last_price, EPS)))
            fresh = np.concatenate([[link], fresh])
        for value in fresh:
            state.returns.append(float(value))
        fresh_n = int(fresh.size)
        state.last_price = float(prices[-1])
        state.last_ts = last_ts

    returns = np.asarray(state.returns, dtype=np.float64)
    n = returns.size
    if n < 8:
        return 0.0

    weights = (np.arange(1, n + 1, dtype=np.float64) / n) ** POWER
    w_sum = float(np.sum(weights)) + EPS

    # Round AI: winsorise at ±WINSOR_SIGMA before the moments.  One bad print
    # (an off-market trade, a feed glitch) contributes r³ to the third moment
    # and pinned the output at ±0.999 for the next five minutes; the tails the
    # skew is *about* are still there at four sigma.
    raw_sd = float(np.sqrt(np.mean(returns**2))) + EPS
    returns = np.clip(returns, -WINSOR_SIGMA * raw_sd, WINSOR_SIGMA * raw_sd)

    mean = float(np.sum(weights * returns) / w_sum)
    centered = returns - mean
    m2 = float(np.sum(weights * centered**2) / w_sum)
    m3 = float(np.sum(weights * centered**3) / w_sum)
    state.last_m2 = m2

    # The standardised third moment is m3 / m2**1.5.  Adding the *absolute*
    # epsilon (1e-12) to that denominator looked harmless but is 60x larger than
    # m2**1.5 for tick-sized returns (~2e-14), so every tape read ~0.01 no
    # matter what its tails did.  Guard the degenerate case instead of padding
    # the denominator.
    if m2 <= EPS:
        return 0.0
    skew = m3 / (m2**1.5)
    state.last_skew = skew
    # Gate by the skew's own standard error sqrt(6/n_eff): a skew that is not
    # two standard errors from zero is noise and votes nothing; the gate is
    # fully open at five.  The magnitude is tanh(skew / 1.5) so a genuinely
    # one-sided tape reads strong without pinning at ±1.000.
    n_eff = (w_sum ** 2) / (float(np.sum(weights**2)) + EPS)
    se = float(np.sqrt(6.0 / max(n_eff, 1.0)))
    t_stat = skew / (se + EPS)
    gate = float(np.clip((abs(t_stat) - GATE_T_LOW) / (GATE_T_HIGH - GATE_T_LOW), 0.0, 1.0))
    trace(ctx, "returns in the horizon", n, "ticks (5 minutes)")
    trace(ctx, "new returns this window", fresh_n, "ticks")
    trace(ctx, "weighted skew", skew, "m3 / m2^1.5 (returns winsorised at ±4σ)")
    trace(ctx, "skew t-stat", t_stat, "skew / sqrt(6 / n_eff)")
    trace(ctx, "significance gate", gate, f"0 below {GATE_T_LOW:g} SE, 1 above {GATE_T_HIGH:g} SE")
    return finite(tanh(skew / SKEW_SCALE) * gate)


DOUBLE_CHECK = "value = tanh(skew / 1.5) x gate, gate = clip((|t| - 2) / 3, 0, 1), t = skew / sqrt(6 / n_eff)"


def double_check(t: dict, asset: str) -> float:
    """Independent re-derivation of the output from the traced intermediates."""
    if "skew t-stat" not in t or "weighted skew" not in t:
        return 0.0
    t_stat = float(t["skew t-stat"])
    gate = max(0.0, min(1.0, (abs(t_stat) - GATE_T_LOW) / (GATE_T_HIGH - GATE_T_LOW)))
    return tanh(float(t["weighted skew"]) / SKEW_SCALE) * gate
