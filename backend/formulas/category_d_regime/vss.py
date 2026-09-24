"""FORMULA 13 - Volatility Surprise Score (VSS).

Compares realised volatility over the last minute with the 5-minute baseline
that the market has "priced in".  A positive surprise means the tape is waking
up - large moves are becoming likely.  The magnitude is then multiplied by the
sign of the recent mean return, so a volatility spike *with* upward drift reads
bullish while a spike with downward drift reads bearish.

Brain mapping: giant fibre (GF) - the fly's startle neuron, which amplifies the
brain's response to everything else when it fires.

Latency budget: < 0.2 ms over <= 300 returns.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite, tanh, trace

NAME = "VSS"
CATEGORY = "D"
TITLE = "Volatility Surprise Score"
BRAIN_NODE = "Giant Fiber (GF)"
DIRECTIONAL = True
LATENCY_MS = 0.2
DESCRIPTION = "Realised 1-minute volatility vs. the 5-minute baseline, signed by drift."

SHORT_WINDOW = 60
LONG_WINDOW = 300
#: The direction is only reported when the drift inside the baseline window is
#: statistically visible (|t| >= 1.5).  The 1-tick volatility is ~5x the 1-tick
#: drift, so a 60-tick mean return flips sign at random - that is where v2.0.0's
#: "volatility surprise" got its random +/- readings from.
MIN_T_STAT = 1.5


class State:
    __slots__ = ("last_sigma_short", "last_sigma_long")

    def __init__(self) -> None:
        self.last_sigma_short = 0.0
        self.last_sigma_long = 0.0

    def to_dict(self) -> dict:
        return {"last_sigma_short": self.last_sigma_short, "last_sigma_long": self.last_sigma_long}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_sigma_short = float(payload.get("last_sigma_short", 0.0))
        obj.last_sigma_long = float(payload.get("last_sigma_long", 0.0))
        return obj


def _returns(prices: np.ndarray) -> np.ndarray:
    if prices.size < 2:
        return np.zeros(0)
    safe = np.maximum(prices, EPS)
    return np.diff(np.log(safe))


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    prices = snapshot.prices(asset)
    if prices.size < 10:
        return 0.0

    returns = _returns(prices)
    if returns.size < 10:
        return 0.0

    short = returns[-min(SHORT_WINDOW, returns.size) :]
    long = returns[-min(LONG_WINDOW, returns.size) :]

    sigma_real = float(np.sqrt(np.mean(short**2)))
    sigma_exp = float(np.sqrt(np.mean(long**2)))
    state.last_sigma_short, state.last_sigma_long = sigma_real, sigma_exp

    if sigma_exp <= EPS:
        return 0.0

    surprise = (sigma_real - sigma_exp) / (sigma_exp + EPS)
    drift = float(np.mean(long))
    t_stat = abs(drift) * float(np.sqrt(long.size)) / (sigma_exp + EPS)
    direction = float(np.sign(drift)) if t_stat >= MIN_T_STAT else 0.0

    trace(ctx, "realised vol (1 min)", sigma_real, "per tick")
    trace(ctx, "baseline vol (5 min)", sigma_exp, "per tick")
    trace(ctx, "vol surprise", surprise, "sigma_short / sigma_long - 1")
    trace(ctx, "drift (baseline window)", drift, "mean log return per tick")
    trace(ctx, "drift t-statistic", t_stat, "|mean| / (sigma / sqrt(n))")
    trace(ctx, "direction used", direction, "+1 / 0 / -1")

    if direction == 0.0:
        return 0.0
    return finite(tanh(surprise) * direction)
