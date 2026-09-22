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

from backend.formulas._util import EPS, finite, tanh

NAME = "VSS"
CATEGORY = "D"
TITLE = "Volatility Surprise Score"
BRAIN_NODE = "Giant Fiber (GF)"
DIRECTIONAL = True
LATENCY_MS = 0.2
DESCRIPTION = "Realised 1-minute volatility vs. the 5-minute baseline, signed by drift."

SHORT_WINDOW = 60
LONG_WINDOW = 300


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
    drift_sign = float(np.sign(np.mean(short)))
    if drift_sign == 0.0:
        return 0.0

    return finite(tanh(surprise) * drift_sign)
