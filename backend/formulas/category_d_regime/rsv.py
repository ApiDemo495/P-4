"""FORMULA 12 - Regime Switch Velocity (RSV).

Most regime indicators tell you which regime you are *in*.  RSV tells you how
fast you are *switching between* them, which fires earlier and is therefore more
actionable.  Eight overlapping 20-tick sub-windows (shifted by 5) give eight
Hurst estimates via rescaled range; the slope across them is the velocity.

    RSV > 0 -> Hurst rising -> trending -> momentum strategies more reliable
    RSV < 0 -> Hurst falling -> mean-reverting -> fade strategies preferred

Brain mapping: circadian clock neuron DN1p - detects transitions between
behavioural states.

Latency budget: < 0.3 ms (8 x O(20)).
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite, tanh

NAME = "RSV"
CATEGORY = "D"
TITLE = "Regime Switch Velocity"
BRAIN_NODE = "Clock DN1p"
DIRECTIONAL = True
LATENCY_MS = 0.3
DESCRIPTION = "Velocity of the Hurst exponent - how fast the market is switching regimes."

WINDOW = 60
SUB_WINDOW = 20
SUB_SHIFT = 5
GAIN = 10.0


class State:
    __slots__ = ("hurst_series", "last_hurst", "last_slope")

    def __init__(self) -> None:
        self.hurst_series: list[float] = []
        self.last_hurst = 0.5
        self.last_slope = 0.0

    def to_dict(self) -> dict:
        return {
            "hurst_series": list(self.hurst_series[-32:]),
            "last_hurst": self.last_hurst,
            "last_slope": self.last_slope,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.hurst_series = [float(x) for x in payload.get("hurst_series", [])][-32:]
        obj.last_hurst = float(payload.get("last_hurst", 0.5))
        obj.last_slope = float(payload.get("last_slope", 0.0))
        return obj


def rescaled_range(segment: np.ndarray) -> float:
    """R/S statistic for one segment (mean-adjusted cumulative range / sigma)."""
    n = segment.size
    if n < 2:
        return 0.0
    mean = float(segment.mean())
    dev = np.cumsum(segment - mean)
    spread = float(dev.max() - dev.min())
    std = float(segment.std(ddof=1))
    if std <= EPS or spread <= EPS:
        return 0.0
    return spread / std


def _hurst_for_window(segment: np.ndarray) -> float:
    rs = rescaled_range(segment)
    if rs <= EPS:
        return 0.5
    h = float(np.log(rs) / np.log(segment.size))
    return float(np.clip(h, 0.0, 1.0))


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    prices = snapshot.prices(asset)
    if prices.size < WINDOW:
        if prices.size < SUB_WINDOW + SUB_SHIFT:
            return 0.0
        prices = prices[-min(prices.size, WINDOW) :]
    else:
        prices = prices[-WINDOW:]

    n_windows = max(1, (prices.size - SUB_WINDOW) // SUB_SHIFT + 1)
    n_windows = min(n_windows, 8)
    hurst: list[float] = []
    for k in range(n_windows):
        start = k * SUB_SHIFT
        segment = prices[start : start + SUB_WINDOW]
        if segment.size < SUB_WINDOW:
            break
        hurst.append(_hurst_for_window(segment))

    if len(hurst) < 2:
        return 0.0

    state.hurst_series.append(hurst[-1])
    if len(state.hurst_series) > 32:
        del state.hurst_series[:-32]

    slope = (hurst[-1] - hurst[0]) / (len(hurst) - 1)
    state.last_hurst = hurst[-1]
    state.last_slope = slope

    return finite(tanh(GAIN * slope))
