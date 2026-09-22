"""FORMULA 15 - Micro-Cycle Phase Estimator (MCPE).

v1.0 used a Hilbert-Huang transform to find micro-cycle phase and cost 1.5 ms.
MCPE gets ~80% of that signal quality in 1% of the compute: detrend the window
with a single OLS fit, count zero crossings, derive the dominant period from the
crossing count, and read the phase at "now".

    MCPE = -cos(phi) * min(1, Z/4)
    cycle trough -> +1 (buy point), cycle peak -> -1 (sell point)

The Z/4 damping is important: with fewer than 4 crossings there is no reliable
cycle to be at the trough of.

Brain mapping: LNv lateral-ventral clock neurons - circadian phase detection.

Latency budget: < 0.1 ms.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite

NAME = "MCPE"
CATEGORY = "E"
TITLE = "Micro-Cycle Phase Estimator"
BRAIN_NODE = "Clock LNv"
DIRECTIONAL = True
LATENCY_MS = 0.1
DESCRIPTION = "Zero-crossing phase of the detrended 60-tick window (peak / trough / mid)."

WINDOW = 60
MIN_CROSSINGS = 4


class State:
    __slots__ = ("last_phase", "last_period", "last_crossings")

    def __init__(self) -> None:
        self.last_phase = 0.0
        self.last_period = 0.0
        self.last_crossings = 0

    def to_dict(self) -> dict:
        return {
            "last_phase": self.last_phase,
            "last_period": self.last_period,
            "last_crossings": self.last_crossings,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_phase = float(payload.get("last_phase", 0.0))
        obj.last_period = float(payload.get("last_period", 0.0))
        obj.last_crossings = int(payload.get("last_crossings", 0))
        return obj


def detrend(prices: np.ndarray) -> np.ndarray:
    n = prices.size
    idx = np.arange(n, dtype=np.float64)
    xm = idx.mean()
    ym = float(prices.mean())
    xc = idx - xm
    denom = float(np.dot(xc, xc))
    beta = float(np.dot(xc, prices - ym) / denom) if denom > EPS else 0.0
    alpha = ym - beta * xm
    return prices - (alpha + beta * idx)


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    prices = snapshot.prices(asset)
    if prices.size < 12:
        return 0.0
    window = prices[-min(WINDOW, prices.size) :]
    n = window.size

    detrended = detrend(window)
    crossings = int(np.sum(detrended[1:] * detrended[:-1] < 0))
    state.last_crossings = crossings

    if crossings == 0:
        return 0.0

    period = 2.0 * (n - 1) / (crossings + EPS)
    state.last_period = period
    if period <= EPS:
        return 0.0

    phase_index = np.fmod(float(n), period)
    phi = 2.0 * np.pi * phase_index / period
    state.last_phase = float(phi)

    raw = -float(np.cos(phi))
    damping = min(1.0, crossings / MIN_CROSSINGS)
    return finite(raw * damping)
