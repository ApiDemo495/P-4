"""FORMULA 1 - Tick Acceleration Impulse (TAI).

Measures the instantaneous "jerk" (third derivative) of price on the last
T ticks using non-uniform finite differences, so it works with the irregular
tick spacing of a real order-driven market.  v1.0's MMF stopped at the second
derivative; TAI goes one level deeper, which is what makes it the *earliest*
detectable onset of a move.

Brain mapping: ORN class Or67d - the strongest pheromone receptor, i.e. the
highest-priority input to the antennal lobe.

Latency budget: < 0.2 ms (three numpy passes over <= 30 ticks).
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, RunningVariance, finite, tanh

NAME = "TAI"
CATEGORY = "A"
TITLE = "Tick Acceleration Impulse"
BRAIN_NODE = "ORN Or67d"
DIRECTIONAL = True
LATENCY_MS = 0.2
DESCRIPTION = "Volume-weighted price jerk (3rd derivative) on the last 30 ticks."

RECENT_JERK_POINTS = 10


class State:
    """EMA of squared jerk over ~300 cycles (0.999 / 0.001 recursion)."""

    __slots__ = ("sigma", "last_jerk")

    def __init__(self) -> None:
        self.sigma = RunningVariance(decay=0.999, initial=0.0)
        self.last_jerk = 0.0

    def to_dict(self) -> dict:
        return {"sigma": self.sigma.to_dict(), "last_jerk": self.last_jerk}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.sigma = RunningVariance.from_dict(payload.get("sigma", {}))
        obj.last_jerk = float(payload.get("last_jerk", 0.0))
        return obj


def _derivatives(prices: np.ndarray, times_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = np.diff(times_ms)
    dt = np.maximum(dt, 1e-3)  # ms - never divide by a zero gap

    velocity = np.diff(prices) / dt
    if velocity.size >= 2:
        acceleration = np.diff(velocity) / (0.5 * (dt[:-1] + dt[1:]))
    else:
        acceleration = np.zeros(0)
    if acceleration.size >= 2:
        jerk = np.diff(acceleration) / ((dt[:-2] + dt[1:-1] + dt[2:]) / 3.0)
    else:
        jerk = np.zeros(0)
    return velocity, acceleration, jerk


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    ticks = snapshot.ticks(asset)
    window = int(params.get("tai_ticks", 30))
    if ticks.shape[0] < 6:
        return 0.0

    window = min(window, ticks.shape[0])
    ticks = ticks[-window:]
    prices = ticks[:, 1]
    times = ticks[:, 0]
    volumes = ticks[:, 2]
    sides = ticks[:, 3]

    _, _, jerk = _derivatives(prices, times)
    if jerk.size == 0:
        return 0.0

    k = min(RECENT_JERK_POINTS, jerk.size)
    jerk_recent = jerk[-k:]
    # Volumes/sides aligned to the LAST k jerk points (jerk lags by 3 ticks).
    v = volumes[-k:]
    s = sides[-k:]

    num = float(np.sum(v * s * jerk_recent))
    den = float(np.sum(v)) + EPS
    j_recent = num / den
    state.last_jerk = j_recent

    state.sigma.update(j_recent)
    # Floor the scale estimator while the EMA is still warming up so the very
    # first cycles cannot produce an absurd |TAI| = 1.
    scale = state.sigma.std(floor=1e-24)
    if not state.sigma.warm():
        scale = max(scale, abs(j_recent) * 3.0, 1e-18)

    return finite(tanh(j_recent / (scale + EPS)))
