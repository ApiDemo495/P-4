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

from collections import deque

import numpy as np

from backend.formulas._util import EPS, RunningVariance, finite, tanh, trace

NAME = "TAI"
CATEGORY = "A"
TITLE = "Tick Acceleration Impulse"
BRAIN_NODE = "ORN Or67d"
DIRECTIONAL = True
LATENCY_MS = 0.2
DESCRIPTION = "Volume-weighted price jerk (3rd derivative) on the last 30 ticks."

#: The jerk is the cubic term of a volume-weighted least-squares fit (6 * c3).
#: The literal third difference of a tick series is dominated by tick noise: its
#: magnitude scales as sigma/dt^3, which on a real tape is orders of magnitude
#: above a genuine acceleration.  The fit is therefore taken over a rolling
#: buffer of the last ``FIT_TICKS`` ticks (30 seconds at 20 ticks/s), which is
#: long enough for a real acceleration to dominate the fit.
FIT_TICKS = 600
#: Smoothing span applied before fitting (removes the tick noise floor).
SMOOTH_SPAN = 5
#: Recency time constant (seconds) of the exponential weighting.  The onset of a
#: move is what TAI is for, so a tick from 10 seconds ago counts e times as much
#: as one from a minute ago.  Without it the impulse only occupies 30 % of the
#: fit and its curvature is diluted by the quiet part of the window.
RECENCY_TAU = 10.0
#: Materiality floor for the *relative* jerk (price-relative per second cubed).
#: 2.6e-6 /s^3 is a curvature that would move the price ~4 bps in ten seconds -
#: a real move.  The noise floor of a recency-weighted fit sits 25x below it,
#: which is why TAI now reads near zero on chop instead of +-1 window to window.
JERK_FLOOR = 2.6e-6


class State:
    """EMA of squared jerk over ~300 cycles (0.999 / 0.001 recursion)."""

    __slots__ = ("sigma", "last_jerk", "buffer")

    def __init__(self) -> None:
        self.sigma = RunningVariance(decay=0.999, initial=0.0)
        self.last_jerk = 0.0
        # (time_ms, price, volume, side) for the last FIT_TICKS ticks.
        self.buffer: deque[tuple[float, float, float, float]] = deque(maxlen=FIT_TICKS)

    def to_dict(self) -> dict:
        return {
            "sigma": self.sigma.to_dict(),
            "last_jerk": self.last_jerk,
            "buffer": [list(row) for row in self.buffer],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.sigma = RunningVariance.from_dict(payload.get("sigma", {}))
        obj.last_jerk = float(payload.get("last_jerk", 0.0))
        for row in payload.get("buffer", []):
            if len(row) == 4:
                obj.buffer.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
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

    # Rolling buffer: the engine hands the formula a parameterised slice, but
    # the cubic fit needs a continuous path, so the buffer is fed the *whole*
    # snapshot window (600 ticks = 60 s at 10 Hz) rather than the last 30 ticks.
    buffer = state.buffer
    buffer.extend(
        (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        for row in snapshot.ticks(asset)
    )
    fit_ticks = np.asarray(buffer, dtype=np.float64)
    if fit_ticks.shape[0] < 12:
        return 0.0
    prices = fit_ticks[:, 1]
    times = (fit_ticks[:, 0] - fit_ticks[0, 0]) / 1000.0  # seconds from buffer start
    volumes = fit_ticks[:, 2]
    sides = fit_ticks[:, 3]
    if times[-1] <= 0:
        return 0.0

    # Light EMA smoothing: kills the tick-level noise floor without moving a
    # sustained acceleration (the impulse lives for hundreds of ticks).
    alpha = 2.0 / (SMOOTH_SPAN + 1.0)
    smooth = np.empty_like(prices)
    smooth[0] = prices[0]
    for i in range(1, prices.size):
        smooth[i] = alpha * prices[i] + (1.0 - alpha) * smooth[i - 1]
    prices = smooth

    # Recency- and volume-weighted cubic fit: c3 is the coefficient of t^3, so
    # jerk = 6*c3.  Volume weighting is the specification's "volume-weighted
    # jerk" (heavy prints dominate the shape of the path); the recency factor
    # concentrates the fit on the last ~20 seconds, where the acceleration is.
    weights = np.sqrt(np.maximum(volumes, 1e-9)) * np.exp(
        -(times[-1] - times) / RECENCY_TAU
    )
    try:
        coeffs = np.polyfit(times, prices, 3, w=weights)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0
    jerk = 6.0 * float(coeffs[0])

    # The aggressor side says which way the tape was being pushed while it
    # accelerated; without it an aggressive sell-off into a rising fit would be
    # scored as bullish.
    # The jerk of the price already carries the direction; the aggressor side is
    # reported for context only.  (Multiplying by it applied the sign twice and
    # flipped the bear tape positive.)
    net_side = float(np.sum(volumes * sides)) / (float(np.sum(volumes)) + EPS)
    j_recent = jerk
    state.last_jerk = j_recent
    state.sigma.update(j_recent)

    # Price-relative jerk, so the same *shape* of tape reads the same on BTC
    # at 68k and on a hypothetical asset at 1k.
    price_ref = float(np.mean(prices))
    j_rel = j_recent / max(abs(price_ref), EPS)
    scale = JERK_FLOOR

    trace(ctx, "ticks in the cubic fit", int(prices.size), "volume-weighted LS fit")
    trace(ctx, "cubic coefficient c3", float(coeffs[0]), "price / s^3")
    trace(ctx, "jerk = 6 * c3", jerk, "price / s^3")
    trace(ctx, "price used for normalising", price_ref, "price units")
    trace(ctx, "relative jerk", j_rel, "1 / s^3")
    trace(ctx, "materiality floor", scale, "1 / s^3 (2.6e-6 ~ 4 bp in ten seconds)")
    trace(ctx, "jerk / floor", j_rel / (scale + EPS), "1.0 = saturating")
    trace(ctx, "net signed volume share", net_side, "+1 all buys, -1 all sells")

    return finite(tanh(j_rel / (scale + EPS)))
