"""FORMULA 15 - Micro-Cycle Phase Estimator (MCPE).

v1.0 used a Hilbert-Huang transform to find micro-cycle phase and cost 1.5 ms.
MCPE gets ~80% of that signal quality in 1% of the compute: detrend the window
with a single OLS fit, count zero crossings, derive the dominant period from the
crossing count, and read the phase at "now".

    MCPE = cos(2*pi*frac) * dir * damping    frac = phase since the last crossing

    Just after an up-crossing is the trough-and-rising part of the cycle (+1,
    the buy point); half a period later is the peak (-1, the sell point).  The
    damping is the product of three confidences: enough crossings, evenly spaced
    crossings, and a dominant cycle amplitude.

The Z/4 damping is important: with fewer than 4 crossings there is no reliable
cycle to be at the trough of.

Brain mapping: LNv lateral-ventral clock neurons - circadian phase detection.

Latency budget: < 0.1 ms.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, Ema, finite, tanh, trace

NAME = "MCPE"
CATEGORY = "E"
TITLE = "Micro-Cycle Phase Estimator"
BRAIN_NODE = "Clock LNv"
DIRECTIONAL = True
LATENCY_MS = 0.1
DESCRIPTION = "Zero-crossing phase of the detrended 60-tick window (peak / trough / mid)."

WINDOW = 60
MIN_CROSSINGS = 4
#: EMA span used to smooth the window before counting crossings.  Without it the
#: zero crossings of the detrended series are the zero crossings of tick noise
#: (a random walk re-crosses its mean every few ticks), so the "period" read out
#: of a quiet tape was ~30 ticks of pure noise.
SMOOTH_SPAN = 5


class State:
    __slots__ = ("last_phase", "last_period", "last_crossings", "last_regularity", "period_ema")

    def __init__(self) -> None:
        self.last_phase = 0.0
        self.last_period = 0.0
        self.last_crossings = 0
        self.last_regularity = 1.0
        self.period_ema = Ema(span=6)

    def to_dict(self) -> dict:
        return {
            "last_phase": self.last_phase,
            "last_period": self.last_period,
            "last_crossings": self.last_crossings,
            "last_regularity": self.last_regularity,
            "period_ema": self.period_ema.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_phase = float(payload.get("last_phase", 0.0))
        obj.last_period = float(payload.get("last_period", 0.0))
        obj.last_crossings = int(payload.get("last_crossings", 0))
        obj.last_regularity = float(payload.get("last_regularity", 1.0))
        obj.period_ema = Ema.from_dict(payload.get("period_ema", {}))
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

    # Causal EMA over the window, then detrend: the cycle survives, the tick
    # noise that used to generate all the crossings does not.
    alpha = 2.0 / (SMOOTH_SPAN + 1.0)
    smoothed = np.empty(n, dtype=np.float64)
    smoothed[0] = window[0]
    for i in range(1, n):
        smoothed[i] = alpha * window[i] + (1.0 - alpha) * smoothed[i - 1]

    detrended = detrend(smoothed)
    # Hysteresis: a crossing only counts once the series has travelled through
    # +-35 % of its own range to the other side.  Without it the tiny jitter
    # around zero produced four "crossings" (with a 46 % regularity score) that
    # then damped the real cycle away.
    band = 0.35 * float(np.std(detrended))
    state_sign = 1 if float(detrended[0]) >= 0 else -1
    crossings = 0
    changes: list[int] = []
    for i, value in enumerate(detrended):
        if state_sign > 0 and value <= -band:
            state_sign = -1
            crossings += 1
            changes.append(i - 1 if i else i)
        elif state_sign < 0 and value >= band:
            state_sign = 1
            crossings += 1
            changes.append(i - 1 if i else i)
    changes_arr = np.asarray(changes, dtype=np.int64)
    signs = np.sign(detrended)
    state.last_crossings = crossings
    trace(ctx, "zero crossings", crossings, "hysteresis crossings in the smoothed window")

    if crossings == 0:
        trace(ctx, "phase", 0.0, "no cycle in this window - pure trend, no phase")
        return 0.0

    # The dominant period comes from the *spacing* of the crossings (a run of
    # evenly spaced crossings is a real cycle; irregular spacing is noise).
    intervals = np.diff(changes_arr).astype(np.float64)
    median_gap = float(np.median(intervals)) if intervals.size else float(n)
    period = 2.0 * median_gap
    # A cycle's period does not change from window to window: track it with an
    # EMA.  An un-smoothed period makes the phase estimate jump by a large
    # fraction of a cycle every window even though the cycle itself is stable.
    state.period_ema.update(period)
    if state.period_ema.ready:
        period = float(state.period_ema.value)
    regularity = 1.0
    if intervals.size >= 2:
        spread = float(np.std(intervals)) / (float(np.mean(intervals)) + EPS)
        regularity = max(0.0, min(1.0, 1.0 - spread))
    state.last_period = period
    state.last_regularity = regularity
    if period <= EPS:
        return 0.0

    # The last crossing index is noisy by a tick or two; the *phase* of the
    # dominant cycle is not.  Fit the crossing-derived period to the detrended
    # window with a single DFT bin, x(t) = A * cos(2*pi*t/P + psi), and read the
    # phase at "now" from that.  The same tape then gives the same phase every
    # window instead of jumping with whichever crossing the noise moved last.
    idx = np.arange(n, dtype=np.float64)
    window_fn = 0.5 - 0.5 * np.cos(2.0 * np.pi * idx / max(1.0, n - 1.0))
    kernel = window_fn * np.exp(-2j * np.pi * idx / period)
    coefficient = complex(np.sum(detrended * kernel))
    amplitude = 2.0 * abs(coefficient) / n
    psi = float(np.angle(coefficient))
    theta = 2.0 * np.pi * (n - 1) / period + psi
    m = theta % (2.0 * np.pi)

    # x(t) = A cos(phi) crosses zero at phi = pi/2 (downward) and 3*pi/2
    # (upward), so the distance from the last crossing decides the fraction of a
    # period we are past it - and the crossing we came through decides the sign.
    since_up = (m - 1.5 * np.pi) % (2.0 * np.pi)
    if since_up < np.pi:
        frac, up_sign = since_up / (2.0 * np.pi), 1.0
        last_crossing = int(round((n - 1) - frac * period))
    else:
        since_down = (m - 0.5 * np.pi) % (2.0 * np.pi)
        frac, up_sign = since_down / (2.0 * np.pi), -1.0
        last_crossing = int(round((n - 1) - frac * period))
    phi = 2.0 * np.pi * frac
    state.last_phase = float(phi)

    # Two independent confidences: how much of the detrended variance the fitted
    # cycle explains (a random walk has no dominant frequency) and how evenly
    # spaced the crossings were.
    coherence = amplitude / (float(np.std(detrended)) + EPS)
    damping = min(1.0, crossings / 3.0) * regularity * min(1.0, coherence / 1.5)

    trace(ctx, "period", period, "ticks")
    trace(ctx, "cycle amplitude", amplitude, "price units")
    trace(ctx, "last crossing index", last_crossing, "index into the window")
    trace(ctx, "fraction of a period since the crossing", frac, "0 = just crossed")
    trace(ctx, "crossing direction", up_sign, "+1 crossed upward / -1 downward")
    trace(ctx, "crossing regularity", regularity, "1.0 = evenly spaced crossings")
    trace(ctx, "cycle coherence", coherence, "amplitude / spread of the window")
    trace(ctx, "damping", damping, "crossings x regularity x coherence")

    # Rising out of an up-crossing is the strongest part of the cycle; the peak
    # (half a period after it) is the sell point, which is why the cosine - not
    # the sine - carries the position.
    raw = float(np.cos(phi)) * up_sign
    return finite(raw * damping)
