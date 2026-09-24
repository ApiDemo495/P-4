"""FORMULA 12 - Regime Switch Velocity (RSV).

Most regime indicators tell you which regime you are *in*.  RSV tells you how
fast you are *switching between* them, which fires earlier and is therefore more
actionable.

Each 600-tick window is measured with detrended fluctuation analysis (DFA):
H < 0.5 is a mean-reverting tape, H > 0.5 is a trending one.  RSV then takes the
slope of those window-by-window estimates over the last 30 windows:

    velocity = OLS slope of the DFA-Hurst series (per window)
    RSV      = tanh(velocity / scale) * sign(drift of the current window)

A *single* window's rescaled-range estimate is far too noisy to difference
(20-tick R/S swings by +/-0.2 and sits at H ~ 0.72 on every tape: v2.0.0 read
random numbers from that).  DFA over 600 returns with scales 4..64 has a
standard error near 0.02, so the slope is a real quantity.  The sign says which
way the market is switching: Hurst rising with an up-drift is a trend forming,
Hurst falling with an up-drift is a rally losing structure.

Brain mapping: circadian clock neuron DN1p - detects transitions between
behavioural states.

Latency budget: < 0.3 ms (DFA over five scales on 600 returns).
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, SelfScale, finite, tanh, trace

NAME = "RSV"
CATEGORY = "D"
TITLE = "Regime Switch Velocity"
BRAIN_NODE = "Clock DN1p"
DIRECTIONAL = True
LATENCY_MS = 0.3
DESCRIPTION = "Velocity of the Hurst exponent - how fast the market is switching regimes."

WINDOW = 600
DFA_SCALES = (4, 8, 16, 32, 64)
HURST_HISTORY = 48
#: A regime switch has to show up as a *trend* in the estimate, so the velocity
#: is regressed over 30 window estimates rather than one window-to-window jump.
LOOKBACK = 30
#: A dozen estimates before the slope means anything: over four or five noisy
#: windows the regression slope is itself a random number.
MIN_POINTS = 12
GAIN = 1.0
#: A Hurst change of 0.005 per window sustained over half an hour is a real
#: regime shift; the adaptive term widens the bar in markets that are already
#: changing constantly.
MEANINGFUL_DELTA_H = 0.005
#: The velocity has no direction of its own - the sign comes from the drift of
#: the scored window, and only when that drift is statistically visible.
MIN_T_STAT = 1.0


class State:
    __slots__ = ("hurst_series", "last_hurst", "last_slope", "velocity_scale")

    def __init__(self) -> None:
        self.hurst_series: list[float] = []
        self.last_hurst = 0.5
        self.last_slope = 0.0
        self.velocity_scale = SelfScale(decay=0.9, floor=MEANINGFUL_DELTA_H)

    def to_dict(self) -> dict:
        return {
            "hurst_series": list(self.hurst_series[-HURST_HISTORY:]),
            "last_hurst": self.last_hurst,
            "last_slope": self.last_slope,
            "velocity_scale": self.velocity_scale.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.hurst_series = [float(x) for x in payload.get("hurst_series", [])][-HURST_HISTORY:]
        obj.last_hurst = float(payload.get("last_hurst", 0.5))
        obj.last_slope = float(payload.get("last_slope", 0.0))
        obj.velocity_scale = SelfScale.from_dict(payload.get("velocity_scale", {}))
        return obj


def dfa_hurst(returns: np.ndarray) -> float:
    """Detrended fluctuation analysis exponent of a return series.

    The profile (cumulative sum of the de-meaned returns) is split into blocks of
    each scale, every block is detrended, and the log of the residual fluctuation
    is regressed on the log of the scale.  The slope *is* the Hurst exponent, and
    unlike rescaled range it is not biased upward on short samples.
    """
    x = np.asarray(returns, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2 * min(DFA_SCALES):
        return 0.5
    profile = np.cumsum(x - float(np.mean(x)))
    logs_x: list[float] = []
    logs_y: list[float] = []
    for scale in DFA_SCALES:
        blocks = profile.size // scale
        if blocks < 2:
            continue
        segment = profile[: blocks * scale].reshape(blocks, scale)
        t = np.arange(scale, dtype=np.float64)
        tt = t - t.mean()
        slopes = (segment * tt).sum(axis=1) / float((tt**2).sum() + EPS)
        residual = segment - segment.mean(axis=1, keepdims=True) - slopes[:, None] * tt
        fluctuation = float(np.sqrt(np.mean(residual**2)))
        if fluctuation > EPS:
            logs_x.append(float(np.log(scale)))
            logs_y.append(float(np.log(fluctuation)))
    if len(logs_x) < 2:
        return 0.5
    return float(np.polyfit(np.asarray(logs_x), np.asarray(logs_y), 1)[0])


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    prices = np.asarray(snapshot.prices(asset), dtype=np.float64)
    if prices.size < 2 * min(DFA_SCALES):
        return 0.0
    window = prices[-min(WINDOW, prices.size) :]
    returns = np.diff(np.log(np.maximum(window, EPS)))
    if returns.size < 2 * min(DFA_SCALES):
        return 0.0

    hurst = dfa_hurst(returns)
    state.hurst_series.append(float(hurst))
    if len(state.hurst_series) > HURST_HISTORY:
        del state.hurst_series[:-HURST_HISTORY]
    state.last_hurst = float(hurst)

    series = np.asarray(state.hurst_series[-LOOKBACK:], dtype=np.float64)
    velocity = 0.0
    if series.size >= MIN_POINTS:
        x = np.arange(series.size, dtype=np.float64)
        velocity = float(np.polyfit(x, series, 1)[0])
    state.last_slope = velocity

    denom = max(MEANINGFUL_DELTA_H, 0.5 * state.velocity_scale.denominator(GAIN))
    state.velocity_scale.update(abs(velocity))

    drift = float(np.mean(returns))
    sigma = float(np.sqrt(np.mean(returns**2)))
    t_stat = abs(drift) * float(np.sqrt(returns.size)) / (sigma + EPS)
    direction = float(np.sign(drift)) if t_stat >= MIN_T_STAT else 0.0

    trace(ctx, "Hurst (DFA, this window)", hurst, "0..1")
    trace(ctx, "window estimates in the series", series.size, "windows")
    trace(ctx, "velocity (slope of H)", velocity, "Hurst per window")
    trace(ctx, "scale used", denom, "Hurst per window")
    trace(ctx, "drift t-statistic", t_stat, "|mean| / (sigma / sqrt(n))")
    trace(ctx, "direction used", direction, "+1 / 0 / -1")

    if direction == 0.0:
        return 0.0
    return finite(tanh(velocity / (denom + EPS)) * direction)
