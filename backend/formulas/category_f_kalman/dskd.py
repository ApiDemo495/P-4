"""FORMULA 18 - Dual-State Kalman Divergence (DSKD).

Two one-dimensional random-walk Kalman filters run on the same tick stream: a
"fast" filter with high process noise that tracks every wiggle, and a "slow"
filter that tracks the underlying level.  When fast sits above slow the move is
real; when they converge the move is over.

v1.0's AKSF used an adaptive Q/R scheme that was fragile.  DSKD keeps the
classic recursion exactly as specified:

    p_pred = p_prev                    (random-walk model)
    P_pred = P_prev + q
    K      = P_pred / (P_pred + R)
    p_new  = p_pred + K * (z - p_pred)
    P_new  = (1 - K) * P_pred

    q_fast = 1.0 (BTC) / 2.0 (PAXG),  q_slow = 0.01,  R = 0.1

Because the filters are updated **per tick** and keep their state between
cycles, this formula is O(1) per tick and never recomputes history.

Brain mapping: campaniform sensilla - two sensitivity levels of the same
proprioceptive sensor.

Latency budget: < 0.01 ms per tick.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, RollingWindow, finite, tanh, trace

NAME = "DSKD"
CATEGORY = "F"
TITLE = "Dual-State Kalman Divergence"
BRAIN_NODE = "Campaniform sensilla"
DIRECTIONAL = True
LATENCY_MS = 0.01
DESCRIPTION = "Fast vs. slow Kalman level estimate - is the move real or noise?"

Q_SLOW = 0.01
R_MEASUREMENT = 0.1
SPREAD_WINDOW = 300
WARMUP_TICKS = 20

#: Both filters run on *relative* prices (price / reference) so the q / R
#: constants mean the same thing on a $68 000 tape and a $2 400 tape.  With
#: absolute prices the gains were ~0.95 and ~0.91 for both filters: they both
#: just copied the latest tick, so the "divergence" was the last tick's noise.
PRICE_SCALE = 1e-8
"""q / R are squared relative moves: 1e-8 = (1 bp)^2."""
DIVERGENCE_FLOOR = 3e-5
"""0.3 bps: a divergence smaller than this is not a level shift."""


class State:
    """Persistent Kalman state.  Fed incrementally from the frozen snapshot."""

    __slots__ = (
        "p_fast",
        "p_slow",
        "P_fast",
        "P_slow",
        "initialised",
        "last_ts",
        "processed",
        "spread_window",
        "last_divergence",
        "reference",
        "div_baseline",
    )

    def __init__(self) -> None:
        self.p_fast = 0.0
        self.p_slow = 0.0
        self.P_fast = 1.0
        self.P_slow = 1.0
        self.initialised = False
        self.last_ts = 0.0
        self.processed = 0
        self.spread_window = RollingWindow(SPREAD_WINDOW)
        self.last_divergence = 0.0
        self.reference = 0.0  # price that the relative series is normalised to
        self.div_baseline = None  # slow EMA of the divergence (one window long)

    def to_dict(self) -> dict:
        return {
            "p_fast": self.p_fast,
            "p_slow": self.p_slow,
            "P_fast": self.P_fast,
            "P_slow": self.P_slow,
            "initialised": self.initialised,
            "last_ts": self.last_ts,
            "processed": self.processed,
            "last_divergence": self.last_divergence,
            "reference": self.reference,
            "div_baseline": self.div_baseline,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.p_fast = float(payload.get("p_fast", 0.0))
        obj.p_slow = float(payload.get("p_slow", 0.0))
        obj.P_fast = float(payload.get("P_fast", 1.0))
        obj.P_slow = float(payload.get("P_slow", 1.0))
        obj.initialised = bool(payload.get("initialised", False))
        obj.last_ts = float(payload.get("last_ts", 0.0))
        obj.processed = int(payload.get("processed", 0))
        obj.last_divergence = float(payload.get("last_divergence", 0.0))
        obj.reference = float(payload.get("reference", 0.0))
        baseline = payload.get("div_baseline")
        obj.div_baseline = None if baseline is None else float(baseline)
        return obj


def _kalman_update(estimate: float, variance: float, q: float, z: float) -> tuple[float, float]:
    p_pred = variance + q
    k = p_pred / (p_pred + R_MEASUREMENT)
    estimate = estimate + k * (z - estimate)
    variance = (1.0 - k) * p_pred
    return estimate, variance


def advance(state: State, ticks: np.ndarray, q_fast: float) -> None:
    """Feed every not-yet-processed tick through both filters (relative space)."""
    if ticks is None or ticks.size == 0:
        return

    times = ticks[:, 0]
    if state.last_ts > 0:
        fresh = ticks[times > state.last_ts]
    else:
        fresh = ticks[-min(ticks.shape[0], 200) :]

    if fresh.size == 0:
        return

    prices = fresh[:, 1]
    if prices.size == 0:
        return
    if state.reference <= 0:
        state.reference = float(prices[0]) or 1.0

    q_fast_rel = q_fast * PRICE_SCALE
    q_slow_rel = Q_SLOW * PRICE_SCALE
    # The measurement noise is one tick of tape noise (~0.5 bps of the price).
    r_rel = 0.25 * PRICE_SCALE

    if not state.initialised:
        state.p_fast = state.p_slow = float(prices[0]) / state.reference
        state.initialised = True

    baseline_alpha = 1.0 / max(1.0, float(ticks.shape[0]))
    for z in prices:
        z_rel = float(z) / state.reference
        state.p_fast, state.P_fast = _kalman_update(state.p_fast, state.P_fast, q_fast_rel, z_rel)
        state.p_slow, state.P_slow = _kalman_update(state.p_slow, state.P_slow, q_slow_rel, z_rel)
        state.processed += 1
        if state.div_baseline is None:
            state.div_baseline = state.p_fast - state.p_slow
        else:
            state.div_baseline += baseline_alpha * (
                (state.p_fast - state.p_slow) - state.div_baseline
            )
        if state.processed > WARMUP_TICKS:
            state.spread_window.push(state.p_fast - state.p_slow)

    state.last_ts = float(times[-1])


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    q_fast = float(params.get("dskd_q_fast", 1.0))
    advance(state, snapshot.ticks(asset), q_fast)

    if not state.initialised or state.spread_window.count < 10:
        return 0.0

    divergence = state.p_fast - state.p_slow          # relative to the reference
    state.last_divergence = divergence
    baseline = state.div_baseline if state.div_baseline is not None else divergence
    fresh = divergence - baseline

    # The scale to beat is the tape's own realised move over this window: a
    # divergence smaller than three quarters of it is indistinguishable from the
    # market's ordinary travel.
    window_ticks = snapshot.ticks(asset)
    window_prices = np.maximum(np.asarray(window_ticks[:, 1], dtype=np.float64), EPS)
    if window_prices.size > 2:
        per_tick = np.diff(np.log(window_prices))
        realised = float(np.std(per_tick)) * float(np.sqrt(per_tick.size))
    else:
        realised = 0.0
    sigma = state.spread_window.std()
    scale = max(sigma, DIVERGENCE_FLOOR, 0.75 * realised)
    trace(ctx, "reference price", state.reference, "the price the series is normalised to")
    trace(ctx, "fast estimate", state.p_fast, "relative")
    trace(ctx, "slow estimate", state.p_slow, "relative")
    trace(ctx, "divergence", divergence, "relative (1e-5 = 1 bp)")
    trace(ctx, "divergence baseline (this window)", baseline, "relative")
    trace(ctx, "fresh divergence (now - baseline)", fresh, "relative - this is what is scored")
    trace(ctx, "realised window move", realised, "relative (sigma x sqrt(n))")
    trace(ctx, "running sigma", sigma, "relative")
    trace(ctx, "scale used", scale, "relative (floor = 0.75 x realised move)")
    return finite(tanh(fresh / (scale + EPS)))
