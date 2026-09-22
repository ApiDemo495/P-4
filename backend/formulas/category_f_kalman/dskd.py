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

from backend.formulas._util import EPS, RollingWindow, finite, tanh

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
        return obj


def _kalman_update(estimate: float, variance: float, q: float, z: float) -> tuple[float, float]:
    p_pred = variance + q
    k = p_pred / (p_pred + R_MEASUREMENT)
    estimate = estimate + k * (z - estimate)
    variance = (1.0 - k) * p_pred
    return estimate, variance


def advance(state: State, ticks: np.ndarray, q_fast: float) -> None:
    """Feed every not-yet-processed tick through both filters."""
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
    if not state.initialised and prices.size:
        state.p_fast = state.p_slow = float(prices[0])
        state.initialised = True

    for z in prices:
        state.p_fast, state.P_fast = _kalman_update(state.p_fast, state.P_fast, q_fast, float(z))
        state.p_slow, state.P_slow = _kalman_update(state.p_slow, state.P_slow, Q_SLOW, float(z))
        state.processed += 1
        if state.processed > WARMUP_TICKS:
            state.spread_window.push(state.p_fast - state.p_slow)

    state.last_ts = float(times[-1])


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    q_fast = float(params.get("dskd_q_fast", 1.0))
    advance(state, snapshot.ticks(asset), q_fast)

    if not state.initialised or state.spread_window.count < 10:
        return 0.0

    divergence = state.p_fast - state.p_slow
    state.last_divergence = divergence

    sigma = state.spread_window.std()
    mid = snapshot.last_price(asset) or abs(state.p_slow) or 1.0
    # Express the scale in relative terms so the same q/R works for a $64 000
    # BTC tape and a $2 400 PAXG tape.
    scale = max(sigma, abs(mid) * 1e-6)
    return finite(tanh(divergence / (scale + EPS)))
