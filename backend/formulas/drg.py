"""Dopamine Reward Gradient (DRG) - the reward-learning meta-parameter.

DRG is not one of the 22 market formulas because it runs on a different
timescale: it consumes the *outcomes of previous cycles* rather than current
market data, and its output modulates the brain (the PAM/PPL1 dopamine nodes)
instead of acting as a signal of its own.

    V_R      = SUM gamma^(R-i) * m_i * o_i / SUM gamma^(R-i),   gamma = 0.95
    delta_TD = (m_R * o_R + gamma * V_R) - V_{R-1}
    DA       = delta^0.8      if delta > 0          (asymmetric: reward is
             = -|delta|^1.2   if delta < 0           discounted less than loss)
    DRG      = tanh(0.1 * DA)

Latency budget: < 0.05 ms.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite, tanh, trace

NAME = "DRG"
CATEGORY = "REWARD"
TITLE = "Dopamine Reward Gradient"
BRAIN_NODE = "PAM / PPL1"
DIRECTIONAL = False
LATENCY_MS = 0.05
DESCRIPTION = "Temporal-difference reward prediction error that gates the dopamine neurons."

OUTCOMES = 20
GAMMA = 0.95
REWARD_EXPONENT = 0.8
LOSS_EXPONENT = 1.2
GAIN = 0.1


class State:
    __slots__ = ("last_drg", "last_td_error", "last_value", "samples")

    def __init__(self) -> None:
        self.last_drg = 0.0
        self.last_td_error = 0.0
        self.last_value = 0.0
        self.samples = 0

    def to_dict(self) -> dict:
        return {
            "last_drg": self.last_drg,
            "last_td_error": self.last_td_error,
            "last_value": self.last_value,
            "samples": self.samples,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_drg = float(payload.get("last_drg", 0.0))
        obj.last_td_error = float(payload.get("last_td_error", 0.0))
        obj.last_value = float(payload.get("last_value", 0.0))
        obj.samples = int(payload.get("samples", 0))
        return obj


def discounted_value(outcomes: np.ndarray, gamma: float = GAMMA) -> float:
    """Discounted, magnitude-weighted value estimate of a slice of history."""
    if outcomes is None or outcomes.size == 0:
        return 0.0
    o = outcomes[:, 0]
    m = outcomes[:, 1]
    r = o.size
    discounts = gamma ** np.arange(r - 1, -1, -1, dtype=np.float64)
    denom = float(np.sum(discounts))
    if denom <= EPS:
        return 0.0
    return float(np.sum(discounts * m * o) / denom)


def compute(
    snapshot,
    state: State,
    params: dict | None = None,
    ctx: dict | None = None,
) -> float:
    """DRG for the current cycle.  ``snapshot.drg_outcomes`` is (R, 2)."""
    outcomes = snapshot.drg_outcomes
    if outcomes is None or outcomes.size == 0:
        state.last_drg = 0.0
        trace(ctx, "outcomes in the window", 0, "cycles")
        return 0.0

    outcomes = outcomes[-OUTCOMES:]
    r = outcomes.shape[0]

    # V_R uses the whole window; V_{R-1} drops the newest observation so that the
    # TD error measures genuine surprise rather than the value including itself.
    value_r = discounted_value(outcomes)
    value_prev = discounted_value(outcomes[:-1]) if r > 1 else 0.0
    state.last_value = value_r

    newest_m = float(outcomes[-1, 1])
    newest_o = float(outcomes[-1, 0])
    delta = (newest_m * newest_o + GAMMA * value_r) - value_prev
    state.last_td_error = delta

    if delta > 0:
        da = delta**REWARD_EXPONENT
    elif delta < 0:
        da = -(abs(delta) ** LOSS_EXPONENT)
    else:
        da = 0.0

    drg = tanh(GAIN * da)
    state.last_drg = drg
    state.samples = r
    trace(ctx, "outcomes in the window", r, "cycles")
    trace(ctx, "newest outcome (m * o)", newest_m * newest_o, "magnitude x result")
    trace(ctx, "discounted value V_R", value_r, "gamma = 0.95")
    trace(ctx, "value V_{R-1} (before the newest)", value_prev, "gamma = 0.95")
    trace(ctx, "TD error (delta)", delta, "(m*o + gamma V_R) - V_{R-1}")
    trace(ctx, "dopamine (asymmetric)", da, "delta^0.8 / -|delta|^1.2")
    trace(ctx, "reward gradient", drg, "tanh(0.1 * dopamine)")
    return finite(drg)
