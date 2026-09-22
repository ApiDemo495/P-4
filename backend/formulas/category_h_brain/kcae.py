"""FORMULA 21 - Kenyon Cell Activation Entropy (KCAE).

Measures the sparsity of the mushroom-body sparse code.  If a handful of Kenyon
Cells carry almost all of the energy, the brain has a crisp, confident
representation of the current market pattern.  If the energy is spread thinly
across all 50 KCs, the brain is confused.

    p_j   = |a_j|^2 / SUM |a_k|^2
    H     = -SUM p_j ln(p_j)
    KCAE  = 1 - H / ln(50)          in [0, 1]

KCAE is **not directional** - it is the confidence term that multiplies the
CCSv2 output.

Brain mapping: none needed; this is a direct read-out of the brain's own state.

Latency budget: < 0.02 ms over 50 nodes.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite

NAME = "KCAE"
CATEGORY = "H"
TITLE = "Kenyon Cell Activation Entropy"
BRAIN_NODE = "Kenyon Cells (native)"
DIRECTIONAL = False
LATENCY_MS = 0.02
DESCRIPTION = "Sparsity of the Kenyon Cell code - the brain's confidence metric."

KC_COUNT = 50


class State:
    __slots__ = ("last_entropy", "last_kcae")

    def __init__(self) -> None:
        self.last_entropy = 0.0
        self.last_kcae = 0.0

    def to_dict(self) -> dict:
        return {"last_entropy": self.last_entropy, "last_kcae": self.last_kcae}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_entropy = float(payload.get("last_entropy", 0.0))
        obj.last_kcae = float(payload.get("last_kcae", 0.0))
        return obj


def from_activations(activations: np.ndarray, state: State | None = None) -> float:
    """KCAE for a Kenyon Cell activation vector."""
    a = np.asarray(activations, dtype=np.float64).ravel()
    if a.size == 0:
        return 0.0

    energy = a * a
    total = float(np.sum(energy))
    if total <= EPS:
        if state is not None:
            state.last_entropy = 0.0
            state.last_kcae = 0.0
        return 0.0

    p = energy / total
    p = np.clip(p, EPS, 1.0)
    entropy = float(-np.sum(p * np.log(p)))
    h_max = float(np.log(a.size))
    if h_max <= EPS:
        return 0.0

    kcae = max(0.0, min(1.0, 1.0 - entropy / h_max))
    if state is not None:
        state.last_entropy = entropy
        state.last_kcae = kcae
    return finite(kcae)


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    ctx = ctx or {}
    # Formula 21's p_j = |a_j|^2 uses absolute values, i.e. it is defined on
    # magnitudes.  The *rectified* Kenyon Cell code is identically zero whenever
    # the ensemble is net-bearish (ReLU clips every negative drive), which would
    # make the brain's confidence zero for every SELL signal.  Measuring the
    # drive preserves the sign-independence the notation implies while the
    # forward pass still uses the sparsified rectified code for propagation.
    values = ctx.get("_kc_drive")
    if values is None:
        values = ctx.get("_kc_activations")
    if values is None:
        return 0.0
    return from_activations(np.asarray(values, dtype=np.float64), state)
