"""FORMULA 14 - Entropy Regime Classifier (ERC).

Sample entropy (SampEn) is a bias-corrected alternative to Shannon entropy: it
counts how often a length-m template recurs *and* continues to length m+1.  Low
SampEn means the tape is repeating itself (trending); high SampEn means it is
irregular (choppy).

ERC is a **pure classifier** - it modulates the other signals and never predicts
direction (that was v1.0's mistake with EWVC).

    ERC = tanh(SampEn - 1.0)
    SampEn 0   -> -0.76 (trending)
    SampEn 1   ->  0.00 (transitional)
    SampEn 2   -> +0.76 (choppy)

The O(N^2) match counting runs in Cython when the extension is built and falls
back to a vectorised numpy path otherwise.

Brain mapping: serotonergic neuron (CSD) - sets global gain / behavioural state.

Latency budget: < 0.5 ms.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite, tanh

NAME = "ERC"
CATEGORY = "D"
TITLE = "Entropy Regime Classifier"
BRAIN_NODE = "Serotonin CSD"
DIRECTIONAL = False
LATENCY_MS = 0.5
DESCRIPTION = "Sample-entropy regime classifier (trending / transitional / choppy)."

WINDOW = 60
TEMPLATE_LENGTH = 2
FALLBACK_SAMPEN = 2.5
"""Returned when there are too few matches to estimate entropy: treat as choppy."""

# --- Cython acceleration, with a transparent numpy fallback ------------------
try:  # pragma: no cover - depends on whether the extension was built
    from backend.formulas.category_d_regime import erc_ext as _cy

    BACKEND = "cython"
except Exception:  # noqa: BLE001
    _cy = None
    BACKEND = "numpy"


def _count_matches_numpy(x: np.ndarray, m: int, r: float) -> int:
    n = x.size - m + 1
    if n < 2:
        return 0
    # (n, n, m) Chebyshev distances; n <= 60 so this is ~7k doubles.
    windows = np.lib.stride_tricks.sliding_window_view(x, m)
    diff = np.abs(windows[:, None, :] - windows[None, :, :]).max(axis=2)
    mask = diff < r
    iu = np.triu_indices(n, k=1)
    return int(mask[iu].sum())


def count_matches(x: np.ndarray, m: int, r: float) -> int:
    if _cy is not None:
        return int(_cy.count_matches(np.ascontiguousarray(x, dtype=np.float64), m, r))
    return _count_matches_numpy(x, m, r)


def sample_entropy(x: np.ndarray, m: int = TEMPLATE_LENGTH, r: float | None = None) -> float:
    """SampEn(m, r, N) for a 1-D series."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < m + 2:
        return 0.0
    if r is None:
        r = 0.2 * float(np.std(x, ddof=1))
    if r <= EPS:
        return 0.0  # perfectly flat series -> maximally predictable

    if _cy is not None:
        return float(_cy.sample_entropy(np.ascontiguousarray(x), m, float(r), FALLBACK_SAMPEN))

    b = _count_matches_numpy(x, m, float(r))
    a = _count_matches_numpy(x, m + 1, float(r))
    if b == 0 or a == 0:
        return FALLBACK_SAMPEN
    return float(-np.log(a / b))


class State:
    __slots__ = ("last_sampen", "sampen_history")

    def __init__(self) -> None:
        self.last_sampen = 1.0
        self.sampen_history: list[float] = []

    def to_dict(self) -> dict:
        return {"last_sampen": self.last_sampen, "sampen_history": list(self.sampen_history[-32:])}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_sampen = float(payload.get("last_sampen", 1.0))
        obj.sampen_history = [float(v) for v in payload.get("sampen_history", [])][-32:]
        return obj


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    prices = snapshot.prices(asset)
    if prices.size < 12:
        return 0.0
    window = prices[-min(WINDOW, prices.size) :]

    r_mult = float(params.get("erc_tolerance_mult", 0.2))
    tolerance = r_mult * float(np.std(window, ddof=1))
    sampen = sample_entropy(window, TEMPLATE_LENGTH, tolerance)

    state.last_sampen = sampen
    state.sampen_history.append(sampen)
    if len(state.sampen_history) > 32:
        del state.sampen_history[:-32]

    return finite(tanh(sampen - 1.0))
