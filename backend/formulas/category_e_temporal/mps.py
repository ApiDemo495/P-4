"""FORMULA 16 - Momentum Persistence Score (MPS).

Are returns autocorrelated?  Positive lag-1/2/3 autocorrelation means the tape
has memory and momentum is likely to continue; negative autocorrelation means it
mean-reverts and the current move should be faded.  Lags are weighted 3:2:1 so
the most recent structure dominates.

    MPS = tanh(3 * (3*rho1 + 2*rho2 + rho3) / 6)

Brain mapping: ORN class Or47b - a persistent odour tracker, i.e. does the scent
trail still lead somewhere.

Latency budget: < 0.15 ms.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite, tanh

NAME = "MPS"
CATEGORY = "E"
TITLE = "Momentum Persistence Score"
BRAIN_NODE = "ORN Or47b"
DIRECTIONAL = True
LATENCY_MS = 0.15
DESCRIPTION = "Weighted lag-1/2/3 autocorrelation of recent returns (persist vs. fade)."

WINDOW = 60
MAX_LAG = 3
GAIN = 3.0

#: Rank-1, 2, 3 autocorrelations matter in a 3:2:1 ratio.
LAG_WEIGHTS = (3.0, 2.0, 1.0)


class State:
    __slots__ = ("last_rho", "last_mps")

    def __init__(self) -> None:
        self.last_rho = (0.0, 0.0, 0.0)
        self.last_mps = 0.0

    def to_dict(self) -> dict:
        return {"last_rho": list(self.last_rho), "last_mps": self.last_mps}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        rho = payload.get("last_rho", [0.0, 0.0, 0.0])
        obj.last_rho = tuple(float(x) for x in (list(rho) + [0.0, 0.0, 0.0])[:3])
        obj.last_mps = float(payload.get("last_mps", 0.0))
        return obj


def autocorrelation(returns: np.ndarray, lag: int) -> float:
    n = returns.size
    if n <= lag + 1:
        return 0.0
    mean = float(returns.mean())
    dev = returns - mean
    denom = float(np.dot(dev, dev))
    if denom <= EPS:
        return 0.0
    return float(np.dot(dev[:-lag], dev[lag:]) / denom)


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    prices = snapshot.prices(asset)
    if prices.size < 12:
        return 0.0
    window = prices[-min(WINDOW, prices.size) :]
    returns = np.diff(np.log(np.maximum(window, EPS)))
    if returns.size < MAX_LAG + 2:
        return 0.0

    rhos = tuple(autocorrelation(returns, lag) for lag in (1, 2, 3))
    state.last_rho = rhos

    weighted = sum(w * r for w, r in zip(LAG_WEIGHTS, rhos)) / sum(LAG_WEIGHTS)
    score = tanh(GAIN * weighted)
    state.last_mps = score
    return finite(score)
