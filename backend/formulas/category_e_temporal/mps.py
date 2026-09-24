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

from collections import deque

import numpy as np

from backend.formulas._util import EPS, finite, tanh, trace

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
#: Rolling horizon of tick returns for the autocorrelation (200 s at 10 ticks/s)
#: and the number of recent ticks that define the drift being projected.
HORIZON = 2000
#: The drift the persistence is projected onto.  60 ticks is ~5x noisier than
#: the trend inside it, so the sign it produced flipped at random; 300 ticks
#: (~30 s) is the shortest sample where the tape's direction is visible.
RECENT_DRIFT_TICKS = 300

#: Rank-1, 2, 3 autocorrelations matter in a 3:2:1 ratio.
LAG_WEIGHTS = (3.0, 2.0, 1.0)


class State:
    __slots__ = ("last_rho", "last_mps", "returns", "last_price", "last_ts")

    def __init__(self) -> None:
        self.last_rho = (0.0, 0.0, 0.0)
        self.last_mps = 0.0
        self.returns: deque[float] = deque(maxlen=HORIZON)
        self.last_price = 0.0  # closes the gap between two windows
        self.last_ts = 0.0  # guards against re-processing the same snapshot

    def to_dict(self) -> dict:
        return {
            "last_rho": list(self.last_rho),
            "last_mps": self.last_mps,
            # The horizon has to survive a process restart or the first windows
            # after a deploy see an empty autocorrelation sample.
            "returns": [float(x) for x in list(self.returns)[-600:]],
            "last_price": self.last_price,
            "last_ts": self.last_ts,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        rho = payload.get("last_rho", [0.0, 0.0, 0.0])
        obj.last_rho = tuple(float(x) for x in (list(rho) + [0.0, 0.0, 0.0])[:3])
        obj.last_mps = float(payload.get("last_mps", 0.0))
        obj.last_price = float(payload.get("last_price", 0.0))
        obj.last_ts = float(payload.get("last_ts", 0.0))
        for value in payload.get("returns", []):
            obj.returns.append(float(value))
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
    prices = np.asarray(snapshot.prices(asset), dtype=np.float64)
    if prices.size < 12:
        return 0.0

    # The horizon must be one continuous return series: appending only the
    # trailing 60 prices per window (v2.0.0) filled the deque with disjoint
    # snippets, and the autocorrelation of snippets has nothing to do with the
    # tape's autocorrelation.  The link return stitches windows together.
    ticks = snapshot.ticks(asset)
    last_ts = float(ticks[-1, 0]) if ticks is not None and ticks.size else 0.0
    fresh_n = 0
    if last_ts > state.last_ts:
        fresh = np.diff(np.log(np.maximum(prices, EPS)))
        if fresh.size and state.last_price > 0.0:
            link = float(np.log(max(prices[0], EPS) / max(state.last_price, EPS)))
            fresh = np.concatenate([[link], fresh])
        for value in fresh:
            state.returns.append(float(value))
        fresh_n = int(fresh.size)
        state.last_price = float(prices[-1])
        state.last_ts = last_ts

    returns = np.asarray(state.returns, dtype=np.float64)
    if returns.size < MAX_LAG + 2:
        return 0.0

    rhos = tuple(autocorrelation(returns, lag) for lag in (1, 2, 3))
    state.last_rho = rhos

    weighted = sum(w * r for w, r in zip(LAG_WEIGHTS, rhos)) / sum(LAG_WEIGHTS)
    # Only *positive* autocorrelation is trend persistence.  A mean-reverting
    # tape (rho < 0) has no trend to continue, so its persistence is 0 rather
    # than -0.87 (that sign then leaked into the output through the drift sign
    # and made a choppy tape read -0.5).
    persistence = tanh(GAIN * max(0.0, weighted))
    state.last_mps = persistence

    # Persistence is a *strength*; a projection neuron needs a direction.  The
    # signed value says "the move that is running tends to continue" (+1) or
    # "it tends to fade" (-1), which is the tradeable statement.
    recent = returns[-min(RECENT_DRIFT_TICKS, returns.size) :]
    drift = float(np.mean(recent)) if recent.size else 0.0
    drift_sign = float(np.sign(drift))

    # How much of the tape's travel is net displacement?  A trend moves in one
    # direction (high ratio); chop retraces itself (low ratio).  Persistence is
    # only worth reporting when the tape is actually going somewhere.
    travel = float(np.sum(np.abs(recent)))
    efficiency = abs(float(np.sum(recent))) / (travel + EPS)
    materiality = min(1.0, 2.0 * efficiency)
    trace(ctx, "returns in the horizon", returns.size, "ticks (200 s)")
    trace(ctx, "new returns this window", fresh_n, "ticks")
    trace(ctx, "rho1 / rho2 / rho3", np.asarray(rhos), "autocorrelation")
    trace(ctx, "weighted rho", weighted, "3:2:1 weights (0 when mean-reverting)")
    trace(ctx, "persistence strength", persistence, "tanh(3 * max(0, weighted rho))")
    trace(ctx, "recent drift", drift, "mean log return")
    trace(ctx, "drift sign", drift_sign, "direction the persistence applies to")
    trace(ctx, "efficiency ratio", efficiency, "net move / total travel")
    trace(ctx, "materiality", materiality, "min(1, 2 x efficiency)")

    if drift_sign == 0.0:
        return 0.0
    return finite(persistence * drift_sign * materiality)
