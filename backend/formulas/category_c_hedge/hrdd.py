"""FORMULA 8 - Hedge Ratio Drift Detector (HRDD)  [NEW in v2.0].

The optimal hedge ratio between BTC and PAXG is not a constant.  HRDD computes
the instantaneous OLS hedge ratio beta over the last 60 synchronised 1-second
returns and compares it with a 60-minute EMA baseline.  A *stretching* ratio
means the pair relationship is under pressure; a *breaking* ratio means the two
assets have decoupled and are trading on their own narratives.

Sign convention:
    BTC:  HRDD = -tanh(d_beta)   (beta falling = BTC decoupling from gold =
                                  independent BTC strength is a bullish tell)
    PAXG: HRDD = +tanh(d_beta)   (mirror)

Brain mapping: bilateral (contralateral) antennal lobe comparison circuit - the
two assets are like two antennae sampling the same wind.

Latency budget: < 0.1 ms.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, Ema, finite, tanh

NAME = "HRDD"
CATEGORY = "C"
TITLE = "Hedge Ratio Drift Detector"
BRAIN_NODE = "Bilateral antennal lobe"
DIRECTIONAL = True
LATENCY_MS = 0.1
DESCRIPTION = "Drift of the instantaneous BTC/PAXG hedge ratio vs. its 60-minute baseline."

BETA_HISTORY = 240  # ~4 hours of 1-minute betas: enough samples for a stable std()


class State:
    __slots__ = ("beta_baseline", "beta_history")

    def __init__(self) -> None:
        self.beta_baseline = Ema(span=60)
        self.beta_history: list[float] = []

    def to_dict(self) -> dict:
        return {
            "beta_baseline": self.beta_baseline.to_dict(),
            "beta_history": list(self.beta_history[-BETA_HISTORY:]),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.beta_baseline = Ema.from_dict(payload.get("beta_baseline", {}))
        obj.beta_history = [float(x) for x in payload.get("beta_history", [])][-BETA_HISTORY:]
        return obj


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    synced = snapshot.synced
    if synced is None or not synced.valid:
        return 0.0
    r_b = synced.btc_returns
    r_g = synced.paxg_returns
    n = min(r_b.size, r_g.size)
    if n < 10:
        return 0.0
    r_b, r_g = r_b[-n:], r_g[-n:]

    b_centered = r_b - r_b.mean()
    denom = float(np.dot(b_centered, b_centered))
    if denom <= EPS:
        return 0.0
    g_centered = r_g - r_g.mean()
    beta_w = float(np.dot(b_centered, g_centered) / denom)

    baseline = state.beta_baseline.value
    if not state.beta_baseline.ready:
        # Warm the baseline before allowing a drift score to be produced.
        state.beta_baseline.update(beta_w)
        state.beta_history.append(beta_w)
        return 0.0

    # Scale by the *prior* dispersion of beta so the current observation does
    # not shrink its own z-score.
    history = np.asarray(state.beta_history[-BETA_HISTORY:], dtype=np.float64)
    hist_std = float(np.std(history, ddof=1)) if history.size > 2 else 0.0
    scale = max(hist_std, abs(beta_w) * 0.25, 1e-6)

    d_beta = (beta_w - baseline) / (scale + EPS)
    score = tanh(d_beta)

    state.beta_baseline.update(beta_w)
    state.beta_history.append(beta_w)
    if len(state.beta_history) > BETA_HISTORY:
        del state.beta_history[:-BETA_HISTORY]

    return finite(-score if asset.upper() == "BTC" else score)
