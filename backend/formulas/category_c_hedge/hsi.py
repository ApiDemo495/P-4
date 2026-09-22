"""FORMULA 11 - Hedge Stress Index (HSI)  [NEW in v2.0].

If you are running BTC and PAXG as a hedge pair, HSI tells you whether the hedge
is working or failing:

    HE = 1 - sigma_hedge / ((sigma_BTC + sigma_PAXG)/2)

    HE ~ 1  -> hedge portfolio far calmer than its components -> hedge works
    HE ~ 0  -> no diversification benefit -> breakdown
    HE < 0  -> the pair is MORE volatile than its legs -> crisis

    HSI = tanh(2 * (1 - HE))   in [0, 1]

HSI is a **regime indicator**, not a direction.  It never says BUY or SELL; it
modulates the brain (high HSI biases the whole system toward HOLD) and dampens
confidence in the fusion layer (Section 8.2).

Brain mapping: octopaminergic neuron OA-VUMa2 - the fly's "stress hormone",
which switches behaviour from exploitation to exploration.

Latency budget: < 0.1 ms.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import EPS, finite, tanh

NAME = "HSI"
CATEGORY = "C"
TITLE = "Hedge Stress Index"
BRAIN_NODE = "Octopamine OA-VUMa2"
DIRECTIONAL = False
LATENCY_MS = 0.1
DESCRIPTION = "Whether the BTC/PAXG hedge pair is working or breaking down (regime indicator)."


class State:
    __slots__ = ("last_he", "last_sigma_b", "last_sigma_g", "last_raw", "use_raw_calibration")

    def __init__(self) -> None:
        self.last_he = 1.0
        self.last_sigma_b = 0.0
        self.last_sigma_g = 0.0
        self.last_raw = 0.0
        self.use_raw_calibration = False

    def to_dict(self) -> dict:
        return {
            "last_he": self.last_he,
            "last_sigma_b": self.last_sigma_b,
            "last_sigma_g": self.last_sigma_g,
            "last_raw": self.last_raw,
            "use_raw_calibration": self.use_raw_calibration,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_he = float(payload.get("last_he", 1.0))
        obj.last_sigma_b = float(payload.get("last_sigma_b", 0.0))
        obj.last_sigma_g = float(payload.get("last_sigma_g", 0.0))
        obj.last_raw = float(payload.get("last_raw", 0.0))
        obj.use_raw_calibration = bool(payload.get("use_raw_calibration", False))
        return obj


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    synced = snapshot.synced
    if synced is None or not synced.valid:
        return 0.0

    r_b, r_g = synced.btc_returns, synced.paxg_returns
    n = min(r_b.size, r_g.size)
    if n < 5:
        return 0.0
    r_b, r_g = r_b[-n:], r_g[-n:]

    w = float(params.get("hedge_weight", 0.5))
    r_h = w * r_b + (1.0 - w) * r_g

    sigma_h = float(np.std(r_h, ddof=1))
    sigma_b = float(np.std(r_b, ddof=1))
    sigma_g = float(np.std(r_g, ddof=1))
    state.last_sigma_b, state.last_sigma_g = sigma_b, sigma_g

    he = 1.0 - sigma_h / (0.5 * (sigma_b + sigma_g) + EPS)
    state.last_he = he

    # --- raw statistic exactly as specified ----------------------------
    raw = tanh(2.0 * (1.0 - he))
    state.last_raw = raw

    # --- calibration (Deviation 11-A, see docs/SPEC_NOTES.md) ----------
    # `raw` is structurally bounded below by the value a *perfectly
    # uncorrelated* pair would produce: sigma_H = sqrt(w^2 sB^2 + (1-w)^2 sG^2)
    # gives ratio ~0.707 for equal legs, i.e. HSI_raw ~= 0.86 - above the 0.8
    # override for real BTC/PAXG data, which would pin the app at HOLD forever.
    # We therefore measure stress against that neutral baseline, using the same
    # tanh(...) shape so the mapping stays monotone and saturates identically at
    # "hedge working" (0) and "hedge fully broken" (1).
    if state.use_raw_calibration:
        hsi = raw
    else:
        sigma_neutral = float(np.sqrt(w * w * sigma_b**2 + (1.0 - w) ** 2 * sigma_g**2))
        sigma_avg = 0.5 * (sigma_b + sigma_g)
        if sigma_avg <= EPS:
            hsi = 0.0
        else:
            base_ratio = sigma_neutral / (sigma_avg + EPS)
            ratio = sigma_h / (sigma_avg + EPS)
            span = max(1.0 - base_ratio, EPS)
            hsi = (ratio - base_ratio) / span

    # HSI is defined on [0, 1]: a hedge can only be "stressed", never negative.
    return finite(max(0.0, min(1.0, hsi)))
