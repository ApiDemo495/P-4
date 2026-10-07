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

from backend.formulas._util import EPS, Ema, finite, tanh, trace

NAME = "HRDD"
CATEGORY = "C"
TITLE = "Hedge Ratio Drift Detector"
BRAIN_NODE = "Bilateral antennal lobe"
DIRECTIONAL = True
LATENCY_MS = 0.1
DESCRIPTION = "Drift of the instantaneous BTC/PAXG hedge ratio vs. its 60-minute baseline."

BETA_HISTORY = 240  # ~4 hours of 1-minute betas: enough samples for a stable std()
#: How many standard errors of the instantaneous beta estimate a change has
#: to clear before it counts as a drift at all.
NOISE_SIGMA = 3.0


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
    if synced is not None and synced.valid and min(synced.btc_returns.size, synced.paxg_returns.size) >= 10:
        r_b = synced.btc_returns
        r_g = synced.paxg_returns
        trace(ctx, "pair basis", float(min(r_b.size, r_g.size)), "tick-grid returns")
    else:
        # Round AP: a thin PAXG tape must not silence the hedge-ratio drift.
        # Fall back to the last 120 one-minute closes of both legs (seeded
        # from public history at start) - a coarser clock, same regression.
        try:
            cb = np.asarray(snapshot.candles("BTC"), dtype=np.float64)
            cg = np.asarray(snapshot.candles("PAXG"), dtype=np.float64)
        except Exception:  # noqa: BLE001
            trace(ctx, "pair basis", 0.0, "no tick grid, no candles")
            return 0.0
        m = min(cb.size, cg.size, 121)
        if m < 11:
            trace(ctx, "pair basis", float(m), "minute closes - need 11 on both legs")
            return 0.0
        cb, cg = cb[-m:], cg[-m:]
        r_b = np.diff(cb) / np.maximum(cb[:-1], EPS)
        r_g = np.diff(cg) / np.maximum(cg[:-1], EPS)
        trace(ctx, "pair basis", float(r_b.size), "one-minute candle returns (tick grid thin)")
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

    # The instantaneous OLS slope has a standard error of its own, and on a tape
    # where the legs are unrelated that error *is* the whole wiggle: dividing by
    # the historical dispersion (v2.0.0) turned pure estimation noise into
    # +/-0.73 readings on the balanced tape.  A change is only a drift when it
    # clears the 3-sigma band of the estimator; the excess is then scored
    # against the pair's own typical beta move.
    n_eff = float(max(2, n))
    sigma_b = float(np.sqrt(float(np.dot(b_centered, b_centered)) / n_eff))
    sigma_g = float(np.sqrt(float(np.dot(g_centered, g_centered)) / n_eff))
    if sigma_b <= EPS:
        return 0.0
    se_beta = sigma_g / (sigma_b * float(np.sqrt(n_eff)) + EPS)

    # Soft signal-to-noise gate: the score fades to zero as the move
    # approaches the estimator's own noise (1 sigma) and reaches full strength
    # at NOISE_SIGMA.  A hard dead zone (v2.0.1) made the formula read exactly
    # 0.000 on every live window, which looks broken rather than calm.
    d_beta = beta_w - baseline
    scale = max(hist_std, se_beta, 1e-6)
    snr = abs(d_beta) / (se_beta + EPS)
    gate = float(np.clip((snr - 1.0) / max(1e-6, NOISE_SIGMA - 1.0), 0.0, 1.0))
    score = tanh(d_beta / (scale + EPS)) * gate
    trace(ctx, "beta this window", beta_w, "PAXG on BTC OLS slope")
    trace(ctx, "beta baseline", baseline, "running mean")
    trace(ctx, "beta deviation", d_beta, "window - baseline")
    trace(ctx, "scale used", scale, "max(history sigma, estimator SE)")
    trace(ctx, "signal-to-noise", snr, "|deviation| / estimator SE")
    trace(ctx, "noise gate", gate, "0 at 1 sigma, 1 at 3 sigma")

    state.beta_baseline.update(beta_w)
    state.beta_history.append(beta_w)
    if len(state.beta_history) > BETA_HISTORY:
        del state.beta_history[:-BETA_HISTORY]

    return finite(-score if asset.upper() == "BTC" else score)


DOUBLE_CHECK = "value = -tanh(beta deviation / scale) x noise gate for BTC (+ for PAXG)"


def double_check(t: dict, asset: str) -> float:
    """Independent re-derivation of the output from the traced intermediates."""
    if "beta deviation" not in t:
        return 0.0
    score = tanh(float(t["beta deviation"]) / (float(t["scale used"]) + EPS)) * float(t["noise gate"])
    return -score if asset.upper() == "BTC" else score
