"""Round BN - Bayesian online change-point detection on the tape.

A 70 s sign bet is a bet that the drift regime running *now* survives the
window.  Every momentum formula assumes it does; every reversal formula
assumes it does not.  Neither knows how old the regime is.  This module does:
it runs Adams & MacKay's Bayesian online change-point detector over the 5 s
returns of the frozen tape and keeps the full posterior over the **run
length** - how many steps since the drift last changed.

Model: returns inside a regime are Normal with unknown mean and precision
(Normal-Gamma conjugate prior ``mu_0, kappa_0, alpha_0, beta_0``); a regime
ends at every step with hazard ``1 / H`` (``H`` = 70 s expressed in steps, so
the prior says "regimes last about one window").  The recursion is exact::

    growth:   P(r_t = r+1) = P(r_{t-1} = r) * pred_r(x_t) * (1 - 1/H)
    change:   P(r_t = 0)   = sum_r P(r_{t-1} = r) * pred_r(x_t) * (1/H)

with ``pred_r`` the Student-t posterior predictive of the run of length r.
From the posterior the engine reads

* ``p_change_window`` - the probability the drift changed inside the last
  window (mass on run lengths shorter than one window),
* ``run_mode`` / ``run_mean`` - the most probable / expected regime age,
* ``mu_hat`` - the run-length-weighted posterior mean of the drift (bps per
  step) and ``sigma_hat`` its predictive dispersion,
* ``score`` - the posterior probability that the *next window's* return is
  positive, mapped to [-1, 1]: ``2 * P(sum of next k returns > 0) - 1``,
  where the sum over a run has mean ``k * mu_hat`` and variance
  ``k * sigma_hat^2`` under the current regime, mixed with a flat (zero-mean)
  predictive with weight ``p_change_window`` - a young regime is not trusted
  to carry the window; an old one is, until the hazard says otherwise.

One signed learned input (``feature:bocpd``) plus a regime label for the
meta-combiner's experts; everything printed in the logic lines.
"""

from __future__ import annotations

import math

import numpy as np

STEP_S = 5.0
MAX_RUN = 400
_lgamma = np.vectorize(math.lgamma, otypes=[np.float64])


def _student_t_logpdf(x: np.ndarray, mu: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """log posterior predictive of a Normal-Gamma: Student-t with 2*alpha dof."""
    nu = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)
    z2 = (x - mu) ** 2 / scale2
    return (_lgamma((nu + 1) / 2) - _lgamma(nu / 2)
            - 0.5 * np.log(nu * math.pi * scale2) - (nu + 1) / 2 * np.log1p(z2 / nu))


def bocpd(ticks: np.ndarray, window_s: float = 70.0, now_ms: float | None = None) -> dict:
    """Run the detector over the tape's 5 s returns.  Never raises."""
    out = {"covered": False, "steps": 0, "run_mode": 0, "run_mean": 0.0, "p_change_window": 0.0,
           "mu_hat": 0.0, "sigma_hat": 0.0, "p_up_window": 0.5, "score": 0.0, "regime": "unknown", "logic": []}
    try:
        if ticks is None or ticks.ndim != 2 or ticks.shape[0] < 50:
            out["logic"] = ["fewer than 50 prints - no regime to date"]
            return out
        tr = ticks[ticks[:, 2] > 0]
        tr = tr if tr.shape[0] >= 50 else ticks
        t = tr[:, 0] / 1000.0
        p = tr[:, 1]
        now = (now_ms / 1000.0) if now_ms else float(t[-1])
        grid = np.arange(now - math.floor((now - t[0]) / STEP_S) * STEP_S, now + 1e-9, STEP_S)
        if grid.size < 30:
            out["logic"] = [f"tape covers {grid.size} steps of {STEP_S:.0f} s - needs 30"]
            return out
        idx = np.searchsorted(t, grid, side="right") - 1
        idx = np.clip(idx, 0, p.size - 1)
        px = p[idx]
        x = np.diff(np.log(px)) * 1e4                      # bps per step
        n = x.size
        k_steps = max(1, int(round(window_s / STEP_S)))
        hazard = 1.0 / k_steps
        # Normal-Gamma prior: zero drift, weakly informed by the tape's own scale
        s0 = max(float(np.std(x)), 1e-3)
        mu0, kappa0, alpha0, beta0 = 0.0, 0.5, 1.5, 0.5 * s0 * s0
        mu = np.array([mu0]); kappa = np.array([kappa0]); alpha = np.array([alpha0]); beta = np.array([beta0])
        logR = np.array([0.0])                              # log P(run length)
        for xi in x:
            lp = _student_t_logpdf(xi, mu, kappa, alpha, beta)
            joint = logR + lp
            growth = joint + math.log1p(-hazard)
            change = np.logaddexp.reduce(joint) + math.log(hazard)
            logR = np.concatenate(([change], growth))
            logR -= np.logaddexp.reduce(logR)
            # sufficient-statistic updates (new run first, then the grown runs)
            kappa_n = kappa + 1.0
            mu_n = (kappa * mu + xi) / kappa_n
            alpha_n = alpha + 0.5
            beta_n = beta + kappa * (xi - mu) ** 2 / (2.0 * kappa_n)
            mu = np.concatenate(([mu0], mu_n)); kappa = np.concatenate(([kappa0], kappa_n))
            alpha = np.concatenate(([alpha0], alpha_n)); beta = np.concatenate(([beta0], beta_n))
            if logR.size > MAX_RUN:                         # truncate the tail, renormalise
                logR = logR[:MAX_RUN]; mu = mu[:MAX_RUN]; kappa = kappa[:MAX_RUN]
                alpha = alpha[:MAX_RUN]; beta = beta[:MAX_RUN]
                logR -= np.logaddexp.reduce(logR)
        R = np.exp(logR)
        runs = np.arange(R.size)
        run_mode = int(runs[np.argmax(R)])
        run_mean = float(np.dot(R, runs))
        p_change = float(R[: min(k_steps, R.size)].sum())
        # run-length-weighted drift and predictive dispersion
        var_pred = beta * (kappa + 1.0) / (alpha * kappa) * np.where(alpha > 1.0, alpha / (alpha - 1.0), 3.0)
        mu_hat = float(np.dot(R, mu))
        sigma_hat = float(math.sqrt(max(np.dot(R, var_pred + (mu - mu_hat) ** 2), 1e-9)))
        # the next window under the current regime, mixed with a flat predictive
        m = k_steps * mu_hat
        s = math.sqrt(k_steps) * sigma_hat
        p_up_regime = 0.5 * (1.0 + math.erf(m / (s * math.sqrt(2.0)))) if s > 0 else 0.5
        p_up = (1.0 - p_change) * p_up_regime + p_change * 0.5
        score = 2.0 * p_up - 1.0
        age_s = run_mean * STEP_S
        regime = ("fresh" if p_change > 0.6 else "young" if age_s < 2 * window_s else "mature" if age_s < 6 * window_s else "old")
        out.update({"covered": True, "steps": int(n), "run_mode": run_mode, "run_mean": round(run_mean, 2),
                    "p_change_window": round(p_change, 4), "mu_hat": round(mu_hat, 4), "sigma_hat": round(sigma_hat, 3),
                    "p_up_window": round(p_up, 4), "score": round(score, 4), "regime": regime,
                    "run_age_s": round(age_s, 1)})
        out["logic"] = [
            f"{n} steps of {STEP_S:.0f} s; hazard 1/{k_steps} per step (regimes expected to last one window)",
            f"posterior run length: mode {run_mode} steps, mean {run_mean:.1f} steps = {age_s:.0f} s -> regime '{regime}'",
            f"P(drift changed inside the last {window_s:.0f} s) = {p_change:.1%}",
            f"regime drift mu_hat = {mu_hat:+.3f} bp/step, predictive sigma = {sigma_hat:.2f} bp/step -> next-window mean {m:+.2f} bp, sd {s:.2f} bp",
            f"P(next window up) = (1 - {p_change:.2f}) x {p_up_regime:.1%} + {p_change:.2f} x 50% = {p_up:.1%} -> score {score:+.3f}",
        ]
    except Exception as exc:  # noqa: BLE001
        out["logic"] = [f"bocpd abstained: {exc}"]
    return out
