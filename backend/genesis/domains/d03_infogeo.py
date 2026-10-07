"""Domain 3 - information-geometric market manifold (210 formulas).

Each window of returns is a point (μ, σ) on the Gaussian statistical manifold
with the Fisher metric g = diag(1/σ², 2/σ²).  That manifold is the hyperbolic
plane (μ/√2, σ), so the geodesic distance is exact:
    d_F = √2 · arccosh(1 + ((μ₁−μ₂)² + 2(σ₁−σ₂)²) / (4σ₁σ₂)).
"""
from __future__ import annotations

import numpy as np

from backend.genesis import ops
from backend.genesis.domains._common import finish
from backend.genesis.spec import grid, make_variants

D = 3
EST = (3, 5, 8, 13, 21, 34, 55)


def _mu_sigma(frame, w):
    return frame.roll("ret_mean", w), frame.roll("ret_std", w) + 1e-7


def fisher_distance(m1, s1, m2, s2):
    with np.errstate(all="ignore"):
        arg = 1.0 + ((m1 - m2) ** 2 + 2.0 * (s1 - s2) ** 2) / (4.0 * s1 * s2)
        return np.sqrt(2.0) * np.arccosh(np.maximum(1.0, arg))


def k_fif(frame, n, norm, w=5, **_):
    mu, sg = _mu_sigma(frame, w)
    d = fisher_distance(mu, sg, ops.shift(mu, 1), ops.shift(sg, 1))
    recent, older = ops.rmean(d, max(3, n // 4)), ops.rmean(ops.shift(d, max(3, n // 4)), n)
    with np.errstate(all="ignore"):
        fif = np.tanh(recent / (older + 1e-9) - 1.0)
    # accelerating distribution change -> go with the fresh drift; stabilising -> fade it
    sig = fif * ops.sign(mu)
    return finish(frame, sig, fif * 100.0, norm, n, 0.5)


def k_ngm(frame, n, norm, w=5, **_):
    mu, sg = _mu_sigma(frame, w)
    dmu = mu - ops.shift(mu, 1)
    eps = (frame.roll("ret_std", 55) ** 2) * 0.25 + 1e-12
    with np.errstate(all="ignore"):
        ngm = sg ** 2 * dmu / (sg ** 2 + eps)
        sig = ngm / (ops.rstd(dmu, n) + 1e-12)
    return finish(frame, sig, ngm, norm, n, 1.0)


def _alpha_div(m1, s1, m2, s2, a):
    """Rényi/α-divergence between two Gaussians (closed form, 0<a<1)."""
    with np.errstate(all="ignore"):
        va = a * s2 ** 2 + (1 - a) * s1 ** 2
        return (a * (m1 - m2) ** 2 / (2 * va)
                - np.log(va / (s1 ** (2 * (1 - a)) * s2 ** (2 * a))) / (2 * (1 - a)))


def _short(w, n):
    return int(min(w, max(2, n // 2)))


def k_alpha(frame, n, norm, w=5, alpha=0.5, **_):
    w = _short(w, n)
    m1, s1 = _mu_sigma(frame, w)
    m2, s2 = _mu_sigma(frame, n)
    div = _alpha_div(m1, s1, m2, s2, alpha)
    sig = np.tanh(div) * ops.sign(m1 - m2)
    return finish(frame, sig, div, norm, n, 1.0)


def k_kl(frame, n, norm, w=5, **_):
    w = _short(w, n)
    m1, s1 = _mu_sigma(frame, w)
    m2, s2 = _mu_sigma(frame, n)
    with np.errstate(all="ignore"):
        kl = np.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5
    sig = np.tanh(kl) * ops.sign(m1 - m2)
    return finish(frame, sig, kl, norm, n, 1.0)


def k_jeffreys(frame, n, norm, w=5, **_):
    w = _short(w, n)
    m1, s1 = _mu_sigma(frame, w)
    m2, s2 = _mu_sigma(frame, n)
    with np.errstate(all="ignore"):
        kl12 = np.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5
        kl21 = np.log(s1 / s2) + (s2 ** 2 + (m1 - m2) ** 2) / (2 * s1 ** 2) - 0.5
        j = kl12 + kl21
    sig = np.tanh(j) * ops.sign(m1 - m2)
    return finish(frame, sig, j, norm, n, 1.0)


def k_amari(frame, n, norm, **_):
    """Amari-Chentsov tensor component T_μμμ = E[(∂_μ ℓ)³] = skew/σ³ for the
    Gaussian score ∂_μ ℓ = (x−μ)/σ²: the cubic skewness of the score."""
    mu, sg = _mu_sigma(frame, n)
    sk = frame.roll("ret_skew", n)
    with np.errstate(all="ignore"):
        T = sk / (sg ** 3)
        sig = sk * (1.0 + np.abs(mu) / sg)
    return finish(frame, sig, T, norm, n, 1.0)


def k_geoflow(frame, n, norm, w=5, **_):
    mu, sg = _mu_sigma(frame, w)
    d = fisher_distance(mu, sg, ops.shift(mu, n), ops.shift(sg, n))
    sig = d * ops.sign(mu - ops.shift(mu, n)) / 3.0
    return finish(frame, sig, d, norm, n, 1.0)


def k_expfam(frame, n, norm, **_):
    mu, sg = _mu_sigma(frame, n)
    with np.errstate(all="ignore"):
        theta1 = mu / sg ** 2          # natural parameter: precision-weighted mean
        sig = theta1 * sg               # = μ/σ, the Sharpe-like e-coordinate
    return finish(frame, sig, theta1, norm, n, 1.0)


def k_mixture(frame, n, norm, **_):
    W = ops.windows(frame.ret, n)
    with np.errstate(all="ignore"):
        up = np.where(W > 0, W, np.nan)
        dn = np.where(W < 0, W, np.nan)
        pi_up = np.nanmean(W > 0, axis=1)
        mu_up = np.nanmean(up, axis=1)
        mu_dn = np.nanmean(dn, axis=1)
        eta = pi_up * np.nan_to_num(mu_up) + (1 - pi_up) * np.nan_to_num(mu_dn)  # m-coordinate (mean)
        sig = eta / (np.nanstd(W, axis=1) + 1e-12) * np.sqrt(n)
    sig[np.isnan(W).any(axis=1)] = np.nan
    return finish(frame, sig, eta, norm, n, 1.0)


def k_student(frame, n, norm, w=5, **_):
    """Student-t leg: ν from excess kurtosis (ν = 6/κ + 4), the t-score's
    Fisher information (ν+1)/(ν+3)/σ² down-weights heavy-tailed windows."""
    mu, sg = _mu_sigma(frame, w)
    kurt = np.clip(frame.roll("ret_kurt", n), 0.05, 50.0)
    nu = 6.0 / kurt + 4.0
    info = (nu + 1.0) / (nu + 3.0)
    dmu = mu - ops.shift(mu, 1)
    with np.errstate(all="ignore"):
        sig = info * dmu / (ops.rstd(dmu, n) + 1e-12)
    return finish(frame, sig, nu, norm, n, 1.0)


def variants():
    g = grid(extra=dict(w=EST))
    return (
        make_variants(D, 1, "Fisher distance", "Fisher information flow (FIF)", k_fif,
                      "d_F between consecutive (μ,σ) windows (exact hyperbolic form); FIF = tanh(mean d recent / mean d older − 1)",
                      "FIF > +30 the distribution is shifting fast (regime change, go with the fresh drift); < −30 stabilising", 2, g)
        + make_variants(D, 2, "natural gradient", "Natural gradient momentum (NGM)", k_ngm,
                        "NGM = σ² · dμ / (σ² + ε): the Fisher-preconditioned gradient of the mean",
                        "momentum that is automatically muted in low-volatility noise and trusted in high-volatility conviction", 2, g)
        + make_variants(D, 3, "α-divergence", "α-divergence recent vs window", k_alpha,
                        "D_α(N(μ_w,σ_w) ‖ N(μ_n,σ_n)) in closed form, oriented by sign(μ_w − μ_n)",
                        "a large divergence in the recent mean's direction = the short horizon has broken from the long one", 2,
                        grid(extra=dict(w=EST, alpha=(0.25, 0.5, 0.75))))
        + make_variants(D, 4, "KL divergence", "KL(recent ‖ window)", k_kl,
                        "KL = ln(σ_n/σ_w) + (σ_w² + (μ_w−μ_n)²)/(2σ_n²) − ½, oriented by the mean shift", "", 2, g)
        + make_variants(D, 5, "Jeffreys divergence", "Jeffreys divergence", k_jeffreys,
                        "J = KL(P‖Q) + KL(Q‖P) between the recent and the window Gaussians", "", 2, g)
        + make_variants(D, 6, "Amari-Chentsov tensor", "Amari-Chentsov T_μμμ", k_amari,
                        "T_μμμ = E[((x−μ)/σ²)³] = skew/σ³, the cubic tensor of the Gaussian family",
                        "skewed score: the asymmetric tail is where the next print is more likely", 2)
        + make_variants(D, 7, "geodesic flow", "Geodesic flow over n", k_geoflow,
                        "exact Fisher-Rao geodesic distance between (μ,σ) now and n candles ago, signed by Δμ", "", 2, g)
        + make_variants(D, 8, "exponential family projection", "Natural parameter θ₁", k_expfam,
                        "θ₁ = μ/σ² (natural coordinate of the Gaussian exponential family), reported as μ/σ", "", 2)
        + make_variants(D, 9, "mixture family", "Up/down mixture m-coordinate", k_mixture,
                        "η = π_up μ_up + (1−π_up) μ_down from the two-component sign mixture, in σ/√n units", "", 2)
        + make_variants(D, 10, "dual connection (Student-t)", "Student-t information-weighted drift", k_student,
                        "ν = 6/κ + 4 from excess kurtosis; drift weighted by the t-score information (ν+1)/(ν+3)",
                        "heavy tails lower the information in the mean shift - the signal shrinks accordingly", 2, g)
    )
