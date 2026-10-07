"""Domain 5 - fractional stochastic calculus indicators (210 formulas)."""
from __future__ import annotations

import numpy as np

from backend.genesis import ops
from backend.genesis.domains._common import finish, momentum_sign, scale_of
from backend.genesis.spec import grid, make_variants

D = 5
ALPHAS = (0.3, 0.5, 0.7)
EPS_MULT = (0.25, 0.5, 1.0)


def k_hurst(frame, n, norm, **_):
    H = frame.roll("hurst", max(n, 16))
    sig = (H - 0.5) * 4.0 * momentum_sign(frame, max(1, n // 4))
    return finish(frame, sig, H, norm, n, 1.0)


def k_malliavin(frame, n, norm, eps_mult=0.5, functional="rv", **_):
    """D_t F by symmetric perturbation of the latest return: F(r₀+ε) − F(r₀−ε) / 2ε."""
    W = ops.windows(frame.ret, n)
    eps = eps_mult * (np.nanstd(W, axis=1) + 1e-9)
    plus, minus = W.copy(), W.copy()
    plus[:, -1] += eps
    minus[:, -1] -= eps

    def F(M):
        with np.errstate(all="ignore"):
            if functional == "skew":
                m, s = np.nanmean(M, axis=1, keepdims=True), np.nanstd(M, axis=1, keepdims=True) + 1e-12
                return np.nanmean(((M - m) / s) ** 3, axis=1)
            if functional == "drawdown":
                cum = np.nancumsum(M, axis=1)
                return np.max(np.maximum.accumulate(cum, axis=1) - cum, axis=1)
            return np.sqrt(np.nansum(M * M, axis=1))
    with np.errstate(all="ignore"):
        mdv = (F(plus) - F(minus)) / (2 * eps)
    mdv[np.isnan(W).any(axis=1)] = np.nan
    sig = mdv if functional != "drawdown" else -mdv
    return finish(frame, sig, mdv, norm, n, 0.7)


def k_clark_ocone(frame, n, norm, eps_mult=0.5, **_):
    """Clark-Ocone integrand for the stochastic position F = (close − min)/(max − min):
    E[D_t F | F_t] ≈ (F(r₀+ε) − F(r₀−ε)) / 2ε - how much the next increment moves the position."""
    W = ops.windows(frame.close, n)
    with np.errstate(all="ignore"):
        eps = eps_mult * (np.nanstd(np.diff(W, axis=1), axis=1) + 1e-9)
        plus, minus = W.copy(), W.copy()
        plus[:, -1] += eps
        minus[:, -1] -= eps

        def F(M):
            lo, hi = M.min(axis=1), M.max(axis=1)
            return (M[:, -1] - lo) / (hi - lo + 1e-12)
        d = (F(plus) - F(minus)) / (2 * eps / (W[:, -1] + 1e-12))  # per unit log move
        pos = F(W)
    # a position near the top with a high integrand = the max is being set now: continuation
    sig = (pos - 0.5) * 2.0 * np.tanh(d / 20.0)
    sig[np.isnan(W).any(axis=1)] = np.nan
    return finish(frame, sig, d, norm, n, 1.0)


def k_rough_vol(frame, n, norm, **_):
    rv = frame.roll("rv", 5)
    eta = ops.safe_log(rv)
    qs = (1, 2, 3, 5)
    lg = []
    for q in qs:
        with np.errstate(all="ignore"):
            lg.append(ops.safe_log(ops.rmean((eta - ops.shift(eta, q)) ** 2, n)))
    X = np.log(np.asarray(qs, dtype=float))
    X = X - X.mean()
    Y = np.vstack(lg)
    with np.errstate(all="ignore"):
        slope = (X[:, None] * (Y - np.nanmean(Y, axis=0))).sum(axis=0) / (X ** 2).sum()
        H = np.clip(slope / 2.0, 0.0, 1.0)
    expanding = ops.sign(rv - ops.shift(rv, 3))
    sig = (0.3 - H) * 3.0 * expanding * ops.sign(frame.ret)
    return finish(frame, sig, H, norm, n, 1.0)


def k_gl(frame, n, norm, alpha=0.5, **_):
    lp = ops.safe_log(frame.close)
    lp = lp - ops.shift(lp, n)   # window-relative level: D^α of a constant is not zero
    d = ops.grunwald_letnikov(lp, alpha, n)
    sig = d / (scale_of(frame, n) * n ** (1 - alpha))
    return finish(frame, sig, d, norm, n, 1.0)


def k_wick(frame, n, norm, order=3, **_):
    z = frame.ret / scale_of(frame, n)
    with np.errstate(all="ignore"):
        if order == 2:
            he = z * z - 1.0           # :x²:
            raw = ops.rmean(he, n)
            sig = raw * ops.sign(frame.ret)
        elif order == 4:
            he = z ** 4 - 6 * z * z + 3  # :x⁴:
            raw = ops.rmean(he, n)
            sig = -raw * momentum_sign(frame) / 3.0
        else:
            he = z ** 3 - 3 * z        # :x³:
            raw = ops.rmean(he, n)
            sig = raw
    return finish(frame, sig, raw, norm, n, 1.0)


def k_fou(frame, n, norm, **_):
    """Fractional Ornstein-Uhlenbeck: κ from the regression of Δx on x (x = log
    price − EMA), with the reversion trusted only when H < 0.5 (anti-persistent)."""
    x = ops.safe_log(frame.close) - ops.safe_log(ops.ema(frame.close, n))
    dx = x - ops.shift(x, 1)
    xp = ops.shift(x, 1)
    with np.errstate(all="ignore"):
        cov = ops.rmean(dx * xp, 3 * n) - ops.rmean(dx, 3 * n) * ops.rmean(xp, 3 * n)
        var = ops.rstd(xp, 3 * n) ** 2 + 1e-12
        kappa = -cov / var
    H = frame.roll("hurst", max(16, n))
    trust = np.clip((0.55 - H) * 5.0, -1.0, 1.0)
    sig = -np.clip(kappa, 0, 2) * x / (ops.rstd(x, 3 * n) + 1e-9) * trust
    return finish(frame, sig, kappa, norm, n, 1.0)


def k_subfbm(frame, n, norm, **_):
    """Sub-fractional BM has increments whose variance decays with time inside
    the window: variance of the first half vs the second half of increments."""
    W = ops.windows(frame.ret, 2 * n)
    with np.errstate(all="ignore"):
        v1 = np.nanvar(W[:, :n], axis=1)
        v2 = np.nanvar(W[:, n:], axis=1)
        raw = np.log((v2 + 1e-12) / (v1 + 1e-12))
    raw[np.isnan(W).any(axis=1)] = np.nan
    sig = np.tanh(raw) * momentum_sign(frame)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_multifractal(frame, n, norm, **_):
    """Generalised Hurst H(q) from structure functions S_q(τ) = E|Δ_τ x|^q at
    τ ∈ {1,2,4}; spectrum width Δ = H(−2) − H(2) (positive for multifractals)."""
    x = np.nancumsum(frame.ret)
    taus = (1, 2, 4)
    Hq = {}
    for q in (-2.0, 2.0):
        ys = []
        for tau in taus:
            with np.errstate(all="ignore"):
                inc = np.abs(x - ops.shift(x, tau)) + 1e-9
                ys.append(ops.safe_log(ops.rmean(inc ** q, n)))
        X = np.log(np.asarray(taus, dtype=float))
        X = X - X.mean()
        Y = np.vstack(ys)
        with np.errstate(all="ignore"):
            Hq[q] = (X[:, None] * (Y - np.nanmean(Y, axis=0))).sum(axis=0) / (X ** 2).sum() / q
    width = Hq[-2.0] - Hq[2.0]
    sig = -(width - 0.2) * 2.0 * momentum_sign(frame)   # wide spectrum = intermittent = fade
    return finish(frame, sig, width, norm, n, 1.0)


def k_volvol(frame, n, norm, **_):
    rv = frame.roll("rv", 5)
    lv = ops.safe_log(rv)
    vov = ops.rstd(lv - ops.shift(lv, 1), n)
    z = (vov - ops.rmean(vov, 55)) / (ops.rstd(vov, 55) + 1e-12)
    sig = -np.clip(z, -3, 3) / 3.0 * momentum_sign(frame)  # unstable vol after a run = exhaustion
    return finish(frame, sig, vov, norm, n, 1.0)


def variants():
    return (
        make_variants(D, 1, "fBm Hurst", "Rescaled-range Hurst gate", k_hurst,
                      "R/S Hurst exponent over n (three sub-scales, least squares); (H − 0.5) × sign of the last n/4 move",
                      "H > 0.5 persistent (follow), H < 0.5 anti-persistent (fade)", 2)
        + make_variants(D, 2, "Malliavin derivative", "Malliavin derivative of volatility (MDV)", k_malliavin,
                        "D F = (F(r₀+ε) − F(r₀−ε)) / 2ε with ε = m·σ, F ∈ {realised vol, skew, max drawdown}",
                        "MDV > 0: volatility is sensitive to the next print - expansion imminent in the direction of r₀", 2,
                        grid(extra=dict(eps_mult=EPS_MULT, functional=("rv", "skew", "drawdown"))))
        + make_variants(D, 3, "Clark-Ocone", "Clark-Ocone position integrand", k_clark_ocone,
                        "integrand of the Clark-Ocone representation for the stochastic position (close − min)/(max − min)",
                        "a position near the top whose integrand is large: the high is being set now", 2, grid(extra=dict(eps_mult=EPS_MULT)))
        + make_variants(D, 4, "rough volatility", "Rough volatility index (RVI)", k_rough_vol,
                        "variogram γ(q) = E|ln RV_{t+q} − ln RV_t|² at q ∈ {1,2,3,5}; H_rough = slope/2",
                        "H_rough < 0.2 very rough: sudden spikes - with expanding vol, go with the last print", 2)
        + make_variants(D, 5, "fractional Itô (Grünwald-Letnikov)", "Fractional momentum D^α log P", k_gl,
                        "Grünwald-Letnikov derivative of order α of log price, weights truncated at n", "", 2,
                        grid(extra=dict(alpha=ALPHAS)))
        + make_variants(D, 6, "Wick product", "Wick (Hermite) moment", k_wick,
                        ":x²: = z²−1, :x³: = z³−3z, :x⁴: = z⁴−6z²+3 of the standardised return, window mean",
                        "the Wick cube is a pure skew-direction reading; the Wick square a volatility surprise", 2,
                        grid(extra=dict(order=(2, 3, 4))))
        + make_variants(D, 7, "fractional Ornstein-Uhlenbeck", "fO-U reversion", k_fou,
                        "κ = −cov(Δx, x)/var(x) for x = log P − EMA_n; reversion −κ·x trusted only when H < 0.55", "", 2)
        + make_variants(D, 8, "sub-fractional BM", "Increment-variance decay", k_subfbm,
                        "ln(var of the last n increments / var of the previous n)", "contracting increments after a move = the move is settling in", 2)
        + make_variants(D, 9, "multifractal spectrum", "Multifractal width", k_multifractal,
                        "width Δ = H(−2) − H(2) from structure functions at τ ∈ {1,2,4}", "wide = intermittent = fade the last move", 2)
        + make_variants(D, 10, "volatility of volatility", "Vol-of-vol exhaustion", k_volvol,
                        "std over n of Δ ln RV₅, as a 55-candle z-score", "", 2)
    )
