"""Round BU - physics-derived estimators for the next window.

Four estimators whose mathematics genuinely applies to a price tape.  None
of them is a "theory of everything" for markets (no such thing exists); each
is a standard tool from statistical physics applied to exactly the quantity
it was built for, with its working traced so the panel can print the logic.

* ``kramers_moyal``   - non-parametric Langevin decomposition.  The first two
  Kramers-Moyal coefficients D1(x) (drift) and D2(x) (diffusion) are measured
  from the tape as conditional moments of the 1 s increments given the state
  x = (price - rolling mean) / rolling sigma.  The window forecast is the
  Peclet number  Pe = D1(x_now) * T / sqrt(2 * D2(x_now) * T): drift over the
  window divided by the diffusion width over the window.  Unlike the physics
  engine's parametric OU fit this assumes no functional form for the drift.

* ``ising_flow``      - 1-D Ising chain on the aggressor signs.  The
  nearest-neighbour coupling is K = atanh(rho_1) (exact for the chain: the
  sign autocorrelation at lag k is tanh(K)^k).  The expected mean sign of the
  next N trades given the last sign s_0 and magnetisation m is
  m_hat = m + (s_0 - m) * rho (1 - rho^N) / (N (1 - rho)), amplified by the
  susceptibility chi = (1 + rho) / (1 - rho) (the chain's response to a small
  field - large near criticality, i.e. when the crowd herds).

* ``fluctuation_response`` - fluctuation-dissipation.  The response of price
  to signed flow is the regression slope lambda = cov(dp, f) / var(f) (Kyle's
  lambda); the fluctuation is var(dp).  Theta = lambda^2 var(f) / var(dp) is
  the share of the price variance that flow explains (R^2).  Flow persists
  with autocorrelation rho_f, so the expected flow-driven move over the next
  window is lambda * rho_f * f_recent * n, scored against sigma_dp * sqrt(n)
  and weighted by Theta.

* ``maxent_window``   - Jaynes' maximum-entropy distribution of the next
  window's return on the empirical support of past window returns, subject
  to two constraints: mean = rho_w * r_current (the lag-1 regression forecast
  of window returns) and variance = the sample variance.  The Lagrange
  multipliers are solved by Newton iteration on the support grid;
  P(up) = sum p_i [x_i > 0].  It is the least-biased forecast consistent with
  those two facts and the observed support (fat tails included).

Every function never raises and reports ``covered=False`` with a reason when
the tape cannot support the estimate.
"""
from __future__ import annotations

import math

import numpy as np

GRID_MS = 1000.0
BINS = 5


def _f(x) -> float:
    v = float(x)
    return v if math.isfinite(v) else 0.0


def _grid_prices(ticks: np.ndarray, step_ms: float = GRID_MS) -> tuple[np.ndarray, np.ndarray]:
    """Last-price sampled on a regular grid (trades and quote prints both carry a price)."""
    t = ticks[:, 0]
    p = ticks[:, 1]
    start = math.ceil(t[0] / step_ms) * step_ms
    grid = np.arange(start, t[-1] + 1e-9, step_ms)
    if grid.size < 4:
        return grid, np.zeros(0)
    idx = np.searchsorted(t, grid, side="right") - 1
    idx = np.clip(idx, 0, p.size - 1)
    return grid, p[idx]


def _trades(ticks: np.ndarray) -> np.ndarray:
    return ticks[(np.abs(ticks[:, 2]) > 0) & (ticks[:, 3] != 0)]


# ----------------------------------------------------------------------------
def kramers_moyal(ticks: np.ndarray, horizon_s: float = 70.0) -> dict:
    out = {"covered": False, "score": 0.0, "logic": []}
    try:
        if ticks is None or len(ticks) < 60:
            out["why"] = "tape too short"
            return out
        _, p = _grid_prices(np.asarray(ticks, dtype=np.float64))
        if p.size < 120:
            out["why"] = f"only {p.size} s of tape on the 1 s grid (needs 120)"
            return out
        x_raw = np.log(np.maximum(p, 1e-9))
        span = max(30, min(int(horizon_s), p.size // 4))
        kernel = np.ones(span) / span
        mean = np.convolve(x_raw, kernel, mode="valid")
        x_tail = x_raw[span - 1:]
        dev = x_tail - mean
        sig = float(np.std(dev)) + 1e-12
        x = dev / sig
        dx = np.diff(x)
        xs = x[:-1]
        if dx.size < 60:
            out["why"] = "not enough increments after detrending"
            return out
        edges = np.quantile(xs, np.linspace(0, 1, BINS + 1))
        which = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, BINS - 1)
        d1 = np.zeros(BINS)
        d2 = np.zeros(BINS)
        count = np.zeros(BINS)
        for b in range(BINS):
            sel = which == b
            count[b] = sel.sum()
            if count[b] >= 5:
                d1[b] = float(np.mean(dx[sel]))
                d2[b] = float(np.mean(dx[sel] ** 2)) / 2.0
            else:
                d1[b] = float(np.mean(dx))
                d2[b] = float(np.mean(dx ** 2)) / 2.0
        now = float(x[-1])
        b_now = int(np.clip(np.searchsorted(edges, now, side="right") - 1, 0, BINS - 1))
        drift = d1[b_now]
        diff = max(d2[b_now], 1e-12)
        T = float(horizon_s)
        peclet = drift * T / math.sqrt(2.0 * diff * T)
        # a linear fit of D1 against x gives the restoring rate (OU limit check)
        slope = float(np.polyfit(xs, dx, 1)[0]) if xs.size > 10 else 0.0
        score = math.tanh(peclet)
        out.update({
            "covered": True, "score": _f(score), "state_sigma": _f(now), "d1": _f(drift), "d2": _f(diff),
            "peclet": _f(peclet), "restoring_rate": _f(-slope), "bin": b_now, "seconds": int(p.size),
            "logic": [
                f"state x = (log p − {span} s mean)/σ = {now:+.2f} σ (bin {b_now + 1}/{BINS}, {int(count[b_now])} samples)",
                f"D1(x) = ⟨Δx | x⟩ = {drift:+.4f} σ/s · D2(x) = ⟨Δx² | x⟩/2 = {diff:.5f} σ²/s",
                f"Péclet over {T:.0f} s = D1·T / √(2·D2·T) = {peclet:+.3f} → score tanh = {score:+.3f}",
                f"linear D1 slope {slope:+.4f}/s ({'restoring' if slope < 0 else 'self-reinforcing'} state)",
            ],
        })
        return out
    except Exception as exc:  # noqa: BLE001
        out["why"] = f"failed: {exc}"
        return out


# ----------------------------------------------------------------------------
def ising_flow(ticks: np.ndarray, horizon_trades: int | None = None) -> dict:
    out = {"covered": False, "score": 0.0, "logic": []}
    try:
        tr = _trades(np.asarray(ticks, dtype=np.float64)) if ticks is not None and len(ticks) else np.zeros((0, 4))
        s = np.sign(tr[:, 3]) if tr.shape[0] else np.zeros(0)
        if s.size < 40:
            out["why"] = f"only {s.size} trades with an aggressor side (needs 40)"
            return out
        s = s[-600:]
        m = float(np.mean(s))
        c = s - m
        var = float(np.mean(c * c))
        rho = float(np.mean(c[1:] * c[:-1]) / var) if var > 1e-12 else 0.0
        rho = max(-0.95, min(0.95, rho))
        K = math.atanh(rho)
        chi = (1.0 + rho) / (1.0 - rho)
        n = horizon_trades or max(10, min(300, s.size // 3))
        if abs(1.0 - rho) > 1e-9:
            persistence = rho * (1.0 - rho ** n) / (n * (1.0 - rho))
        else:
            persistence = 1.0
        s0 = float(s[-1])
        m_hat = m + (s0 - m) * persistence
        score = math.tanh(m_hat * min(chi, 10.0))
        out.update({
            "covered": True, "score": _f(score), "magnetisation": _f(m), "rho1": _f(rho), "coupling": _f(K),
            "susceptibility": _f(chi), "m_hat": _f(m_hat), "trades": int(s.size), "horizon_trades": int(n),
            "logic": [
                f"magnetisation m = ⟨s⟩ over {s.size} aggressor signs = {m:+.3f}",
                f"lag-1 sign autocorrelation ρ₁ = {rho:+.3f} → chain coupling K = atanh ρ₁ = {K:+.3f}",
                f"susceptibility χ = (1+ρ₁)/(1−ρ₁) = {chi:.2f} ({'herding' if chi > 2 else 'independent crowd'})",
                f"expected mean sign of the next {n} trades m̂ = m + (s₀ − m)·ρ(1−ρⁿ)/(n(1−ρ)) = {m_hat:+.3f}",
                f"score = tanh(m̂ · χ) = {score:+.3f}",
            ],
        })
        return out
    except Exception as exc:  # noqa: BLE001
        out["why"] = f"failed: {exc}"
        return out


# ----------------------------------------------------------------------------
def fluctuation_response(ticks: np.ndarray, horizon_s: float = 70.0) -> dict:
    out = {"covered": False, "score": 0.0, "logic": []}
    try:
        if ticks is None or len(ticks) < 60:
            out["why"] = "tape too short"
            return out
        tk = np.asarray(ticks, dtype=np.float64)
        grid, p = _grid_prices(tk)
        if p.size < 90:
            out["why"] = f"only {p.size} s on the 1 s grid (needs 90)"
            return out
        tr = _trades(tk)
        if tr.shape[0] < 30:
            out["why"] = f"only {tr.shape[0]} trades with volume"
            return out
        signed = tr[:, 2] * np.sign(tr[:, 3])
        flow = np.zeros(p.size)
        idx = np.clip(np.searchsorted(grid, tr[:, 0], side="left"), 0, p.size - 1)
        np.add.at(flow, idx, signed)
        dp = np.diff(np.log(np.maximum(p, 1e-9))) * 1e4            # bps per second
        f = flow[1:]
        vf = float(np.var(f))
        vp = float(np.var(dp))
        if vf < 1e-18 or vp < 1e-12:
            out["why"] = "no flow or price variance on the grid"
            return out
        lam = float(np.cov(dp, f)[0, 1] / vf)
        theta = max(0.0, min(1.0, lam * lam * vf / vp))
        fc = f - f.mean()
        rho_f = float(np.mean(fc[1:] * fc[:-1]) / vf)
        rho_f = max(-0.95, min(0.95, rho_f))
        recent = float(np.mean(f[-10:]))
        n = float(horizon_s)
        expected = lam * rho_f * recent * n
        width = math.sqrt(vp * n)
        gate = 0.05 + 0.95 * theta            # flow never explains nothing at all
        score = math.tanh(expected / (width + 1e-9)) * gate
        out.update({
            "covered": True, "score": _f(score), "lambda_bps_per_unit": _f(lam), "theta": _f(theta),
            "rho_flow": _f(rho_f), "recent_flow": _f(recent), "expected_bps": _f(expected), "width_bps": _f(width),
            "logic": [
                f"response λ = cov(Δp, f)/var(f) = {lam:+.3f} bp per unit flow (1 s grid, {p.size} s)",
                f"fluctuation var(Δp) = {vp:.4f} bp²/s · Θ = λ²·var(f)/var(Δp) = {theta:.3f} (share of price variance flow explains)",
                f"flow persistence ρ_f = {rho_f:+.3f} · recent flow {recent:+.4f}/s",
                f"expected flow-driven move over {n:.0f} s = λ·ρ_f·f·n = {expected:+.2f} bp vs diffusion width {width:.2f} bp",
                f"score = tanh(expected/width)·(0.05 + 0.95 Θ) = {score:+.3f}",
            ],
        })
        return out
    except Exception as exc:  # noqa: BLE001
        out["why"] = f"failed: {exc}"
        return out


# ----------------------------------------------------------------------------
def _maxent_fit(support: np.ndarray, mean: float, var: float, iters: int = 60) -> tuple[np.ndarray, np.ndarray, bool]:
    """p_i ∝ exp(λ₁ x_i + λ₂ x_i²) on the grid ``support`` with ⟨x⟩ = mean, ⟨x²⟩ = var + mean²."""
    x = support
    targets = np.array([mean, var + mean * mean])
    lam = np.zeros(2)
    feats = np.column_stack([x, x * x])
    ok = False
    for _ in range(iters):
        z = feats @ lam
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        mom = feats.T @ p
        g = mom - targets
        cov = (feats * p[:, None]).T @ feats - np.outer(mom, mom)
        try:
            step = np.linalg.solve(cov + np.eye(2) * 1e-12, g)
        except np.linalg.LinAlgError:
            break
        lam -= np.clip(step, -50, 50)
        if np.max(np.abs(g) / (np.abs(targets) + 1e-12)) < 1e-6:
            ok = True
            break
    z = feats @ lam
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    return p, lam, ok


def maxent_window(ticks: np.ndarray, horizon_s: float = 70.0) -> dict:
    out = {"covered": False, "score": 0.0, "logic": []}
    try:
        if ticks is None or len(ticks) < 60:
            out["why"] = "tape too short"
            return out
        _, p = _grid_prices(np.asarray(ticks, dtype=np.float64))
        n = int(horizon_s)
        if p.size < 3 * n:
            out["why"] = f"tape covers {p.size} s (needs 3 windows = {3 * n} s)"
            return out
        lp = np.log(np.maximum(p, 1e-9))
        stride = max(1, n // 10)
        r = (lp[n::stride] - lp[:-n:stride]) * 1e4       # overlapping window returns, bps
        if r.size < 12:
            out["why"] = "too few window returns"
            return out
        current = float(lp[-1] - lp[-1 - n]) * 1e4
        # lag-1 regression of non-overlapping consecutive window returns
        lag = max(1, n // stride)
        a, b = r[:-lag], r[lag:]
        va = float(np.var(a))
        rho_w = float(np.cov(a, b)[0, 1] / va) if va > 1e-12 and a.size > 5 else 0.0
        rho_w = max(-0.9, min(0.9, rho_w))
        mu_next = rho_w * (current - float(r.mean())) + float(r.mean())
        var = float(np.var(r))
        if var < 1e-9:
            out["why"] = "window returns have no variance"
            return out
        support = np.unique(np.round(r, 3))
        if support.size < 8:
            support = np.linspace(r.min(), r.max(), 24)
        # the constrained mean must lie inside the support
        mu_next = float(np.clip(mu_next, support.min() + 1e-6, support.max() - 1e-6))
        var_c = min(var, float(np.var(support)) * 0.99)
        probs, lam, ok = _maxent_fit(support, mu_next, var_c)
        p_up = float(probs[support > 0].sum())
        p_tie = float(probs[np.abs(support) < 1e-9].sum())
        ent = float(-np.sum(probs * np.log(probs + 1e-300)))
        score = 2.0 * p_up - 1.0
        out.update({
            "covered": True, "score": _f(score), "p_up": _f(p_up), "p_tie": _f(p_tie), "rho_window": _f(rho_w),
            "mu_next_bps": _f(mu_next), "sigma_bps": _f(math.sqrt(var)), "current_bps": _f(current),
            "lambda1": _f(lam[0]), "lambda2": _f(lam[1]), "entropy_nats": _f(ent), "support": int(support.size),
            "converged": bool(ok), "windows": int(r.size),
            "logic": [
                f"{r.size} overlapping {n} s returns · current window {current:+.1f} bp · lag-1 ρ_w = {rho_w:+.3f}",
                f"constraints: ⟨r⟩ = ρ_w·(r₀ − r̄) + r̄ = {mu_next:+.2f} bp · σ = {math.sqrt(var):.2f} bp · support {support.size} observed values",
                f"max-entropy p ∝ exp(λ₁ r + λ₂ r²): λ₁ = {lam[0]:+.4f}, λ₂ = {lam[1]:+.5f} ({'converged' if ok else 'partial'}) · H = {ent:.2f} nats",
                f"P(up) = Σ p[r>0] = {p_up:.3f} · P(tie) = {p_tie:.3f} → score 2P−1 = {score:+.3f}",
            ],
        })
        return out
    except Exception as exc:  # noqa: BLE001
        out["why"] = f"failed: {exc}"
        return out


ESTIMATORS = {
    "physics:kramers_moyal": ("Kramers–Moyal drift", kramers_moyal),
    "physics:ising": ("Ising flow chain", ising_flow),
    "physics:fdr": ("fluctuation–dissipation", fluctuation_response),
    "physics:maxent": ("max-entropy window", maxent_window),
}


def estimate_all(ticks: np.ndarray, horizon_s: float = 70.0) -> dict:
    """Run the four estimators; ``values`` holds the covered scores by input name."""
    out: dict = {"estimators": {}, "values": {}}
    for key, (label, fn) in ESTIMATORS.items():
        res = fn(ticks) if fn is ising_flow else fn(ticks, horizon_s)
        res["label"] = label
        out["estimators"][key] = res
        if res.get("covered"):
            out["values"][key] = float(res["score"])
    return out
