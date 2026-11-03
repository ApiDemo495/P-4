"""Round AJ - the kinetic mechanisms of the physics layer.

Every mechanism here reads only the frozen tape of the window (BTC and PAXG
prints: ``[t_ms, price, volume, side]``), so it is computable on every real
feed - a Kraken WebSocket in a Codespace, a REST tape, the simulator - and
never prints 0.00 because some third-party telemetry host was unreachable.

Direction convention (shared with ``micro.py``): ``+1`` = BTC strengthens
against PAXG, ``-1`` = the opposite, ``0`` = the mechanism is a regime or a
cost and carries no side.  Each mechanism also returns ``active`` (False when
the tape is too short to estimate it) with the reason, so the engine excludes
it from the Kelly blend instead of feeding it a zero.

Mechanisms
----------
hawkes        branching ratio n of the trade-arrival process (self-excitation).
              Clustered arrivals carry the flow's sign forward; n → 1 is the
              critical regime where the cascade is about to exhaust.
kinetic       signed momentum flux  p = Σ side·volume·|Δln P|  of the last
              60 s, z-scored against the trailing ten minutes; kinetic energy
              ½ Σ volume·(Δln P)² is the "temperature" of that flux.
entropy       Shannon entropy of the trade-sign sequence (blocks of 3 prints).
              A sequence with low entropy is an informed trader working an
              order; high entropy is noise and gives nothing to follow.
diffusion     heat transfer between the two legs: lagged cross-correlation of
              5-second BTC and PAXG returns; when one leg leads, the leader's
              last move predicts the follower's next move.
temperature   realised variance of the BTC/PAXG ratio over the last minute
              against the trailing ten minutes.  A hot tape (shock) shrinks
              every directional vote: it is the drag term of the layer.
"""
from __future__ import annotations

from backend.core.numfmt import fx, fxs

import math

import numpy as np

from backend.core import config as cfg
from backend.physics.constants import sign_of, soft_gate

HAWKES_BIN_S = 1.0
HAWKES_WINDOW_S = 300.0
HAWKES_CRITICAL = 0.95
KINETIC_WINDOW_S = float(cfg.window_seconds())   # one window (70 s); historical name
KINETIC_BASE_S = 600.0
ENTROPY_BLOCK = 3
ENTROPY_PRINTS = 240
ENTROPY_INFORMED = 0.85          # normalised H below this ⇒ structured flow
DIFFUSION_GRID_S = 5.0
DIFFUSION_WINDOW_S = 300.0
DIFFUSION_MAX_LAG = 3
TEMPERATURE_HOT = 2.0            # T_rel above this ⇒ full drag


def _recent(ticks: np.ndarray, seconds: float) -> np.ndarray:
    if ticks is None or ticks.shape[0] == 0:
        return np.zeros((0, 4))
    t_end = float(ticks[-1, 0])
    return ticks[ticks[:, 0] >= t_end - seconds * 1000.0]


def _inactive(key: str, section: str, name: str, why: str, value: float = 0.0) -> dict:
    return {"key": key, "section": section, "name": name, "value": value, "direction": 0, "edge_bps": 0.0,
            "active": False, "source": "tape", "logic": why}


# ------------------------------------------------------------------ Hawkes
def hawkes(snapshot) -> dict:
    key, sec, name = "hawkes", "8.6", "Hawkes self-excitation"
    ticks = snapshot.ticks("BTC")
    win = _recent(ticks, HAWKES_WINDOW_S)
    if win.shape[0] < 40:
        return _inactive(key, sec, name, f"{win.shape[0]} prints in {HAWKES_WINDOW_S:.0f} s - need 40 to fit the branching ratio")
    t = win[:, 0] / 1000.0
    span = max(HAWKES_BIN_S * 10, float(t[-1] - t[0]))
    n_bins = int(span // HAWKES_BIN_S) + 1
    counts = np.histogram(t, bins=n_bins, range=(float(t[0]), float(t[0]) + n_bins * HAWKES_BIN_S))[0]
    mean = float(counts.mean())
    var = float(counts.var())
    if mean <= 0:
        return _inactive(key, sec, name, "no arrivals")
    # Hawkes with exponential kernel: Var/Mean of bin counts → 1/(1-n)² for
    # bins much longer than the kernel.  Invert for n, floor at Poisson.
    vmr = var / mean
    # Round BH: a branching ratio is physically in [0, 1); under-dispersed
    # arrivals (VMR < 1) are reported as the floor 0.001, never a dead 0.000
    n = max(0.001, min(0.999, 1.0 - 1.0 / math.sqrt(max(1e-6, vmr))))
    W = cfg.window_seconds()
    last = _recent(win, W)
    vol = np.abs(last[:, 2])
    flow = float(np.sum(vol * np.sign(last[:, 3])) / max(1e-12, vol.sum())) if last.shape[0] else 0.0
    critical = n >= HAWKES_CRITICAL
    # Round BH: soft gates - memory (n vs 0.15) and side (flow vs 0.05) scale
    # the edge continuously; a critical cascade damps it to a sliver instead
    # of switching it off, so the mechanism always carries its side.
    gate = soft_gate(n, 0.15) * soft_gate(flow, 0.05) * (0.1 if critical else 1.0)
    direction = sign_of(flow)
    edge = min(6.0, max(n, 0.02) * max(abs(flow), 1e-3) * 8.0) * gate if direction else 0.0
    regime = ("critical cascade (n ≥ 0.95): the chain is about to exhaust - edge damped ×0.1" if critical
              else "near-Poisson arrivals: little memory to follow (weak)" if n < 0.15
              else "nearly balanced flow: self-excitation has a faint side" if abs(flow) < 0.05
              else f"self-excited {'buying' if flow > 0 else 'selling'} carries forward")
    return {
        "key": key, "section": sec, "name": name, "value": n, "unit": " n", "direction": direction, "edge_bps": edge,
        "active": True, "source": "tape", "branching_ratio": n, "vmr": vmr, "flow_60s": flow, "arrivals_per_s": mean / HAWKES_BIN_S,
        "critical": critical,
        "logic": (f"λ(t) = μ + Σ α·e^(−β(t−tᵢ)); {win.shape[0]} arrivals in {span:.0f} s binned at {HAWKES_BIN_S:.0f} s: "
                  f"mean {fx(mean, 2)}, Var/Mean = {fx(vmr, 2)} ⇒ n = 1 − 1/√(Var/Mean) = {fx(n, 3)}; "
                  f"signed flow ({W:g} s) {fxs(flow, 3)} ⇒ {regime}"),
    }


# ----------------------------------------------------------------- kinetic
def kinetic(snapshot) -> dict:
    key, sec, name = "kinetic", "8.7", "Momentum flux (kinetic energy)"
    ticks = snapshot.ticks("BTC")
    base = _recent(ticks, KINETIC_BASE_S)
    if base.shape[0] < 60:
        return _inactive(key, sec, name, f"{base.shape[0]} prints in {KINETIC_BASE_S:.0f} s - need 60 for a baseline")
    price = base[:, 1]
    ok = price > 0
    base = base[ok]
    lr = np.diff(np.log(base[:, 1]))
    vol = np.abs(base[1:, 2])
    side = np.sign(base[1:, 3])
    t = base[1:, 0] / 1000.0
    flux = side * vol * np.abs(lr)                   # signed momentum per print
    ke = 0.5 * vol * lr ** 2                         # kinetic energy per print
    t_end = t[-1]
    # Rolling 60 s sums on a 10 s stride for the baseline distribution.
    starts = np.arange(t[0], t_end - KINETIC_WINDOW_S + 1e-9, 10.0)
    if starts.size < 6:
        return _inactive(key, sec, name, "less than 2 minutes of tape for the momentum baseline")
    sums = np.array([flux[(t >= s) & (t < s + KINETIC_WINDOW_S)].sum() for s in starts])
    now_mask = t >= t_end - KINETIC_WINDOW_S
    p_now = float(flux[now_mask].sum())
    ke_now = float(ke[now_mask].sum())
    ke_base = float(ke.sum()) * KINETIC_WINDOW_S / max(1.0, t_end - t[0])
    scale = float(np.std(sums)) if sums.size > 2 else 0.0
    if scale <= 1e-18:
        return _inactive(key, sec, name, "momentum flux has no variance yet", value=0.0)
    z = p_now / scale
    direction = sign_of(z)
    edge = min(5.0, abs(z) * 1.5) * soft_gate(z, 1.0) if direction else 0.0      # Round BH: soft |z| gate
    heat = ke_now / max(1e-18, ke_base)
    return {
        "key": key, "section": sec, "name": name, "value": z, "unit": " σ", "direction": direction, "edge_bps": edge,
        "active": True, "source": "tape", "momentum_now": p_now, "momentum_sigma": scale, "ke_ratio": heat,
        "logic": (f"p = Σ side·v·|Δln P| over {cfg.window_seconds():g} s = {p_now:+.3e}; σ_p from {sums.size} trailing windows = {scale:.3e} ⇒ z = {fxs(z, 2)}; "
                  f"KE = ½Σ v·(Δln P)² = {ke_now:.3e} ({fx(heat, 2)}× the 10-min average); "
                  f"gate z²/(z²+1) = {fx(soft_gate(z, 1.0), 2)} ⇒ {'follow the flux' if abs(z) >= 1.0 else 'faint momentum, small edge'}"),
    }


# ----------------------------------------------------------------- entropy
def entropy(snapshot) -> dict:
    key, sec, name = "entropy", "8.8", "Trade-sign entropy (information)"
    ticks = snapshot.ticks("BTC")
    if ticks is None or ticks.shape[0] < 3 * ENTROPY_BLOCK * 8:
        n = 0 if ticks is None else int(ticks.shape[0])
        return _inactive(key, sec, name, f"{n} prints - need {3 * ENTROPY_BLOCK * 8} for an entropy estimate")
    recent = ticks[-ENTROPY_PRINTS:]
    s = np.sign(recent[:, 3])
    s = s[s != 0]
    if s.size < 3 * ENTROPY_BLOCK * 8:
        return _inactive(key, sec, name, "trade sides unknown on this tape (no aggressor flag)")
    bits = (s > 0).astype(int)
    n_blocks = bits.size // ENTROPY_BLOCK
    blocks = bits[:n_blocks * ENTROPY_BLOCK].reshape(n_blocks, ENTROPY_BLOCK)
    symbols = blocks @ (2 ** np.arange(ENTROPY_BLOCK))
    counts = np.bincount(symbols, minlength=2 ** ENTROPY_BLOCK).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    h = float(-(p * np.log2(p)).sum())
    h_norm = h / ENTROPY_BLOCK
    vol = np.abs(recent[-100:, 2])
    flow = float(np.sum(vol * np.sign(recent[-100:, 3])) / max(1e-12, vol.sum()))
    informed = h_norm < ENTROPY_INFORMED
    direction = sign_of(flow)
    # Round BH: how far below the informed line the entropy sits (soft), times the flow's side strength
    structure = soft_gate(max(0.0, ENTROPY_INFORMED - h_norm) + 0.02, 0.1)
    edge = min(6.0, 10.0 * abs(flow) * structure * soft_gate(flow, 0.05)) if direction else 0.0
    runs = int(np.sum(np.diff(s) != 0)) + 1
    return {
        "key": key, "section": sec, "name": name, "value": h_norm, "unit": " H", "direction": direction, "edge_bps": edge,
        "active": True, "source": "tape", "entropy_bits": h, "runs": runs, "flow_100": flow, "informed": informed,
        "logic": (f"H = −Σ p(block)·log₂ p over {n_blocks} blocks of {ENTROPY_BLOCK} signs = {fx(h, 3)} bits of {ENTROPY_BLOCK} "
                  f"(normalised {fx(h_norm, 3)}); {runs} runs in {s.size} prints; net flow (100) {fxs(flow, 3)} ⇒ "
                  f"{'structured: an informed order is being worked - follow its sign' if informed else 'near-random sequence: nothing to follow'}"),
    }


# --------------------------------------------------------------- diffusion
def _grid_returns(ticks: np.ndarray, t0: float, t1: float, step: float) -> np.ndarray:
    grid = np.arange(t0, t1 + 1e-9, step)
    t = ticks[:, 0] / 1000.0
    idx = np.searchsorted(t, grid, side="right") - 1
    px = np.where(idx >= 0, ticks[np.maximum(idx, 0), 1], np.nan)
    return np.diff(np.log(px))


def diffusion(snapshot) -> dict:
    key, sec, name = "diffusion", "8.9", "Cross-leg heat transfer (lead-lag)"
    btc, paxg = snapshot.ticks("BTC"), snapshot.ticks("PAXG")
    if btc.shape[0] < 20 or paxg.shape[0] < 5:
        return _inactive(key, sec, name, "both legs need prints (BTC ≥ 20, PAXG ≥ 5)")
    t1 = min(btc[-1, 0], paxg[-1, 0]) / 1000.0
    t0 = max(btc[0, 0], paxg[0, 0]) / 1000.0
    t0 = max(t0, t1 - DIFFUSION_WINDOW_S)
    if t1 - t0 < 20 * DIFFUSION_GRID_S:
        return _inactive(key, sec, name, f"only {t1 - t0:.0f} s of overlapping tape - need {20 * DIFFUSION_GRID_S:.0f}")
    rb = _grid_returns(btc, t0, t1, DIFFUSION_GRID_S)
    rp = _grid_returns(paxg, t0, t1, DIFFUSION_GRID_S)
    ok = np.isfinite(rb) & np.isfinite(rp)
    rb, rp = rb[ok], rp[ok]
    n = rb.size
    if n < 20 or rb.std() <= 0 or rp.std() <= 0:
        return _inactive(key, sec, name, "one leg has not moved on the 5 s grid yet")
    zb, zp = (rb - rb.mean()) / rb.std(), (rp - rp.mean()) / rp.std()
    thresh = 2.0 / math.sqrt(n)
    pred_b = pred_p = 0.0
    used = []
    # Round BH: every lag contributes, weighted by its own significance
    # (ρ² / (ρ² + thresh²)) - a significant lag counts nearly fully, an
    # insignificant one a sliver, so the prediction is never a dead 0.00
    for k in range(1, DIFFUSION_MAX_LAG + 1):
        c_p_leads = float(np.mean(zp[:-k] * zb[k:]))       # PAXG at t−k vs BTC at t
        c_b_leads = float(np.mean(zb[:-k] * zp[k:]))       # BTC at t−k vs PAXG at t
        w_p = c_p_leads * c_p_leads / (c_p_leads * c_p_leads + thresh * thresh)
        w_b = c_b_leads * c_b_leads / (c_b_leads * c_b_leads + thresh * thresh)
        pred_b += w_p * c_p_leads * (rb.std() / rp.std()) * rp[-k]
        pred_p += w_b * c_b_leads * (rp.std() / rb.std()) * rb[-k]
        if abs(c_p_leads) > thresh:
            used.append(f"PAXG→BTC lag {k}: ρ {fxs(c_p_leads, 2)}")
        if abs(c_b_leads) > thresh:
            used.append(f"BTC→PAXG lag {k}: ρ {fxs(c_b_leads, 2)}")
    rel = pred_b - pred_p                                    # predicted next ratio move
    d_coeff = float(np.mean(zb * zp))                        # contemporaneous coupling
    direction = sign_of(rel)
    edge = min(4.0, abs(rel) * 1e4) * soft_gate(rel * 1e4, 0.2) if direction else 0.0   # Round BH: soft 0.2 bp gate
    return {
        "key": key, "section": sec, "name": name, "value": rel * 1e4, "unit": " bp", "direction": direction, "edge_bps": edge,
        "active": True, "source": "tape", "coupling": d_coeff, "lags_used": used, "samples": n,
        "logic": (f"∂u/∂t = D∇²u on the two legs: {n} returns on a {DIFFUSION_GRID_S:.0f} s grid, contemporaneous ρ = {fxs(d_coeff, 2)}, "
                  f"significance |ρ| > 2/√n = {fx(thresh, 2)}; "
                  + (f"leads: {'; '.join(used)}; predicted next-step BTC {fxs(pred_b * 1e4, 2)} bp − PAXG {fxs(pred_p * 1e4, 2)} bp = {fxs(rel * 1e4, 2)} bp"
                     if used else f"no lag is significant (all weighted by ρ²/(ρ²+thresh²)): faint lead {fxs(rel * 1e4, 2)} bp - the legs mostly diffuse independently")),
    }


# ------------------------------------------------------------- temperature
def temperature(snapshot) -> dict:
    key, sec, name = "temperature", "8.10", "Tape temperature (variance shock)"
    btc = snapshot.ticks("BTC")
    base = _recent(btc, KINETIC_BASE_S)
    if base.shape[0] < 60:
        return _inactive(key, sec, name, f"{base.shape[0]} prints in 10 min - need 60", value=1.0)
    t = base[:, 0] / 1000.0
    lr = np.diff(np.log(base[:, 1]))
    t = t[1:]
    t_end = t[-1]
    W = cfg.window_seconds()
    now = lr[t >= t_end - W]
    span = max(W, t_end - t[0])
    var_now = float(np.sum(now ** 2)) if now.size else 0.0
    var_base = float(np.sum(lr ** 2)) * W / span
    if var_base <= 0:
        return _inactive(key, sec, name, "no price variance in the baseline", value=1.0)
    t_rel = var_now / var_base
    # Round BH: softplus drag - a cool tape still carries a sliver of drag that
    # grows smoothly through T_rel = 1 instead of switching on from 0.00
    x = (t_rel - 1.0) / (TEMPERATURE_HOT - 1.0) * 0.6
    drag = min(0.6, 0.05 * math.log1p(math.exp(max(-40.0, min(40.0, x / 0.05)))))
    sigma60_bps = math.sqrt(var_now) * 1e4
    return {
        "key": key, "section": sec, "name": name, "value": t_rel, "unit": " T", "direction": 0, "edge_bps": 0.0,
        "active": True, "source": "tape", "drag": drag, "sigma_60s_bps": sigma60_bps, "prints_60s": int(now.size),
        "logic": (f"½k·T ≡ ½⟨v²⟩: realised variance last {W:g} s = {var_now:.3e} ({now.size} prints) vs {W:g} s share of the 10-min baseline "
                  f"{var_base:.3e} ⇒ T_rel = {fx(t_rel, 2)}; σ₆₀ = {sigma60_bps:.1f} bp; "
                  f"drag on every directional vote = {fx(drag, 2)}" + (" (shock: votes shrink)" if t_rel > 1.0 else " (cool tape: faint drag)")),
    }
