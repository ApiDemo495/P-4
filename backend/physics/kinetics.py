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

import math

import numpy as np

HAWKES_BIN_S = 1.0
HAWKES_WINDOW_S = 300.0
HAWKES_CRITICAL = 0.95
KINETIC_WINDOW_S = 60.0
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
    n = max(0.0, min(0.999, 1.0 - 1.0 / math.sqrt(max(1.0, vmr))))
    last = _recent(win, 60.0)
    vol = np.abs(last[:, 2])
    flow = float(np.sum(vol * np.sign(last[:, 3])) / max(1e-12, vol.sum())) if last.shape[0] else 0.0
    critical = n >= HAWKES_CRITICAL
    if critical or n < 0.15 or abs(flow) < 0.05:
        direction = 0
    else:
        direction = int(np.sign(flow))
    edge = 0.0 if direction == 0 else min(6.0, n * abs(flow) * 8.0)
    regime = ("critical cascade (n ≥ 0.95): the chain is about to exhaust - stand aside" if critical
              else "Poisson arrivals: no memory to follow" if n < 0.15
              else "balanced flow: self-excitation has no side to carry" if abs(flow) < 0.05
              else f"self-excited {'buying' if flow > 0 else 'selling'} carries forward")
    return {
        "key": key, "section": sec, "name": name, "value": n, "unit": " n", "direction": direction, "edge_bps": edge,
        "active": True, "source": "tape", "branching_ratio": n, "vmr": vmr, "flow_60s": flow, "arrivals_per_s": mean / HAWKES_BIN_S,
        "critical": critical,
        "logic": (f"λ(t) = μ + Σ α·e^(−β(t−tᵢ)); {win.shape[0]} arrivals in {span:.0f} s binned at {HAWKES_BIN_S:.0f} s: "
                  f"mean {mean:.2f}, Var/Mean = {vmr:.2f} ⇒ n = 1 − 1/√(Var/Mean) = {n:.3f}; "
                  f"signed flow (60 s) {flow:+.3f} ⇒ {regime}"),
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
    direction = int(np.sign(z)) if abs(z) >= 1.0 else 0
    edge = min(5.0, max(0.0, abs(z) - 0.5) * 1.5) if direction else 0.0
    heat = ke_now / max(1e-18, ke_base)
    return {
        "key": key, "section": sec, "name": name, "value": z, "unit": " σ", "direction": direction, "edge_bps": edge,
        "active": True, "source": "tape", "momentum_now": p_now, "momentum_sigma": scale, "ke_ratio": heat,
        "logic": (f"p = Σ side·v·|Δln P| over 60 s = {p_now:+.3e}; σ_p from {sums.size} trailing windows = {scale:.3e} ⇒ z = {z:+.2f}; "
                  f"KE = ½Σ v·(Δln P)² = {ke_now:.3e} ({heat:.2f}× the 10-min average); "
                  f"{'|z| ≥ 1 ⇒ follow the flux' if direction else '|z| < 1 ⇒ no net momentum'}"),
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
    direction = int(np.sign(flow)) if (informed and abs(flow) > 0.05) else 0
    edge = min(6.0, (ENTROPY_INFORMED - h_norm) / ENTROPY_INFORMED * 10.0 * abs(flow)) if direction else 0.0
    runs = int(np.sum(np.diff(s) != 0)) + 1
    return {
        "key": key, "section": sec, "name": name, "value": h_norm, "unit": " H", "direction": direction, "edge_bps": edge,
        "active": True, "source": "tape", "entropy_bits": h, "runs": runs, "flow_100": flow, "informed": informed,
        "logic": (f"H = −Σ p(block)·log₂ p over {n_blocks} blocks of {ENTROPY_BLOCK} signs = {h:.3f} bits of {ENTROPY_BLOCK} "
                  f"(normalised {h_norm:.3f}); {runs} runs in {s.size} prints; net flow (100) {flow:+.3f} ⇒ "
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
    for k in range(1, DIFFUSION_MAX_LAG + 1):
        c_p_leads = float(np.mean(zp[:-k] * zb[k:]))       # PAXG at t−k vs BTC at t
        c_b_leads = float(np.mean(zb[:-k] * zp[k:]))       # BTC at t−k vs PAXG at t
        if abs(c_p_leads) > thresh:
            pred_b += c_p_leads * (rb.std() / rp.std()) * rp[-k]
            used.append(f"PAXG→BTC lag {k}: ρ {c_p_leads:+.2f}")
        if abs(c_b_leads) > thresh:
            pred_p += c_b_leads * (rp.std() / rb.std()) * rb[-k]
            used.append(f"BTC→PAXG lag {k}: ρ {c_b_leads:+.2f}")
    rel = pred_b - pred_p                                    # predicted next ratio move
    d_coeff = float(np.mean(zb * zp))                        # contemporaneous coupling
    direction = int(np.sign(rel)) if abs(rel) * 1e4 >= 0.2 else 0
    edge = min(4.0, abs(rel) * 1e4) if direction else 0.0
    return {
        "key": key, "section": sec, "name": name, "value": rel * 1e4, "unit": " bp", "direction": direction, "edge_bps": edge,
        "active": True, "source": "tape", "coupling": d_coeff, "lags_used": used, "samples": n,
        "logic": (f"∂u/∂t = D∇²u on the two legs: {n} returns on a {DIFFUSION_GRID_S:.0f} s grid, contemporaneous ρ = {d_coeff:+.2f}, "
                  f"significance |ρ| > 2/√n = {thresh:.2f}; "
                  + (f"leads: {'; '.join(used)}; predicted next-step BTC {pred_b * 1e4:+.2f} bp − PAXG {pred_p * 1e4:+.2f} bp = {rel * 1e4:+.2f} bp"
                     if used else "no lag is significant: the legs diffuse independently this window")),
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
    now = lr[t >= t_end - 60.0]
    span = max(60.0, t_end - t[0])
    var_now = float(np.sum(now ** 2)) if now.size else 0.0
    var_base = float(np.sum(lr ** 2)) * 60.0 / span
    if var_base <= 0:
        return _inactive(key, sec, name, "no price variance in the baseline", value=1.0)
    t_rel = var_now / var_base
    drag = max(0.0, min(0.6, (t_rel - 1.0) / (TEMPERATURE_HOT - 1.0) * 0.6)) if t_rel > 1.0 else 0.0
    sigma60_bps = math.sqrt(var_now) * 1e4
    return {
        "key": key, "section": sec, "name": name, "value": t_rel, "unit": " T", "direction": 0, "edge_bps": 0.0,
        "active": True, "source": "tape", "drag": drag, "sigma_60s_bps": sigma60_bps, "prints_60s": int(now.size),
        "logic": (f"½k·T ≡ ½⟨v²⟩: realised variance last 60 s = {var_now:.3e} ({now.size} prints) vs 60 s share of the 10-min baseline "
                  f"{var_base:.3e} ⇒ T_rel = {t_rel:.2f}; σ₆₀ = {sigma60_bps:.1f} bp; "
                  f"drag on every directional vote = {drag:.2f}" + (" (shock: votes shrink)" if drag > 0 else " (cool tape)")),
    }
