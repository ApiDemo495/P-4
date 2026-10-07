"""Domain 7 - quantum probability market states (210 formulas).

An indicator bank (momentum over several horizons, order imbalance, taker
flow, VWAP deviation, cross-asset lead) is read as a set of measurements of
a three-state system |bull⟩, |bear⟩, |neutral⟩.  The density matrix ρ has
the classical probabilities on its diagonal and the coherences
ρ_ij = √(p_i p_j)(1 − agreement) off it; S(ρ) = −Tr ρ ln ρ is the von
Neumann entropy.  Everything below is computed per candle from ρ.
"""
from __future__ import annotations

import numpy as np

from backend.genesis import ops
from backend.genesis.domains._common import finish, scale_of
from backend.genesis.spec import grid, make_variants

D = 7
BANKS = ("full", "momentum", "flow")
LN3 = np.log(3.0)


def _bank(frame, n, which="full"):
    """Indicator bank -> (N, m) matrix of readings in [-1, 1]."""
    s = scale_of(frame, n)
    cols = []
    if which in ("full", "momentum"):
        for k in (1, 2, 3, 5, 8, 13):
            cols.append(np.tanh((frame.close - ops.shift(frame.close, k)) / frame.close / (s * np.sqrt(k)) / 2))
        cols.append(np.tanh((frame.close - ops.ema(frame.close, n)) / frame.close / (s * np.sqrt(n)) / 2))
    if which in ("full", "flow"):
        with np.errstate(all="ignore"):
            flow = (frame.buy_vol - frame.sell_vol) / (frame.buy_vol + frame.sell_vol + 1e-12)
        cols.append(ops.rmean(flow, max(2, n // 3)))
        cols.append(ops.rmean(frame.imbalance, max(2, n // 3)))
        cols.append(np.tanh((frame.close - frame.vwap) / frame.close / s / 2))
        cols.append(np.tanh(ops.rmean(frame.other_ret, 3) / (ops.rstd(frame.other_ret, n) + 1e-9)))
    M = np.vstack(cols).T
    return M


def _rho(frame, n, bank="full", band=0.15):
    M = _bank(frame, n, bank)
    valid = ~np.isnan(M)
    cnt = valid.sum(axis=1)
    bull = np.where(valid, M > band, False).sum(axis=1)
    bear = np.where(valid, M < -band, False).sum(axis=1)
    neut = cnt - bull - bear
    with np.errstate(all="ignore"):
        p = np.vstack([bull, bear, neut]).T / np.maximum(cnt, 1)[:, None]
        conf = np.nanmean(np.abs(M), axis=1)
        agreement = np.abs(np.nanmean(np.sign(M) * (np.abs(M) > band), axis=1))
    coh = 1.0 - np.nan_to_num(agreement)
    N = frame.n
    rho = np.zeros((N, 3, 3))
    for i in range(3):
        rho[:, i, i] = p[:, i]
        for j in range(i + 1, 3):
            c = np.sqrt(p[:, i] * p[:, j]) * coh
            rho[:, i, j] = c
            rho[:, j, i] = c
    bad = cnt < 3
    return rho, p, conf, agreement, bad


def _entropy(rho):
    ev = np.linalg.eigvalsh(rho)
    ev = np.clip(ev, 1e-12, None)
    ev = ev / ev.sum(axis=1, keepdims=True)
    return -(ev * np.log(ev)).sum(axis=1)


def k_mdm(frame, n, norm, bank="full", mode="level", **_):
    rho, p, conf, agr, bad = _rho(frame, n, bank)
    S = _entropy(rho)
    dom = np.sign(p[:, 0] - p[:, 1])
    if mode == "trend":
        dS = S - ops.shift(S, 3)
        sig = np.where(dS < 0, -dS, 0.0) * 3.0 * dom     # collapsing superposition: prepare to trade
    else:
        sig = np.where(S < 0.3 * LN3, 1.0, np.where(S > 0.8 * LN3, 0.0, (0.8 * LN3 - S) / (0.5 * LN3))) * dom * conf
    sig = np.asarray(sig, dtype=float)
    sig[bad] = np.nan
    return finish(frame, sig, S, norm, n, 1.0)


def k_qim(frame, n, norm, bank="full", **_):
    M = _bank(frame, n, bank)
    with np.errstate(all="ignore"):
        amp = np.sqrt(np.abs(M))
        phase = np.where(M > 0, 0.0, np.where(M < 0, np.pi, np.pi / 2))
        A = np.nansum(amp * np.exp(1j * phase), axis=1)
        p_q = np.abs(A.real) ** 2 / (np.abs(A) ** 2 + 1e-12)
        p_c = np.nansum(np.where(M > 0, np.abs(M), 0), axis=1) / (np.nansum(np.abs(M), axis=1) + 1e-12)
        qim = (p_q - p_c) * 100.0
        sig = np.sign(A.real) * np.clip(qim / 20.0, -1, 1)
    sig[np.isnan(M).sum(axis=1) > M.shape[1] - 3] = np.nan
    return finish(frame, sig, qim, norm, n, 1.0)


def _sqrtm_psd(R):
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 0, None)
    return (V * np.sqrt(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def k_fidelity(frame, n, norm, bank="full", **_):
    rho, p, conf, agr, bad = _rho(frame, n, bank)
    k = max(1, n // 3)
    sigma = np.roll(rho, k, axis=0)
    sq = _sqrtm_psd(rho)
    inner = sq @ sigma @ sq
    F = np.trace(_sqrtm_psd(inner), axis1=1, axis2=2) ** 2
    F = np.clip(F.real, 0, 1)
    F[:k] = np.nan
    dom = np.sign(p[:, 0] - p[:, 1])
    sig = (1.0 - F) * dom * conf * 2.0   # a state that changed: follow the new dominant side
    sig[bad] = np.nan
    return finish(frame, sig, F, norm, n, 1.0)


def k_bloch(frame, n, norm, bank="full", **_):
    rho, p, conf, agr, bad = _rho(frame, n, bank)
    with np.errstate(all="ignore"):
        pb = p[:, 0] + p[:, 1] + 1e-12
        z = (p[:, 0] - p[:, 1]) / pb
        x = 2 * rho[:, 0, 1] / pb
        r = np.sqrt(x * x + z * z)
    sig = z * r                     # Bloch vector: purity-weighted polarisation
    sig[bad] = np.nan
    return finish(frame, sig, r, norm, n, 1.0)


def k_walk(frame, n, norm, steps=8, **_):
    """Hadamard-type quantum walk on the price lattice; the coin angle each step
    is set by that candle's order imbalance (θ = π/4 + imbalance·π/8)."""
    imb = np.nan_to_num(ops.rmean(frame.imbalance, 2))
    flow = np.nan_to_num((frame.buy_vol - frame.sell_vol) / (frame.buy_vol + frame.sell_vol + 1e-12))
    coin = np.clip(0.5 * imb + 0.5 * flow, -1, 1)
    N = frame.n
    raw = ops.nan(N)
    L = steps
    start = max(steps, N - int(frame.eval_tail))
    for t in range(start, N):
        psi = np.zeros((2 * L + 1, 2), dtype=complex)
        psi[L, 0] = 1 / np.sqrt(2)
        psi[L, 1] = 1j / np.sqrt(2)
        for s in range(steps):
            th = np.pi / 4 + coin[t - steps + 1 + s] * np.pi / 8
            C = np.array([[np.cos(th), np.sin(th)], [np.sin(th), -np.cos(th)]])
            psi = psi @ C.T
            new = np.zeros_like(psi)
            new[1:, 0] = psi[:-1, 0]     # coin 0 moves right (up)
            new[:-1, 1] = psi[1:, 1]     # coin 1 moves left (down)
            psi = new
        prob = (np.abs(psi) ** 2).sum(axis=1)
        raw[t] = prob[L + 1:].sum() - prob[:L].sum()
    sig = raw
    return finish(frame, sig, raw, norm, n, 1.0)


def k_povm(frame, n, norm, bank="full", **_):
    rho, p, conf, agr, bad = _rho(frame, n, bank)
    r = frame.ret / scale_of(frame, n)
    up = ops.rmean((r > 0.3).astype(float), n)
    dn = ops.rmean((r < -0.3).astype(float), n)
    fl = 1 - up - dn
    with np.errstate(all="ignore"):
        E_up = np.stack([np.diag(v) for v in np.vstack([up, dn * 0.2, fl * 0.5]).T])
        E_dn = np.stack([np.diag(v) for v in np.vstack([up * 0.2, dn, fl * 0.5]).T])
        sig = np.trace(rho @ E_up, axis1=1, axis2=2) - np.trace(rho @ E_dn, axis1=1, axis2=2)
    sig = np.asarray(sig) * 2.0
    sig[bad] = np.nan
    return finish(frame, sig, sig, norm, n, 1.0)


def k_channel(frame, n, norm, bank="full", **_):
    rho, p, conf, agr, bad = _rho(frame, n, bank)
    pdep = 1.0 - np.nan_to_num(agr)
    I3 = np.eye(3)[None]
    rho2 = (1 - pdep)[:, None, None] * rho + pdep[:, None, None] * I3 / 3.0
    diag = np.diagonal(rho2, axis1=1, axis2=2)
    sig = (diag[:, 0] - diag[:, 1]) * 2.0
    sig[bad] = np.nan
    return finish(frame, sig, pdep, norm, n, 1.0)


def k_decoherence(frame, n, norm, bank="full", **_):
    rho, p, conf, agr, bad = _rho(frame, n, bank)
    coh = np.abs(rho[:, 0, 1]) + np.abs(rho[:, 0, 2]) + np.abs(rho[:, 1, 2])
    rate = -(coh - ops.shift(coh, 2))
    dom = np.sign(p[:, 0] - p[:, 1])
    sig = np.clip(rate * 4.0, 0, 1) * dom * conf
    sig[bad] = np.nan
    return finish(frame, sig, rate, norm, n, 1.0)


def k_entanglement(frame, n, norm, **_):
    """Two-qubit state of (this leg, other leg) built from their bull amplitudes
    with the rolling correlation as the entangling phase; entropy of the
    reduced density matrix.  Entangled legs: follow the other leg's last move."""
    a = np.clip(0.5 + 0.5 * np.tanh(ops.rmean(frame.ret, 3) / (scale_of(frame, n) * 0.6)), 0.02, 0.98)
    b = np.clip(0.5 + 0.5 * np.tanh(ops.rmean(frame.other_ret, 3) / (ops.rstd(frame.other_ret, n) + 1e-9) / 0.6), 0.02, 0.98)
    with np.errstate(all="ignore"):
        corr = (ops.rmean(frame.ret * frame.other_ret, n) - ops.rmean(frame.ret, n) * ops.rmean(frame.other_ret, n)) / (
            frame.roll("ret_std", n) * ops.rstd(frame.other_ret, n) + 1e-12)
    c = np.nan_to_num(np.clip(corr, -1, 1))
    # |ψ⟩ = √(1−|c|)(|a⟩⊗|b⟩) + √|c| (c>0 ? |00⟩+|11⟩ : |01⟩+|10⟩)/√2
    S = ops.nan(frame.n)
    for t in range(frame.n):
        if np.isnan(a[t]) or np.isnan(b[t]):
            continue
        qa = np.array([np.sqrt(a[t]), np.sqrt(1 - a[t])])
        qb = np.array([np.sqrt(b[t]), np.sqrt(1 - b[t])])
        prod = np.outer(qa, qb)
        bell = (np.eye(2) if c[t] >= 0 else np.fliplr(np.eye(2))) / np.sqrt(2)
        psi = np.sqrt(1 - abs(c[t])) * prod + np.sqrt(abs(c[t])) * bell
        psi /= np.linalg.norm(psi) + 1e-12
        red = psi @ psi.T
        ev = np.clip(np.linalg.eigvalsh(red), 1e-12, 1)
        S[t] = float(-(ev * np.log2(ev)).sum())
    sig = S * np.sign(c) * ops.sign(frame.other_ret) + (1 - S) * np.sign(a - 0.5)
    return finish(frame, sig, S, norm, n, 1.0)


def k_qbayes(frame, n, norm, bank="full", **_):
    """Lüders update: ρ' = M ρ M† / Tr(·) with M = diag(√L) and likelihoods from
    the latest standardised return under the three states."""
    rho, p, conf, agr, bad = _rho(frame, n, bank)
    z = np.nan_to_num(frame.ret / scale_of(frame, n))
    L = np.vstack([np.exp(-0.5 * (z - 0.7) ** 2), np.exp(-0.5 * (z + 0.7) ** 2), np.exp(-0.5 * z ** 2)]).T + 1e-6
    M = np.stack([np.diag(np.sqrt(v)) for v in L])
    post = M @ rho @ M
    tr = np.trace(post, axis1=1, axis2=2) + 1e-12
    post = post / tr[:, None, None]
    d = np.diagonal(post, axis1=1, axis2=2)
    sig = (d[:, 0] - d[:, 1]) * 2.0
    sig[bad] = np.nan
    return finish(frame, sig, d[:, 0], norm, n, 1.0)


def variants():
    g = grid(extra=dict(bank=BANKS))
    return (
        make_variants(D, 1, "density matrix", "Market density matrix (MDM)", k_mdm,
                      "ρ_ii = share of indicators bull/bear/neutral, ρ_ij = √(p_i p_j)(1 − agreement); S = −Tr ρ ln ρ",
                      "S < 0.3 collapsed → trade the dominant side; S > 0.8 superposed → no trade; falling S → prepare", 2,
                      grid(extra=dict(bank=BANKS, mode=("level", "level", "trend"))))
        + make_variants(D, 2, "quantum interference", "Quantum interference momentum (QIM)", k_qim,
                        "amplitudes α_i = √conf_i·e^{iφ_i}, φ = 0/π/π/2; P_q = |Re ΣA|²/|ΣA|²; QIM = (P_q − P_classical)·100",
                        "QIM > +10 constructive interference (signals amplify), < −10 destructive (they cancel)", 2, g)
        + make_variants(D, 3, "quantum fidelity", "Fidelity to the state n/3 candles ago", k_fidelity,
                        "F(ρ,σ) = (Tr √(√ρ σ √ρ))²", "low fidelity = the state changed: follow the new dominant side", 3, g)
        + make_variants(D, 4, "Wigner function (Bloch)", "Bloch polarisation", k_bloch,
                        "two-level reduction: z = p_bull − p_bear, x = 2 Re ρ_12; signal z·|r|", "", 2, g)
        + make_variants(D, 5, "quantum walk", "Quantum walk on the price lattice", k_walk,
                        "discrete-time coined walk, coin angle θ_t = π/4 + (½ imbalance + ½ taker flow)·π/8, P(right) − P(left) after k steps", "", 3,
                        grid(extra=dict(steps=(4, 6, 8, 10, 13, 16, 21))))
        + make_variants(D, 6, "POVM measurement", "POVM up/down effect difference", k_povm,
                        "Tr(ρ E_up) − Tr(ρ E_down) with effects from the window's realised up/down/flat frequencies", "", 2, g)
        + make_variants(D, 7, "quantum channel", "Depolarised state", k_channel,
                        "Λ(ρ) = (1−p)ρ + p·I/3 with p = 1 − agreement; diagonal polarisation after the channel", "", 2, g)
        + make_variants(D, 8, "decoherence rate", "Coherence decay", k_decoherence,
                        "−Δ₂ Σ|ρ_ij| (i<j): positive when the superposition is collapsing", "", 2, g)
        + make_variants(D, 9, "entanglement entropy", "BTC-PAXG entanglement entropy", k_entanglement,
                        "entropy of the reduced density matrix of a two-qubit state with correlation-weighted Bell component", "", 3)
        + make_variants(D, 10, "quantum Bayes", "Lüders-rule posterior", k_qbayes,
                        "ρ' = MρM†/Tr with M = diag(√L(z)); posterior p_bull − p_bear", "", 2, g)
    )
