"""Domain 9 - categorical market morphisms (210 formulas).

Timeframes are functors F_n from market states to readings; a natural
transformation η: F_n ⇒ F_m is natural when η_X = F_m(X) − F_n(X) is
consistent across states X.  Overlapping windows form a (pre)sheaf whose
gluing condition is checked on the overlap; limits/colimits fuse timeframes;
the Yoneda embedding represents a state by its similarities to every other.
"""
from __future__ import annotations

import numpy as np

from backend.genesis import ops
from backend.genesis.domains._common import finish, scale_of
from backend.genesis.spec import grid, make_variants

D = 9
PAIRS = ((3, 8), (5, 13), (8, 21), (13, 34), (21, 55), (34, 89), (55, 144))
TFS = (3, 5, 8, 13, 21, 34)


def _composite(frame, n):
    """One timeframe's composite reading: standardised drift, flow and VWAP position."""
    key = ("d9comp", n)
    hit = frame.cache.get(key)
    if hit is not None:
        return hit
    s = scale_of(frame, n)
    drift = ops.rmean(frame.ret, n) / s * np.sqrt(n)
    with np.errstate(all="ignore"):
        flow = ops.rmean((frame.buy_vol - frame.sell_vol) / (frame.buy_vol + frame.sell_vol + 1e-12), n)
        vw = (frame.close - ops.rmean(frame.vwap, n)) / frame.close / (s * np.sqrt(n))
    parts = [np.tanh(drift / 2), np.nan_to_num(np.tanh(flow * 3), nan=0.0) * (~np.isnan(frame.buy_vol)), np.tanh(vw)]
    out = np.nanmean(np.vstack(parts), axis=0)
    frame.cache[key] = out
    return out


def k_ints(frame, n, norm, short=5, long=20, **_):
    Fs, G = _composite(frame, short), _composite(frame, long)
    eta = G - Fs
    with np.errstate(all="ignore"):
        sd = ops.rstd(eta, n)
        ints = sd / (np.abs(ops.rmean(eta, n)) + sd + 1e-9)   # 0 = η keeps one sign (natural), 1 = pure noise
    gate = np.where(ints < 0.3, 1.0, np.where(ints > 0.7, 0.0, (0.7 - ints) / 0.4))
    sig = gate * np.sign(G + Fs) * np.minimum(1, np.abs(G + Fs))
    return finish(frame, sig, ints, norm, n, 1.0)


def k_sheaf(frame, n, norm, overlap=0.33, **_):
    L = max(4, n)
    o = max(2, int(round(L * overlap)))
    shiftB = L - o                         # window B starts shiftB candles after A
    # section of A over the overlap: overlap drift standardised by A's scale; same for B
    sA = ops.rstd(ops.shift(frame.ret, shiftB), L) + 1e-9      # A = candles [t−shiftB−L+1, t−shiftB] ∪ overlap … (A ends at t−shiftB+o)
    sB = ops.rstd(frame.ret, L) + 1e-9
    ov = ops.rmean(ops.shift(frame.ret, L - o), o)             # overlap region = oldest o candles of B
    with np.errstate(all="ignore"):
        secA, secB = np.tanh(ov / sA * np.sqrt(o)), np.tanh(ov / sB * np.sqrt(o))
        sci = 1.0 - np.abs(secA - secB) / 2.0
    direction = _composite(frame, L)
    sig = np.where(sci > 0.8, 1.0, np.where(sci < 0.5, 0.0, (sci - 0.5) / 0.3)) * direction
    return finish(frame, sig, sci, norm, n, 1.0)


def _stack(frame):
    key = ("d9stack",)
    hit = frame.cache.get(key)
    if hit is None:
        hit = np.vstack([_composite(frame, k) for k in TFS])
        frame.cache[key] = hit
    return hit


def k_limit(frame, n, norm, **_):
    S = _stack(frame)
    with np.errstate(all="ignore"):
        agree = (np.sign(S) == np.sign(S[0])).all(axis=0)
        lim = np.where(agree, np.sign(S[0]) * np.nanmin(np.abs(S), axis=0), 0.0)
    raw = np.nanmin(np.abs(S), axis=0)
    return finish(frame, ops.rmean(lim, max(1, n // 5)), raw, norm, n, 1.0)


def k_colimit(frame, n, norm, **_):
    S = _stack(frame)
    with np.errstate(all="ignore"):
        idx = np.nanargmax(np.abs(np.nan_to_num(S)), axis=0)
        col = S[idx, np.arange(S.shape[1])]
        agreement = np.abs(np.nanmean(np.sign(S), axis=0))
    sig = col * agreement
    return finish(frame, ops.rmean(sig, max(1, n // 5)), col, norm, n, 1.0)


def k_adjunction(frame, n, norm, **_):
    """price ⊣ volume: regress price drift on volume drift over the window; the
    counit residual is the price move volume does not account for."""
    pm = ops.rmean(frame.ret, 3) / scale_of(frame, n)
    vm = ops.rmean(frame.vol_ret, 3) / (ops.rstd(frame.vol_ret, n) + 1e-9)
    with np.errstate(all="ignore"):
        cov = ops.rmean(pm * vm, n) - ops.rmean(pm, n) * ops.rmean(vm, n)
        beta = cov / (ops.rstd(vm, n) ** 2 + 1e-9)
        resid = pm - beta * vm
    sig = resid
    return finish(frame, sig, beta, norm, n, 1.0)


def k_monad(frame, n, norm, **_):
    E = ops.ema(frame.close, n)
    EE = ops.ema(E, n)
    with np.errstate(all="ignore"):
        mu = (E - EE) / frame.close / (scale_of(frame, n) * np.sqrt(n))   # μ: T² ⇒ T deviation
    return finish(frame, mu, E - EE, norm, n, 1.0)


def k_kan(frame, n, norm, short=5, **_):
    c = _composite(frame, short)
    lan, ran = ops.rmax(c, n), ops.rmin(c, n)
    sig = 0.5 * (lan + ran)
    return finish(frame, sig, lan - ran, norm, n, 1.0)


def k_yoneda(frame, n, norm, pattern=5, **_):
    """Hom(−, X): correlation of the last ``pattern`` returns with every earlier
    pattern in the window; similarity-weighted mean of what followed them."""
    P = ops.windows(frame.ret, pattern)
    nxt = ops.shift(frame.ret, -1)
    raw = ops.nan(frame.n)
    n = max(n, 21)
    start = max(n + pattern, frame.n - int(frame.eval_tail))
    Pz = (P - np.nanmean(P, axis=1, keepdims=True)) / (np.nanstd(P, axis=1, keepdims=True) + 1e-12)
    for t in range(start, frame.n):
        cur = Pz[t]
        if np.isnan(cur).any():
            continue
        past = Pz[t - n: t - 1]
        fut = nxt[t - n: t - 1]
        ok = ~(np.isnan(past).any(axis=1) | np.isnan(fut))
        if ok.sum() < 5:
            continue
        sim = past[ok] @ cur / pattern
        w = np.clip(sim, 0, None) ** 2
        if w.sum() <= 0:
            continue
        raw[t] = float((w * fut[ok]).sum() / w.sum())
    sig = raw / scale_of(frame, n) * 2
    return finish(frame, sig, raw, norm, n, 1.0)


def k_topos(frame, n, norm, **_):
    """Subobject classifier over nested windows: the truth value of 'up' is the
    largest k such that all timeframes ≤ k read up, divided by the number of
    timeframes (a Heyting-algebra of opens)."""
    S = _stack(frame)
    up = S > 0.05
    dn = S < -0.05
    K = len(TFS)

    def truth(mask):
        out = np.zeros(mask.shape[1])
        alive = np.ones(mask.shape[1], dtype=bool)
        for k in range(K):
            alive &= mask[k]
            out += alive
        return out / K
    tu, td = truth(up), truth(dn)
    sig = tu - td
    return finish(frame, ops.rmean(sig, max(1, n // 5)), tu, norm, n, 1.0)


def k_functor(frame, n, norm, lag=1, **_):
    """Cross-asset functor: the other leg's composite, transported along the
    naturality square with the rolling sign of the legs' co-movement."""
    s_o = ops.rstd(frame.other_ret, n) + 1e-9
    other_comp = np.tanh(ops.rmean(frame.other_ret, 5) / s_o * np.sqrt(5) / 2)
    with np.errstate(all="ignore"):
        corr = (ops.rmean(frame.ret * ops.shift(frame.other_ret, lag), n) - ops.rmean(frame.ret, n) * ops.rmean(ops.shift(frame.other_ret, lag), n)) / (
            frame.roll("ret_std", n) * s_o + 1e-12)
    sig = ops.shift(other_comp, lag) * np.clip(corr * 3, -1, 1)
    return finish(frame, sig, corr, norm, n, 1.0)


def variants():
    g_pairs = grid(extra=dict(short=[p[0] for p in PAIRS], long=[p[1] for p in PAIRS]))
    return (
        make_variants(D, 1, "natural transformation", "Indicator natural transformation score (INTS)", k_ints,
                      "η_t = G_long(t) − F_short(t); INTS = std_n(η) / mean_n|η|",
                      "INTS < 0.3 timeframes aligned → trade with conviction; > 0.7 conflicting → stay out", 2, g_pairs)
        + make_variants(D, 2, "sheaf cohomology", "Sheaf consistency index (SCI)", k_sheaf,
                        "sections of two overlapping windows restricted to the overlap; SCI = 1 − |s_A − s_B|/2", 
                        "SCI > 0.8 the data glue → reliable; < 0.5 contradictory → no trade", 2,
                        grid(extra=dict(overlap=(0.25, 0.33, 0.5))))
        + make_variants(D, 3, "limit", "Limit (product) fusion", k_limit,
                        "when every timeframe agrees in sign, the smallest |reading| (the cone apex); else 0", "", 2)
        + make_variants(D, 4, "colimit", "Colimit (coproduct) fusion", k_colimit,
                        "the largest |reading| across timeframes, scaled by their sign agreement", "", 2)
        + make_variants(D, 5, "adjunction", "Price ⊣ volume counit residual", k_adjunction,
                        "price drift − β·volume drift (β from the window regression): the move volume does not explain", "", 2)
        + make_variants(D, 6, "monad", "Monad multiplication μ: T² ⇒ T", k_monad,
                        "EMA_n − EMA_n(EMA_n) in σ√n units (the DEMA correction term)", "", 2)
        + make_variants(D, 7, "Kan extension", "Left/right Kan extension average", k_kan,
                        "Lan = max, Ran = min of the short composite over the long window; ½(Lan + Ran)", "", 2,
                        grid(extra=dict(short=(3, 5, 8))))
        + make_variants(D, 8, "Yoneda", "Yoneda similarity predictor", k_yoneda,
                        "Hom(−, X): squared positive correlation of the last p returns with every earlier pattern; weighted mean of what followed", "", 3,
                        grid(extra=dict(pattern=(3, 5, 8))))
        + make_variants(D, 9, "topos", "Heyting truth of 'up'", k_topos,
                        "largest k with all timeframes ≤ k reading up (minus the same for down), / K", "", 2)
        + make_variants(D, 10, "enriched / derived (cross-asset functor)", "Cross-asset functor transport", k_functor,
                        "other leg's composite transported with sign(corr(r_t, r_other,t−lag))", "", 2,
                        grid(extra=dict(lag=(0, 1, 2))))
    )
