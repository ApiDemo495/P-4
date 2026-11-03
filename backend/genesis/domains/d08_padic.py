"""Domain 8 - p-adic analysis of price movements (210 formulas).

Price changes in ticks are integers; v_p(k) is the exponent of the prime p in
k, |k|_p = p^{−v_p(k)} the p-adic absolute value, d_p(i, j) = |ΔP_i − ΔP_j|_p
the ultrametric.  Lags k likewise live on the p-adic tree, which gives a
natural hierarchical weighting of look-backs (multiples of p^m are 'closer
to zero').
"""
from __future__ import annotations

import numpy as np

from backend.genesis import ops
from backend.genesis.domains._common import finish, momentum_sign, scale_of
from backend.genesis.spec import grid, make_variants

D = 8
PRIMES = (2, 3, 5, 7, 11, 13, 17)


def _ticks(frame, what="price"):
    if what == "volume":
        return np.round(np.nan_to_num(frame.volume) * 1e4)
    if what == "trades":
        return np.nan_to_num(frame.trades)
    return np.round(np.nan_to_num(frame.close - frame.open) / frame.tick_size)


def _v(frame, p, what="price"):
    return ops.padic_valuation(_ticks(frame, what), p)


def k_pavt(frame, n, norm, p=2, what="price", **_):
    v = _v(frame, p, what)
    raw = ops.rmean(v, n) - ops.rmean(v, 5 * n)
    # round changes = algorithmic = fade; irregular = organic = follow
    sig = np.where(raw > 0.5, -1.0, np.where(raw < -0.25, 1.0, -raw * 2.0)) * momentum_sign(frame) * np.minimum(1, np.abs(raw) * 2)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_ultrametric(frame, n, norm, p=2, **_):
    t = _ticks(frame)
    W = ops.windows(t, n + 1)
    cur = W[:, -1:]
    with np.errstate(all="ignore"):
        diff = np.abs(W[:, :-1] - cur)
        # v_p of each difference, vectorised over the window
        v = np.zeros_like(diff)
        live = diff > 0
        d = diff.copy()
        while True:
            m = live & (np.mod(d, p) == 0)
            if not m.any():
                break
            v[m] += 1
            d[m] = d[m] // p
        dist = np.where(diff == 0, 0.0, p ** (-v))
        raw = np.nanmean(dist, axis=1)   # 1 = nothing in the window resembles this candle (p-adically)
    raw[np.isnan(W).any(axis=1)] = np.nan
    sig = (raw - 0.6) * 3.0 * momentum_sign(frame)   # a p-adically unique candle = anomaly: follow it
    return finish(frame, sig, raw, norm, n, 1.0)


def k_interpolation(frame, n, norm, p=2, **_):
    k = np.arange(1, n + 1, dtype=float)
    w = p ** (-ops.padic_valuation(k, p))   # |k|_p: lags at multiples of p^m count less
    W = ops.windows(frame.ret, n)[:, ::-1]
    with np.errstate(all="ignore"):
        raw = np.nansum(W * w, axis=1) / w.sum()
    raw[np.isnan(W).any(axis=1)] = np.nan
    sig = raw / scale_of(frame, n) * np.sqrt(n)
    return finish(frame, sig, raw, norm, n, 1.0)


def _level(p, m, cap=128):
    while m > 1 and p ** m > cap:
        m -= 1
    return m


def k_wavelet(frame, n, norm, p=2, m=2, **_):
    m = _level(p, m)
    big, small = p ** m, p ** (m - 1)
    raw = ops.rmean(frame.ret, small) - ops.rmean(frame.ret, big)   # p-adic Haar detail at level m
    sig = raw / scale_of(frame, n) * np.sqrt(small)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_volkenborn(frame, n, norm, p=2, m=2, **_):
    big = p ** _level(p, m)
    raw = ops.rsum(frame.ret, big) / big   # p^{-m} Σ_{k<p^m} f(k)
    sig = raw / scale_of(frame, n) * np.sqrt(big)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_hensel(frame, n, norm, p=2, k=2, **_):
    """Residue-class pattern predictor: lift the current ΔP residue mod p^k and
    average the returns that followed the same residue class in the window."""
    t = _ticks(frame)
    mod = p ** k
    res = np.mod(t, mod)
    nxt = ops.shift(frame.ret, -1)
    R = ops.windows(res, n + 1)[:, :-1]
    X = ops.windows(nxt, n + 1)[:, :-1]
    cur = res[:, None]
    with np.errstate(all="ignore"):
        same = (R == cur)
        cnt = same.sum(axis=1)
        raw = np.nansum(np.where(same, X, 0.0), axis=1) / np.maximum(cnt, 1)
        conf = np.sqrt(cnt / n)
    raw[np.isnan(R).any(axis=1)] = np.nan
    sig = raw / scale_of(frame, n) * conf * 2
    return finish(frame, sig, raw, norm, n, 1.0)


def _legendre(k, p):
    k = np.asarray(k, dtype=np.int64) % p
    out = np.zeros(len(k))
    for i, a in enumerate(k):
        if a == 0:
            continue
        out[i] = 1.0 if pow(int(a), (p - 1) // 2, p) == 1 else -1.0
    return out


def k_lfunction(frame, n, norm, p=3, s=1.0, **_):
    k = np.arange(1, n + 1)
    chi = _legendre(k, p) if p > 2 else np.where(k % 2 == 1, 1.0, -1.0)
    w = chi * k ** (-s)
    W = ops.windows(frame.ret, n)[:, ::-1]
    with np.errstate(all="ignore"):
        raw = np.nansum(W * w, axis=1)
    raw[np.isnan(W).any(axis=1)] = np.nan
    sig = raw / scale_of(frame, n) / np.sqrt((w ** 2).sum())
    return finish(frame, sig, raw, norm, n, 1.0)


def k_iwasawa(frame, n, norm, p=2, **_):
    levels = [p ** m for m in range(1, 6) if p ** m <= max(n, p)]
    if len(levels) < 2:
        levels = [p, p * p]
    sums = [ops.rsum(frame.ret, L) for L in levels]
    with np.errstate(all="ignore"):
        logs = np.vstack([ops.safe_log(np.abs(s_) + 1e-12) for s_ in sums])
        X = np.arange(len(levels), dtype=float)
        X = X - X.mean()
        slope = (X[:, None] * (logs - np.nanmean(logs, axis=0))).sum(axis=0) / (X ** 2).sum()
    # λ-invariant proxy: growth of |Σ r| along the tower, signed by the top-level sum
    sig = np.clip(slope, -2, 2) / 2.0 * np.sign(sums[-1])
    return finish(frame, sig, slope, norm, n, 1.0)


def k_hodge(frame, n, norm, p=2, **_):
    v = _v(frame, p)
    cls = np.minimum(v, 2)
    nxt = ops.shift(frame.ret, -1)
    Cw = ops.windows(cls, n + 1)[:, :-1]
    Xw = ops.windows(nxt, n + 1)[:, :-1]
    with np.errstate(all="ignore"):
        same = Cw == cls[:, None]
        cnt = same.sum(axis=1)
        raw = np.nansum(np.where(same, Xw, 0.0), axis=1) / np.maximum(cnt, 1)
    raw[np.isnan(Cw).any(axis=1)] = np.nan
    sig = raw / scale_of(frame, n) * np.sqrt(cnt / n) * 2
    return finish(frame, sig, raw, norm, n, 1.0)


def k_hecke(frame, n, norm, p=2, **_):
    """Hecke operator on the return 'q-expansion': (T_p a)_k = a_{pk} + a_{k/p}
    (weight-1 normalisation); the Hecke-transformed window sum vs the plain sum."""
    n = int(min(n, max(3, 300 // p)))
    W = ops.windows(frame.ret, n * p + 1)[:, ::-1]   # a_0 = latest
    idx = np.arange(1, n + 1)
    with np.errstate(all="ignore"):
        a_pk = W[:, idx * p]
        a_kp = np.where(idx % p == 0, W[:, np.maximum(idx // p, 0)], 0.0)
        hecke = np.nansum(a_pk + a_kp, axis=1)
        plain = np.nansum(W[:, idx], axis=1)
        raw = hecke - plain
    raw[np.isnan(W).any(axis=1)] = np.nan
    sig = raw / scale_of(frame, n) / np.sqrt(n)
    return finish(frame, sig, raw, norm, n, 1.0)


def variants():
    gp = grid(extra=dict(p=PRIMES))
    return (
        make_variants(D, 1, "p-adic valuation", "p-adic valuation trend (PAVT)", k_pavt,
                      "v_p(ΔP in ticks); PAVT = mean_n v_p − mean_5n v_p",
                      "PAVT > 0.5 unusually round moves = algorithmic = fade; < −0.25 irregular = organic = follow", 2,
                      grid(extra=dict(p=PRIMES, what=("price", "price", "volume"))))
        + make_variants(D, 2, "ultrametric distance", "Ultrametric regime tree depth (UMRT)", k_ultrametric,
                        "mean over the window of d_p = p^{−v_p(ΔP_t − ΔP_i)}; 1 = the candle shares no p-adic digits with any recent one",
                        "a p-adically unique candle is an anomaly - breakout or crash in its own direction", 2, gp)
        + make_variants(D, 3, "p-adic interpolation", "|k|_p-weighted momentum", k_interpolation,
                        "Σ_k r_{t−k} |k|_p / Σ|k|_p - lags on deeper branches of the p-adic tree weigh less", "", 2, gp)
        + make_variants(D, 4, "p-adic wavelet", "p-adic Haar detail", k_wavelet,
                        "mean_{p^{m−1}} r − mean_{p^m} r: the level-m detail coefficient on the p-ary tree", "", 2,
                        grid(extra=dict(p=PRIMES, m=(1, 2, 3))))
        + make_variants(D, 5, "p-adic integration", "Volkenborn block mean", k_volkenborn,
                        "p^{−m} Σ_{k<p^m} r_{t−k} in σ√(p^m) units", "", 2, grid(extra=dict(p=PRIMES, m=(1, 2, 3))))
        + make_variants(D, 6, "Hensel lifting", "Residue-class lift predictor", k_hensel,
                        "mean next return after window candles with ΔP ≡ ΔP_t (mod p^k), weighted by √(count/n)", "", 2,
                        grid(extra=dict(p=PRIMES, k=(1, 2, 3))))
        + make_variants(D, 7, "p-adic L-function", "Character-weighted momentum", k_lfunction,
                        "Σ_k χ(k) k^{−s} r_{t−k} with χ the Legendre symbol mod p (parity for p = 2)", "", 2,
                        grid(extra=dict(p=PRIMES, s=(0.5, 1.0, 1.5))))
        + make_variants(D, 8, "Iwasawa theory", "Tower growth λ", k_iwasawa,
                        "slope of ln|Σ_{k<p^m} r| along the tower m = 1..5, signed by the top sum", "", 2, gp)
        + make_variants(D, 9, "p-adic Hodge", "Valuation-class drift", k_hodge,
                        "mean next return after candles in the same v_p class {0, 1, ≥2}", "", 2, gp)
        + make_variants(D, 10, "p-adic modular form", "Hecke-filtered momentum", k_hecke,
                        "Σ_k (a_{pk} + a_{k/p}) − Σ_k a_k over the return q-expansion", "", 2, gp)
    )
