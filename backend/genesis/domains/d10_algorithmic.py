"""Domain 10 - algorithmic information market complexity (210 formulas).

Returns are symbolised (binary up/down, or ternary up/flat/down with a band
of ±¼σ) and parsed with Lempel-Ziv 76; c(n)·log_a(n)/n → 1 for random
strings.  Block entropies, Markov predictors, runs tests, compression and
MDL model order give the remaining sub-categories.
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from backend.genesis import ops
from backend.genesis.domains._common import finish, momentum_sign, scale_of
from backend.genesis.spec import grid, make_variants

D = 10
N_LZ = (16, 32, 64, 128, 256, 384, 512)
SOURCES = ("binary", "ternary", "volume", "imbalance")


def _symbols(frame, source):
    key = ("d10sym", source)
    hit = frame.cache.get(key)
    if hit is not None:
        return hit
    if source == "ternary":
        band = 0.25 * scale_of(frame, 21)
        sym, a = ops.ternarize(frame.ret / band, 1.0), 3
    elif source == "volume":
        sym, a = ops.binarize(frame.vol_ret), 2
    elif source == "imbalance":
        sym, a = ops.binarize(np.nan_to_num(frame.imbalance)), 2
    else:
        sym, a = ops.binarize(frame.ret), 2
    frame.cache[key] = (sym, a)
    return sym, a


def _stride(frame, n):
    """Long symbol windows change slowly: evaluate every n/32 candles and hold
    (the live reading at the last candle is always computed exactly)."""
    return max(1, n // 32) if frame.eval_tail > 1 else 1


def _positions(frame, n):
    start = max(n - 1, frame.n - int(frame.eval_tail))
    st = _stride(frame, n)
    pos = list(range(start, frame.n, st))
    if pos and pos[-1] != frame.n - 1:
        pos.append(frame.n - 1)
    return pos


def _hold(out):
    """Forward-fill the strided evaluations."""
    last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def _lz_series(frame, n, source):
    key = ("d10lz", source, n)
    hit = frame.cache.get(key)
    if hit is None:
        sym, a = _symbols(frame, source)
        out = ops.nan(frame.n)
        norm_ = math.log(n, a) / n
        for t in _positions(frame, n):
            out[t] = ops.lz76(sym[t - n + 1: t + 1]) * norm_
        start = max(n - 1, frame.n - int(frame.eval_tail))
        out[start:] = _hold(out[start:])
        frame.cache[key] = out
        hit = out
    return hit


def k_lzmc(frame, n, norm, source="binary", **_):
    raw = _lz_series(frame, n, source)
    gate = np.where(raw < 0.4, 1.0, np.where(raw > 0.8, 0.0, (0.8 - raw) / 0.4))
    sig = gate * momentum_sign(frame)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_ami(frame, n, norm, **_):
    a = ops.binarize(frame.ret)
    b = ops.binarize(frame.other_ret)
    joint = a * 2 + b
    out = ops.nan(frame.n)
    norm_ab = math.log(n, 2) / n
    for t in _positions(frame, n):
        if np.isnan(frame.other_ret[t - n + 1: t + 1]).any():
            continue
        la = ops.lz76(a[t - n + 1: t + 1]) * norm_ab
        lb = ops.lz76(b[t - n + 1: t + 1]) * norm_ab
        lab = ops.lz76(joint[t - n + 1: t + 1]) * (math.log(n, 4) / n)
        out[t] = la + lb - lab
    out = _hold(out)
    sig = np.clip(out, 0, 1) * ops.sign(frame.other_ret) * np.minimum(1, np.abs(frame.other_ret) / (ops.rstd(frame.other_ret, n) + 1e-9))
    return finish(frame, sig, out, norm, n, 1.0)


def _markov_predict(sym, a, n, k, t):
    seg = sym[t - n + 1: t + 1]
    if len(seg) < k + 2:
        return np.nan, np.nan
    ctx = tuple(seg[-k:]) if k else ()
    counts = np.zeros(a)
    for i in range(k, len(seg)):          # context seg[i-k:i] -> next symbol seg[i]
        if tuple(seg[i - k: i]) == ctx:
            counts[seg[i]] += 1
    tot = counts.sum()
    if tot < 3:
        return np.nan, tot
    p = (counts + 0.5) / (tot + 0.5 * a)
    return (p[-1] - p[0]), tot


def k_block_entropy(frame, n, norm, k=2, source="binary", **_):
    sym, a = _symbols(frame, source)
    out = ops.nan(frame.n)
    rate = ops.nan(frame.n)
    for t in _positions(frame, n):
        seg = sym[t - n + 1: t + 1]
        # block entropies H_k and H_{k+1}
        hs = []
        for kk in (k, k + 1):
            blocks = np.lib.stride_tricks.sliding_window_view(seg, kk)
            codes = (blocks * (a ** np.arange(kk))).sum(axis=1)
            _, cnt = np.unique(codes, return_counts=True)
            p = cnt / cnt.sum()
            hs.append(float(-(p * np.log2(p)).sum()))
        rate[t] = hs[1] - hs[0]
        pred, _tot = _markov_predict(sym, a, n, k, t)
        out[t] = pred
    out, rate = _hold(out), _hold(rate)
    gate = np.clip(1.0 - rate / math.log2(a), 0, 1)
    sig = out * (0.5 + gate)
    return finish(frame, sig, rate, norm, n, 1.0)


def k_conditional(frame, n, norm, **_):
    p = ops.binarize(frame.ret)
    v = ops.binarize(frame.vol_ret)
    joint = p * 2 + v
    out = ops.nan(frame.n)
    for t in _positions(frame, n):
        lv = ops.lz76(v[t - n + 1: t + 1]) * math.log(n, 2) / n
        lpv = ops.lz76(joint[t - n + 1: t + 1]) * math.log(n, 4) / n
        out[t] = lpv - lv  # K(price | volume)
    out = _hold(out)
    with np.errstate(all="ignore"):
        corr = ops.rmean(np.sign(frame.ret) * np.sign(frame.vol_ret), n)
    sig = np.clip(1 - out, 0, 1) * np.sign(corr) * ops.sign(frame.vol_ret)
    return finish(frame, sig, out, norm, n, 1.0)


def k_kolmogorov(frame, n, norm, levels=8, **_):
    z = frame.ret / scale_of(frame, 55)
    q = np.clip(np.round(z * 2) + levels // 2, 0, levels - 1).astype(np.uint8)
    out = ops.nan(frame.n)
    for t in _positions(frame, n):
        seg = q[t - n + 1: t + 1].tobytes()
        out[t] = len(zlib.compress(seg, 9)) / max(1, len(seg))
    out = _hold(out)
    rnd = ops.rmean(out, 55)
    sig = np.clip((rnd - out) * 10, -1, 1) * momentum_sign(frame)
    return finish(frame, sig, out, norm, n, 1.0)


def k_effective_dim(frame, n, norm, lags=8, **_):
    X = np.vstack([ops.shift(frame.ret, k) for k in range(lags)]).T
    out = ops.nan(frame.n)
    start = max(n + lags, frame.n - int(frame.eval_tail))
    for t in range(start, frame.n):
        M = X[t - n + 1: t + 1]
        if np.isnan(M).any():
            continue
        M = M - M.mean(axis=0)
        ev = np.linalg.eigvalsh(M.T @ M)[::-1]
        ev = np.clip(ev, 0, None)
        if ev.sum() <= 0:
            continue
        cum = np.cumsum(ev) / ev.sum()
        out[t] = (np.searchsorted(cum, 0.9) + 1) / lags
    sig = (1 - out) * momentum_sign(frame)
    return finish(frame, sig, out, norm, n, 1.0)


def k_runs(frame, n, norm, **_):
    s = np.sign(np.nan_to_num(frame.ret))
    s = np.where(s == 0, 1, s)
    change = (s != ops.shift(s, 1)).astype(float)
    runs = ops.rsum(change, n) + 1
    pos = ops.rsum((s > 0).astype(float), n)
    neg = n - pos
    with np.errstate(all="ignore"):
        mu = 2 * pos * neg / n + 1
        var = (mu - 1) * (mu - 2) / (n - 1)
        z = (runs - mu) / (np.sqrt(np.maximum(var, 1e-9)))
    sig = -np.clip(z, -3, 3) / 3 * s
    return finish(frame, sig, z, norm, n, 1.0)


def _rules(sym):
    """A small library of 'programs' over the binary symbol stream: each maps the
    last three symbols to a predicted next symbol."""
    s1, s2, s3 = sym, ops.shift(sym, 1), ops.shift(sym, 2)
    return [
        s1, 1 - s1, s2, 1 - s2, s3, 1 - s3,
        (s1 == s2).astype(float) * s1 + (s1 != s2) * (1 - s1),
        ((s1 + s2 + s3) >= 2).astype(float), ((s1 + s2 + s3) <= 1).astype(float),
        (s1 * s2), 1 - (s1 * s2), np.maximum(s1, s2), 1 - np.maximum(s1, s2),
        ((s1 == s2) & (s2 == s3)).astype(float) * (1 - s1) + ((s1 != s2) | (s2 != s3)) * s1,
        s1 * (1 - s3) + (1 - s1) * s3, (s1 + s3) % 2,
    ]


def k_omega(frame, n, norm, **_):
    sym = ops.binarize(frame.ret).astype(float)
    nxt = ops.shift(sym, -1)
    preds = _rules(sym)
    hits = np.vstack([ops.rmean((p == nxt).astype(float), n) for p in preds])  # accuracy up to t−1 … shift by 1
    hits = np.vstack([ops.shift(h, 1) for h in hits])
    good = hits > 0.6
    omega = good.mean(axis=0)   # halting-probability analogue: mass of successful programs
    with np.errstate(all="ignore"):
        vote = np.nansum(np.where(good, (np.vstack(preds) * 2 - 1) * (hits - 0.5), 0.0), axis=0)
    sig = vote / 2.0
    return finish(frame, sig, omega, norm, n, 1.0)


def k_depth(frame, n, norm, source="binary", **_):
    sym, a = _symbols(frame, source)
    out = ops.nan(frame.n)
    rng = np.random.default_rng(7)
    for t in _positions(frame, n):
        seg = sym[t - n + 1: t + 1]
        c = ops.lz76(seg)
        cs = np.mean([ops.lz76(rng.permutation(seg)) for _ in range(3)])
        out[t] = (cs - c) / max(1.0, cs)   # structure beyond the symbol frequencies
    out = _hold(out)
    sig = np.clip(out * 4, -1, 1) * momentum_sign(frame)
    return finish(frame, sig, out, norm, n, 1.0)


def k_sophistication(frame, n, norm, source="binary", **_):
    sym, a = _symbols(frame, source)
    out = ops.nan(frame.n)
    order = ops.nan(frame.n)
    for t in _positions(frame, n):
        seg = sym[t - n + 1: t + 1]
        best, best_k, best_pred = np.inf, 0, np.nan
        for k in range(0, 4):
            if len(seg) < k + 4:
                break
            ctxs = {}
            for i in range(k, len(seg) - 1):
                ctxs.setdefault(tuple(seg[i - k: i]), np.zeros(a))[seg[i]] += 1
            ll = 0.0
            for cnt in ctxs.values():
                p = (cnt + 0.5) / (cnt.sum() + 0.5 * a)
                ll += float((cnt * np.log2(p)).sum())
            mdl = -ll + 0.5 * (a - 1) * (a ** k) * math.log2(max(2, len(seg)))
            if mdl < best:
                best, best_k = mdl, k
                cnt = ctxs.get(tuple(seg[len(seg) - k:]) if k else ())
                if cnt is not None and cnt.sum() >= 2:
                    p = (cnt + 0.5) / (cnt.sum() + 0.5 * a)
                    best_pred = p[-1] - p[0]
                else:
                    best_pred = np.nan
        out[t] = best_pred
        order[t] = best_k
    out, order = _hold(out), _hold(order)
    return finish(frame, out, order, norm, n, 1.0)


def variants():
    g_lz = grid(windows=N_LZ, extra=dict(source=SOURCES))
    return (
        make_variants(D, 1, "LZ complexity", "Lempel-Ziv market complexity (LZMC)", k_lzmc,
                      "c(s)·log_a(n)/n from the LZ76 exhaustive parse of the symbolised returns",
                      "LZMC > 0.8 random → don't trade; < 0.4 structured → patterns exploitable (follow the last move)", 2, g_lz)
        + make_variants(D, 2, "algorithmic mutual information", "Algorithmic mutual information BTC↔PAXG", k_ami,
                        "LZ(X) + LZ(Y) − LZ(X,Y) over the two legs' sign strings (joint alphabet 4)",
                        "high AMI = synchronised legs → follow the other leg's last move", 3, grid(windows=N_LZ))
        + make_variants(D, 3, "block entropy", "Block-entropy-rate Markov predictor", k_block_entropy,
                        "h = H_{k+1} − H_k of k-blocks; prediction P(up | last k symbols) − P(down | ·) with add-½ smoothing", "", 3,
                        grid(windows=N_LZ, extra=dict(k=(1, 2, 3), source=("binary", "ternary", "binary"))))
        + make_variants(D, 4, "conditional complexity", "K(price | volume)", k_conditional,
                        "LZ(price, volume) − LZ(volume): the part of price not computable from volume", "", 3, grid(windows=N_LZ))
        + make_variants(D, 5, "Kolmogorov complexity", "Compression-ratio complexity", k_kolmogorov,
                        "zlib-9 compressed size / raw size of the 8-level quantised returns", "", 2,
                        grid(windows=N_LZ, extra=dict(levels=(4, 8, 16))))
        + make_variants(D, 6, "effective dimension", "Effective dimension of the lag space", k_effective_dim,
                        "principal components needed for 90 % of the variance of the (n × lags) lagged-return matrix, / lags", "", 3,
                        grid(extra=dict(lags=(4, 8, 13))))
        + make_variants(D, 7, "Martin-Löf randomness", "Wald-Wolfowitz runs statistic", k_runs,
                        "z of the number of sign runs against its expectation 2 n₊ n₋ / n + 1",
                        "too few runs (z < 0) = persistence → follow; too many = alternation → fade", 2)
        + make_variants(D, 8, "Chaitin Ω", "Halting mass of successful programs", k_omega,
                        "share of a 16-rule program library with > 60 % accuracy over the window; their accuracy-weighted vote", "", 2)
        + make_variants(D, 9, "logical depth", "Logical depth (structure beyond frequencies)", k_depth,
                        "(LZ of three shuffles − LZ of the sequence) / LZ shuffled", "", 3, grid(windows=N_LZ, extra=dict(source=SOURCES)))
        + make_variants(D, 10, "sophistication", "MDL model-order predictor", k_sophistication,
                        "Markov order k ∈ {0..3} chosen by MDL (−log-lik + ½(a−1)a^k log₂ n); that model's P(up) − P(down)", "", 3,
                        grid(windows=N_LZ, extra=dict(source=("binary", "ternary"))))
    )
