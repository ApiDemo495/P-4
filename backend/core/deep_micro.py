"""Deep market-microstructure reasoning, from microseconds to the minute.

Round J measured the crowd with *ramps*: "a move of three typical sizes in a
second is a panic".  That is honest but shallow - it says *what* the tape did,
not *why the crowd did it* or *how confident the engine should be about it*.
This module adds the layer underneath: the quantitative-finance formulas that
describe how a crowd behaves at the tick level, run on the same tape, at the
same 0.5 s cadence, and folded into the emotion reading through a Bayesian
filter that shows its work.

Every formula below is textbook microstructure, expressed in the tape's own
units so it reads the same on a dead simulator tape and on a violent live one:

=====  ==========================================  ==================================
band   formula                                     what it tells the crowd model
=====  ==========================================  ==================================
µs     Hawkes branching ratio  n = 1 - 1/sqrt(F)   self-excitation: are prints
       (F = Fano factor of arrival counts)         causing more prints (a cascade)?
µs     microprice  P* = (Pa*Qb + Pb*Qa)/(Qa+Qb)     where the next print is leaning
µs     quote-stuffing ratio                        prints without price = noise
s      VPIN  (Easley / Lopez de Prado / O'Hara)    order-flow toxicity (informed vs
                                                   emotional flow)
s      Kyle's lambda  dP = lambda * signed volume  price impact per unit of flow -
       (least squares, with R^2)                   a thin, pushable tape has a big
                                                   lambda
s      Lo-MacKinlay variance ratio  VR(q)          > 1 momentum, < 1 mean reversion
s..min generalised Hurst exponent  H               persistence of the move
s..min Bandt-Pompe permutation entropy             disorder / predictability
s..min Lillo-Farmer sign memory  gamma             how long the herd keeps trading
                                                   the same way
µs..min Haar wavelet energy spectrum                where the action is - which
                                                   timescale carries the variance
s      three-state HMM forward filter              calm / trend / stress regime
s      momentum-ignition detector                  burst-then-fade = retail bait
=====  ==========================================  ==================================

The output of all of that is (a) a *posterior* over the eight emotions,
obtained by a discrete Bayesian filter with sticky transitions (so belief is
carried from sample to sample and updated by log-likelihood ratios instead of
being recomputed from scratch), (b) a *reasoning chain*: an ordered list of
steps - formula, inputs, value, reading, and which emotions the step pushes -
that the panel renders verbatim, and (c) three new manipulation detectors
(ignition, stuffing, spoofing) that the manipulation read and the fusion
dampener consume.

Everything is a pure function of the tape plus a small ``DeepState`` (the
posterior carried between samples), so the Signal Lock Protocol still holds:
the locked reading of a frozen snapshot is reproducible bit for bit.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import permutations
from typing import Any

import numpy as np

from backend.core import geometry
from backend.core.timebase import format_us, now_us

EMOTION_NAMES = (
    "FEAR", "PANIC", "CAPITULATION", "DENIAL", "HOPE", "EUPHORIA", "FOMO", "COMPLACENCY",
)

#: The three bands the multi-scale formulas are reported on: name, label,
#: the clock the returns are sampled on (0 = every tick) and the aggregation
#: ``q`` for the variance ratio (so VR compares 4 ticks vs 1 tick, 1 s vs
#: 250 ms, and 5 s vs 1 s).
BANDS = (
    ("micro", "microseconds to 250 ms", 0.0, 4),
    ("seconds", "250 ms to 5 s", 250.0, 4),
    ("window", "1 s to the minute", 1000.0, 5),
)

#: Naive-Bayes log-odds table.  Each row is an emotion, each key an evidence
#: variable in [-1, 1] or [0, 1] (see ``_evidence``), the value the log-odds
#: per unit of evidence.  Positive means the evidence argues *for* the emotion.
LIKELIHOOD: dict[str, dict[str, float]] = {
    "FEAR": {
        "formula_down": 1.5, "formula_up": -1.0,
        "down_mid": 2.0, "down_slow": 1.0, "stress": 1.5, "blowout": 1.0,
        "toxicity": 0.8, "book_pressure": -0.8, "news": -0.8, "calm": -1.5,
        # Round Z geometry: cavities opening in the book and critical slowing
        # down are fear *before* the market orders arrive.
        "tearing": 1.2, "csd": 1.0, "cooling": 0.8,
    },
    "PANIC": {
        "formula_down": 1.2,
        "down_fast": 2.5, "cascade": 2.0, "stress": 1.5, "toxicity": 1.2,
        "blowout": 1.2, "momentum": 0.3, "calm": -2.0,
        "tearing": 1.0, "chaotic": 1.2, "polarised": 0.8,
    },
    "CAPITULATION": {
        "formula_down": 1.0,
        "drawdown": 2.5, "down_slow": 1.5, "toxicity": 1.0, "momentum": -0.5,
        "herd_memory": 0.8, "persistence": 0.6, "calm": -1.5,
        "cooling": 1.2, "csd": 0.6,
    },
    "DENIAL": {
        "formula_down": 1.2, "formula_up": -0.8,
        "drawdown": 2.0, "down_slow": 0.8, "up_fast": 1.0, "momentum": -1.0,
        "disorder": 0.4, "book_pressure": 0.6, "news": -0.5, "stress": 0.5, "calm": -1.0,
    },
    "HOPE": {
        "formula_up": 1.5, "formula_down": -1.0,
        "up_mid": 1.5, "up_slow": 1.0, "calm": 0.5, "book_pressure": 0.8,
        "news": 0.8, "stress": -0.8, "momentum": 0.5,
    },
    "EUPHORIA": {
        "formula_up": 1.5, "formula_down": -0.8,
        "up_slow": 2.0, "up_mid": 1.0, "runup": 2.0, "persistence": 1.0, "herd_memory": 1.0,
        "calm": -0.5, "news": 0.8, "momentum": 0.8,
    },
    "FOMO": {
        "formula_up": 1.2,
        "up_fast": 2.5, "cascade": 2.0, "toxicity": 1.0, "momentum": 1.0,
        "herd_memory": 1.2, "calm": -1.5,
        "heating": 1.0, "chaotic": 0.8, "polarised": 0.8,
    },
    "COMPLACENCY": {
        "formula_conviction": -1.5,
        "calm": 2.5, "disorder": 1.0, "stress": -2.0, "down_fast": -1.5,
        "up_fast": -1.5, "down_mid": -1.5, "up_mid": -1.5, "cascade": -1.0,
        "toxicity": -0.8,
        "tearing": -1.0, "csd": -0.8, "chaotic": -0.8, "polarised": -0.6,
    },
}

#: Plain-language names for the evidence variables, used in the chain.
EVIDENCE_LABEL = {
    "down_fast": "fast sell-off (1 s)", "up_fast": "fast ramp (1 s)",
    "down_mid": "sell-off over 5 s", "up_mid": "ramp over 5 s",
    "down_slow": "minute-scale decline", "up_slow": "minute-scale advance",
    "drawdown": "drawdown from the window high", "runup": "run-up from the window low",
    "toxicity": "order-flow toxicity (VPIN)", "cascade": "self-excitation (Hawkes)",
    "momentum": "variance-ratio momentum", "persistence": "Hurst persistence",
    "disorder": "permutation entropy", "herd_memory": "sign memory (Lillo-Farmer)",
    "stress": "stress regime", "calm": "calm regime", "blowout": "spread blow-out",
    "book_pressure": "book pressure (microprice)", "news": "news tone",
    "formula_up": "22-formula consensus (bullish)", "formula_down": "22-formula consensus (bearish)",
    "formula_conviction": "22-formula conviction |C|",
    "tearing": "book manifold tearing (TDA persistence)", "chaotic": "attractor divergence (Lyapunov)",
    "csd": "critical slowing down (a1 ↑, variance ↑)", "cooling": "heat BTC→PAXG (entropy production)",
    "heating": "heat PAXG→BTC (entropy production)", "polarised": "non-classical decision interference",
}


def _safe(value: Any, fallback: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    return v if math.isfinite(v) else fallback


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return float(min(high, max(low, value)))


def _ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clip((value - low) / (high - low), 0.0, 1.0)


# ---------------------------------------------------------------------------
# State carried between samples
# ---------------------------------------------------------------------------
@dataclass
class DeepState:
    """The Bayesian filter's memory: the posterior after the previous sample.

    ``forgetting`` is the probability, per sample, that the crowd's state
    re-draws from the uniform prior - the "sticky transition" of the filter.
    At 0.5 s cadence 0.1 gives a belief half-life of roughly 3 seconds - a
    little slower than the panel's EMA, so the belief is the steadier of the
    two readings and a single burst cannot flip it.  ``temperature`` softens the
    naive-Bayes likelihood (whose evidence variables are correlated) so the
    posterior is a calibrated belief rather than a 99.9 % certainty.
    """

    posterior: dict[str, float] = field(
        default_factory=lambda: {name: 1.0 / len(EMOTION_NAMES) for name in EMOTION_NAMES}
    )
    forgetting: float = 0.1
    temperature: float = 1.4
    samples: int = 0


# ---------------------------------------------------------------------------
# The formulas
# ---------------------------------------------------------------------------
def hawkes_branching(times_ms: np.ndarray, bin_ms: float = 250.0) -> dict:
    """Self-excitation of the print process.

    For a Hawkes process with branching ratio ``n`` the Fano factor of the
    counts in bins much longer than the kernel is ``F = 1 / (1 - n)^2``, so
    ``n = 1 - 1 / sqrt(F)``.  ``n`` near 0 is a Poisson tape (every print is
    independent news); ``n`` near 1 is a cascade (prints are causing prints -
    a herd).  We also report the current intensity against the baseline.
    """
    out = {"branching_ratio": 0.0, "fano": 1.0, "intensity_hz": 0.0, "baseline_hz": 0.0,
           "excitation": 0.0, "bins": 0}
    if times_ms.size < 8:
        return out
    span = float(times_ms[-1] - times_ms[0])
    if span <= bin_ms * 4:
        return out
    edges = np.arange(times_ms[0], times_ms[-1] + bin_ms, bin_ms)
    counts, _ = np.histogram(times_ms, bins=edges)
    counts = counts[:-1] if counts.size > 1 else counts   # the last bin is partial
    mean = float(np.mean(counts)) if counts.size else 0.0
    if mean <= 0:
        return out
    fano = float(np.var(counts) / mean)
    n = 1.0 - 1.0 / math.sqrt(max(fano, 1e-9)) if fano > 1.0 else 0.0
    baseline = mean * 1000.0 / bin_ms
    last_window = times_ms[times_ms >= times_ms[-1] - 1000.0]
    intensity = float(last_window.size)  # prints in the last second
    out.update({
        "branching_ratio": round(_clip(n, 0.0, 0.99), 4),
        "fano": round(fano, 4),
        "intensity_hz": round(intensity, 3),
        "baseline_hz": round(baseline, 3),
        # instantaneous excitation: how far above the baseline the tape is now
        "excitation": round(_ramp(intensity / baseline if baseline > 0 else 1.0, 1.5, 5.0), 4),
        "bins": int(counts.size),
    })
    return out


def vpin(ticks: np.ndarray, buckets: int = 24) -> dict:
    """Volume-synchronised probability of informed trading.

    Volume is cut into ``buckets`` equal-volume buckets; in each the absolute
    buy/sell imbalance divided by the bucket volume is the probability that
    the flow is one-sided.  The mean over the buckets is VPIN: 0 balanced,
    1 completely one-sided.  A rising VPIN with no news is the signature of
    an emotional (as opposed to informed) crowd.
    """
    out = {"vpin": 0.0, "buckets": 0, "last_bucket": 0.0, "trend": 0.0}
    if ticks.shape[0] < 16:
        return out
    qty = np.abs(ticks[:, 2])
    side = np.sign(ticks[:, 3]) if ticks.shape[1] > 3 else np.sign(np.diff(ticks[:, 1], prepend=ticks[0, 1]))
    total = float(qty.sum())
    if total <= 0:
        return out
    bucket_volume = total / buckets
    cumulative = np.cumsum(qty)
    index = np.minimum((cumulative / bucket_volume).astype(int), buckets - 1)
    buys = np.bincount(index, weights=np.where(side > 0, qty, 0.0), minlength=buckets)
    sells = np.bincount(index, weights=np.where(side < 0, qty, 0.0), minlength=buckets)
    volume = buys + sells
    imbalance = np.abs(buys - sells) / np.where(volume > 0, volume, 1.0)
    used = volume > 0
    values = imbalance[used]
    if values.size == 0:
        return out
    half = max(1, values.size // 2)
    out.update({
        "vpin": round(float(values.mean()), 4),
        "buckets": int(values.size),
        "last_bucket": round(float(values[-1]), 4),
        "trend": round(float(values[-half:].mean() - values[:half].mean()), 4),
    })
    return out


def kyle_lambda(ticks: np.ndarray, bucket_ms: float = 1000.0) -> dict:
    """Price impact per unit of signed flow (Kyle 1985), by least squares.

    Returns are bucketed on the time clock; ``lambda`` is the slope of the
    mid-price change (bps) on the signed volume of the bucket and ``r2`` how
    much of the price change the flow explains.  A big lambda with a good fit
    means a *pushable* tape - the crowd's own orders move the price, which is
    exactly the condition under which a 60 s window is easy to manipulate.
    """
    out = {"lambda_bps": 0.0, "r2": 0.0, "buckets": 0, "impact_norm": 0.0}
    if ticks.shape[0] < 20:
        return out
    t = ticks[:, 0]
    edges = np.arange(t[0], t[-1] + bucket_ms, bucket_ms)
    if edges.size < 6:
        return out
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, edges.size - 2)
    signed = np.bincount(idx, weights=ticks[:, 2] * np.sign(ticks[:, 3]), minlength=edges.size - 1)
    last_price = np.zeros(edges.size - 1)
    seen = np.zeros(edges.size - 1, dtype=bool)
    for k in range(edges.size - 1):
        members = np.where(idx == k)[0]
        if members.size:
            last_price[k] = ticks[members[-1], 1]
            seen[k] = True
    if seen.sum() < 6:
        return out
    prices = last_price[seen]
    flow = signed[seen]
    dp = np.diff(prices) / prices[:-1] * 10_000.0
    x = flow[1:]
    if x.size < 5 or float(np.var(x)) <= 1e-12:
        return out
    x_c = x - x.mean(); y_c = dp - dp.mean()
    slope = float((x_c * y_c).sum() / (x_c * x_c).sum())
    pred = slope * x_c
    ss_res = float(((y_c - pred) ** 2).sum()); ss_tot = float((y_c ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # normalise: impact of one typical bucket's flow, in units of the typical move
    typical_flow = float(np.median(np.abs(x))) or 1.0
    typical_move = float(np.median(np.abs(dp))) or 1e-6
    out.update({
        "lambda_bps": round(slope, 6),
        "r2": round(_clip(r2, 0.0, 1.0), 4),
        "buckets": int(x.size),
        "impact_norm": round(_clip(abs(slope) * typical_flow / typical_move, 0.0, 5.0), 4),
    })
    return out


def variance_ratio(returns: np.ndarray, q: int) -> tuple[float, float]:
    """Lo-MacKinlay VR(q) and its asymptotic z-statistic (homoskedastic form)."""
    n = returns.size
    if n < q * 4 or q < 2:
        return 1.0, 0.0
    mu = float(returns.mean())
    var_1 = float(((returns - mu) ** 2).sum() / (n - 1))
    if var_1 <= 1e-18:
        return 1.0, 0.0
    agg = np.convolve(returns, np.ones(q), mode="valid")
    m = (n - q + 1) * (1 - q / n)
    var_q = float(((agg - q * mu) ** 2).sum() / max(m, 1e-9))
    vr = var_q / (q * var_1)
    z = (vr - 1.0) / math.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * n))
    return float(vr), float(z)


def hurst_exponent(returns: np.ndarray, scales: tuple[int, ...] = (1, 2, 4, 8, 16)) -> float:
    """Generalised Hurst exponent from the scaling of the variance of q-sums.

    ``Var(sum of q returns) ~ q^(2H)``; the slope of log-variance on log-q,
    halved, is H.  0.5 random walk, > 0.5 persistent (trend-following crowd),
    < 0.5 anti-persistent (a market that snaps back - fades and denial).
    """
    usable = [q for q in scales if returns.size >= q * 8]
    if len(usable) < 3:
        return 0.5
    xs, ys = [], []
    for q in usable:
        agg = np.convolve(returns, np.ones(q), mode="valid")
        v = float(np.var(agg))
        if v > 1e-24:
            xs.append(math.log(q)); ys.append(math.log(v))
    if len(xs) < 3:
        return 0.5
    slope = float(np.polyfit(xs, ys, 1)[0])
    return _clip(slope / 2.0, 0.0, 1.0)


def permutation_entropy(prices: np.ndarray, order: int = 3) -> float:
    """Bandt-Pompe permutation entropy, normalised to [0, 1].

    Counts the ordinal patterns of consecutive ``order``-tuples; 1 means every
    pattern is equally likely (pure noise), 0 means one pattern dominates (a
    perfectly ordered move).  A crowd in a clean trend has low entropy; a bored
    or confused one, high.
    """
    n = prices.size
    if n < order * 10:
        return 1.0
    perms = {p: i for i, p in enumerate(permutations(range(order)))}
    counts = np.zeros(len(perms))
    windows = np.lib.stride_tricks.sliding_window_view(prices, order)
    ranks = np.argsort(windows, axis=1)
    for row in ranks:
        counts[perms[tuple(int(x) for x in row)]] += 1
    p = counts[counts > 0] / counts.sum()
    return _clip(float(-(p * np.log(p)).sum() / math.log(len(perms))), 0.0, 1.0)


def sign_memory(sides: np.ndarray, lags: tuple[int, ...] = (1, 2, 3, 5, 8, 13)) -> dict:
    """Autocorrelation of trade signs and the Lillo-Farmer decay exponent.

    Trade signs are long-memory: the ACF decays as ``lag^(-gamma)``.  A small
    gamma (slow decay) means the herd keeps trading the same way for a long
    time; the strength at lag 1 is the instantaneous herding.
    """
    out = {"acf": {}, "gamma": 1.0, "memory": 0.0}
    s = sides[sides != 0]
    n = s.size
    if n < 40:
        return out
    s = s - s.mean()
    denom = float((s * s).sum())
    if denom <= 0:
        return out
    acf = {}
    for lag in lags:
        if n > lag + 10:
            acf[lag] = float((s[:-lag] * s[lag:]).sum() / denom)
    out["acf"] = {str(k): round(v, 4) for k, v in acf.items()}
    positive = [(lag, v) for lag, v in acf.items() if v > 0.01 and lag >= 1]
    if len(positive) >= 3:
        xs = np.log([lag for lag, _ in positive]); ys = np.log([v for _, v in positive])
        out["gamma"] = round(_clip(-float(np.polyfit(xs, ys, 1)[0]), 0.0, 3.0), 4)
    # memory in [0,1]: the *signed* mean ACF over the lags (so a strictly
    # alternating tape, positive only at even lags, reads as no herd), scaled
    # so a mean of 0.3 saturates.
    out["memory"] = round(_ramp(float(np.mean(list(acf.values()))) if acf else 0.0, 0.0, 0.3), 4)
    return out


def haar_spectrum(returns: np.ndarray, resolution_us: float, levels: int = 6) -> list[dict]:
    """Energy of the return series per dyadic scale (a Haar wavelet transform).

    Scale ``2^k`` ticks, labelled in microseconds using the tape's median
    interval, and the share of the total variance that lives there.  The band
    with the largest share is where the crowd is *acting*; a tape whose energy
    sits at the finest scale is being traded tick by tick (bots and panic), one
    whose energy sits at coarse scales is being traded on a story.
    """
    x = returns.astype(float)
    energies = []
    for level in range(1, levels + 1):
        if x.size < 4:
            break
        even = x[: x.size // 2 * 2].reshape(-1, 2)
        detail = (even[:, 0] - even[:, 1]) / math.sqrt(2.0)
        approx = (even[:, 0] + even[:, 1]) / math.sqrt(2.0)
        energies.append((2 ** level, float((detail ** 2).sum())))
        x = approx
    total = sum(e for _, e in energies) or 1.0
    return [
        {
            "scale_ticks": scale,
            "scale_us": round(scale * resolution_us, 1),
            "scale_label": format_us(scale * resolution_us) if resolution_us else f"{scale} ticks",
            "energy": round(e, 10),
            "share": round(e / total, 4),
        }
        for scale, e in energies
    ]


def regime_filter(returns_bps: np.ndarray, typical_bps: float, stickiness: float = 0.95) -> dict:
    """Three-state Gaussian HMM forward filter over 1 s returns.

    States: calm (0.8 sigma, no drift), trend (1 sigma with drift), stress
    (3 sigma), where sigma is the tape's own 1 s standard deviation.  The transition matrix is sticky; the forward pass
    over the window's returns gives the posterior of the *current* regime.
    """
    out = {"calm": 1 / 3, "trend": 1 / 3, "stress": 1 / 3, "label": "calm", "steps": 0}
    if returns_bps.size < 6 or typical_bps <= 1e-9:
        return out
    # ``typical`` is a median absolute move; for a Gaussian that is 0.6745 sigma.
    sigma = typical_bps / 0.6745
    sig = np.array([0.8, 1.0, 3.0]) * sigma
    drift = np.array([0.0, float(np.sign(returns_bps[-6:].sum())) * 0.8 * sigma, 0.0])
    off = (1 - stickiness) / 2
    a = np.full((3, 3), off); np.fill_diagonal(a, stickiness)
    belief = np.full(3, 1 / 3)
    for r in returns_bps:
        like = np.exp(-0.5 * ((r - drift) / sig) ** 2) / sig
        belief = (belief @ a) * like
        s = belief.sum()
        belief = belief / s if s > 0 else np.full(3, 1 / 3)
    names = ("calm", "trend", "stress")
    out.update({names[i]: round(float(belief[i]), 4) for i in range(3)})
    out["label"] = names[int(np.argmax(belief))]
    out["steps"] = int(returns_bps.size)
    return out


def microprice(book: Any) -> dict:
    """Microprice and depth pressure from the top of the book."""
    out = {"microprice_bps": 0.0, "pressure_top5": 0.0, "pressure_top20": 0.0, "spread_bps": 0.0}
    try:
        bids = np.asarray(book[0], dtype=float); asks = np.asarray(book[1], dtype=float)
        pb, qb = float(bids[0, 0]), float(bids[0, 1])
        pa, qa = float(asks[0, 0]), float(asks[0, 1])
    except (IndexError, TypeError, ValueError):
        return out
    if not (pa > pb > 0) or qa + qb <= 0:
        return out
    mid = (pa + pb) / 2.0
    micro = (pa * qb + pb * qa) / (qa + qb)
    out["microprice_bps"] = round((micro - mid) / mid * 10_000.0, 5)
    out["spread_bps"] = round((pa - pb) / mid * 10_000.0, 5)
    for levels, key in ((5, "pressure_top5"), (20, "pressure_top20")):
        b = float(bids[:levels, 1].sum()); s = float(asks[:levels, 1].sum())
        out[key] = round((b - s) / (b + s), 4) if b + s > 0 else 0.0
    return out


def ignition(times_ms: np.ndarray, prices: np.ndarray, typical_1s_bps: float) -> dict:
    """Momentum-ignition detector: a burst that fades.

    Looks for the largest 1 s move in the last ~15 s, then measures how much of
    it was retraced within the following 5 s.  A big move (>= 2 typical) that
    gives back most of itself (>= 60%) with a print burst on the way in is the
    classic bait: retail chases the burst, the initiator sells into them.
    """
    out = {"score": 0.0, "burst_bps": 0.0, "retrace": 0.0, "age_s": 0.0}
    if prices.size < 30 or typical_1s_bps <= 1e-9:
        return out
    now = times_ms[-1]
    start = int(np.searchsorted(times_ms, now - 15_000.0))
    t = times_ms[start:]; p = prices[start:]
    if p.size < 10:
        return out
    earlier = np.searchsorted(t, t - 1000.0, side="right") - 1
    valid = earlier >= 0
    if not np.any(valid):
        return out
    move = np.zeros(p.size)
    move[valid] = (p[valid] / p[earlier[valid]] - 1.0) * 10_000.0
    k = int(np.argmax(np.abs(move)))
    burst = float(move[k])
    if abs(burst) < 2.0 * typical_1s_bps:
        return out
    after = p[k:][t[k:] <= t[k] + 5000.0]
    if after.size < 3:
        return out
    end = float(after[-1])
    peak = float(p[k])
    origin = float(p[earlier[k]]) if earlier[k] >= 0 else peak
    total = peak - origin
    retrace = (peak - end) / total if abs(total) > 1e-12 else 0.0
    retrace = _clip(retrace, 0.0, 1.5)
    burst_norm = _ramp(abs(burst) / typical_1s_bps, 2.0, 5.0)
    score = burst_norm * _ramp(retrace, 0.4, 0.9)
    out.update({
        "score": round(score, 4),
        "burst_bps": round(burst, 4),
        "retrace": round(retrace, 4),
        "age_s": round(float(now - t[k]) / 1000.0, 3),
    })
    return out


def quote_stuffing(times_ms: np.ndarray, prices: np.ndarray, tick_rate_hz: float) -> float:
    """Prints without price: a burst of activity that moves nothing.

    The last second's print rate against the tape's rate, discounted by how
    much price actually changed - a high rate with a flat price is noise
    injected to slow other participants down (or to simulate interest).
    """
    if prices.size < 20 or tick_rate_hz <= 0:
        return 0.0
    now = times_ms[-1]
    start = int(np.searchsorted(times_ms, now - 1000.0))
    recent = prices[start:]
    if recent.size < 4:
        return 0.0
    rate = recent.size / max(0.25, float(now - times_ms[start]) / 1000.0)
    surge = _ramp(rate / tick_rate_hz, 2.0, 6.0)
    changed = float(np.count_nonzero(np.diff(recent))) / max(1, recent.size - 1)
    return round(surge * (1.0 - changed), 4)


# ---------------------------------------------------------------------------
# Evidence, Bayesian filter, reasoning chain
# ---------------------------------------------------------------------------
def _evidence(
    f: dict[str, float], deep: dict, formula_consensus: float = 0.0, formula_voters: int = 0
) -> dict[str, float]:
    """Compress the raw formulas into the bounded evidence variables the
    likelihood table understands (each in [-1, 1] or [0, 1]).

    ``formula_consensus`` is the weighted vote of the 22 directional formulas
    (``backend.core.prediction.consensus``, in [-1, 1]).  It enters the filter
    as its own evidence so the crowd reading is pulled toward what the
    formulas measure - the emotions describe *how* the crowd behaves around
    the formulas' view, they never out-vote it (the fusion gives the formulas
    the vote and the crowd only a bounded confidence modifier).
    """
    z1 = _safe(f.get("z_1s")); z5 = _safe(f.get("z_5s")); z60 = _safe(f.get("z_60s"))
    geo = deep.get("geometry") or {}
    bands = deep["bands"]
    reg = deep["regime"]
    c = _clip(_safe(formula_consensus), -1.0, 1.0) if formula_voters >= 3 else 0.0
    return {
        "formula_up": _ramp(c, 0.05, 0.50),
        "formula_down": _ramp(-c, 0.05, 0.50),
        "formula_conviction": _ramp(abs(c), 0.10, 0.60),
        "down_fast": _ramp(-z1, 0.5, 3.0), "up_fast": _ramp(z1, 0.5, 3.0),
        "down_mid": _ramp(-z5, 0.5, 3.0), "up_mid": _ramp(z5, 0.5, 3.0),
        "down_slow": _ramp(-z60, 0.5, 3.0), "up_slow": _ramp(z60, 0.5, 3.0),
        "drawdown": _ramp(_safe(f.get("z_drawdown")), 1.0, 4.0),
        "runup": _ramp(_safe(f.get("z_run_up")), 1.0, 4.0),
        "toxicity": _ramp(_safe(deep["flow"]["vpin"]), 0.35, 0.85),
        "cascade": _safe(deep["hawkes"]["branching_ratio"]),
        "momentum": _clip(_safe(bands["seconds"]["variance_ratio"]) - 1.0, -1.0, 1.0),
        "persistence": _clip((_safe(bands["seconds"]["hurst"], 0.5) - 0.5) * 4.0, -1.0, 1.0),
        "disorder": _safe(bands["seconds"]["entropy"], 1.0),
        "herd_memory": _safe(deep["flow"]["sign_memory"]),
        "stress": _safe(reg.get("stress")),
        "calm": _safe(reg.get("calm")),
        # A wide spread is only evidence when the tape is actually moving, and a
        # sub-basis-point spread cannot be a fear signal however wide it "blew".
        "blowout": (
            _ramp(_safe(f.get("spread_blowout"), 1.0), 1.6, 4.0)
            * (0.5 if _safe(f.get("spread_bps")) < 1.0 else 1.0)
            * _ramp(max(abs(z1), abs(z5)), 0.3, 1.0)
        ),
        "book_pressure": _clip(_safe(deep["book"]["pressure_top5"]), -1.0, 1.0),
        "news": _clip(_safe(f.get("news_sentiment")), -1.0, 1.0),
        # Round Z: the geometric layer (backend/core/geometry.py)
        "tearing": _safe((geo.get("tda") or {}).get("tearing")),
        "chaotic": _safe((geo.get("takens") or {}).get("chaotic")),
        "csd": _safe((geo.get("csd") or {}).get("csd")),
        "cooling": (_safe((geo.get("thermo") or {}).get("entropy_production"))
                    if _safe((geo.get("thermo") or {}).get("flux_J")) < 0 else 0.0),
        "heating": (_safe((geo.get("thermo") or {}).get("entropy_production"))
                    if _safe((geo.get("thermo") or {}).get("flux_J")) > 0 else 0.0),
        "polarised": _safe((geo.get("quantum") or {}).get("polarisation")),
    }


def bayesian_update(state: DeepState | None, evidence: dict[str, float]) -> dict:
    """One step of the discrete Bayesian filter over the eight emotions.

    ``prior = (1 - rho) * previous + rho * uniform`` (sticky transitions),
    ``log L_e = sum_j w_ej * evidence_j / temperature``,
    ``posterior ∝ prior * exp(log L)``.

    Returns the posterior, the per-emotion log-likelihood with its largest
    contributions (the "why"), the posterior entropy (how sure the filter is),
    and the KL divergence from the prior (how much this sample changed its
    mind - the *surprise*).
    """
    n = len(EMOTION_NAMES)
    uniform = 1.0 / n
    if state is None:
        prior = {name: uniform for name in EMOTION_NAMES}
        rho, temperature = 1.0, 1.4
    else:
        rho, temperature = state.forgetting, state.temperature
        prior = {
            name: (1 - rho) * state.posterior.get(name, uniform) + rho * uniform
            for name in EMOTION_NAMES
        }
    log_like: dict[str, float] = {}
    contributions: dict[str, list[dict]] = {}
    for name in EMOTION_NAMES:
        parts = []
        total = 0.0
        for key, weight in LIKELIHOOD[name].items():
            value = evidence.get(key, 0.0)
            term = weight * value / max(temperature, 1e-6)
            total += term
            if abs(term) > 1e-6:
                parts.append({"evidence": key, "label": EVIDENCE_LABEL.get(key, key),
                              "value": round(value, 4), "weight": weight, "log_odds": round(term, 4)})
        parts.sort(key=lambda p: abs(p["log_odds"]), reverse=True)
        log_like[name] = total
        contributions[name] = parts[:4]
    peak = max(log_like.values())
    unnormalised = {name: prior[name] * math.exp(log_like[name] - peak) for name in EMOTION_NAMES}
    z = sum(unnormalised.values()) or 1.0
    posterior = {name: v / z for name, v in unnormalised.items()}
    entropy = -sum(p * math.log(p) for p in posterior.values() if p > 0) / math.log(n)
    kl = sum(p * math.log(p / prior[name]) for name, p in posterior.items() if p > 0)
    if state is not None:
        state.posterior = dict(posterior)
        state.samples += 1
    argmax = max(posterior, key=posterior.get)
    ranked = sorted(posterior.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "posterior": {name: round(p, 5) for name, p in posterior.items()},
        "argmax": argmax,
        "argmax_probability": round(posterior[argmax], 4),
        "runner_up": ranked[1][0] if len(ranked) > 1 else "",
        "margin": round(ranked[0][1] - ranked[1][1], 4) if len(ranked) > 1 else 0.0,
        "entropy": round(entropy, 4),
        "certainty": round(1.0 - entropy, 4),
        "surprise_kl": round(kl, 4),
        "log_likelihood": {name: round(v, 4) for name, v in log_like.items()},
        "evidence_for": {name: contributions[name] for name in (argmax, ranked[1][0])},
        "forgetting": rho,
        "prior_samples": state.samples if state is not None else 0,
    }


def _feeds(step_key: str, direction: float, strength: float = 1.0) -> list[str]:
    """Which emotions a step argues for, from the likelihood table - nothing
    when the evidence is too small to move the filter."""
    winners = []
    if abs(strength) < 0.1:
        return winners
    for name, table in LIKELIHOOD.items():
        w = table.get(step_key)
        if w and w * direction > 0:
            winners.append(name.capitalize())
    return winners


def reasoning_chain(f: dict[str, float], deep: dict, evidence: dict[str, float]) -> list[dict]:
    """The ordered, human-readable derivation of the reading."""
    h = deep["hawkes"]; fl = deep["flow"]; b = deep["bands"]; bk = deep["book"]
    reg = deep["regime"]; man = deep["manipulation"]; spec = deep["spectrum"]
    top_scale = max(spec, key=lambda s: s["share"]) if spec else None
    steps: list[dict] = []

    def add(name, formula, inputs, value, unit, reads, timescale, key=None, direction=1.0):
        strength = evidence.get(key, 0.0) if key else 0.0
        steps.append({
            "step": len(steps) + 1, "name": name, "formula": formula, "inputs": inputs,
            "value": value, "unit": unit, "reads": reads, "timescale": timescale,
            "evidence": round(strength, 4) if key else None,
            "feeds": _feeds(key, direction, strength) if key else [],
        })

    n = h["branching_ratio"]
    add("Hawkes self-excitation", "n = 1 − 1/√F,  F = Var(N)/E[N] over 250 ms bins",
        {"fano": h["fano"], "bins": h["bins"], "intensity_hz": h["intensity_hz"], "baseline_hz": h["baseline_hz"]},
        n, "branching ratio",
        ("cascade: prints are causing prints" if n >= 0.6 else
         "some clustering of prints" if n >= 0.3 else "independent arrivals (Poisson tape)"),
        "micro", "cascade", 1.0)
    add("Microprice lean", "P* = (Pa·Qb + Pb·Qa)/(Qa+Qb);  lean = (P* − mid)/mid",
        {"pressure_top5": bk["pressure_top5"], "pressure_top20": bk["pressure_top20"], "spread_bps": bk["spread_bps"]},
        bk["microprice_bps"], "bps",
        ("book leans to the bid: next print more likely up" if bk["microprice_bps"] > 0.02 else
         "book leans to the ask: next print more likely down" if bk["microprice_bps"] < -0.02 else
         "balanced top of book"),
        "micro", "book_pressure", 1.0 if evidence["book_pressure"] >= 0 else -1.0)
    add("Order-flow toxicity", "VPIN = mean_k |V_buy,k − V_sell,k| / V_k  over 24 volume buckets",
        {"buckets": fl["vpin_buckets"], "last_bucket": fl["vpin_last"], "trend": fl["vpin_trend"]},
        fl["vpin"], "probability",
        ("one-sided flow: the crowd is all on one side" if fl["vpin"] >= 0.6 else
         "moderately one-sided flow" if fl["vpin"] >= 0.4 else "two-sided, balanced flow"),
        "seconds", "toxicity", 1.0)
    add("Kyle's lambda", "ΔP_bps = λ · signed volume  (OLS over 1 s buckets)",
        {"r2": fl["kyle_r2"], "buckets": fl["kyle_buckets"], "impact_norm": fl["impact_norm"]},
        fl["kyle_lambda_bps"], "bps per unit",
        ("pushable tape: the crowd's own orders move price" if fl["impact_norm"] >= 1.0 and fl["kyle_r2"] >= 0.3 else
         "flow explains little of the price change" if fl["kyle_r2"] < 0.15 else "normal price impact"),
        "seconds")
    vr = b["seconds"]["variance_ratio"]
    add("Variance ratio", "VR(q) = Var(r_q)/(q·Var(r_1)),  1 s vs 250 ms",
        {"z_stat": b["seconds"]["vr_z"], "micro_vr": b["micro"]["variance_ratio"], "window_vr": b["window"]["variance_ratio"]},
        vr, "ratio",
        ("momentum: moves continue" if vr > 1.15 else "mean reversion: moves snap back" if vr < 0.85 else "random-walk scaling"),
        "seconds", "momentum", 1.0 if evidence["momentum"] >= 0 else -1.0)
    hurst = b["seconds"]["hurst"]
    add("Hurst exponent", "Var(Σ_q r) ∝ q^{2H}  (log-log slope over q = 1…16)",
        {"micro_H": b["micro"]["hurst"], "window_H": b["window"]["hurst"]},
        hurst, "H",
        ("persistent: the crowd keeps pushing" if hurst > 0.58 else
         "anti-persistent: the crowd fades every move" if hurst < 0.42 else "no memory in the path"),
        "seconds to minute", "persistence", 1.0 if evidence["persistence"] >= 0 else -1.0)
    ent = b["seconds"]["entropy"]
    add("Permutation entropy", "H_p = −Σ p(π) ln p(π) / ln(3!)  over ordinal patterns of 3",
        {"micro": b["micro"]["entropy"], "window": b["window"]["entropy"]},
        ent, "normalised",
        ("ordered path: one story is driving the tape" if ent < 0.75 else
         "noisy path: no consensus in the crowd" if ent > 0.93 else "mixed"),
        "seconds to minute", "disorder", 1.0)
    add("Sign memory", "ACF_ε(ℓ) ∝ ℓ^{−γ}  (Lillo–Farmer long memory of trade signs)",
        {"acf": fl["sign_acf"], "gamma": fl["sign_gamma"]},
        fl["sign_memory"], "0–1",
        ("herd: the same side keeps hitting" if fl["sign_memory"] >= 0.6 else
         "some same-side clustering" if fl["sign_memory"] >= 0.3 else "signs alternate - no herd"),
        "seconds", "herd_memory", 1.0)
    add("Wavelet energy", "E_k = Σ d_k²  (Haar detail coefficients at scale 2^k ticks)",
        {"scales": [(s["scale_label"], s["share"]) for s in spec]},
        top_scale["share"] if top_scale else 0.0, f"share at {top_scale['scale_label']}" if top_scale else "share",
        (f"the action is at the {top_scale['scale_label']} scale" if top_scale else "no spectrum"),
        "micro to minute")
    add("Regime filter", "P(s_t | r_1:t) ∝ P(r_t | s_t) Σ_s' A_{s's} P(s_{t−1} | r_1:t−1)",
        {"calm": reg["calm"], "trend": reg["trend"], "stress": reg["stress"], "steps": reg["steps"]},
        reg["label"], "state",
        {"calm": "calm regime: small, balanced 1 s returns", "trend": "trending regime: drift in the 1 s returns",
         "stress": "stress regime: returns three times the usual size"}[reg["label"]],
        "seconds", {"stress": "stress", "calm": "calm"}.get(reg["label"]), 1.0)
    add("Momentum-ignition", "burst ≥ 2σ_1s then retrace ≥ 60% within 5 s",
        {"burst_bps": man["ignition_burst_bps"], "retrace": man["ignition_retrace"], "age_s": man["ignition_age_s"]},
        man["ignition"], "score",
        ("bait: a burst that has already faded" if man["ignition"] >= 0.5 else "no ignition pattern"),
        "seconds")
    add("Quote stuffing", "surge(prints/s ÷ tape rate) × (1 − share of prints that moved price)",
        {"tick_rate_hz": round(_safe(f.get("tick_rate_hz")), 3)},
        man["stuffing"], "score",
        ("activity without price: noise injection" if man["stuffing"] >= 0.4 else "prints carry price"),
        "micro")
    add("Spoofing proxy", "|book pressure| high while flow (VPIN) is balanced and price is flat",
        {"pressure_top20": bk["pressure_top20"], "vpin": fl["vpin"]},
        man["spoofing"], "score",
        ("resting size that is not being traded against - likely for show" if man["spoofing"] >= 0.5 else "book size is being consumed"),
        "micro")
    post = deep["posterior"]
    add("Bayesian filter", "posterior ∝ prior_sticky × exp(Σ_j w_ej · evidence_j)",
        {"certainty": post["certainty"], "surprise_kl": post["surprise_kl"], "margin": post["margin"],
         "top_evidence": [p["label"] for p in post["evidence_for"].get(post["argmax"], [])[:3]]},
        f"{post['argmax'].capitalize()} {post['argmax_probability'] * 100:.0f}%", "belief",
        (f"the filter believes {post['argmax'].capitalize()} "
         f"({post['argmax_probability'] * 100:.0f}%, runner-up {post['runner_up'].capitalize()})"),
        "all bands")
    c = _safe(deep.get("formula_consensus")); voters = int(deep.get("formula_voters", 0))
    add("22-formula cross-check",
        "C = Σ_i w_cat(i)·clip(F_i, −1, 1) / Σ_i w_cat(i)  over the directional formulas",
        {"consensus": round(c, 4), "voters": voters,
         "for": round(evidence["formula_up"], 4), "against": round(evidence["formula_down"], 4)},
        round(c, 4), "consensus",
        ("the formulas have no directional vote yet - the crowd reading stands alone" if voters < 3 else
         f"{voters} formulas lean BUY: the filter favours the buying emotions" if c > 0.05 else
         f"{voters} formulas lean SELL: the filter favours the selling emotions" if c < -0.05 else
         f"{voters} formulas are split: the crowd reading is not pulled either way"),
        "window", "formula_up" if c >= 0 else "formula_down", 1.0)
    return steps


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def analyze(
    tape: Any,
    asset: str,
    f: dict[str, float],
    *,
    state: DeepState | None = None,
    recent: int = 1200,
    formula_consensus: float = 0.0,
    formula_voters: int = 0,
) -> dict:
    """Run every formula on the tape and return the deep block.

    ``f`` is the flat feature dictionary the emotion ramps already computed
    (``backend.core.emotions.features``) - the z-scores and news tone are
    shared so the two layers describe the same tape.
    """
    started = time.perf_counter()
    empty = {"available": False, "reason": "not enough tape for deep formulas"}
    ticks = np.asarray(tape.ticks(asset), dtype=np.float64)
    if ticks.ndim != 2 or ticks.shape[0] < 32:
        return empty
    if ticks.shape[0] > recent:
        ticks = ticks[-recent:]
    times_ms = ticks[:, 0]; prices = ticks[:, 1]
    sides = np.sign(ticks[:, 3]) if ticks.shape[1] > 3 else np.sign(np.diff(prices, prepend=prices[0]))
    resolution_us = _safe(f.get("resolution_us"))
    log_p = np.log(np.maximum(prices, 1e-12))
    tick_returns = np.diff(log_p)

    # --- time-clock returns (250 ms, 1 s, 5 s) for the bands ---------------
    def clock_returns(step_ms: float) -> np.ndarray:
        edges = np.arange(times_ms[0], times_ms[-1] + step_ms, step_ms)
        idx = np.searchsorted(times_ms, edges, side="right") - 1
        idx = idx[idx >= 0]
        series = log_p[np.unique(idx)]
        return np.diff(series) * 10_000.0

    bands = {}
    for name, label, clock_ms, q in BANDS:
        r = tick_returns * 10_000.0 if clock_ms <= 0 else clock_returns(clock_ms)
        levels = np.cumsum(r)
        vr, vz = variance_ratio(r, q)
        bands[name] = {
            "label": label,
            "clock": "tick" if clock_ms <= 0 else (f"{clock_ms / 1000:g} s" if clock_ms >= 1000 else f"{int(clock_ms)} ms"),
            "q": q,
            "samples": int(r.size),
            "hurst": round(hurst_exponent(r), 4),
            "variance_ratio": round(vr, 4),
            "vr_z": round(vz, 3),
            "entropy": round(permutation_entropy(levels), 4),
        }

    one_s = clock_returns(1000.0)
    typical_1s = _safe(f.get("scale_1s_bps"))
    if typical_1s <= 1e-9 and one_s.size:
        typical_1s = float(np.median(np.abs(one_s))) or 1e-6

    hawkes = hawkes_branching(times_ms)
    toxicity = vpin(ticks)
    kyle = kyle_lambda(ticks)
    memory = sign_memory(sides)
    spectrum = haar_spectrum(tick_returns[-512:], resolution_us)
    regime = regime_filter(one_s[-60:], typical_1s)
    book = microprice(tape.book(asset))
    ign = ignition(times_ms, prices, typical_1s)
    stuffing = quote_stuffing(times_ms, prices, _safe(f.get("tick_rate_hz")))
    flat = 1.0 - _ramp(abs(_safe(f.get("z_1s"))), 0.3, 1.5)
    spoofing = round(_ramp(abs(book["pressure_top20"]), 0.25, 0.7) * (1.0 - _ramp(toxicity["vpin"], 0.3, 0.7)) * flat, 4)

    deep = {
        "available": True,
        "ticks": int(ticks.shape[0]),
        "bands": bands,
        "hawkes": hawkes,
        "flow": {
            "vpin": toxicity["vpin"], "vpin_buckets": toxicity["buckets"],
            "vpin_last": toxicity["last_bucket"], "vpin_trend": toxicity["trend"],
            "kyle_lambda_bps": kyle["lambda_bps"], "kyle_r2": kyle["r2"],
            "kyle_buckets": kyle["buckets"], "impact_norm": kyle["impact_norm"],
            "sign_acf": memory["acf"], "sign_gamma": memory["gamma"], "sign_memory": memory["memory"],
        },
        "book": book,
        "spectrum": spectrum,
        "regime": regime,
        "manipulation": {
            "ignition": ign["score"], "ignition_burst_bps": ign["burst_bps"],
            "ignition_retrace": ign["retrace"], "ignition_age_s": ign["age_s"],
            "stuffing": stuffing, "spoofing": spoofing,
            "toxicity": round(_ramp(toxicity["vpin"], 0.35, 0.85), 4),
            "pushable": round(_ramp(kyle["impact_norm"], 0.8, 2.5) * _ramp(kyle["r2"], 0.15, 0.6), 4),
        },
    }
    try:
        deep["geometry"] = geometry.analyze(tape, asset, one_s)
    except Exception as exc:  # noqa: BLE001
        deep["geometry"] = {"available": False, "reason": str(exc)[:120]}
    evidence = _evidence(f, deep, formula_consensus, formula_voters)
    deep["formula_consensus"] = round(_safe(formula_consensus), 4)
    deep["formula_voters"] = int(formula_voters)
    deep["evidence"] = {k: round(v, 4) for k, v in evidence.items()}
    deep["posterior"] = bayesian_update(state, evidence)
    deep["chain"] = reasoning_chain(f, deep, evidence)
    deep["compute_us"] = int((time.perf_counter() - started) * 1e6)
    deep["at_us"] = int(_safe(f.get("at_us")) or now_us())
    return deep


def compact(deep: dict) -> dict:
    """The streamed form: posterior, verdicts and the chain without the raw
    ACF/spectrum arrays (those stay on REST and in the PULSE)."""
    if not deep or not deep.get("available"):
        return dict(deep or {})
    post = deep["posterior"]
    return {
        "available": True,
        "ticks": deep["ticks"],
        "compute_us": deep["compute_us"],
        "formula_consensus": deep.get("formula_consensus", 0.0),
        "formula_voters": deep.get("formula_voters", 0),
        "bands": deep["bands"],
        "hawkes": {k: deep["hawkes"][k] for k in ("branching_ratio", "intensity_hz", "baseline_hz", "excitation")},
        "flow": {k: deep["flow"][k] for k in ("vpin", "kyle_lambda_bps", "kyle_r2", "impact_norm", "sign_memory", "sign_gamma")},
        "book": deep["book"],
        "regime": deep["regime"],
        "manipulation": deep["manipulation"],
        "geometry": {
            k: v for k, v in (deep.get("geometry") or {}).items()
            if k in ("available", "live_measurements", "stress", "read")
        } | {
            "tda": {k: (deep.get("geometry") or {}).get("tda", {}).get(k) for k in ("tearing", "cavities", "max_persistence", "betti0")},
            "takens": {k: (deep.get("geometry") or {}).get("takens", {}).get(k) for k in ("lyapunov", "chaotic")},
            "csd": {k: (deep.get("geometry") or {}).get("csd", {}).get(k) for k in ("csd", "a1_now", "variance_ratio")},
            "thermo": {k: (deep.get("geometry") or {}).get("thermo", {}).get(k) for k in ("entropy_production", "heat_direction", "flux_J", "affinity_X")},
            "quantum": {k: (deep.get("geometry") or {}).get("quantum", {}).get(k) for k in ("interference", "polarisation")},
        } if (deep.get("geometry") or {}).get("available") else {"available": False},
        "spectrum": [{"scale_label": s["scale_label"], "share": s["share"]} for s in deep["spectrum"]],
        "posterior": {
            k: post[k] for k in ("posterior", "argmax", "argmax_probability", "runner_up",
                                 "margin", "entropy", "certainty", "surprise_kl", "evidence_for")
        },
        "chain": [
            {k: step[k] for k in ("step", "name", "formula", "value", "unit", "reads", "timescale", "feeds")}
            for step in deep["chain"]
        ],
    }
