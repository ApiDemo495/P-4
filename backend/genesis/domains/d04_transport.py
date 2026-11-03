"""Domain 4 - optimal transport market metrics (210 formulas).

The per-minute transport readings of the order book (W₁/W₂ shift of each
side, Sinkhorn cost, centroids, barycenter deviation, displacement
extrapolation) are produced by ``book_ot.minute_readings`` when a minute
closes and live in the candle store; the kernels here aggregate them over
k minutes, orient them, and add the trade-flow and cross-asset transport
problems.  Minutes that were bootstrapped from REST history carry NaN in
these columns, so the pool scores them only on live-tape minutes.
"""
from __future__ import annotations

import numpy as np

from backend.genesis import ops
from backend.genesis.domains._common import finish, scale_of
from backend.genesis.spec import grid, make_variants

D = 4
K_MIN = (1, 2, 3, 5, 8, 13, 21)


def _agg(x, k):
    return ops.rsum(x, k) if k > 1 else x


def k_wobs(frame, n, norm, p=1, **_):
    b = _agg(frame.ot["w2_bid" if p == 2 else "w1_bid"], n)
    a = _agg(frame.ot["w2_ask" if p == 2 else "w1_ask"], n)
    raw = b - a
    # which way did the moving side go?  centroid drift says: bids moving closer to mid = pressing
    drift = -(frame.ot["bid_centroid"] - ops.shift(frame.ot["bid_centroid"], n)) + (
        frame.ot["ask_centroid"] - ops.shift(frame.ot["ask_centroid"], n))
    sig = raw * ops.sign(drift) / (ops.rstd(raw, 55) + 1e-9)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_tftc(frame, n, norm, **_):
    w1 = ops.rmean(frame.ot["tftc_w1"], n)
    d = ops.rmean(frame.ot["tftc_dir"], n)
    raw = w1 * ops.sign(d)
    sig = raw / (ops.rstd(w1, 55) + 1e-9)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_sinkhorn(frame, n, norm, **_):
    b, a = _agg(frame.ot["sink_bid"], n), _agg(frame.ot["sink_ask"], n)
    raw = b - a
    drift = -(frame.ot["bid_centroid"] - ops.shift(frame.ot["bid_centroid"], n)) + (
        frame.ot["ask_centroid"] - ops.shift(frame.ot["ask_centroid"], n))
    sig = raw * ops.sign(drift) / (ops.rstd(raw, 55) + 1e-9)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_entropic_gap(frame, n, norm, **_):
    """Regularisation gap: Sinkhorn cost − W₂² grows with how spread the mass
    is (entropy smears concentrated walls).  A side whose gap collapses has
    formed a wall - price tends to move away from a wall."""
    with np.errstate(all="ignore"):
        gb = ops.rmean(frame.ot["sink_bid"] - frame.ot["w2_bid"] ** 2, n)
        ga = ops.rmean(frame.ot["sink_ask"] - frame.ot["w2_ask"] ** 2, n)
        raw = (ga - gb) / (ga + gb + 1e-9)   # positive: bids more concentrated (bid wall)
    sig = raw
    return finish(frame, sig, raw, norm, n, 1.0)


def k_plan_drift(frame, n, norm, **_):
    cb, ca = frame.ot["bid_centroid"], frame.ot["ask_centroid"]
    raw = -(cb - ops.shift(cb, n)) + (ca - ops.shift(ca, n))  # bids closing in (+), asks backing off (+)
    sig = raw / (ops.rstd(raw, 55) + 1e-9)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_barycenter(frame, n, norm, **_):
    b, a = ops.rmean(frame.ot["bary_bid"], n), ops.rmean(frame.ot["bary_ask"], n)
    raw = b - a  # the side that strayed furthest from its own barycenter is the active side
    imb = ops.rmean(frame.imbalance, n)
    sig = np.abs(raw) * ops.sign(imb) / (ops.rstd(raw, 55) + 1e-9)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_displacement(frame, n, norm, **_):
    raw = ops.rmean(frame.ot["ot_extrap_imb"], n)
    sig = raw * 10.0
    return finish(frame, sig, raw, norm, n, 1.0)


def k_multimarginal(frame, n, norm, **_):
    """Straightness of the book's path over 2n minutes: W₁(0,2n) against the
    sum of the two legs (triangle inequality).  Straight = persistent shift."""
    cb = frame.ot["bid_centroid"]
    ca = frame.ot["ask_centroid"]
    with np.errstate(all="ignore"):
        legs = _agg(frame.ot["w1_bid"] + frame.ot["w1_ask"], 2 * n)
        chord = np.abs(cb - ops.shift(cb, 2 * n)) + np.abs(ca - ops.shift(ca, 2 * n))
        raw = chord / (legs + 1e-9)
    drift = -(cb - ops.shift(cb, 2 * n)) + (ca - ops.shift(ca, 2 * n))
    sig = np.clip(raw, 0, 1) * ops.sign(drift)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_unbalanced(frame, n, norm, tau=1.0, **_):
    """Unbalanced OT: mass is created/destroyed at a KL price τ.  The signed
    mass change of each side (log depth ratio) weighted by τ, bids minus asks."""
    with np.errstate(all="ignore"):
        db = np.log(frame.bid_depth + 1e-9) - np.log(ops.shift(frame.bid_depth, n) + 1e-9)
        da = np.log(frame.ask_depth + 1e-9) - np.log(ops.shift(frame.ask_depth, n) + 1e-9)
    raw = tau * (db - da)
    sig = raw / (ops.rstd(raw, 55) + 1e-9)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_gromov(frame, n, norm, **_):
    """Gromov-Wasserstein between the return windows of this asset and the
    other leg: in 1-D with uniform weights the optimal coupling is monotone or
    anti-monotone, so GW² = min(cost_mono, cost_anti) over intra-distance
    matrices.  Anti cheaper = the legs are structurally anti-coupled."""
    A, B = ops.windows(frame.ret, n), ops.windows(frame.other_ret, n)
    raw = ops.nan(frame.n)
    start = max(n - 1, frame.n - int(frame.eval_tail))
    for t in range(start, frame.n):
        a, b = A[t], B[t]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)
        sa, sb = np.sort(a), np.sort(b)
        Da, Db = np.abs(sa[:, None] - sa[None, :]), np.abs(sb[:, None] - sb[None, :])
        mono = float(np.mean((Da - Db) ** 2))
        anti = float(np.mean((Da - Db[::-1, ::-1]) ** 2))
        raw[t] = (anti - mono) / (anti + mono + 1e-12)  # +1 strongly co-structured, −1 anti
    last_other = ops.sign(frame.other_ret)
    sig = raw * last_other * np.minimum(1.0, np.abs(frame.other_ret) / scale_of(frame, n))
    return finish(frame, sig, raw, norm, n, 1.0)


def variants():
    g = grid(windows=K_MIN)
    return (
        make_variants(D, 1, "W₁ / W₂ distance", "Wasserstein order-book shift (WOBS)", k_wobs,
                      "W_p between each side's depth distribution (bps from mid) at minute open and close, summed over k minutes; bid − ask, oriented by centroid drift",
                      "WOBS > 0 bids reshaped more than asks and closed in on mid = buyers getting aggressive", 2,
                      grid(windows=K_MIN, extra=dict(p=(1, 2))))
        + make_variants(D, 2, "transport plan (trade flow)", "Trade-flow transport cost (TFTC)", k_tftc,
                        "W₁ between buy-print and sell-print price distributions in the minute × sign(mean buys − mean sells), averaged over k",
                        "buyers paying a premium over sellers = conviction behind the prints", 2, g)
        + make_variants(D, 3, "Sinkhorn", "Entropic book shift", k_sinkhorn,
                        "Sinkhorn cost (ε = 2 bp², 60 scalings, cost (i−j)²) of moving the minute-open side onto the minute-close side; bid − ask",
                        "", 2, g)
        + make_variants(D, 4, "entropic regularisation", "Regularisation-gap wall detector", k_entropic_gap,
                        "(Sinkhorn − W₂²) per side; the side with the smaller gap is the more concentrated (a wall)",
                        "positive: bid wall below - price tends to lift away from it", 2, g)
        + make_variants(D, 5, "transport plan (centroid)", "Plan centroid drift", k_plan_drift,
                        "first moment of the transport plan = centroid shift of each side over k minutes; bids closing in (+), asks backing off (+)", "", 2, g)
        + make_variants(D, 6, "barycenter", "Barycenter deviation", k_barycenter,
                        "W₁ of each side from the Wasserstein barycenter (quantile average) of the last 5 snapshots; the side further from its barycenter is active, direction from imbalance", "", 2, g)
        + make_variants(D, 7, "displacement interpolation", "Geodesic extrapolation of the book", k_displacement,
                        "McCann interpolation extended to t = 2: q₂ = q_open + 2 (q_close − q_open) per side; extrapolated (ask − bid) centroid imbalance",
                        "where the book is heading one minute out, if the current displacement continues", 2, g)
        + make_variants(D, 8, "multi-marginal transport", "Book path straightness", k_multimarginal,
                        "chord (centroid move over 2k) / Σ legs (W₁ per minute): 1 = straight persistent shift, 0 = churn", "", 2, g)
        + make_variants(D, 9, "unbalanced OT", "Unbalanced mass creation", k_unbalanced,
                        "τ · (Δ log bid depth − Δ log ask depth) over k minutes - the KL-priced mass term of unbalanced OT",
                        "mass appearing on the bid = support building", 2, grid(windows=K_MIN, extra=dict(tau=(0.5, 1.0, 2.0))))
        + make_variants(D, 10, "Gromov-Wasserstein", "Cross-asset Gromov-Wasserstein coupling", k_gromov,
                        "GW² = min(monotone, anti-monotone) alignment cost of the two legs' intra-distance matrices; (anti − mono)/(anti + mono)",
                        "when the legs are co-structured follow the other leg's last move, when anti-structured fade it", 3)
    )
