"""Domain 6 - tropical geometry of OHLCV data (210 formulas).

Tropical semiring (ℝ ∪ {−∞}, ⊕ = max, ⊗ = +).  A tropical polynomial
f(x) = ⊕_k (a_k ⊗ x^{⊗k}) = max_k (a_k + k x) is piecewise linear; its corner
locus (where two terms tie) is the tropical hypersurface.  OHLC is tropical
by birth: high = ⊕ of prices, low = the min-plus dual.
"""
from __future__ import annotations

import numpy as np

from backend.genesis import ops
from backend.genesis.domains._common import finish, scale_of
from backend.genesis.spec import grid, make_variants

D = 6
DEGREES = (2, 3, 4, 5, 6, 7, 8)


def _coeffs(frame, n, d):
    """a_0 = median r, a_1 = range, a_2 = skew·range, a_3 = 0.01·kurt, a_k≥4 = a_3 / k."""
    W = ops.windows(frame.ret, n)
    with np.errstate(all="ignore"):
        a0 = np.nanmedian(W, axis=1)
        rngw = np.nanmax(W, axis=1) - np.nanmin(W, axis=1)
        a1 = rngw
        a2 = frame.roll("ret_skew", n) * rngw
        a3 = frame.roll("ret_kurt", n) * 0.01 * rngw
    coeffs = [a0, a1, a2, a3][: d + 1]
    for k in range(4, d + 1):
        coeffs.append(a3 / k)
    return coeffs


def _tmp_terms(frame, n, d, x):
    coeffs = _coeffs(frame, n, d)
    terms = np.vstack([c + k * x for k, c in enumerate(coeffs)])  # (d+1, N)
    return coeffs, terms


def k_tmp(frame, n, norm, d=3, semiring="max", **_):
    x = frame.bar_ret / scale_of(frame, n)
    coeffs, terms = _tmp_terms(frame, n, d, x)
    with np.errstate(all="ignore"):
        val = np.nanmax(terms, axis=0) if semiring == "max" else np.nanmin(terms, axis=0)
        k_star = np.nanargmax(np.nan_to_num(terms, nan=-1e9), axis=0) if semiring == "max" else \
            np.nanargmin(np.nan_to_num(terms, nan=1e9), axis=0)
    # the winning monomial's degree says how extreme the move is: k ≥ 2 extreme -> continuation,
    # k = 0 base rate -> reversion; sign from the move itself
    sig = np.where(k_star >= 2, 1.0, np.where(k_star == 0, -1.0, 0.0)) * np.sign(x) * np.minimum(1.0, np.abs(x) / 2.0)
    sig[np.isnan(val)] = np.nan
    return finish(frame, sig, val, norm, n, 1.0)


def k_root(frame, n, norm, d=3, **_):
    x = frame.bar_ret / scale_of(frame, n)
    coeffs, _terms = _tmp_terms(frame, n, d, x)
    root12 = coeffs[0] - coeffs[1]          # where a_0 = a_1 + x
    root23 = coeffs[1] - coeffs[2]          # where a_1 + x = a_2 + 2x
    with np.errstate(all="ignore"):
        above = x > root23
        below = x < root12
        sig = np.where(above, np.sign(x), np.where(below, -np.sign(x), 0.0)) * np.minimum(1.0, np.abs(x))
        raw = x - root23
    return finish(frame, sig, raw, norm, n, 1.0)


def k_discriminant(frame, n, norm, d=3, **_):
    x = frame.bar_ret / scale_of(frame, n)
    _c, terms = _tmp_terms(frame, n, d, x)
    with np.errstate(all="ignore"):
        srt = np.sort(np.nan_to_num(terms, nan=-1e9), axis=0)
        gap = srt[-1] - srt[-2]      # distance to the tropical hypersurface (tie of the top two)
        gap[np.isnan(terms).any(axis=0)] = np.nan
    # deep inside a cell: the regime is unambiguous - follow; at the corner locus: boundary - stand down
    sig = np.tanh(gap) * np.sign(x)
    return finish(frame, sig, gap, norm, n, 1.0)


def _lower_hull(y):
    pts = list(enumerate(y))
    hull = []
    for p in pts:
        while len(hull) >= 2 and (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-1][0]) >= (p[1] - hull[-1][1]) * (hull[-1][0] - hull[-2][0]):
            hull.pop()
        hull.append(p)
    return hull


def k_newton(frame, n, norm, **_):
    """Newton polygon of the window: lower convex hull of (i, cumulative log
    return); its first and last slopes are the extreme tropical roots."""
    x = np.nancumsum(frame.ret)
    W = ops.windows(x, n)
    raw = ops.nan(frame.n)
    start = max(n - 1, frame.n - int(frame.eval_tail))
    for t in range(start, frame.n):
        y = W[t]
        if np.isnan(y).any():
            continue
        h = _lower_hull(y - y[0])
        if len(h) < 2:
            continue
        s_first = (h[1][1] - h[0][1]) / max(1, h[1][0] - h[0][0])
        s_last = (h[-1][1] - h[-2][1]) / max(1, h[-1][0] - h[-2][0])
        raw[t] = s_last - s_first  # convexity of the path: accelerating up (+)
    sig = raw / scale_of(frame, n)
    return finish(frame, sig, raw, norm, n, 1.0)


def _transition_matrix(W, bins=3):
    """A_ij = max return seen on a transition from state i to state j (max-plus),
    states = terciles of the window's returns."""
    r = W
    q1, q2 = np.quantile(r[:-1], [1 / 3, 2 / 3])
    st = np.digitize(r, [q1, q2])
    A = np.full((bins, bins), -np.inf)
    for i in range(len(r) - 1):
        A[st[i], st[i + 1]] = max(A[st[i], st[i + 1]], r[i + 1])
    return A


def _maxplus_eigen(A):
    """Max-plus eigenvalue = maximum cycle mean (Karp), via tropical powers."""
    n = len(A)
    best = -np.inf
    P = A.copy()
    for k in range(1, n + 1):
        d = np.diag(P)
        d = d[np.isfinite(d)]
        if d.size:
            best = max(best, d.max() / k)
        # tropical matrix product P = P ⊗ A
        P = np.max(P[:, :, None] + A[None, :, :], axis=1)
    return best


def k_eigen(frame, n, norm, **_):
    W = ops.windows(frame.ret, n)
    raw = ops.nan(frame.n)
    start = max(n - 1, frame.n - int(frame.eval_tail))
    for t in range(start, frame.n):
        w = W[t]
        if np.isnan(w).any():
            continue
        A = _transition_matrix(w)
        lam_max = _maxplus_eigen(A)
        lam_min = -_maxplus_eigen(np.where(np.isfinite(A), -A, -np.inf))
        raw[t] = (lam_max + lam_min)  # best cycle up vs worst cycle down
    sig = raw / scale_of(frame, n)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_determinant(frame, n, norm, **_):
    """Tropical determinant (permanent) of the 3×3 Hankel matrix of returns:
    max over permutations of Σ r_{i+σ(i)} - the best combinatorial momentum,
    minus the min-plus one (worst)."""
    import itertools
    r = frame.ret
    perms = list(itertools.permutations(range(3)))
    H = np.stack([ops.shift(r, i + j) for i in range(3) for j in range(3)], axis=1).reshape(-1, 3, 3)
    with np.errstate(all="ignore"):
        sums = np.stack([H[:, [0, 1, 2], list(p)].sum(axis=1) for p in perms], axis=1)
        raw = np.max(sums, axis=1) + np.min(sums, axis=1)
    sig = ops.rmean(raw, n) / scale_of(frame, n)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_linear_space(frame, n, norm, **_):
    """Hilbert projective (tropical) distance from today's log-OHLC to the window
    median log-OHLC: max_i(x_i − y_i) − min_i(x_i − y_i); signed by which of
    high/low carries the deviation."""
    X = np.vstack([ops.safe_log(c) for c in (frame.open, frame.high, frame.low, frame.close)]).T
    Y = np.vstack([ops.rmedian(X[:, i], n) for i in range(4)]).T
    with np.errstate(all="ignore"):
        diff = X - Y
        raw = diff.max(axis=1) - diff.min(axis=1)
        sig = (diff[:, 1] + diff[:, 2]) / (raw + 1e-12) * np.minimum(1.0, raw / scale_of(frame, n))
    return finish(frame, sig, raw, norm, n, 1.0)


def k_curve(frame, n, norm, **_):
    """Tropical curve in (x = return, y = Δlog volume): f = max(a, b + x, c + y, d + x + y);
    the active monomial tells which quadrant of the (price, volume) plane we are in."""
    x = frame.bar_ret / scale_of(frame, n)
    y = frame.vol_ret / (ops.rstd(frame.vol_ret, n) + 1e-9)
    W = ops.windows(x, n)
    V = ops.windows(y, n)
    with np.errstate(all="ignore"):
        a = np.nanmedian(W, axis=1) + np.nanmedian(V, axis=1)
        b = np.nanquantile(W, 0.8, axis=1)
        c = np.nanquantile(V, 0.8, axis=1)
        d = b + c - a
        terms = np.vstack([a, b + x, c + y, d + x + y])
        k = np.argmax(np.nan_to_num(terms, nan=-1e9), axis=0)
    sig = np.where(k == 3, 1.0, np.where(k == 1, 0.5, np.where(k == 2, -0.3, -0.5))) * np.sign(x) * np.minimum(1, np.abs(x))
    sig[np.isnan(terms).any(axis=0)] = np.nan
    return finish(frame, sig, k.astype(float), norm, n, 1.0)


def k_minplus(frame, n, norm, **_):
    """Morphology = min-plus / max-plus convolution: opening (erode then dilate)
    and closing (dilate then erode) of log price with a flat n-structuring
    element; closing − opening = tropical trend residual."""
    lp = ops.safe_log(frame.close)
    erode = ops.rmin(lp, n)
    dilate = ops.rmax(lp, n)
    opening = ops.rmax(erode, n)
    closing = ops.rmin(dilate, n)
    raw = (lp - 0.5 * (opening + closing))
    sig = raw / (0.5 * (closing - opening) + 1e-9)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_tsrl(frame, n, norm, **_):
    """Tropical support/resistance: f(p) = max_k (v_k − |p − p_k|/scale) over the
    window's volume-at-price; corners (ties) are tropical S/R levels; TSRL =
    distance from the close to the nearest corner, signed towards it."""
    P = ops.windows(frame.vwap, n)
    Vv = ops.windows(frame.volume, n)
    raw = ops.nan(frame.n)
    sig = ops.nan(frame.n)
    start = max(n - 1, frame.n - int(frame.eval_tail))
    for t in range(start, frame.n):
        p, v = P[t], Vv[t]
        if np.isnan(p).any() or np.isnan(v).any() or p.max() <= p.min():
            continue
        grid_p = np.linspace(p.min(), p.max(), 48)
        scale = (p.max() - p.min()) / 8.0 + 1e-12
        vn = v / (v.max() + 1e-12)
        F = vn[None, :] - np.abs(grid_p[:, None] - p[None, :]) / scale  # (grid, k)
        top2 = np.sort(F, axis=1)[:, -2:]
        corner = np.abs(top2[:, 1] - top2[:, 0]) < 0.05
        if not corner.any():
            continue
        levels = grid_p[corner]
        close = frame.close[t]
        j = int(np.argmin(np.abs(levels - close)))
        dist = (levels[j] - close) / close
        raw[t] = dist * 1e4
        sig[t] = np.tanh(dist / (scale_of(frame, n)[t] + 1e-9)) * 0.5  # drawn towards the nearest tropical level
    return finish(frame, sig, raw, norm, n, 1.0)


def variants():
    return (
        make_variants(D, 1, "tropical polynomial", "Tropical momentum polynomial (TMP)", k_tmp,
                      "TMP(x) = max_k (a_k + k·x), x = standardised bar return, a_0 = median r, a_1 = range, a_2 = skew·range, a_3 = 0.01·kurt·range",
                      "the winning monomial's degree: ≥ 2 extreme move (continuation), 0 base rate (reversion)", 2,
                      grid(extra=dict(d=DEGREES, semiring=("max", "max", "min"))))
        + make_variants(D, 2, "tropical root", "Tropical roots (critical thresholds)", k_root,
                        "root₁₂ = a₀ − a₁, root₂₃ = a₁ − a₂; x > root₂₃ extreme → continuation, x < root₁₂ sub-normal → reversion", "", 2,
                        grid(extra=dict(d=DEGREES)))
        + make_variants(D, 3, "tropical discriminant", "Distance to the corner locus", k_discriminant,
                        "gap between the two largest monomials: 0 on the tropical hypersurface (regime boundary)",
                        "deep inside a cell the regime is unambiguous; on the locus, stand down", 2, grid(extra=dict(d=DEGREES)))
        + make_variants(D, 4, "Newton polygon", "Newton polygon convexity", k_newton,
                        "lower convex hull of (i, cumulative return); last hull slope − first hull slope", "", 3)
        + make_variants(D, 5, "tropical eigenvector", "Max-plus eigenvalue of the transition matrix", k_eigen,
                        "λ_max = maximum cycle mean (Karp) of A_ij = max return on tercile transition i→j, plus the min-plus dual", "", 3)
        + make_variants(D, 6, "tropical determinant", "Tropical permanent of the return Hankel matrix", k_determinant,
                        "max_σ Σ r_{i+σ(i)} + min_σ Σ r_{i+σ(i)} over the 3×3 Hankel matrix, window mean", "", 2)
        + make_variants(D, 7, "tropical linear space", "Hilbert projective distance of OHLC", k_linear_space,
                        "d(x, y) = max_i(x_i − y_i) − min_i(x_i − y_i) between today's log OHLC and the window median", "", 2)
        + make_variants(D, 8, "tropical curve", "Price-volume tropical curve cell", k_curve,
                        "active monomial of max(a, b + x, c + y, d + x + y) in (return, Δlog volume)", "", 2)
        + make_variants(D, 9, "min-plus algebra", "Morphological trend residual", k_minplus,
                        "closing − opening of log price (min-plus/max-plus convolutions with a flat window)", "", 2)
        + make_variants(D, 10, "tropical intersection", "Tropical support-resistance locus (TSRL)", k_tsrl,
                        "corner locus of f(p) = max_k (v_k − |p − p_k|/scale) over volume-at-price; signed distance to the nearest corner", "", 3)
    )
