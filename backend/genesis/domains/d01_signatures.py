"""Domain 1 - rough path signatures of price trajectories (210 formulas).

Level-1 / level-2 / level-3 iterated integrals of a 2-D path X = (x, y)
over the last n candles, the Lévy area A = S^{xy} - S^{yx}, signature
kernels against reference paths, log-signature, path development (Chen),
signature distance, cross-asset signatures and normalised signatures.
"""
from __future__ import annotations

import numpy as np

from backend.genesis import ops
from backend.genesis.domains._common import finish, path_pair, scale_of
from backend.genesis.spec import grid, make_variants

D = 1
PATHS = ("price_volume", "price_trades", "price_depth", "logret_vol")


def _sig2(x: np.ndarray, y: np.ndarray, n: int):
    """Level-1 and level-2 signature terms over trailing windows of n+1 points.
    Returns S_x, S_y, S_xy, S_yx (each length-N arrays)."""
    X, Y = ops.windows(x, n + 1), ops.windows(y, n + 1)
    dX, dY = np.diff(X, axis=1), np.diff(Y, axis=1)
    cX = np.cumsum(dX, axis=1) - dX  # path value before each increment
    cY = np.cumsum(dY, axis=1) - dY
    with np.errstate(all="ignore"):
        S_x, S_y = np.nansum(dX, axis=1), np.nansum(dY, axis=1)
        S_xy = np.nansum(cX * dY, axis=1)  # ∫ x dy
        S_yx = np.nansum(cY * dX, axis=1)  # ∫ y dx
    bad = np.isnan(dX).any(axis=1) | np.isnan(dY).any(axis=1)
    for a in (S_x, S_y, S_xy, S_yx):
        a[bad] = np.nan
    return S_x, S_y, S_xy, S_yx, dX, dY, cX, cY


def k_l1(frame, n, norm, path="price_volume", **_):
    x, _y = path_pair(frame, path)
    S = x - ops.shift(x, n)
    raw = S / (x + 1e-12)
    return finish(frame, raw / scale_of(frame, n) / np.sqrt(n), raw, norm, n, 1.0)


def k_levy(frame, n, norm, path="price_volume", **_):
    x, y = path_pair(frame, path)
    S_x, S_y, S_xy, S_yx, *_ = _sig2(x, y, n)
    area = S_xy - S_yx
    with np.errstate(all="ignore"):
        l2sm = area / (np.abs(S_x) * np.abs(S_y) + 1e-4)
    return finish(frame, np.clip(l2sm, -1, 1), area, norm, n, 0.3)


def k_l3(frame, n, norm, path="price_volume", **_):
    x, y = path_pair(frame, path)
    S_x, S_y, S_xy, S_yx, dX, dY, cX, cY = _sig2(x, y, n)
    # S^{xyx} = ∫∫∫ dx dy dx : running (∫ x dy) integrated against dx
    with np.errstate(all="ignore"):
        run_xy = np.cumsum(cX * dY, axis=1) - cX * dY
        S_xyx = np.nansum(run_xy * dX, axis=1)
        scale = (np.nanstd(dX, axis=1) ** 2) * np.nanstd(dY, axis=1) * n ** 1.5 + 1e-12
        sig = S_xyx / scale
    return finish(frame, sig, S_xyx, norm, n, 1.0)


def k_kernel(frame, n, norm, path="logret_vol", ref="line", **_):
    """Signature kernel K(X, Y) = Σ_level <S_l(X), S_l(Y)>, truncated at level 2,
    against an up-trend reference (constant positive increments, constant
    volume) minus the same against the mirrored down-trend."""
    x, y = path_pair(frame, path)
    S_x, S_y, S_xy, S_yx, dX, dY, *_ = _sig2(x, y, n)
    sx, sy = np.nanstd(dX, axis=1) + 1e-12, np.nanstd(dY, axis=1) + 1e-12
    S_x, S_y, S_xy, S_yx = S_x / sx, S_y / sy, S_xy / (sx * sy), S_yx / (sx * sy)
    if ref == "line":
        rx, ry = np.full(n, 1.0), np.zeros(n)
    elif ref == "accel":
        rx, ry = np.linspace(0.2, 1.8, n), np.zeros(n)
    else:  # climax: price up with rising volume
        rx, ry = np.full(n, 1.0), np.linspace(-1, 1, n)
    cx, cy = np.cumsum(rx) - rx, np.cumsum(ry) - ry
    R = dict(x=rx.sum(), y=ry.sum(), xy=(cx * ry).sum(), yx=(cy * rx).sum())
    with np.errstate(all="ignore"):
        k_up = S_x * R["x"] + S_y * R["y"] + 0.5 * (S_x ** 2) * 0.5 * (R["x"] ** 2) + S_xy * R["xy"] + S_yx * R["yx"]
        k_dn = -S_x * R["x"] + S_y * R["y"] + 0.5 * (S_x ** 2) * 0.5 * (R["x"] ** 2) - S_xy * R["xy"] - S_yx * R["yx"]
        svt = (k_up - k_dn) / (n * n)
    return finish(frame, svt, k_up - k_dn, norm, n, 1.0)


def k_logsig(frame, n, norm, path="price_volume", **_):
    x, y = path_pair(frame, path)
    S_x, S_y, S_xy, S_yx, dX, dY, *_ = _sig2(x, y, n)
    area = 0.5 * (S_xy - S_yx)  # log-signature level-2 coordinate
    with np.errstate(all="ignore"):
        sig = area / (np.nanstd(dX, axis=1) * np.nanstd(dY, axis=1) * n + 1e-12)
    return finish(frame, sig * ops.sign(S_x), area, norm, n, 1.0)


def k_chen(frame, n, norm, path="price_volume", **_):
    """Path development by Chen: S(X*Y) = S(X) ⊗ S(Y).  The level-2 cross term
    of the concatenation of two halves is S¹_X ⊗ S¹_Y; its antisymmetric part
    says whether the second half continued the first in (x, y) space."""
    x, y = path_pair(frame, path)
    h = max(2, n // 2)
    ax = x - ops.shift(x, h)                 # second half increments
    ay = y - ops.shift(y, h)
    bx = ops.shift(x, h) - ops.shift(x, 2 * h)  # first half
    by = ops.shift(y, h) - ops.shift(y, 2 * h)
    with np.errstate(all="ignore"):
        cross = bx * ay - by * ax
        sx, sy = frame.roll("close_std", 2 * h) + 1e-12, ops.rstd(y, 2 * h) + 1e-12
        sig = cross / (sx * sy) * ops.sign(ax + bx)
    return finish(frame, sig, cross, norm, n, 1.0)


def k_sigdist(frame, n, norm, path="price_volume", **_):
    x, y = path_pair(frame, path)
    S_x, S_y, S_xy, S_yx, dX, dY, *_ = _sig2(x, y, n)
    cur = np.vstack([S_x, S_y, S_xy - S_yx]).T
    prev = np.vstack([ops.shift(c, n) for c in cur.T]).T
    with np.errstate(all="ignore"):
        scale = np.nanstd(cur, axis=0) + 1e-12
        dist = np.sqrt(np.nansum(((cur - prev) / scale) ** 2, axis=1))
        dist[np.isnan(cur).any(axis=1) | np.isnan(prev).any(axis=1)] = np.nan
        sig = dist * ops.sign(S_x) / 3.0
    return finish(frame, sig, dist, norm, n, 1.0)


def k_sigvol(frame, n, norm, **_):
    """Signature of the (time, return) path: Lévy area between clock time and
    cumulative return is positive when returns arrived late (accelerating)."""
    t = np.arange(frame.n, dtype=float)
    r = np.nancumsum(frame.ret)
    S_t, S_r, S_tr, S_rt, dT, dR, *_ = _sig2(t, r, n)
    with np.errstate(all="ignore"):
        area = S_tr - S_rt
        sig = area / (n * n * (np.nanstd(dR, axis=1) + 1e-12))
    return finish(frame, sig, area, norm, n, 1.0)


def k_cross(frame, n, norm, **_):
    S_a, S_b, S_ab, S_ba, dA, dB, *_ = _sig2(np.nancumsum(frame.ret), np.nancumsum(frame.other_ret), n)
    with np.errstate(all="ignore"):
        area = S_ab - S_ba
        lead = area / (np.nanstd(dA, axis=1) * np.nanstd(dB, axis=1) * n + 1e-12)
        # positive area: this asset led the other; negative: the other led -> follow its last move
        sig = np.where(lead < 0, -lead * ops.sign(S_b), lead * ops.sign(S_a))
    return finish(frame, sig, area, norm, n, 1.0)


def k_normsig(frame, n, norm, path="price_volume", **_):
    x, y = path_pair(frame, path)
    S_x, S_y, S_xy, S_yx, dX, dY, *_ = _sig2(x, y, n)
    with np.errstate(all="ignore"):
        length = np.nansum(np.sqrt((dX / (np.nanstd(dX, axis=1, keepdims=True) + 1e-12)) ** 2
                                   + (dY / (np.nanstd(dY, axis=1, keepdims=True) + 1e-12)) ** 2), axis=1) + 1e-12
        sig = (S_xy - S_yx) / (np.nanstd(dX, axis=1) * np.nanstd(dY, axis=1) + 1e-12) / length
    return finish(frame, sig, S_xy - S_yx, norm, n, 1.0)


def variants():
    P = dict(path=PATHS)
    return (
        make_variants(D, 1, "L1 signature", "Level-1 signature drift", k_l1,
                      "S¹ = x[n] − x[0] over the last n candles, scaled by σ_ret·√n",
                      "positive: the path ended above where it started by more than noise", 2, grid(extra=P))
        + make_variants(D, 2, "L2 Lévy area", "Level-2 signature momentum (L2SM)", k_levy,
                        "S^{xy} = Σ (x_i − x_0)(y_{i+1} − y_i), S^{yx} likewise; A = S^{xy} − S^{yx}; L2SM = A / (|S^x||S^y| + ε)",
                        "L2SM > 0.3 price leads volume up (genuine momentum); < −0.3 down; |L2SM| < 0.1 decoupled", 2, grid(extra=P))
        + make_variants(D, 3, "L3 signature", "Level-3 term S^{xyx}", k_l3,
                        "S^{xyx} = ∫∫∫ dx dy dx computed as Σ (running ∫x dy) · dx, scaled by σ_x²σ_y n^{3/2}",
                        "sign tells whether price pushes were reinforced by volume and then price again", 2, grid(extra=P))
        + make_variants(D, 4, "signature kernel", "Signature volatility tensor (SVT)", k_kernel,
                        "K(X,Y)=Σ_{l≤2}⟨S_l(X),S_l(Y)⟩ against an up-trend reference minus the mirrored down reference",
                        "SVT > 0 path resembles an up-trend signature, < 0 a down-trend, small = noise", 2,
                        grid(extra=dict(path=("logret_vol", "price_volume", "price_trades"), ref=("line", "accel", "climax"))))
        + make_variants(D, 5, "log-signature", "Log-signature area", k_logsig,
                        "level-2 log-signature coordinate ½(S^{xy} − S^{yx}) scaled by σ_xσ_y n, oriented by sign S^x",
                        "large |value| with the sign of the drift: the volume path wrapped around the price move", 2, grid(extra=P))
        + make_variants(D, 6, "path development (Chen)", "Chen concatenation cross term", k_chen,
                        "S(X*Y)=S(X)⊗S(Y): antisymmetric part of S¹_X⊗S¹_Y for first/second half of the window",
                        "positive: second half continued the first half in (x, y) space", 2, grid(extra=P))
        + make_variants(D, 7, "signature distance", "Signature distance to previous window", k_sigdist,
                        "‖(S^x,S^y,A)_now − (·)_prev‖ in units of their own spread, oriented by sign S^x",
                        "a path whose shape just changed a lot, in the direction of its drift", 2, grid(extra=P))
        + make_variants(D, 8, "signature volatility", "Time-return Lévy area", k_sigvol,
                        "Lévy area of the (clock time, cumulative return) path: ∫t dr − ∫r dt",
                        "positive: returns arrived late in the window (acceleration), negative: early (fading)", 2)
        + make_variants(D, 9, "cross-asset signature", "BTC↔PAXG lead-lag area", k_cross,
                        "Lévy area of (cum ret this asset, cum ret other asset); negative area = other asset led",
                        "when the other leg led, follow its last move; when this leg led, follow its own", 2)
        + make_variants(D, 10, "normalised signature", "Length-normalised Lévy area", k_normsig,
                        "Lévy area divided by the total variation (path length) of the standardised path",
                        "area per unit of path travelled - a cleaner momentum reading on choppy windows", 2, grid(extra=P))
    )
