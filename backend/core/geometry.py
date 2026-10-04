"""The geometric emotion layer (Round Z).

"The human emotion is never seen directly - it is mapped through the physical
and geometric deformation it causes in the numbers."  Five measurements on the
live tape and book, each a real computation with a stated formula, bounded to
[0, 1] (stress-like) or [-1, 1] (signed) so the Bayesian emotion filter can
take them as evidence next to the microstructure formulas:

1. **Topological data analysis of the book** - zero-dimensional persistent
   homology of the liquidity surface.  Levels are points on the price axis
   weighted by depth; a *cavity* is a gap between consecutive populated levels
   that is wide relative to the tick grid.  The persistence of a cavity is its
   width in units of the median gap; the Betti-0 curve counts the components
   alive at each scale.  ``tda_tearing`` = share of the top-of-book depth that
   sits behind cavities with persistence >= 3 (liquidity "holes" opening is
   the geometric signature of limit orders being pulled before a stampede).

2. **Phase-space reconstruction (Takens)** - delay embedding of the 1 s log
   return series (dimension 3, lag 1).  ``lyapunov`` is the Rosenstein local
   divergence rate of nearest-neighbour trajectories; positive = the
   attractor is spiralling (chaotic feedback), ~0 = stable.

3. **Critical slowing down** - lag-1 autocorrelation and variance of the 1 s
   returns on a rolling window, each compared with the previous window.
   Rising a1 *and* rising variance is the universal precursor of a phase
   transition (Scheffer et al.); ``csd`` ramps 0..1 on their joint rise.

4. **Non-equilibrium thermodynamics of BTC <-> PAXG** - capital flux
   J = signed BTC flow − signed PAXG flow (normalised), affinity X = the
   30 s return differential; entropy production σ = J·X ≥ 0 when flow
   follows return (heat moving down the gradient, i.e. a rotation out of the
   hot asset into the cold one); ``entropy_production`` is σ scaled to [0,1],
   ``heat_direction`` says which way capital is moving.

5. **Non-commutative (quantum) decision probability** - the order effect of
   two binary "questions" asked of the tape: A = "price up over the next tick"
   and B = "aggressor was a buyer".  Classical probability demands
   P(A) = P(A∧B) + P(A∧¬B).  On a tape the conditioning order matters when
   the crowd is polarised; ``interference`` = P(A) − [P(B)P(A|B) + P(¬B)P(A|¬B)]
   estimated on two interleaved halves of the window (so the two orders of
   evaluation are genuinely different samples).  |interference| > 0 is the
   signature of context-dependent (non-Bayesian) decisions.

Every number is returned with its inputs so the panel can show the geometry,
not just the verdict.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

EPS = 1e-12


def _safe(x: Any, fallback: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return fallback
    return v if math.isfinite(v) else fallback


def _ramp(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(min(1.0, max(0.0, (v - lo) / (hi - lo))))


# ---------------------------------------------------------------------------
# 1. TDA of the order book
# ---------------------------------------------------------------------------
def book_topology(book: Any) -> dict:
    """0-dim persistent homology of the liquidity surface (both sides)."""
    try:
        arr = np.asarray(book, dtype=np.float64)
        bids, asks = arr[0], arr[1]
    except Exception:  # noqa: BLE001
        return {"available": False, "tearing": 0.0, "cavities": 0, "max_persistence": 0.0, "betti0": []}
    out_sides = []
    for side in (bids, asks):
        px = side[:, 0]
        qty = side[:, 1]
        live = (px > 0) & (qty > 0)
        px, qty = px[live], qty[live]
        if px.size < 4:
            continue
        order = np.argsort(px)
        px, qty = px[order], qty[order]
        gaps = np.diff(px)
        med = float(np.median(gaps)) if gaps.size else 0.0
        if med <= 0:
            continue
        persistence = gaps / med                      # cavity width in median-gap units
        cavities = persistence >= 3.0
        # depth that sits *behind* a cavity, from the best price outward
        behind = 0.0
        total = float(qty.sum())
        if side is bids:
            # best bid is the highest price: walk downward
            seen_cavity = False
            for i in range(px.size - 1, -1, -1):
                if seen_cavity:
                    behind += qty[i]
                if i > 0 and cavities[i - 1]:
                    seen_cavity = True
        else:
            seen_cavity = False
            for i in range(px.size):
                if seen_cavity:
                    behind += qty[i]
                if i < cavities.size and cavities[i]:
                    seen_cavity = True
        # Betti-0 curve: components alive as the scale grows (median-gap units)
        scales = (1.0, 2.0, 3.0, 5.0, 8.0)
        betti0 = [int(1 + np.sum(persistence > s)) for s in scales]
        out_sides.append({
            "levels": int(px.size),
            "cavities": int(cavities.sum()),
            "max_persistence": round(float(persistence.max()), 3),
            "depth_behind_cavities": round(behind / total, 4) if total > 0 else 0.0,
            "betti0": betti0,
        })
    if not out_sides:
        return {"available": False, "tearing": 0.0, "cavities": 0, "max_persistence": 0.0, "betti0": []}
    tearing = float(np.mean([s["depth_behind_cavities"] for s in out_sides]))
    return {
        "available": True,
        "tearing": round(tearing, 4),
        "cavities": int(sum(s["cavities"] for s in out_sides)),
        "max_persistence": round(max(s["max_persistence"] for s in out_sides), 3),
        "betti0": [int(sum(x)) for x in zip(*(s["betti0"] for s in out_sides))],
        "scales": [1, 2, 3, 5, 8],
        "sides": out_sides,
        "formula": "persistence_i = gap_i / median(gap); cavity if >= 3; tearing = depth behind cavities / depth",
    }


# ---------------------------------------------------------------------------
# 2. Takens embedding + Rosenstein Lyapunov
# ---------------------------------------------------------------------------
def takens_lyapunov(series: np.ndarray, dim: int = 3, lag: int = 1, horizon: int = 4) -> dict:
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size - (dim - 1) * lag
    if n < 20:
        return {"available": False, "lyapunov": 0.0, "points": int(max(0, n))}
    emb = np.column_stack([x[i * lag: i * lag + n] for i in range(dim)])
    sd = float(emb.std()) or 1.0
    emb = emb / sd
    # nearest neighbour (temporal separation >= dim, so a point's own trail is excluded)
    d = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=2)
    idx = np.arange(n)
    d[np.abs(idx[:, None] - idx[None, :]) < dim] = np.inf
    usable = n - horizon
    if usable < 10:
        return {"available": False, "lyapunov": 0.0, "points": int(n)}
    nn = np.argmin(d[:usable, :usable], axis=1)
    d0 = d[np.arange(usable), nn]
    ok = np.isfinite(d0) & (d0 > EPS)
    if ok.sum() < 8:
        return {"available": False, "lyapunov": 0.0, "points": int(n)}
    div = []
    for k in range(1, horizon + 1):
        dk = np.linalg.norm(emb[np.arange(usable) + k] - emb[nn + k], axis=1)
        good = ok & (dk > EPS)
        div.append(float(np.mean(np.log(dk[good] / d0[good]))) if good.any() else 0.0)
    ks = np.arange(1, horizon + 1, dtype=np.float64)
    slope = float(np.polyfit(ks, np.asarray(div), 1)[0]) if len(div) >= 2 else 0.0
    return {
        "available": True,
        "lyapunov": round(slope, 4),             # per step (1 s)
        "divergence": [round(v, 4) for v in div],
        "points": int(n),
        "dim": dim, "lag": lag,
        "chaotic": round(_ramp(slope, 0.05, 0.6), 4),
        "formula": "Rosenstein: lambda = slope of <ln d_k/d_0> vs k on a (3,1) delay embedding of 1 s returns",
    }


# ---------------------------------------------------------------------------
# 3. Critical slowing down
# ---------------------------------------------------------------------------
def critical_slowing_down(returns: np.ndarray, window: int = 30) -> dict:
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2 * window:
        return {"available": False, "csd": 0.0}

    def stats(seg: np.ndarray) -> tuple[float, float]:
        seg = seg - seg.mean()
        var = float(np.mean(seg * seg))
        a1 = float(np.sum(seg[1:] * seg[:-1]) / (np.sum(seg * seg) + EPS))
        return a1, var

    a_prev, v_prev = stats(r[-2 * window:-window])
    a_now, v_now = stats(r[-window:])
    d_a1 = a_now - a_prev
    v_ratio = v_now / (v_prev + EPS)
    csd = _ramp(d_a1, 0.05, 0.4) * _ramp(v_ratio, 1.1, 2.5)
    return {
        "available": True,
        "a1_now": round(a_now, 4), "a1_prev": round(a_prev, 4),
        "variance_ratio": round(v_ratio, 3),
        "csd": round(float(csd), 4),
        "formula": "csd = ramp(Δa1, .05→.4) × ramp(var_now/var_prev, 1.1→2.5) on 1 s returns",
    }


# ---------------------------------------------------------------------------
# 4. BTC <-> PAXG entropy production
# ---------------------------------------------------------------------------
def entropy_production(btc_ticks: Any, paxg_ticks: Any, seconds: float = 30.0) -> dict:
    def flow_and_return(ticks: Any) -> tuple[float, float] | None:
        try:
            t = np.asarray(ticks, dtype=np.float64)
        except Exception:  # noqa: BLE001
            return None
        if t.ndim != 2 or t.shape[0] < 10:
            return None
        t_end = t[-1, 0]
        sel = t[t[:, 0] >= t_end - seconds * 1000.0]
        if sel.shape[0] < 5:
            return None
        notional = sel[:, 1] * sel[:, 2]
        flow = float(np.sum(notional * sel[:, 3]) / (np.sum(notional) + EPS))   # [-1, 1]
        ret = float(math.log(sel[-1, 1] / sel[0, 1])) * 1e4 if sel[0, 1] > 0 else 0.0
        return flow, ret

    b = flow_and_return(btc_ticks)
    p = flow_and_return(paxg_ticks)
    if b is None or p is None:
        return {"available": False, "entropy_production": 0.0, "heat_direction": "none"}
    J = 0.5 * (b[0] - p[0])            # capital flux BTC->PAXG is negative J
    X = (b[1] - p[1]) / 10.0           # affinity: return differential, 10 bps = 1
    sigma = J * X                      # > 0: flow follows the gradient
    direction = "BTC→PAXG (cooling)" if J < -0.05 else "PAXG→BTC (heating)" if J > 0.05 else "balanced"
    return {
        "available": True,
        "flux_J": round(J, 4), "affinity_X": round(X, 4),
        "entropy_production": round(_ramp(abs(sigma), 0.02, 0.5), 4),
        "sigma_raw": round(sigma, 5),
        "heat_direction": direction,
        "btc_flow": round(b[0], 4), "paxg_flow": round(p[0], 4),
        "btc_ret_bps": round(b[1], 2), "paxg_ret_bps": round(p[1], 2),
        "formula": "σ = J·X, J = ½(flow_BTC − flow_PAXG), X = (r_BTC − r_PAXG)/10 bps over 30 s",
    }


# ---------------------------------------------------------------------------
# 5. Non-commutative decision probability
# ---------------------------------------------------------------------------
def quantum_interference(ticks: Any) -> dict:
    try:
        t = np.asarray(ticks, dtype=np.float64)
    except Exception:  # noqa: BLE001
        return {"available": False, "interference": 0.0}
    if t.ndim != 2 or t.shape[0] < 60:
        return {"available": False, "interference": 0.0}
    px, side = t[:, 1], t[:, 3]
    up = (np.diff(px) > 0).astype(float)           # A: next tick up
    buyer = (side[:-1] > 0)                         # B: aggressor was a buyer
    # Two interleaved halves: P(A) measured on one, the conditional
    # decomposition on the other - two genuinely different "orders of asking".
    first, second = slice(0, None, 2), slice(1, None, 2)
    pa = float(up[first].mean()) if up[first].size else 0.5
    b2, a2 = buyer[second], up[second]
    pb = float(b2.mean()) if b2.size else 0.5
    pa_b = float(a2[b2].mean()) if b2.any() else pa
    pa_nb = float(a2[~b2].mean()) if (~b2).any() else pa
    classical = pb * pa_b + (1.0 - pb) * pa_nb
    interference = pa - classical
    return {
        "available": True,
        "p_up": round(pa, 4), "p_buyer": round(pb, 4),
        "p_up_given_buyer": round(pa_b, 4), "p_up_given_seller": round(pa_nb, 4),
        "classical": round(classical, 4),
        "interference": round(interference, 4),
        "polarisation": round(_ramp(abs(interference), 0.03, 0.2), 4),
        "formula": "I = P(A) − [P(B)P(A|B) + P(¬B)P(A|¬B)], A = next tick up, B = buyer aggressor; halves interleaved",
    }


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------
def analyze(tape: Any, asset: str, one_second_returns_bps: np.ndarray | None = None) -> dict:
    """All five measurements on one tape; never raises."""
    asset = asset.upper()
    other = "PAXG" if asset == "BTC" else "BTC"
    out: dict = {"available": False}
    try:
        ticks = tape.ticks(asset)
        book = tape.book(asset)
        try:
            other_ticks = tape.ticks(other)
        except Exception:  # noqa: BLE001
            other_ticks = None
        r1 = np.asarray(one_second_returns_bps if one_second_returns_bps is not None else [], dtype=np.float64)
        if r1.size < 20 and ticks is not None and len(ticks) > 20:
            # fall back to per-tick log returns in bps
            px = np.asarray(ticks)[:, 1]
            r1 = np.diff(np.log(px[px > 0])) * 1e4
        tda = book_topology(book)
        tak = takens_lyapunov(r1[-240:])
        csd = critical_slowing_down(r1[-120:])
        ent = (entropy_production(ticks, other_ticks) if asset == "BTC"
               else entropy_production(other_ticks, ticks))
        qi = quantum_interference(ticks)
        # One stress number for the emotion filter: the geometry's own vote.
        parts = [tda.get("tearing", 0.0) if tda.get("available") else 0.0,
                 tak.get("chaotic", 0.0) if tak.get("available") else 0.0,
                 csd.get("csd", 0.0) if csd.get("available") else 0.0,
                 ent.get("entropy_production", 0.0) if ent.get("available") else 0.0,
                 qi.get("polarisation", 0.0) if qi.get("available") else 0.0]
        live = sum(1 for d in (tda, tak, csd, ent, qi) if d.get("available"))
        stress = float(np.mean(parts)) if live else 0.0
        out = {
            "available": live > 0,
            "live_measurements": live,
            "stress": round(stress, 4),
            "tda": tda, "takens": tak, "csd": csd, "thermo": ent, "quantum": qi,
            "read": _read(tda, tak, csd, ent, qi),
        }
    except Exception as exc:  # noqa: BLE001
        out = {"available": False, "reason": str(exc)[:160]}
    return out


def _read(tda: dict, tak: dict, csd: dict, ent: dict, qi: dict) -> str:
    bits = []
    if tda.get("available"):
        bits.append(f"book manifold {'tearing' if tda['tearing'] > 0.25 else 'intact'} "
                    f"({tda['cavities']} cavities, max persistence {tda['max_persistence']:.1f})")
    if tak.get("available"):
        bits.append(f"attractor {'spiralling' if tak['lyapunov'] > 0.2 else 'stable'} (λ {tak['lyapunov']:+.2f}/s)")
    if csd.get("available"):
        bits.append(f"critical slowing {'rising' if csd['csd'] > 0.3 else 'absent'} "
                    f"(a1 {csd['a1_now']:+.2f}, var×{csd['variance_ratio']:.2f})")
    if ent.get("available"):
        bits.append(f"heat {ent['heat_direction']} (σ {ent['sigma_raw']:+.3f})")
    if qi.get("available"):
        bits.append(f"decision logic {'non-classical' if qi['polarisation'] > 0.3 else 'classical'} "
                    f"(I {qi['interference']:+.3f})")
    return "; ".join(bits) or "no geometry yet"
