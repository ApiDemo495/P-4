"""Regime classification and regime-gated activation (Part 5 of the spec).

Five regimes from the tape itself:
    trending        Hurst > 0.58 and |drift| above noise
    mean_reverting  Hurst < 0.45
    high_vol        realised vol in the top quintile of the last 500 minutes
    low_vol         realised vol in the bottom quintile
    cascade         a 3-minute move beyond 4σ with one-sided taker flow
(`high_vol`/`cascade` take precedence over the Hurst regimes; `cascade` over all.)

Each regime activates a subset of the active pool by domain, exactly the
sizes the specification lists (80 / 70 / 60 / 50 / 40 of 200), filled in
fitness order from the preferred domains first and the rest of the pool
after.
"""
from __future__ import annotations

import numpy as np

from backend.genesis import ops

REGIMES = ("trending", "mean_reverting", "high_vol", "low_vol", "cascade")

#: regime -> (preferred domains, how many of the active 200 to run)
GATES: dict[str, tuple[tuple[int, ...], int]] = {
    "trending": ((1, 5, 6, 9), 80),           # momentum + signature families
    "mean_reverting": ((2, 4, 3), 70),        # topology + transport
    "high_vol": ((7, 10, 3), 60),             # quantum + entropy
    "low_vol": ((8, 6), 50),                  # p-adic + tropical
    "cascade": ((4, 5, 7), 40),               # liquidation / transport
}


def classify(frame) -> dict:
    """Regime of the latest candle, with the numbers behind it."""
    n = frame.n
    if n < 60:
        return {"regime": "low_vol", "hurst": None, "vol_pct": None, "drift_z": 0.0, "note": "warming up"}
    H = frame.roll("hurst", 64)
    h = float(H[-1]) if np.isfinite(H[-1]) else 0.5
    rv = frame.roll("rv", 15)
    hist = rv[-500:]
    hist = hist[np.isfinite(hist)]
    cur = float(rv[-1]) if np.isfinite(rv[-1]) else float("nan")
    pct = float(np.mean(hist < cur)) if hist.size and np.isfinite(cur) else 0.5
    sigma = float(frame.roll("ret_std", 60)[-1]) if np.isfinite(frame.roll("ret_std", 60)[-1]) else 1e-6
    move3 = float(np.nansum(frame.ret[-3:]))
    drift_z = move3 / (sigma * np.sqrt(3) + 1e-12)
    with np.errstate(all="ignore"):
        flow = np.nansum(frame.buy_vol[-3:] - frame.sell_vol[-3:]) / (np.nansum(frame.buy_vol[-3:] + frame.sell_vol[-3:]) + 1e-12)
    one_sided = abs(flow) > 0.5 if np.isfinite(flow) else False
    if abs(drift_z) > 4.0 and (one_sided or np.isnan(flow)):
        regime = "cascade"
    elif pct >= 0.8:
        regime = "high_vol"
    elif pct <= 0.2:
        regime = "low_vol"
    elif h > 0.58 and abs(drift_z) > 0.8:
        regime = "trending"
    elif h < 0.45:
        regime = "mean_reverting"
    else:
        regime = "trending" if abs(drift_z) > 1.2 else "mean_reverting"
    return {"regime": regime, "hurst": round(h, 3), "vol_pct": round(pct, 3), "drift_z": round(drift_z, 2),
            "flow": None if not np.isfinite(flow) else round(float(flow), 3),
            "note": f"H={h:.2f} · vol pct {pct:.0%} · 3-min move {drift_z:+.1f}σ"}


def regime_labels(frame) -> np.ndarray:
    """Per-candle regime labels for the whole frame (used by autopsies and the
    regime-stability metric).  Vectorised version of :func:`classify`."""
    n = frame.n
    out = np.array(["low_vol"] * n, dtype=object)
    if n < 60:
        return out
    H = np.nan_to_num(frame.roll("hurst", 64), nan=0.5)
    rv = frame.roll("rv", 15)
    pct = ops.rrank(rv, 500) * 0.5 + 0.5
    sigma = frame.roll("ret_std", 60)
    move3 = ops.rsum(frame.ret, 3)
    with np.errstate(all="ignore"):
        dz = move3 / (sigma * np.sqrt(3) + 1e-12)
    dz = np.nan_to_num(dz)
    pct = np.nan_to_num(pct, nan=0.5)
    out[:] = np.where(H < 0.45, "mean_reverting", np.where(np.abs(dz) > 1.2, "trending", "mean_reverting"))
    out[(H > 0.58) & (np.abs(dz) > 0.8)] = "trending"
    out[pct <= 0.2] = "low_vol"
    out[pct >= 0.8] = "high_vol"
    out[np.abs(dz) > 4.0] = "cascade"
    return out


def gate(active: list, regime: str) -> list:
    """Pick the regime's subset of the active (fitness-ordered) formulas."""
    preferred, size = GATES.get(regime, ((), 80))
    first = [f for f in active if f.spec.domain in preferred]
    rest = [f for f in active if f.spec.domain not in preferred]
    return (first + rest)[:size]
