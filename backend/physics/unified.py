"""Sections 11-12: multi-mechanism Kelly, the thermodynamic band and the
composite metrics (TSR, phase angle).

The physical weight ``w_composite`` (Sections 1-2 blended by alpha) is the
centre of a band of half-width ``delta_w``; the microstructural Kelly weight
is clamped into that band (11.4).  The layer's single output is a signed vote
``2·(w_final − ½)`` for BTC against PAXG, which fusion weighs beside the brain
and the AI agents.
"""
from __future__ import annotations

import math

import numpy as np

from backend.physics import constants as K

MECHANISMS = ("amm", "fragmentation", "ou", "as_spread", "peg", "vpin", "pendulum", "energy_mass")


def kelly(mechanisms: dict[str, dict], history: dict[str, list[float]]) -> dict:
    """f = (Σ + ridge·I)⁻¹ μ  with μ the signed expected edge (bp) per mechanism."""
    names = [m for m in MECHANISMS if m in mechanisms]
    mu = np.array([float(mechanisms[m].get("direction", 0)) * float(mechanisms[m].get("edge_bps", 0.0))
                   for m in names])
    # Σ from the trailing history of signed edges; diagonal prior when short.
    cols = []
    for m in names:
        h = history.get(m) or []
        cols.append(np.array(h[-120:], dtype=float) if len(h) >= 10 else None)
    n = len(names)
    sigma = np.eye(n) * 4.0          # prior: 2 bp standard deviation per mechanism
    if all(c is not None for c in cols) and n > 0:
        L = min(len(c) for c in cols)
        M = np.column_stack([c[-L:] for c in cols])
        if L >= 10:
            sigma = np.cov(M.T) if n > 1 else np.array([[float(np.var(M))]])
            sigma = np.atleast_2d(sigma) + np.eye(n) * K.KELLY_RIDGE
    try:
        f = np.linalg.solve(sigma + np.eye(n) * K.KELLY_RIDGE, mu) if n else np.zeros(0)
    except np.linalg.LinAlgError:
        f = mu / np.maximum(np.diag(sigma), 1e-9)
    f = np.clip(f, -K.KELLY_CAP, K.KELLY_CAP)
    tilt = float(f.sum())            # net BTC tilt in "fractions of capital"
    w_micro = 0.5 + 0.5 * math.tanh(tilt / 0.6)
    return {
        "names": names, "mu_bps": [round(float(v), 4) for v in mu],
        "sigma_diag": [round(float(v), 4) for v in np.diag(sigma)] if n else [],
        "f": [round(float(v), 4) for v in f], "tilt": tilt, "w_micro": w_micro,
        "logic": (f"μ = signed edge per mechanism (bp); Σ from the last {min((len(history.get(m) or []) for m in names), default=0)} cycles"
                  f"{'' if all(c is not None for c in cols) else ' (diagonal prior until 10 cycles)'}; "
                  f"f = Σ⁻¹μ clipped to ±{K.KELLY_CAP}; Σf = {tilt:+.4f} ⇒ w_micro = ½ + ½·tanh(Σf/0.6) = {w_micro:.4f}"),
    }


def clamp_weight(w_micro: float, w_composite: float, delta_w: float) -> tuple[float, bool]:
    lo, hi = w_composite - delta_w, w_composite + delta_w
    clamped = w_micro < lo or w_micro > hi
    return max(lo, min(hi, w_micro)), clamped


def window_vote(w_micro: float, w_composite: float) -> tuple[float, float]:
    """The 60-second direction comes from the microstructure; the physical
    weight is a *portfolio* target that moves over hours, so it acts as drag
    on votes that lean against it rather than as the direction itself
    (deviation from 11.4, documented in SPEC_NOTES T)."""
    raw = 2.0 * (w_micro - 0.5)
    lean = 2.0 * (w_composite - 0.5)            # + ⇒ physics wants more BTC
    against = max(0.0, -math.copysign(1.0, raw) * lean) if raw != 0 else 0.0
    drag = min(0.6, against)
    return raw * (1.0 - drag), drag


def thermodynamic_sharpe(total_edge_bps: float, theta: float) -> float:
    """TSR = (Y_total / V) / sqrt(ΔS_BTC / S_max): return per unit of entropy
    budget consumed; ΔS/S_max over one cycle equals Θ."""
    if theta <= 0:
        return 0.0
    return (total_edge_bps / 1e4) / math.sqrt(theta) * 100.0   # scaled so 1 bp at Θ=1 reads 0.01


def phase_label(phi: float) -> str:
    if abs(abs(phi) - math.pi / 2) < math.radians(5):
        return "phase transition: flatten and wait"
    return "energy-dominant: accumulate BTC" if phi > 0 else "mass-dominant: accumulate PAXG" if phi < 0 else "balanced"
