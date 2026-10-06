"""Sections 11-12: multi-mechanism Kelly, the thermodynamic band and the
composite metrics (TSR, phase angle).

Round AI: the blend runs over the *active* mechanisms only; the layer's
single output is the signed vote ``2·(w_micro − ½)·(1 − drag)`` for BTC
against PAXG (see ``engine.py``), which fusion weighs beside the brain and
the AI agents.
"""
from __future__ import annotations

import math

import numpy as np

from backend.physics import constants as K

MECHANISMS = ("ou", "vpin", "hawkes", "kinetic", "entropy", "diffusion", "pendulum", "temperature", "as_spread",
              "fragmentation", "peg")


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
