"""Constants of the physics layer (Round AJ: planetary sections retired).

Exact SI constants are exact.  Everything under *modelled magnitudes* is a
published-order-of-magnitude estimate, labelled ``model`` wherever it reaches
the UI, and replaced by telemetry when a live source answers.
"""
from __future__ import annotations

import math

# ----------------------------------------------------------------- exact SI
K_B = 1.380649e-23          # J/K   Boltzmann constant (2019 SI, exact)
LN2 = math.log(2.0)

# ------------------------------------------------------- soft gates (Round BH)
def soft_gate(x: float, threshold: float) -> float:
    """Continuous replacement for ``1 if |x| >= threshold else 0``:
    ``x² / (x² + θ²)`` - ½ at the threshold, → 1 above it, → 0 below it but
    never exactly 0 while x ≠ 0.  A mechanism below its threshold now says
    "weak" with a small edge instead of a dead 0.00 (the user's report:
    half the app read 0.00; an honest small number carries information, a
    hard zero carries none)."""
    x = abs(float(x))
    th = max(1e-12, abs(float(threshold)))
    return x * x / (x * x + th * th)


def sign_of(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


# ------------------------------------------------ microstructure (Section 8)
VPIN_BUCKETS = 50
VPIN_CRIT = 0.40
OU_EQUILIBRIUM_WINDOW_S = 300.0
OU_MIN_SHARPE = 1.5
AS_RISK_AVERSION = 50.0                # per unit return (δ in return units)
PENDULUM_BINS_S = 30.0

# ------------------------------------------------------------ peg (Section 10)
PAXG_MINT_BURN_FEE = 0.0002            # 2 bps protocol fee (model)
GAS_FEE_FRACTION = 0.0001              # gas as a fraction of a 1-oz ticket (model)

# ------------------------------------------------- unified (Sections 11-12)
DELTA_W_DEFAULT = 0.15                 # band around the physical weight (spec example: 0.05)
KELLY_RIDGE = 1e-3
KELLY_CAP = 0.25
PENDULUM_OMEGA0 = 2.0 * math.pi / 3600.0   # natural period ~1 h (model prior)

