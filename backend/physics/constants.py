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

