"""Physical constants and modelled magnitudes for the thermodynamic layer.

Exact SI constants are exact.  Everything under *modelled magnitudes* is a
published-order-of-magnitude estimate, labelled ``model`` wherever it reaches
the UI, and replaced by telemetry when a live source answers.
"""
from __future__ import annotations

import math

# ----------------------------------------------------------------- exact SI
K_B = 1.380649e-23          # J/K   Boltzmann constant (2019 SI, exact)
LN2 = math.log(2.0)
C_LIGHT = 299_792_458.0     # m/s
SOLAR_CONSTANT = 1361.0     # W/m^2 at 1 AU
OMEGA_EARTH = 2.0 * math.pi / 86164.0905   # rad/s sidereal
EARTH_RADIUS_M = 6_371_000.0
TROY_OUNCE_KG = 0.0311035
GOLD_Z, GOLD_A = 79, 197

# ------------------------------------------------------- Landauer (Section 1)
BITS_ERASED_PER_DOUBLE_SHA = 1024      # two 512-bit compressions
ASIC_JUNCTION_K = 338.0                # ~65 C fleet-average junction temperature
AMBIENT_DIURNAL_SWING_K = 6.0          # ambient swing that junction tracks
THETA_STAR = 0.85                      # critical thermal occupancy
THERMAL_HEADROOM = 0.12                # S_max = (1 + headroom) x 30-day peak
LAMBDA_THERMAL = 12.0                  # sigmoid steepness

# -------------------------------------------------------- solar (Section 2)
MU_SOLAR = 6.0
# Major proof-of-work regions: (name, lat, lon, share of global hashrate,
# photovoltaic fraction of local supply, baseload fraction).  Shares are the
# Cambridge CBECI order of magnitude, normalised below.
MINING_HUBS = (
    ("Texas, US", 31.5, -99.0, 0.26, 0.35, 0.65),
    ("Georgia/NY, US", 36.0, -80.0, 0.12, 0.15, 0.85),
    ("Alberta/Quebec, CA", 52.0, -105.0, 0.07, 0.05, 0.95),
    ("Paraguay/Argentina", -25.0, -57.5, 0.04, 0.20, 0.80),
    ("Norway/Sweden", 64.0, 17.0, 0.03, 0.02, 0.98),
    ("Kazakhstan", 48.0, 68.0, 0.09, 0.10, 0.90),
    ("Russia (Siberia)", 56.0, 93.0, 0.10, 0.03, 0.97),
    ("Sichuan/Xinjiang, CN", 36.0, 95.0, 0.14, 0.25, 0.75),
    ("Ethiopia/Oman/UAE", 15.0, 45.0, 0.07, 0.40, 0.60),
    ("Bhutan/Malaysia", 20.0, 100.0, 0.04, 0.15, 0.85),
    ("Australia", -27.0, 140.0, 0.02, 0.45, 0.55),
    ("Iceland", 64.5, -18.0, 0.02, 0.0, 1.0),
)
GRID_PEAK_LOCAL_HOUR = 19.0            # evening demand peak
GRID_PEAK_PREMIUM = 0.35               # C_grid swings +-35 % around 1

# --------------------------------------------- energy-mass ratio (Section 4)
FLEET_EFFICIENCY_J_PER_HASH = 22e-12   # ~22 J/TH fleet average (model)
BLOCK_INTERVAL_S = 600.0
BLOCK_REWARD_BTC = 3.125 + 0.08        # subsidy after the 2024 halving + typical fees
GOLD_REFINE_J_PER_OZ = 4.4e9           # ~140 GJ/kg mine-to-dore energy intensity
GOLD_TRANSPORT_J_PER_OZ = 1.5e7
GOLD_CUSTODY_J_PER_OZ_PER_YEAR = 2.0e7
GOLD_REST_MASS_J = TROY_OUNCE_KG * C_LIGHT ** 2   # 2.796e15 J (never 'accessible')

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

# ------------------------------------------------------- fallback telemetry
MODEL_HASHRATE_H_PER_S = 9.5e20        # ~950 EH/s (model, 2025-26 magnitude)
MODEL_XAU_USD = None                   # unknown without a source: use PAXG itself
