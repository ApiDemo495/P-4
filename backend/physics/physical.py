"""Sections 1, 2 and 4: Landauer thermal valve, planetary solar flux and the
work-to-rest-mass ratio.

Each function returns a plain dict with ``value`` (the BTC-weight or signed
quantity the layer uses), the intermediates the UI prints as *logic*, and a
``source`` tag.  No randomness anywhere: the same inputs give the same output.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from backend.physics import constants as K


def sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------
# Section 2 helpers: where is the Sun over the mining fleet right now?
# --------------------------------------------------------------------------
def solar_declination(now: float) -> float:
    """Spencer's approximation of the declination angle (rad) from day of year."""
    day = datetime.fromtimestamp(now, tz=timezone.utc).timetuple().tm_yday
    g = 2.0 * math.pi / 365.0 * (day - 1)
    return (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))


def irradiance(lat_deg: float, lon_deg: float, now: float) -> float:
    """I(phi, lambda, t) = I0 * max(0, sin d sin phi + cos d cos phi cos(hour angle))  [W/m^2]."""
    phi = math.radians(lat_deg)
    delta = solar_declination(now)
    utc_hours = (now % 86400.0) / 3600.0
    local_solar_hours = (utc_hours + lon_deg / 15.0) % 24.0
    hour_angle = math.radians(15.0 * (local_solar_hours - 12.0))
    cos_zenith = math.sin(delta) * math.sin(phi) + math.cos(delta) * math.cos(phi) * math.cos(hour_angle)
    return K.SOLAR_CONSTANT * max(0.0, cos_zenith)


def local_hour(lon_deg: float, now: float) -> float:
    return ((now % 86400.0) / 3600.0 + lon_deg / 15.0) % 24.0


def fleet_ambient_offset_k(now: float) -> float:
    """Hashrate-weighted diurnal ambient offset (K): warmest ~15:00 local."""
    total = sum(h[3] for h in K.MINING_HUBS)
    offset = 0.0
    for _name, _lat, lon, share, _pv, _base in K.MINING_HUBS:
        hour = local_hour(lon, now)
        offset += share / total * K.AMBIENT_DIURNAL_SWING_K * 0.5 * math.cos(2 * math.pi * (hour - 15.0) / 24.0)
    return offset


# --------------------------------------------------------------------------
# Section 1: Landauer thermal valve
# --------------------------------------------------------------------------
def landauer(hashrate: float, hashrate_series: list[float], now: float, source: str) -> dict:
    t_avg = K.ASIC_JUNCTION_K + fleet_ambient_offset_k(now)
    per_bit = K.K_B * t_avg * K.LN2                                    # J per erased bit
    s_dot = hashrate * K.BITS_ERASED_PER_DOUBLE_SHA * per_bit           # W (= J/s dissipated at the floor)
    peak = max([hashrate, *hashrate_series]) if hashrate_series else hashrate
    s_max = peak * (1.0 + K.THERMAL_HEADROOM) * K.BITS_ERASED_PER_DOUBLE_SHA * K.K_B * K.ASIC_JUNCTION_K * K.LN2
    theta = s_dot / s_max if s_max > 0 else 0.0
    w_thermal = 1.0 - sigmoid(K.LAMBDA_THERMAL * (theta - K.THETA_STAR))
    cycle_entropy_j = s_dot * 60.0
    return {
        "key": "landauer", "section": "1", "name": "Landauer thermal valve",
        "value": w_thermal, "unit": "w_BTC",
        "theta": theta, "theta_star": K.THETA_STAR, "t_avg_k": t_avg,
        "landauer_j_per_bit": per_bit, "s_dot_w": s_dot, "s_max_w": s_max,
        "cycle_dissipation_j": cycle_entropy_j, "hashrate_h_s": hashrate,
        "source": source,
        "logic": (
            f"E_bit = k_B·T·ln2 = {per_bit:.3e} J at T = {t_avg:.1f} K; "
            f"Ṡ = H·1024·E_bit = {hashrate:.3e} H/s × 1024 × {per_bit:.2e} = {s_dot:.3e} W; "
            f"Ṡ_max = (1+{K.THERMAL_HEADROOM})·peak-30d = {s_max:.3e} W; "
            f"Θ = Ṡ/Ṡ_max = {theta:.4f}; w_thermal = 1 − σ({K.LAMBDA_THERMAL:g}·(Θ − {K.THETA_STAR})) = {w_thermal:.4f}"
        ),
    }


# --------------------------------------------------------------------------
# Section 2: solar flux / energy opportunity cost
# --------------------------------------------------------------------------
def solar(hashrate: float, now: float) -> dict:
    total_share = sum(h[3] for h in K.MINING_HUBS)
    p_total = 0.0        # normalised: 1.0 = fleet at full baseload + full-sun PV
    grid_cost = 0.0
    hubs = []
    for name, lat, lon, share, pv_frac, base_frac in K.MINING_HUBS:
        w = share / total_share
        irr = irradiance(lat, lon, now)
        supply = base_frac + pv_frac * irr / K.SOLAR_CONSTANT
        hour = local_hour(lon, now)
        demand = 1.0 + K.GRID_PEAK_PREMIUM * math.cos(2 * math.pi * (hour - K.GRID_PEAK_LOCAL_HOUR) / 24.0)
        p_total += w * supply
        grid_cost += w * demand
        hubs.append({"hub": name, "irradiance_w_m2": round(irr, 1), "local_hour": round(hour, 2),
                     "supply": round(supply, 3), "grid_demand": round(demand, 3), "share": round(w, 3)})
    # C_hash = P_total / (H · ε): with the fleet's own efficiency this is the
    # marginal J per hash; normalised against the fleet's design point so that
    # Ω compares like with like.
    design = sum(h[3] / total_share * (h[5] + 0.5 * h[4]) for h in K.MINING_HUBS)
    c_hash = design / max(p_total, 1e-9)        # scarce supply ⇒ a dear hash (relative J/hash)
    omega = c_hash / max(grid_cost, 1e-9)       # Ω = C_hash / C_grid, Section 2.4
    w_solar = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, K.MU_SOLAR * (omega - 1.0)))))
    terminator_km = K.EARTH_RADIUS_M / 1000.0 * K.OMEGA_EARTH * 60.0
    hubs.sort(key=lambda h: -h["irradiance_w_m2"])
    return {
        "key": "solar", "section": "2", "name": "Planetary solar flux",
        "value": w_solar, "unit": "w_BTC",
        "omega": omega, "p_total_norm": p_total, "grid_cost_norm": grid_cost,
        "declination_deg": math.degrees(solar_declination(now)),
        "terminator_km_per_cycle": terminator_km, "hubs": hubs[:6],
        "sunlit_share": round(sum(h["share"] for h in hubs if h["irradiance_w_m2"] > 0), 3),
        "source": "model (geometry exact; hub table modelled)",
        "logic": (
            f"δ = {math.degrees(solar_declination(now)):+.2f}°, terminator sweeps {terminator_km:.1f} km/cycle; "
            f"P_total = Σ A·η·I + baseload = {p_total:.3f} of design {design:.3f}; "
            f"C_grid = {grid_cost:.3f}; Ω = C_hash/C_grid = {omega:.4f}; "
            f"w_solar = 1/(1+e^{{{K.MU_SOLAR:g}(Ω−1)}}) = {w_solar:.4f}; "
            f"sunlit hashrate share {sum(h['share'] for h in hubs if h['irradiance_w_m2'] > 0):.0%}"
        ),
    }


def blend_alpha(theta_hist: list[float], omega_hist: list[float]) -> float:
    """alpha = Var[Theta] / (Var[Theta] + Var[Omega]) over the trailing hour."""
    if len(theta_hist) < 3 or len(omega_hist) < 3:
        return 0.5
    vt = float(np.var(theta_hist))
    vo = float(np.var(omega_hist))
    if vt + vo <= 1e-18:
        return 0.5
    return vt / (vt + vo)


# --------------------------------------------------------------------------
# Section 4: work-to-rest-mass ratio
# --------------------------------------------------------------------------
def energy_mass(hashrate: float, market_ratio: float, ratio_hist: list[tuple[float, float, float]],
                now: float, source: str) -> dict:
    """R(t) = E_BTC / E_PAXG_eff and the phase angle of its motion (12.4).

    ``ratio_hist`` holds (time, R, market_ratio) samples kept by the engine.
    """
    e_btc = hashrate * K.FLEET_EFFICIENCY_J_PER_HASH * K.BLOCK_INTERVAL_S / K.BLOCK_REWARD_BTC   # J per marginal BTC
    e_paxg = K.GOLD_REFINE_J_PER_OZ + K.GOLD_TRANSPORT_J_PER_OZ + K.GOLD_CUSTODY_J_PER_OZ_PER_YEAR
    r = e_btc / e_paxg
    # Motion: compare the energy ratio's log-drift with the market ratio's
    # log-drift over the stored history (≥ 2 samples, ≤ 1 h).
    dln_r = dln_p = 0.0
    r_dot = 0.0
    if len(ratio_hist) >= 1:
        t0, r0, p0 = ratio_hist[0]
        dt = max(1.0, now - t0)
        if r0 > 0 and r > 0:
            dln_r = math.log(r / r0)
            r_dot = (r - r0) / dt
        if p0 > 0 and market_ratio > 0:
            dln_p = math.log(market_ratio / p0)
    phase = math.atan2(r_dot, r * K.PENDULUM_OMEGA0) if r > 0 else 0.0
    # Section 4.4 per-cycle update: ΔE absorbed this cycle by each substrate.
    de_btc = hashrate * K.FLEET_EFFICIENCY_J_PER_HASH * 60.0 / (K.BLOCK_REWARD_BTC * 60.0 / K.BLOCK_INTERVAL_S)
    de_paxg = K.GOLD_CUSTODY_J_PER_OZ_PER_YEAR * 60.0 / (365.25 * 86400.0)
    drift = (de_btc - de_paxg) / (e_btc + e_paxg)
    # Signed vote: energy embedding growing faster than the market ratio ⇒
    # BTC under-priced in energy terms ⇒ +; the opposite ⇒ −.  Scaled so a
    # 1 % divergence per hour reads as a full vote.
    divergence = dln_r - dln_p
    value = max(-1.0, min(1.0, divergence / 0.01))
    return {
        "key": "energy_mass", "section": "4", "name": "Work-to-rest-mass ratio (E = mc²)",
        "value": value, "unit": "vote",
        "e_btc_j": e_btc, "e_paxg_eff_j": e_paxg, "e_paxg_rest_mass_j": K.GOLD_REST_MASS_J,
        "ratio_oz_per_btc": r, "market_ratio": market_ratio, "phase_angle_rad": phase,
        "dln_r": dln_r, "dln_market": dln_p, "cycle_log_drift": drift,
        "source": source,
        "logic": (
            f"E_BTC = H·ε·600 s / reward = {hashrate:.3e}×{K.FLEET_EFFICIENCY_J_PER_HASH:.0e}×600/{K.BLOCK_REWARD_BTC:.3f} = {e_btc:.3e} J/BTC; "
            f"E_PAXG_eff = refine+transport+custody = {e_paxg:.3e} J/oz (rest mass m·c² = {K.GOLD_REST_MASS_J:.3e} J, not accessible); "
            f"R = {r:.1f} oz-equivalent/BTC vs market {market_ratio:.2f}; "
            f"Δln R = {dln_r:+.5f}, Δln P = {dln_p:+.5f} ⇒ vote {value:+.3f}; "
            f"Φ = atan(Ṙ/(R·ω₀)) = {math.degrees(phase):+.2f}°; "
            f"P⁽ⁿ⁺¹⁾ = P⁽ⁿ⁾·exp((ΔE_BTC−ΔE_PAXG)/(E_BTC+E_PAXG)) ⇒ {drift * 1e4:+.4f} bp/cycle"
        ),
    }

