"""Round T - the thermodynamic capital layer (BTC = work, PAXG = rest mass)."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from backend.agents import fusion as fusion_module
from backend.formulas import synthetic
from backend.physics import constants as K
from backend.physics import micro, physical, unified
from backend.physics.engine import PhysicsEngine

ROOT = Path(__file__).resolve().parents[2]


def _report(key: str, n: int = 6):
    eng = PhysicsEngine()
    for snap in synthetic.scenario(key).snapshots[-n:]:
        report = eng.compute(snap, "BTC")
    return report


def test_landauer_constant_and_valve_direction():
    e_bit = K.K_B * 300.0 * K.LN2
    assert abs(e_bit - 2.87e-21) < 0.02e-21          # Section 1.1 at 300 K
    now = time.time()
    hot = physical.landauer(1.0e21, [1.0e21], now, "model")       # at the 30-day peak ⇒ Θ ≈ 1/1.12
    cool = physical.landauer(0.6e21, [1.0e21], now, "model")
    assert hot["theta"] > cool["theta"]
    assert hot["value"] < cool["value"]              # saturation shifts weight to PAXG
    assert 0.0 < hot["value"] < 1.0 and "k_B" in hot["logic"]


def test_solar_geometry_is_exact_and_diurnal():
    # Equator at local solar noon on an equinox: cos(zenith) ≈ 1 ⇒ I ≈ I0.
    equinox = 1774483200.0                            # 2026-03-26 00:00 UTC (declination ≈ +2°)
    noon_lon0 = equinox + 12 * 3600
    assert physical.irradiance(0.0, 0.0, noon_lon0) > 0.99 * K.SOLAR_CONSTANT
    assert physical.irradiance(0.0, 0.0, equinox) == 0.0          # midnight
    terminator = K.EARTH_RADIUS_M / 1000 * K.OMEGA_EARTH * 60
    assert abs(terminator - 27.87) < 0.1                           # Section 2.1
    a = physical.solar(1e21, noon_lon0)
    b = physical.solar(1e21, noon_lon0 + 12 * 3600)
    assert a["omega"] != b["omega"] and 0 < a["value"] < 1


def test_energy_mass_ratio_and_phase():
    now = time.time()
    em = physical.energy_mass(9.5e20, 27.0, [(now - 1800, 800.0, 27.5)], now, "model")
    assert em["ratio_oz_per_btc"] > 0 and em["e_paxg_rest_mass_j"] > 1e15
    assert em["dln_r"] > 0 and em["dln_market"] < 0 and em["value"] > 0   # energy outran the market ⇒ BTC
    assert -math.pi / 2 <= em["phase_angle_rad"] <= math.pi / 2


def test_vpin_and_ou_read_the_tape():
    bull = synthetic.scenario("BULL").snapshots[-1]
    v = micro.vpin(bull)
    assert 0.0 <= v["value"] <= 1.0 and v["buckets"] == 50
    ou = micro.ornstein_uhlenbeck(bull)
    assert ou["kappa_per_min"] > 0 and ou["samples"] >= 30
    assert (ou["direction"] != 0) == ou["execute"]
    pend = micro.pendulum(bull)
    assert 0.0 <= pend["value"] <= 1.0
    spread = micro.avellaneda_stoikov(bull, 60.0)
    assert 0.0 < spread["value"] < 50.0            # half-spread in bp, sane
    assert spread["direction"] == 0


def test_window_vote_comes_from_microstructure_with_physical_drag():
    assert unified.window_vote(0.5, 0.3) == (0.0, 0.0)
    with_lean, drag = unified.window_vote(0.8, 0.3)       # BTC vote against a PAXG lean
    against, drag2 = unified.window_vote(0.2, 0.3)        # PAXG vote with the lean
    assert 0 < with_lean < 0.6 and drag > 0
    assert against == -0.6 and drag2 == 0.0
    w, clamped = unified.clamp_weight(0.95, 0.4, 0.15)
    assert w == 0.55 and clamped


def test_engine_report_is_complete_fast_and_honest():
    r = _report("BULL")
    assert r["pair"] == "BTC/PAXG" and -1 <= r["vote"] <= 1 and 0 < r["confidence"] < 1
    assert {m["key"] for m in r["mechanisms"]} == set(unified.MECHANISMS)
    assert all("logic" in m and "source" in m for m in r["mechanisms"])
    assert r["elapsed_us"] < 20_000
    assert "never a guaranteed yield" in r["composite"]["note"]
    assert "not implemented" in r["composite"]["note"]
    # PAXG is the mirror of BTC
    eng = PhysicsEngine()
    snap = synthetic.scenario("BEAR").snapshots[-1]
    b = eng.compute(snap, "BTC")["vote"]
    p = eng.compute(snap, "PAXG")["vote"]
    assert abs(b + p) < 1e-9


def test_fusion_weighs_the_layer_but_never_lets_it_veto():
    physics = {"vote": -0.9, "confidence": 0.8, "live_inputs": 0, "weights": {"w_final": 0.3}}
    with_layer = fusion_module.fuse({}, ccs_value=0.6, ccs_confidence=0.8, hsi=0.2, physics=physics)
    without = fusion_module.fuse({}, ccs_value=0.6, ccs_confidence=0.8, hsi=0.2)
    assert "physics" in with_layer.contributions and "physics" not in without.contributions
    assert with_layer.contributions["physics"]["status"] == "MODEL"
    # model-only telemetry counts half the configured weight
    assert abs(with_layer.contributions["physics"]["weight"] - 0.5 * fusion_module.cfg.SETTINGS.weight_physics) < 1e-9
    assert with_layer.decision == "BUY"                     # brain 0.4 vs physics 0.1: the brain still decides
    assert with_layer.score < without.score                 # ... but the layer counted
    assert "thermodynamic layer" in with_layer.reasoning
    assert with_layer.to_dict()["physics"] is physics


def test_scope_is_declared_in_the_api_and_docs():
    from backend.api import routes_physics

    status = {s["section"]: s["status"] for s in routes_physics.SECTIONS}
    for missing in ("3", "6", "7", "9"):
        assert status[missing] == "not implemented"
    notes = (ROOT / "docs/SPEC_NOTES.md").read_text()
    assert "## T" in notes and "cannot lose" in notes
    assert np.isfinite(K.GOLD_REST_MASS_J)
