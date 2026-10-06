"""Round T - the thermodynamic capital layer (BTC = work, PAXG = rest mass)."""
from __future__ import annotations

from pathlib import Path

from backend.agents import fusion as fusion_module
from backend.formulas import synthetic
from backend.physics import constants as K
from backend.physics import kinetics, micro, unified
from backend.physics.engine import PhysicsEngine

ROOT = Path(__file__).resolve().parents[2]


def _report(key: str, n: int = 6):
    eng = PhysicsEngine()
    for snap in synthetic.scenario(key).snapshots[-n:]:
        report = eng.compute(snap, "BTC")
    return report


def test_kinetic_mechanisms_read_the_tape_and_flag_inactivity():
    bull = synthetic.scenario("BULL").snapshots[-1]
    bear = synthetic.scenario("BEAR").snapshots[-1]
    h = kinetics.hawkes(bull)
    assert h["active"] and 0.0 <= h["value"] < 1.0 and "Var/Mean" in h["logic"]
    e_up, e_dn = kinetics.entropy(bull), kinetics.entropy(bear)
    assert e_up["active"] and 0.0 <= e_up["value"] <= 1.0
    assert e_up["direction"] >= 0 >= e_dn["direction"]            # follows the informed flow's sign
    t = kinetics.temperature(bull)
    assert t["active"] and t["direction"] == 0 and 0.0 <= t["drag"] <= 0.6
    # A synthetic window carries 60 s of tape: the two-minute mechanisms say
    # so explicitly instead of voting 0.00.
    k = kinetics.kinetic(bull)
    d = kinetics.diffusion(bull)
    assert k["active"] is False and "baseline" in k["logic"]
    assert d["active"] is False and "overlapping tape" in d["logic"]


def test_ou_edge_is_cost_gated_and_capped():
    bear = synthetic.scenario("BEAR").snapshots[-1]
    ou = micro.ornstein_uhlenbeck(bear)
    assert ou["active"] and ou["cost_bps"] >= 0.5 and ou["half_life_s"] > 0
    if ou["execute"]:
        assert ou["edge_bps"] <= max(1.0, ou["sigma60_bps"]) + 1e-9
        assert ou["expected_bps"] > ou["cost_bps"]
    else:
        assert ou["direction"] == 0 and ou["edge_bps"] == 0.0


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


def test_blend_excludes_inactive_mechanisms_and_gates_on_cost():
    r = _report("BULL")
    assert set(r["active"]).isdisjoint(r["inactive"])
    assert all(m["key"] in r["inactive"] for m in r["mechanisms"] if not m["active"])
    assert set(r["kelly"]["names"]) == set(r["active"])
    assert r["composite"]["cost_bps"] > 0 and "net_edge_bps" in r["composite"]
    w, clamped = unified.clamp_weight(0.95, 0.5, 0.15)
    assert w == 0.65 and clamped


def test_engine_report_is_complete_fast_and_honest():
    r = _report("BULL")
    assert r["pair"] == "BTC/PAXG" and -1 <= r["vote"] <= 1 and 0 < r["confidence"] < 1
    assert {m["key"] for m in r["mechanisms"]} == set(unified.MECHANISMS)
    assert all("logic" in m and "source" in m for m in r["mechanisms"])
    assert r["elapsed_us"] < 20_000
    assert "never a guaranteed yield" in r["composite"]["note"]
    assert "not implemented" in r["composite"]["note"]
    assert all(m["active"] or m["direction"] == 0 for m in r["mechanisms"])
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
    assert "physics layer" in with_layer.reasoning
    assert with_layer.to_dict()["physics"] is physics


def test_scope_is_declared_in_the_api_and_docs():
    from backend.api import routes_physics

    status = {s["section"]: s["status"] for s in routes_physics.SECTIONS}
    for missing in ("3", "6", "7", "9"):
        assert status[missing] == "not implemented"
    for retired in ("1", "2", "4", "5"):
        assert status[retired] == "retired"
    notes = (ROOT / "docs/SPEC_NOTES.md").read_text()
    assert "## T" in notes and "cannot lose" in notes
    assert K.VPIN_CRIT == 0.40
