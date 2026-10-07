"""Round AR - the brain's own dopamine and the three-minds analyst."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.core.dopamine import DopamineBrain, LAMBDA
from backend.core.triune import TriuneAnalyst

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- dopamine
def test_dopamine_is_a_prediction_error_not_a_reward():
    b = DopamineBrain()
    first = b.observe(10.0, 1.0, 0.5)
    assert first["delta"] == 10.0 and b.phasic > 0.5          # unexpected win: burst
    # the same win, repeated, is expected more and more - habituation
    bursts = [b.observe(10.0, 1.0, 0.5)["phasic"] for _ in range(6)]
    assert bursts[-1] < bursts[0] and b.expectation > 5.0
    # a merely expected win barely moves dopamine
    assert abs(bursts[-1]) < 0.35


def test_losses_hurt_about_twice_as_much_and_confident_failures_more():
    b = DopamineBrain()
    row = b.observe(-10.0, -1.0, 0.2)
    assert row["u"] == pytest.approx(-LAMBDA * 10.0) and row["phasic"] < 0
    low_conf = DopamineBrain().observe(-10.0, -1.0, 0.1)["phasic"]
    high_conf = DopamineBrain().observe(-10.0, -1.0, 0.9)["phasic"]
    assert high_conf < low_conf                                  # the confident miss dips deeper
    win_low = DopamineBrain().observe(10.0, 1.0, 0.1)["phasic"]
    win_high = DopamineBrain().observe(10.0, 1.0, 0.9)["phasic"]
    assert win_high < win_low                                    # an expected win is discounted


def test_the_brain_guards_against_its_own_hot_hand_and_tilt():
    b = DopamineBrain()
    for _ in range(6):
        b.observe(12.0, 1.0, 0.5)
    assert b.mood in ("ELATED", "CONTENT") and b.win_streak == 6
    assert b.overconfidence > 0 and b.appetite < 1.0 and b.appetite >= 0.5
    t = DopamineBrain()
    for _ in range(5):
        t.observe(-12.0, -1.0, 0.6)
    assert t.mood in ("FRUSTRATED", "DISAPPOINTED") and t.tilt > 0 and t.appetite < 1.0
    # the appetite is a guard: it never exceeds 1
    assert DopamineBrain().appetite == 1.0
    d = t.to_dict()
    assert d["available"] and d["guard"] and "tilt guard" in d["guard"][0]
    assert any("Schultz" in e for e in d["equations"])


def test_flat_windows_teach_without_a_transient_and_bursts_fade():
    b = DopamineBrain()
    b.observe(10.0, 1.0, 0.5)
    before = b.phasic
    b.tick_window()
    assert 0 < b.phasic < before
    flat = b.observe(0.0, 0.0, 0.5)
    assert flat["phasic"] == 0.0 and b.expectation < 10.0


# ------------------------------------------------------------------ triune
def test_minds_are_built_from_the_three_kinds_of_evidence():
    class R:
        def __init__(self, d, c, ok=True):
            self.decision, self.confidence, self.ok = d, c, ok

    m = TriuneAnalyst.minds_from(crowd_tone=0.5, social_sentiment=0.2, news_niv=-0.1,
                                 agents={"gemini": R("BUY", 0.8), "github": R("SELL", 0.4), "local": R("BUY", 0.5, ok=False)},
                                 ccs_value=-0.4, formula_consensus=-0.2, physics_vote=0.1)
    assert m["human"] == pytest.approx(0.6 * 0.5 + 0.2 * 0.2 - 0.2 * 0.1, abs=1e-4)
    assert m["ai"] == pytest.approx((0.8 - 0.4) / 2, abs=1e-4) and m["ai_answered"] == 2
    assert m["data"] == pytest.approx(-0.2 - 0.06 + 0.02, abs=1e-4)


def test_the_analyst_keeps_score_and_fades_a_crowd_that_is_reliably_wrong():
    a = TriuneAnalyst()
    # 40 windows: the data mind is right 80 %, the human mind wrong 80 %, AI silent
    for i in range(40):
        up = (i % 2) == 0
        data_right = (i % 5) != 0                        # 80 % right
        human_right = (i % 5) == 1                       # 20 % right
        data_vote = (0.6 if up else -0.6) * (1 if data_right else -1)
        human_vote = (0.5 if up else -0.5) * (1 if human_right else -1)
        a.remember(i, "BTC", {"human": human_vote, "ai": 0.0, "data": data_vote}, "FEAR", "BUY")
        a.score(i, up, 8.0 if up else -8.0)
    rep = a.report()
    assert rep["scored"] == 40
    assert rep["minds"]["data"]["hit_rate"] == pytest.approx(0.8, abs=0.01)
    assert rep["minds"]["human"]["hit_rate"] < 0.3 and rep["minds"]["human"]["contrarian"] is True
    assert rep["minds"]["ai"]["n"] == 0
    verdict = a.analyse({"human": 0.5, "ai": 0.0, "data": -0.5}, emotion="FEAR")
    # the crowd is read inverted, so both now say SELL and corroborate
    assert verdict["used"]["human"] == -0.5 and verdict["side"] == "SELL" and verdict["corroboration"] == 1.0
    assert verdict["weights"]["data"] > verdict["weights"]["human"]
    assert verdict["maturity"] == 1.0 and "inverted" in verdict["notes"][0]
    assert "FEAR" in rep["minds"]["data"]["by_emotion"]


def test_a_split_lowers_confidence_and_an_immature_record_stays_a_prior():
    a = TriuneAnalyst()
    v = a.analyse({"human": 0.6, "ai": -0.6, "data": 0.1})
    assert v["corroboration"] == 0.0 and v["confidence_multiplier"] == 0.85 and v["maturity"] == 0.0
    assert all(w == pytest.approx(0.15) for w in v["weights"].values())
    agree = a.analyse({"human": 0.6, "ai": 0.6, "data": 0.4})
    assert agree["corroboration"] == 1.0 and agree["confidence_multiplier"] == 1.10


def test_the_manager_wires_dopamine_into_the_brain_and_the_analyst_into_the_fusion():
    src = (ROOT / "backend/core/cycle_manager.py").read_text()
    assert "self.dopamine.observe(pnl_bps, outcome" in src
    assert "self.brain.dopamine_phasic = self.dopamine.brain_gain()" in src
    assert "self.triune.score(signal.cycle_number" in src and "_apply_triune" in src
    assert 'risk["size_multiplier"]' in src
    brain = (ROOT / "backend/brain/brain.py").read_text()
    assert "0.5 * float(self.dopamine_phasic)" in brain
    app_js = (ROOT / "backend/web/app.js").read_text()
    index = (ROOT / "backend/web/index.html").read_text()
    assert "function renderBrainMood" in app_js and 'id="brain-mood"' in index and 'id="w-triune"' in index
    assert len(re.findall(r"setInterval\(", app_js)) == 1
    routes = (ROOT / "backend/api/routes_brain.py").read_text()
    assert "/api/brain/dopamine" in routes and "/api/triune" in routes
