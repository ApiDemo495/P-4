"""Round BI - the last-hour hindsight review."""
from __future__ import annotations

import time
from pathlib import Path

from backend.core import hindsight as H


def _fill(rev: H.HindsightReviewer, n: int = 20, *, good: str = "A", bad: str = "B", biased: str = "C") -> None:
    now = time.time()
    for i in range(n):
        move = 10.0 if i % 2 == 0 else -10.0
        inputs = {good: 0.5 if move > 0 else -0.5, bad: -0.5 if move > 0 else 0.5, biased: 0.3 + (0.1 if move > 0 else -0.1)}
        groups = {good: "formula", bad: "formula", biased: "system"}
        rev.record(1000 + i, "BUY" if move > 0 else "SELL", 0.6, inputs, groups, at=now - (n - i) * 70.0)
        rev.settle(1000 + i, move, cost_bps=0.5)


def test_hour_review_grades_boosts_damps_and_inverts():
    rev = H.HindsightReviewer()
    _fill(rev, 20)
    a, b, c = rev.reviews["A"], rev.reviews["B"], rev.reviews["C"]
    assert a.hits == 20 and a.misses == 0 and a.action == "boosted" and a.multiplier == H.MULT_CEIL
    assert b.misses == 20 and b.inverted and b.action == "inverted"
    # the inverted input is read the other way round - its reading now agrees with the move
    assert rev.apply("B", -0.5) > 0
    # C leaned up all hour (bias 0.3) while half the windows fell: re-centred
    assert c.offset != 0.0 and abs(c.offset) <= 0.5 * c.spread + 1e-12
    assert "re-centred" in c.reason
    # bounded corrections only
    for cr in rev.reviews.values():
        assert H.MULT_FLOOR <= cr.multiplier <= H.MULT_CEIL


def test_inputs_below_min_decided_only_watch_and_are_not_changed():
    rev = H.HindsightReviewer()
    _fill(rev, H.MIN_DECIDED - 1)
    assert all(cr.action == "watching" for cr in rev.reviews.values())
    out, changed = rev.apply_map({"A": 0.4, "B": -0.2})
    assert out == {"A": 0.4, "B": -0.2} and changed == {}


def test_rows_older_than_an_hour_roll_off_and_state_round_trips():
    rev = H.HindsightReviewer()
    _fill(rev, 12)
    rev.record(5, "BUY", 0.5, {"A": 0.9}, {"A": "formula"}, at=time.time() - 2 * 3600)
    rev.settle(5, -50.0)
    assert all(l["cycle"] != 5 for l in rev.reviews["A"].losses)
    payload = rev.to_dict()
    assert payload["available"] and payload["windows_in_hour"] == 12 and payload["table"]
    assert payload["table"][0]["net_bps"] <= payload["table"][-1]["net_bps"]
    fresh = H.HindsightReviewer()
    assert fresh.import_state(rev.export_state()) == 13
    assert fresh.reviews["A"].hits == rev.reviews["A"].hits


def test_inputs_from_covers_every_group():
    class R:
        def __init__(self, d, c):
            self.decision, self.confidence = d, c
    inputs, groups = H.HindsightReviewer.inputs_from(
        formula_values={"VSD": 0.2, "HSI": 0.7, "TAI": -0.1}, directional={"VSD": "a", "TAI": "a"},
        agents={"gemini": R("SELL", 0.7), "github": R(None, None)}, ccs_value=0.3, physics_vote=-0.2,
        genesis_vote=0.1, niv=0.05, crowd_tone=-0.3, forward_vote=0.2, spec_score=0.4,
        features={"return_1s_bps": 3.0, "tick_rate_hz": 9.0})
    assert "HSI" not in inputs and inputs["agent:gemini"] == -0.7 and "agent:github" not in inputs
    assert set(groups.values()) == {"formula", "agent", "reasoning", "physics", "genesis", "emotion", "feature"}
    assert "feature:return_1s_bps" in inputs and "feature:tick_rate_hz" not in inputs


def test_dashboard_and_routes_carry_the_hindsight_panel():
    root = Path(__file__).resolve().parents[2]
    html = (root / "backend/web/index.html").read_text(encoding="utf-8")
    js = (root / "backend/web/app.js").read_text(encoding="utf-8")
    routes = (root / "backend/api/routes_brain.py").read_text(encoding="utf-8")
    assert 'id="hindsight-card"' in html and 'data-panel="hindsight"' in html
    assert "function renderHindsight" in js and "data.hindsight" in js
    assert '"/api/hindsight"' in routes


def test_opening_audit_grades_sixty_candles_through_every_layer():
    import numpy as np
    from backend.core.forward import ForwardExpectancy
    from backend.core.opening_audit import run_audit
    from backend.physics.engine import PhysicsEngine
    rng = np.random.default_rng(3)
    n = 7200
    end = int(time.time() * 1000) // 1000 * 1000
    t = np.arange(end - n * 1000, end, 1000, dtype=np.float64)
    px = 60000 * np.exp(np.cumsum(rng.normal(0, 2e-4, n)))
    vol = rng.gamma(2, 0.3, n)
    bars = np.column_stack([t, px, px * 1.0002, px * 0.9998, px, vol, vol * rng.uniform(0.3, 0.7, n)])
    hs = H.HindsightReviewer()
    rep = run_audit("BTC", bars, None, hs, physics=PhysicsEngine(), forward=ForwardExpectancy())
    assert rep.phase == "done", rep.error
    assert rep.graded >= 55 and rep.market["candles"] == rep.graded
    assert {"formula", "hedge", "physics", "emotion", "feature"} <= set(rep.layers)
    assert rep.summary.startswith("Last hour on BTC") and "Changed by majority" in rep.summary
    # the corrections are in force for the live windows immediately
    assert hs.reviewed_decided >= 50 and any(cr.action != "watching" for cr in hs.reviews.values())
    out, changed = hs.apply_map({name: 0.3 for name in list(hs.reviews)[:10]})
    assert changed
    html = (Path(__file__).resolve().parents[2] / "backend/web/index.html").read_text(encoding="utf-8")
    assert 'id="hs-audit"' in html and 'id="hs-audit-layers"' in html


def test_meta_combiner_learns_walk_forward_and_only_decides_when_earned():
    import numpy as np
    from backend.core import meta as M
    rng = np.random.default_rng(7)
    rev = H.HindsightReviewer()
    now = time.time()
    n = 200
    for i in range(n):
        move = float(rng.normal(0, 10))
        good = 0.6 * np.sign(move) + rng.normal(0, 0.5)       # informative
        noise = rng.normal(0, 1)                               # not
        inputs = {"GOOD": good, "NOISE": noise, "fusion:recipe": float(rng.normal(0, 1))}
        groups = {"GOOD": "formula", "NOISE": "formula", "fusion:recipe": "reasoning"}
        rev.record(i, "BUY", 0.5, inputs, groups, at=now - (n - i) * 70.0)
        rev.rows[-1].move_bps, rev.rows[-1].cost_bps, rev.rows[-1].settled = move, 0.5, True
    mc = M.MetaCombiner()
    graded = mc.build_record(list(rev.rows))
    assert graded >= 100 and mc.fit is not None
    w = dict(zip(mc.fit.names, mc.fit.w))
    assert w["GOOD"] > 0.3 and abs(w["NOISE"]) < abs(w["GOOD"]) / 2
    rate, k = mc.trailing()
    assert rate is not None and rate > 0.6 and mc.may_decide()[0]
    pred = mc.predict({"GOOD": 0.9, "NOISE": 0.0, "fusion:recipe": -0.5})
    assert pred["side"] == "BUY" and pred["drivers"][0]["name"] == "GOOD"
    # a fresh combiner with no graded record only watches
    fresh = M.MetaCombiner()
    fresh.refit(list(rev.rows))
    assert fresh.fit is not None and not fresh.may_decide()[0]
    d = mc.to_dict()
    assert d["decides"] and d["fit"]["top_weights"][0]["name"] == "GOOD"
    html = (Path(__file__).resolve().parents[2] / "backend/web/index.html").read_text(encoding="utf-8")
    assert 'id="hs-meta"' in html
