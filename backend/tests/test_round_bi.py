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


def test_temporal_arbitrage_prices_rolls_and_settles_the_twin():
    """Round BL: the synthetic twin (60 s -> roll -> 10 s) is priced on the
    tape with the exact variance decomposition, rolled at :60 from the first
    leg, settled strictly, and the ledger is the realised spread."""
    import numpy as np
    from backend.core import temporal_arb as ta

    rng = np.random.default_rng(7)
    n = 20 * 60 * 10
    t = np.arange(n) * 100.0
    # strongly mean-reverting tape at the roll scale: an AR(1) in 10 s blocks
    blocks = n // 100
    br = np.zeros(blocks)
    for i in range(1, blocks):
        br[i] = -0.6 * br[i - 1] + rng.normal(0, 3.0)
    r = np.repeat(br / 100.0, 100) + rng.normal(0, 0.05, n)
    p = 60000 * np.exp(np.cumsum(r) / 1e4)
    ticks = np.c_[t, p, np.ones(n), np.sign(r)]
    pr = ta.price_structures(ticks, "BUY", 0.3)
    assert pr["covered"] and pr["pairs"] >= ta.MIN_PAIRS
    # the decomposition sigma_70^2 = s60^2 + s10^2 + 2 rho s60 s10 holds on the tape
    assert abs(pr["sigma_70_model"] - pr["sigma_70"]) / pr["sigma_70"] < 0.15
    assert pr["rho_roll"] < -ta.ACT_RHO and pr["roll_rule"].startswith("fade")
    assert pr["spread_bps2"] < 0 and pr["p_win_10"] > 0.5
    assert len(pr["logic"]) >= 5 and "arcsin" in " ".join(pr["logic"])
    eng = ta.TemporalArbEngine()
    tr = eng.price(5, 100.0, "BUY", 0.3, ticks, now=0.0)
    assert tr.roll_at == 60.0 and tr.due_at == 70.0
    eng.roll(tr, 100.10)                       # leg A +10 bp -> fade -> SELL
    assert tr.leg_b_side == "SELL"
    eng.settle(tr, 100.05)                     # 10 s leg -5 bp: SELL wins; continuous +5 bp wins
    assert tr.leg_a_won and tr.leg_b_won and tr.continuous_won
    assert tr.twin_pnl > tr.continuous_pnl     # ~+15 vs +5 bp
    d = eng.to_dict()
    assert d["ledger"]["windows"] == 1 and d["ledger"]["twin_better"] == 1
    # a flat 70 s print is a loss for the continuous bet; the twin can still earn
    tr2 = eng.price(6, 100.0, "BUY", 0.3, ticks, now=100.0)
    eng.roll(tr2, 100.08)
    eng.settle(tr2, 100.0)
    assert tr2.flat_70 and not tr2.continuous_won and tr2.twin_pnl > 0
    assert eng.to_dict()["ledger"]["flat_windows"] == 1
    # persistence round-trips the ledger
    fresh = ta.TemporalArbEngine()
    fresh.import_state(eng.export_state())
    assert fresh.windows == 2 and fresh.twin_better == 2
    # a short tape abstains instead of inventing a price
    assert ta.price_structures(ticks[:50], "BUY", 0.5)["covered"] is False
    html = open("backend/web/index.html", encoding="utf-8").read()
    js = open("backend/web/app.js", encoding="utf-8").read()
    assert 'id="twin-card"' in html and 'data-panel="twin"' in html
    assert "function renderTemporalArb" in js and 'case "TWIN"' in js


def test_hawkes_cascade_and_watchdog_guards():
    """Round BM: the Hawkes cascade is measured on the tape (buy cascade ->
    positive projected flow, calm Poisson tape -> branching near 0), and the
    watchdog / stage budgets exist in the cycle loop."""
    import numpy as np
    from backend.core.hawkes import hawkes_flow

    rng = np.random.default_rng(3)
    # a clustered buy cascade at the end of 4 minutes of calm tape
    n = 4 * 60 * 5
    t = np.sort(rng.uniform(0, 240_000, n))
    q = rng.exponential(0.05, n)
    side = rng.choice([-1.0, 1.0], n)
    burst_t = 240_000 - rng.exponential(1500, 120)
    burst = np.c_[burst_t, np.full(120, 60000.0), rng.exponential(0.3, 120), np.ones(120)]
    base = np.c_[t, np.full(n, 60000.0), q, side]
    ticks = np.vstack([base, burst])
    ticks = ticks[np.argsort(ticks[:, 0])]
    h = hawkes_flow(ticks, 70.0, now_ms=240_000.0)
    assert h["covered"] and h["lambda_buy"] > h["lambda_sell"] and h["score"] > 0.3
    assert 0.0 <= h["branching"] <= 0.9 and len(h["logic"]) == 4
    calm = hawkes_flow(base, 70.0, now_ms=240_000.0)
    assert calm["covered"] and abs(calm["score"]) < 0.6 and calm["branching"] < h["branching"] + 0.3
    assert hawkes_flow(base[:20], 70.0)["covered"] is False
    src = open("backend/core/cycle_manager.py", encoding="utf-8").read()
    assert "async def _watchdog_loop" in src and "_dump_stacks" in src and "os._exit(75)" in src
    assert "asyncio.wait_for(self._prefetch(deadline)" in src
    assert "asyncio.wait_for(self._open_window(deadline)" in src
    assert "set_default_executor" in src
    assert 'feature:hawkes' in src and 'feature:twin_edge' in src
    html = open("backend/web/index.html", encoding="utf-8").read()
    js = open("backend/web/app.js", encoding="utf-8").read()
    assert 'id="w-hawkes"' in html and 'id="c-ring-fg"' in html
    assert "function renderHawkes" in js and "function renderWinRing" in js


def test_bocpd_dates_the_regime_and_regime_experts_route_walk_forward():
    """Round BN: the change-point detector finds a drift that began two
    windows ago; the meta-combiner fits one expert per regime and routes to
    it only where its walk-forward record in that regime beats the global."""
    import numpy as np
    from backend.core.bocpd import bocpd
    from backend.core import meta as M

    rng = np.random.default_rng(2)
    n = 20 * 60 * 10
    t = np.arange(n) * 100.0
    r = rng.normal(0, 0.3, n)
    r[-1400:] += 0.08                                     # drift up for the last 140 s
    p = 60000 * np.exp(np.cumsum(r) / 1e4)
    b = bocpd(np.c_[t, p, np.ones(n), np.sign(r)])
    assert b["covered"] and b["score"] > 0.3 and 60 <= b["run_age_s"] <= 260
    assert b["regime"] in ("young", "mature") and len(b["logic"]) == 5
    flat = bocpd(np.c_[t, np.full(n, 60000.0) * np.exp(np.cumsum(rng.normal(0, 0.3, n)) / 1e4), np.ones(n), np.ones(n)])
    assert flat["covered"] and abs(flat["score"]) < abs(b["score"])
    assert bocpd(np.c_[t[:100], p[:100], np.ones(100), np.ones(100)])["covered"] is False

    # regime experts: in regime 2 the sign of GOOD flips, so only an expert can learn it
    rev = H.HindsightReviewer()
    now = time.time()
    total = 320
    for i in range(total):
        regime = 2.0 if i % 2 else 1.0
        move = float(rng.normal(0, 10))
        good = (0.8 if regime == 1.0 else -0.8) * np.sign(move) + rng.normal(0, 0.4)
        inputs = {"GOOD": good, "fusion:recipe": float(rng.normal(0, 1)), "regime:bocpd": regime}
        groups = {"GOOD": "formula", "fusion:recipe": "reasoning", "regime:bocpd": "feature"}
        rev.record(i, "BUY", 0.5, inputs, groups, at=now - (total - i) * 70.0)
        rev.rows[-1].move_bps, rev.rows[-1].cost_bps, rev.rows[-1].settled = move, 0.5, True
    mc = M.MetaCombiner()
    mc.build_record(list(rev.rows))
    assert "regime:bocpd" not in mc.fit.names                # a routing key, never a feature
    assert "young" in mc.experts and "mature" in mc.experts
    e_rate, g_rate, k = mc.expert_record("mature")
    assert k >= M.EXPERT_MIN_GRADED and e_rate > g_rate and e_rate > 0.7
    pred = mc.predict({"GOOD": 0.9, "fusion:recipe": 0.0, "regime:bocpd": 2.0})
    assert pred["expert_used"] and pred["side"] == "SELL" and pred["regime"] == "mature"
    d = mc.to_dict()
    assert any(e["in_use"] for e in d["experts"])
    fresh = M.MetaCombiner()
    fresh.import_state(mc.export_state())
    assert fresh.expert_record("mature")[2] == k
    html = open("backend/web/index.html", encoding="utf-8").read()
    js = open("backend/web/app.js", encoding="utf-8").read()
    assert 'id="w-regime"' in html and 'id="hs-meta-experts"' in html
    assert "function renderRegime" in js
