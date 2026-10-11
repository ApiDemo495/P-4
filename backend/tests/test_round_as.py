"""Round AS - the competition rule, the ladder and the boundary pick."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.core.competition import (
    BoundaryPicker, CompetitionLedger, boundary_features, expected_windows_to_streak, streak_probability,
)

ROOT = Path(__file__).resolve().parents[2]


def test_the_rule_is_last_trade_vs_last_trade_and_a_tie_loses():
    led = CompetitionLedger()
    led.enter(1, "BUY", 83000.0)
    assert led.settle(1, 83000.0).tie is True and led.streak == 0 and led.ties == 1
    led.enter(2, "BUY", 83000.0)
    assert led.settle(2, 83000.01).won is True and led.streak == 1
    led.enter(3, "SELL", 83000.0)
    assert led.settle(3, 82999.99).won is True and led.streak == 2
    led.enter(4, "SELL", 83000.0)
    assert led.settle(4, 83000.01).won is False and led.streak == 0 and led.best_streak == 2
    assert led.streak_history[-1] == 2
    d = led.to_dict()
    assert d["scored"] == 4 and d["wins"] == 2 and d["win_rate"] == 0.5
    assert "tie is a loss" in d["rule"]


def test_the_ladder_is_the_users_ladder_and_the_arithmetic_is_honest():
    led = CompetitionLedger()
    for i in range(1, 21):
        led.enter(i, "BUY", 100.0)
        led.settle(i, 101.0)
    d = led.to_dict()
    assert [r["streak"] for r in d["ladder"]] == [5, 20, 50, 72]
    assert d["ladder"][0]["hit"] == 1 and d["ladder"][1]["hit"] == 1 and d["ladder"][1]["reached"]
    assert "$30" in d["ladder"][2]["prize"] and d["next_rung"] == 50 and d["to_next_rung"] == 30
    # 20/20 wins is still shrunk toward 0.5 - no "50-streak every afternoon"
    assert d["p_estimate"] < 0.9
    assert streak_probability(0.8, 10) == pytest.approx(0.1074, abs=1e-3)
    assert expected_windows_to_streak(0.5, 5) == pytest.approx(62.0)
    assert expected_windows_to_streak(0.0, 5) is None and expected_windows_to_streak(1.0, 5) == 5.0


def _tape(entry=83000.0, last_side=1.0, drift_ticks=0.0, n=120, now_ms=1_000_000.0):
    rows = []
    for i in range(n):
        t = now_ms - (n - i) * 100.0
        px = entry - drift_ticks * 0.01 * (1 - i / n)
        rows.append((t, round(px, 2), 0.05, 1.0 if i % 2 else -1.0))
    rows[-1] = (now_ms - 50.0, entry, 0.05, last_side)
    return np.asarray(rows, dtype=np.float64)


def test_boundary_features_read_the_entry_print_microprice_and_short_flow():
    f = boundary_features(_tape(last_side=1.0), bid=82999.99, ask=83000.01, bid_qty=9.0, ask_qty=1.0,
                          now_ms=1_000_000.0, pin_rate=0.0, tick=0.01)
    assert f.entry == 83000.0 and f.entry_side == 1.0
    # heavy bid -> microprice above the entry (ask*bid_qty dominates)
    assert f.microprice > f.entry and f.gap_ticks > 0 and f.book_imbalance == pytest.approx(0.8)
    assert f.prints_5s == 50 and abs(f.ofi5) <= 1.0
    assert boundary_features(np.zeros((0, 4)), 1, 2, 1, 1, 0.0, 0.0) is None


def test_the_picker_keeps_a_confident_recipe_and_flips_a_coin_flip_on_the_print():
    pk = BoundaryPicker()
    strong = boundary_features(_tape(last_side=1.0), 82999.99, 83000.01, 1.0, 9.0, 1_000_000.0, 0.0, 0.01)
    keep = pk.pick(1, strong, "BUY", 0.80)
    assert keep.side == "BUY" and not keep.flipped and "recipe kept" in keep.note
    # a 51 % BUY whose entry print sat at the ask with a heavy ask -> SELL
    flip = pk.pick(2, strong, "BUY", 0.51)
    assert flip.side == "SELL" and flip.flipped and flip.p_up < 0.5
    d = flip.as_dict()
    assert d["p_side"] > 0.5 and set(d["contributions"]) == set(BoundaryPicker.PRIOR)


def test_a_pinned_tape_breaks_the_tie_toward_the_microprice():
    pk = BoundaryPicker()
    f = boundary_features(_tape(last_side=0.0), 82999.99, 83000.01, 5.0, 5.0, 1_000_000.0, 0.4, 0.01)
    f.entry_side, f.ofi5, f.book_imbalance, f.gap_ticks = 0.0, 0.0, 0.0, 0.3   # balanced, microprice a shade above
    pick = pk.pick(3, f, "SELL", 0.5)
    assert pick.side == "BUY" and "tie broken toward the microprice" in pick.note


def test_the_picker_learns_online_and_ties_teach_nothing():
    pk = BoundaryPicker()
    f = boundary_features(_tape(last_side=1.0), 82999.99, 83000.01, 1.0, 9.0, 1_000_000.0, 0.0, 0.01)
    w0 = dict(pk.w)
    pk.pick(1, f, "BUY", 0.6)
    pk.learn(1, None)
    assert pk.updates == 0 and pk.w == w0
    for i in range(2, 40):
        pk.pick(i, f, "BUY", 0.6)
        pk.learn(i, False)                   # the recipe keeps saying up; the tape keeps going down
    assert pk.updates == 38 and pk.w["recipe"] < w0["recipe"] and pk.w["recipe"] >= 0.2
    assert pk.w["bounce"] < w0["bounce"]     # buy-print -> down got reinforced
    d = pk.to_dict()
    assert d["updates"] == 38 and d["log_loss_200"] is not None


def test_the_engine_enters_on_the_last_trade_and_settles_the_competition():
    src = (ROOT / "backend/core/cycle_manager.py").read_text()
    assert "self.market.last_trade(self.asset)" in src and "_boundary_pick(draft, cycle, price)" in src
    assert "self.competition.enter(cycle, locked.signal" in src
    assert "self.competition.settle(signal.cycle_number, settle_px)" in src
    assert "self.picker.learn(signal.cycle_number" in src
    assert 'payload["competition"]' in src and 'payload["boundary_pick"]' in src
    hub = (ROOT / "backend/data/market_hub.py").read_text()
    assert "def last_trade(self, asset: str)" in hub and "def l1(self, asset: str)" in hub
    index = (ROOT / "backend/web/index.html").read_text()
    app = (ROOT / "backend/web/app.js").read_text()
    assert 'id="competition-cell"' in index and 'id="c-ladder"' in index and 'id="c-pick"' in index
    assert "function renderCompetition" in app and "msg.data.competition" in app
    assert "/api/competition" in (ROOT / "backend/api/routes_brain.py").read_text()
