"""Round BD - five local models voting at once, and four horizons (55 / 60 /
70 / 120 s) issued at every boundary and settled on their own clocks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.agents.local_model_agent import thread_budget
from backend.agents.local_pool import MAX_MODELS, LocalModelPool
from backend.agents.model_fetch import read_manifest
from backend.core.horizons import HORIZONS, HorizonBook, horizon_stats

ROOT = Path(__file__).resolve().parents[2]


def test_the_manifest_is_back_to_the_one_baked_model_per_machine():
    """Round BF: the user asked to drop the five-model set and keep the old
    one - Qwen2.5-1.5B on 4+ cores, Qwen2.5-0.5B on 2-3 cores, slot 1.  The
    five slots stay available for uploads."""
    assert MAX_MODELS == 5
    small = read_manifest(ROOT / "models.json", cores=2)
    big = read_manifest(ROOT / "models.json", cores=4)
    assert [e["file"] for e in small] == ["qwen2.5-0.5b-instruct-q4_k_m.gguf"]
    assert [e["file"] for e in big] == ["qwen2.5-1.5b-instruct-q4_k_m.gguf"]
    raw = json.loads((ROOT / "models.json").read_text())
    assert len(raw["models"]) == 2 and all(m["slot"] == 1 for m in raw["models"])


def test_the_pool_has_five_slots_and_threads_are_shared(tmp_path):
    pool = LocalModelPool(store_dir=tmp_path)
    assert len(pool.slots) == 5
    assert thread_budget(expected_models=5) >= 1
    assert thread_budget(expected_models=1) >= thread_budget(expected_models=5)
    assert thread_budget(expected_models=5) <= 4


def _tape(seconds: int, drift_bps_per_s: float, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = seconds * 4
    t = 1.7e12 + np.arange(n) * 250.0
    steps = drift_bps_per_s / 4 / 1e4 + rng.normal(0, 0.8e-4, n)
    px = 60000.0 * np.cumprod(1.0 + steps)
    return np.column_stack([t, px, np.ones(n), np.where(steps > 0, 1.0, -1.0)])


def test_horizon_stats_read_a_trending_tape_at_each_horizon():
    tape = _tape(900, 0.6)
    for h in HORIZONS:
        st = horizon_stats(tape, h)
        assert st["covered"] and st["n"] > 10 and st["drift"] > 0.2, (h, st)
        assert 0.0 <= st["p_tie"] <= 1.0 and st["sigma_bps"] > 0
    assert not horizon_stats(_tape(100, 0.0), 120)["covered"]       # shorter than two horizons


def test_the_contest_lane_is_the_lock_and_every_lane_settles_strictly():
    book = HorizonBook(contest_horizon=70)
    picks = book.issue(1, 60000.0, "SELL", 0.3, _tape(900, 0.6), now=1000.0)
    by_h = {p.horizon: p for p in picks}
    assert by_h[70].side == "SELL" and by_h[70].score == -0.3                # never disagrees with the lock
    assert by_h[120].due_at == 1120.0 and by_h[55].due_at == 1055.0
    assert all(p.logic and any("PIN@" in line for line in p.logic) for p in picks)
    win_px = 59990.0 if by_h[55].side == "SELL" else 60010.0
    book.settle(by_h[55], win_px, now=1055.0)                               # the lane's own side came true: win
    book.settle(by_h[60], 60000.0, now=1060.0)                              # unchanged: tie = loss
    book.settle(by_h[70], 60010.0, now=1070.0)                              # rose: loss
    d = book.to_dict()
    lanes = {l["horizon"]: l for l in d["lanes"]}
    assert lanes[55]["wins"] == 1 and lanes[55]["streak"] == 1
    assert lanes[60]["ties"] == 1 and lanes[60]["losses"] == 1 and lanes[60]["win_rate"] == 0.0
    assert lanes[70]["losses"] == 1 and lanes[120]["settled"] == 0
    restored = HorizonBook(contest_horizon=70)
    restored.import_state(book.export_state())
    assert restored.lanes[55].wins == 1 and restored.lanes[60].ties == 1


def test_the_dashboard_and_api_carry_the_four_lanes():
    js = (ROOT / "backend" / "web" / "app.js").read_text()
    html = (ROOT / "backend" / "web" / "index.html").read_text()
    assert "function renderHorizons" in js and 'case "HORIZON"' in js
    assert 'id="w-horizons"' in html
    routes = (ROOT / "backend" / "api" / "routes_brain.py").read_text()
    assert '"/api/horizons"' in routes
    settings = (ROOT / "backend" / "web" / "settings.html").read_text()
    assert 'id="model-slot-5"' in settings and 'value="5">slot 5' in settings
