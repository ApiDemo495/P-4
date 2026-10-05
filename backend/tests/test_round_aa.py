"""Round AA - risk engine, lock weights, inverse-RL utility, feed diagnostics."""
from __future__ import annotations

import numpy as np
import pytest

from backend.core import config as cfg
from backend.core import lock_weights, risk_engine, utility_inversion


def test_risk_engine_stop_is_the_mae_quantile_and_target_grows_with_edge():
    flat = risk_engine.size_levels("BTC", 20.0, edge=0.0, settings=cfg.SETTINGS)
    assert flat.mae_z == pytest.approx(1.2816, abs=1e-3)
    assert flat.sl_bps == pytest.approx(1.2816 * 20.0, rel=1e-3)
    assert flat.rr == pytest.approx(cfg.SETTINGS.rr_target, rel=1e-3)
    strong = risk_engine.size_levels("BTC", 20.0, edge=0.9, settings=cfg.SETTINGS)
    assert strong.sl_bps == flat.sl_bps and strong.rr > flat.rr and strong.rr <= risk_engine.RR_CEIL
    wide = risk_engine.size_levels("BTC", 2.0, spread_bps=4.0, settings=cfg.SETTINGS)
    assert wide.sl_bps >= 12.0 and "spread floor" in wide.method
    longer = risk_engine.size_levels("BTC", 20.0, horizon_seconds=240.0, settings=cfg.SETTINGS)
    assert longer.sigma_window_bps == pytest.approx(40.0)


def test_lock_weight_table_is_base_times_reliability_times_availability():
    rows = [
        {"source": "f:TAI", "reliability": 0.7, "n": 40},
        {"source": "f:VSD", "reliability": 0.7, "n": 40},
        {"source": "brain:CCSv2", "reliability": 0.3, "n": 40},
        {"source": "physics:layer", "reliability": 0.5, "n": 2},
    ]
    table = lock_weights.build(cfg.SETTINGS, rows)
    assert table.rows["formulas"].reliability > 1.0 > table.rows["drosophila"].reliability
    assert table.rows["physics"].reliability == pytest.approx(1.0)
    assert table.rows["gemini"].hit_rate is None and table.rows["gemini"].reliability == 1.0
    table.mark("drosophila")
    table.mark("formulas")
    table.mark("physics", 0.5)
    d = table.as_dict()
    assert d["physics"]["availability"] == 0.5 and d["gemini"]["present"] is False
    assert sum(d[k]["share"] for k in ("drosophila", "formulas", "physics")) == pytest.approx(1.0, abs=1e-6)
    assert lock_weights.reliability_multiplier(0.7, 1e9) == pytest.approx(1.4)
    assert lock_weights.reliability_multiplier(0.0, 1e9) == lock_weights.MULT_FLOOR


def _tape(loss_averse: bool):
    rng = np.random.default_rng(3)
    n = 900
    t = np.arange(n) * 100.0
    px = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    r = np.r_[0.0, np.diff(px)]
    if loss_averse:
        side = np.where(r < 0, -3.0, 1.0)
    else:
        side = np.where(r < 0, -1.0, 3.0)
    ticks = np.column_stack([t, px, np.abs(rng.normal(1, 0.2, n)) * np.abs(side), np.sign(side)])
    book = np.zeros((2, 25, 2))
    book[0, :, 0] = px[-1] * (1 - np.arange(1, 26) * 0.0004)
    book[1, :, 0] = px[-1] * (1 + np.arange(1, 26) * 0.0004)
    book[:, :, 1] = 1.0
    book[:, 20:, 1] = 5.0

    class Tape:
        def ticks(self, asset):
            return ticks

        def book(self, asset):
            return book

    return Tape()


def test_inverse_rl_reads_loss_aversion_and_tail_weighting_from_the_tape():
    scared = utility_inversion.analyze(_tape(True), "BTC")
    greedy = utility_inversion.analyze(_tape(False), "BTC")
    assert scared["available"] and scared["live_measurements"] == 3
    assert scared["lambda"]["lambda"] > 2.0 > greedy["lambda"]["lambda"]
    assert scared["intensity"]["loss_averse"] > 0.5 and greedy["intensity"]["gain_chasing"] > 0.5
    assert scared["alpha"]["alpha"] < 1.0 and scared["intensity"]["tail_fear"] > 0
    assert "lambda" in scared["method"].lower() or "λ" in scared["method"]


def test_inverse_rl_never_raises_on_an_empty_tape():
    class Empty:
        def ticks(self, asset):
            return np.zeros((0, 4))

        def book(self, asset):
            return None

    out = utility_inversion.analyze(Empty(), "BTC")
    assert out == {"available": False, "reason": "fewer than 30 ticks"}


def test_deep_model_carries_the_utility_evidence():
    from backend.core import deep_micro
    for key in ("loss_averse", "gain_chasing", "risk_averse", "risk_seeking", "tail_fear", "complacent"):
        assert key in deep_micro.EVIDENCE_LABEL
    assert deep_micro.LIKELIHOOD["PANIC"]["loss_averse"] > 0
    assert deep_micro.LIKELIHOOD["COMPLACENCY"]["complacent"] > 0
    assert deep_micro.LIKELIHOOD["FOMO"]["gain_chasing"] > 0


def test_window_clock_is_absolute_and_carries_the_phase():
    from backend.core import window_clock
    started, ends = 1_000_000.0, 1_000_060.0
    marks = window_clock.grid_marks(started, ends, 15.0, 30.0)
    assert [m["offset_seconds"] for m in marks] == [15.0, 30.0, 45.0]
    assert marks[1]["parts"] == ["formulas", "news"]
    block = window_clock.describe(
        started, ends, now=started + 42.0, period_seconds=60.0, cycle_id=7, marks=marks,
        minute_aligned=True, freshness_max_age_seconds=90.0, scoring_horizon_seconds=60.0)
    assert block["window_ends_at_ms"] == 1_000_060_000 and block["seconds_remaining"] == 18.0
    assert block["phase"]["label"] == "late" and block["phase"]["progress"] == pytest.approx(0.7)
    assert block["next_tick"]["offset_seconds" if "offset_seconds" in block["next_tick"] else "seconds_until"] == 3.0
    assert [t["done"] for t in block["ticks"]] == [True, True, False]
    late = window_clock.phase(started, ends, started + 59.0)
    assert late["label"] == "closing" and late["remaining_seconds"] == 1.0
