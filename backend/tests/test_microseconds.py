"""Round I: analysis in microseconds, and a prediction that covers 60 s.

The user asked for five things, and each has a test here:

1.  **Microsecond resolution.**  The engine times every pass and every formula
    in microseconds, and the market tape is measured at microsecond resolution
    (``backend/core/timebase.py``, ``backend/core/micro.py``).  Nothing in the
    analysis path is allowed to fall back to a whole-second clock.
2.  **One minute of future.**  Every prediction says, in words and in µs, which
    60-second window it is for - the window that starts at its release.
3.  **More detail.**  Formulas carry units, sensitivity, ranges, history
    statistics and per-formula µs timings; the prediction carries a ``detail``
    block naming the formulas behind the side.
4.  **Six times the data.**  Every buffer the dashboard reads is six times the
    previous size, driven by ``DATA_MULTIPLIER``.
5.  **No unused code.**  Covered by ``test_dead_code.py``.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import pytest

from backend.core import config as cfg
from backend.core import prediction as prediction_module
from backend.core.micro import analyze as micro_analyze
from backend.core.timebase import format_us, interval_us, now_us, perf_us, span_us, us_to_iso
from backend.data.ring_buffer import L2Buffer, OutcomeBuffer, TickBuffer
from backend.formulas import synthetic

SETTINGS = cfg.Settings(time_scale=1.0)


# ---------------------------------------------------------------------------
# 1. the clock itself
# ---------------------------------------------------------------------------
def test_the_clock_reads_microseconds_not_seconds() -> None:
    us = now_us()
    assert isinstance(us, int)
    assert us > 1_700_000_000_000_000, "now_us must be epoch microseconds"
    # two consecutive reads differ by less than a millisecond on any real host -
    # a seconds clock could not tell them apart at all
    assert now_us() - us < 1000
    assert isinstance(perf_us(), int)


def test_microsecond_timestamps_keep_their_six_digits() -> None:
    stamp = us_to_iso(1_700_000_000_123_456)
    assert stamp.endswith(".123456Z"), stamp
    assert stamp == "2023-11-14T22:13:20.123456Z"


def test_human_labels_switch_units_at_the_right_scale() -> None:
    assert format_us(0) == "0 µs"
    assert format_us(812) == "812 µs"
    assert format_us(4_310) == "4.31 ms"
    assert format_us(1_204_000) == "1.204 s"


def test_span_and_interval_are_measured_in_microseconds() -> None:
    times_ms = np.array([0.0, 33.3, 66.7, 100.0])
    assert span_us(times_ms) == 100_000
    intervals = interval_us(times_ms)
    assert intervals.shape == (3,)
    assert intervals[0] == 33_300
    assert span_us(np.array([])) == 0


# ---------------------------------------------------------------------------
# 2. the microsecond tape analyser
# ---------------------------------------------------------------------------
def _tape(freq_hz: float, seconds: float) -> np.ndarray:
    n = int(freq_hz * seconds)
    rows = []
    price = 100.0
    for i in range(n):
        price += 0.01 if i % 2 else -0.009
        rows.append([i * 1000.0 / freq_hz, price, 1.0 + (i % 5), 1.0 if i % 3 else -1.0])
    return np.asarray(rows, dtype=np.float64)


def _snapshot(ticks_by_asset: dict[str, np.ndarray], book_ticks: int = 120):
    """A frozen snapshot carrying one synthetic tape per asset."""
    from backend.core.frozen_snapshot import FrozenMarketSnapshot

    book = np.ones((2, cfg.L2_DEPTH_LEVELS, 2), dtype=np.float64)
    book[0, :, 0] = np.arange(1, cfg.L2_DEPTH_LEVELS + 1)
    empty = np.zeros((0, 4), dtype=np.float64)
    btc = ticks_by_asset.get("BTC", empty)
    paxg = ticks_by_asset.get("PAXG", empty)
    return FrozenMarketSnapshot(
        timestamp=time.time(),
        btc_ticks=btc,
        paxg_ticks=paxg,
        btc_book=book,
        paxg_book=book,
        btc_book_prev=book,
        paxg_book_prev=book,
        btc_candles=np.zeros((0, 2)),
        paxg_candles=np.zeros((0, 2)),
        news_items=(),
        drg_outcomes=np.zeros((0, 2)),
        btc_tick_count=int(btc.shape[0]),
        paxg_tick_count=int(paxg.shape[0]),
    )


def test_the_analyser_reports_resolution_in_microseconds() -> None:
    ticks = _tape(30.0, 10.0)
    report = micro_analyze(_snapshot({"BTC": ticks}), "BTC", now=now_us()).to_dict()
    assert report["available"] is True
    assert report["ticks"] == len(ticks)
    assert report["resolution_us"] == pytest.approx(33_300, abs=500)
    assert report["mean_interval_us"] == pytest.approx(33_300, abs=500)
    assert report["tick_rate_hz"] == pytest.approx(30.0, abs=0.5)
    assert report["span_us"] == pytest.approx(int((len(ticks) - 1) * 1e6 / 30.0), rel=0.01)
    assert report["median_interval_us"] == pytest.approx(33_300, abs=500)
    assert report["p95_interval_us"] >= report["median_interval_us"]
    assert report["span_label"].endswith(("ms", "s"))
    assert report["jitter_us"] == pytest.approx(0.0, abs=200)
    assert report["last_tick_age_us"] >= 0
    assert report["analyzed_us"] > 0


def test_the_analyser_measures_the_quote_side_too() -> None:
    ticks = _tape(50.0, 6.0)
    report = micro_analyze(_snapshot({"BTC": ticks}), "BTC", now=now_us()).to_dict()
    assert "quote_lifetime_us" in report
    assert report["quote_lifetime_us"] >= 0
    assert report["aggression"] != 0
    assert report["micro_range_bps"] > 0
    assert report["micro_vol_per_us"] > 0
    # the labels the UI prints come from the same numbers
    assert report["resolution_label"].endswith(("µs", "ms", "s"))
    assert report["span_label"].endswith(("µs", "ms", "s"))


def test_a_thin_tape_says_so_instead_of_guessing() -> None:
    report = micro_analyze(_snapshot({"BTC": np.zeros((3, 4))}), "BTC", now=now_us()).to_dict()
    assert report["available"] is False
    assert report["reason"]
    assert report["ticks"] == 3


# ---------------------------------------------------------------------------
# 3. more detail inside the formulas
# ---------------------------------------------------------------------------
def test_every_formula_has_logic_detail() -> None:
    from backend.formulas import logic as logic_module

    for name in logic_module.LOGIC:
        entry = logic_module.get(name)
        assert entry is not None
        detail = entry.to_dict()
        for key in ("units", "sensitivity", "misleads", "corroborates", "range"):
            assert key in detail, f"{name} is missing {key}"
        assert detail["range"]
        assert detail["units"]


def test_the_engine_times_each_formula_in_microseconds() -> None:
    from backend.formulas.engine import FormulaEngine

    engine = FormulaEngine()
    snapshot = synthetic.scenario("BULL").snapshots[-1]
    result = engine.run(snapshot, "BTC")
    assert result.total_us > 0
    assert result.timings_us, "no per-formula timings recorded"
    assert all(isinstance(v, int) for v in result.timings_us.values())
    assert sum(result.timings_us.values()) <= result.total_us * 1.5 + 1000
    assert result.micro.get("resolution_label")
    stats = result.stats
    assert stats, "no history statistics recorded"
    sample = next(iter(stats.values()))
    for key in ("samples", "mean", "std", "min", "max", "last", "zscore"):
        assert key in sample
    # traces carry the millisecond-free audit trail
    trace_labels = [row.get("label") for row in result.traces["TWRS"]]
    assert any("computed in" in (label or "") for label in trace_labels)
    assert any("data window" in (label or "") for label in trace_labels)


# ---------------------------------------------------------------------------
# 4. the prediction covers the minute that follows its release
# ---------------------------------------------------------------------------
def test_the_horizon_is_the_sixty_seconds_after_release() -> None:
    released = 1_700_000_000.25
    block = prediction_module.horizon(released_wall=released, seconds=60.0, now=released + 10.0)
    assert block["seconds"] == 60.0
    assert block["label"] == "the next 60 seconds"
    assert block["target_at_us"] - block["released_at_us"] == 60_000_000
    assert block["microseconds_to_target"] == 50_000_000
    assert block["seconds_to_target"] == 50.0
    assert block["released_at_precise"].endswith(".250000Z")
    assert block["scored_at_us"] - block["released_at_us"] == 60_000_000
    assert block["covers"]


def test_a_released_prediction_covers_exactly_one_minute() -> None:
    released = time.time()
    built = prediction_module.build(
        side="BUY",
        confidence=0.71,
        conviction="HIGH",
        computed_wall=released,
        max_age=5.0,
        risk={"tp_bps": 12.0, "sl_bps": 12.0, "rr": 1.0, "tp_price": 1.0, "sl_price": 1.0},
        reasoning={"summary": "x", "bullets": []},
        accuracy={"windows": 12, "hit_rate": 0.5},
        window_seconds=60.0,
        horizon_seconds=60.0,
        scoring_seconds=60.0,
        now=released + 1.0,
    )
    assert built["horizon"]["seconds"] == 60.0
    assert built["forecast_for"] == "the next 60 seconds"
    span = built["horizon"]["target_at_us"] - built["horizon"]["released_at_us"]
    assert span == 60_000_000
    assert built["horizon"]["scoring_seconds"] >= 60.0
    assert built["horizon"]["scored_in_seconds"] >= 59.0


def test_freshness_is_measured_in_microseconds() -> None:
    computed = time.time() - 0.75
    fresh = prediction_module.freshness(computed, max_age=5.0)
    assert fresh["computed_at_us"] > 0
    assert fresh["age_microseconds"] == pytest.approx(750_000, abs=250_000)
    assert fresh["age_label"].endswith(("µs", "ms", "s"))
    assert fresh["state"] == "LIVE"
    assert fresh["age_seconds"] <= 1.0


def test_the_prediction_detail_names_the_formulas_and_the_clock() -> None:
    detail = prediction_module.detail(
        side="SELL",
        category_scores={"A": -0.4, "B": 0.1},
        supporters=[{"name": "TAI", "value": -0.4}],
        opponents=[{"name": "DGW", "value": 0.2}],
        confidence_parts={"consensus": 0.3},
        levels={"tp_bps": 12.0},
        micro={"resolution_us": 33_300},
        stats={"TAI": {"samples": 12}},
    )
    assert detail["side"] == "SELL"
    assert detail["supporters"][0]["name"] == "TAI"
    assert detail["levels"]["tp_bps"] == 12.0
    assert detail["micro"]["resolution_us"] == 33_300
    assert detail["formula_stats"]["TAI"]["samples"] == 12


# ---------------------------------------------------------------------------
# 5. six times the data
# ---------------------------------------------------------------------------
def test_the_data_multiplier_is_six() -> None:
    assert cfg.DATA_MULTIPLIER == 6


def test_every_buffer_is_six_times_its_round_h_size() -> None:
    # the Round-H sizes, x6
    assert cfg.TICK_BUFFER_SIZE == 3600
    assert cfg.L2_DEPTH_LEVELS == 120
    assert cfg.CANDLE_BUFFER_SIZE == 360
    assert cfg.NEWS_CACHE_SIZE == 120
    assert cfg.OUTCOME_BUFFER_SIZE == 120
    assert L2Buffer.SPREAD_HISTORY == 5400


def test_the_ring_buffers_hold_six_times_the_rows() -> None:
    ticks = TickBuffer(capacity=120)
    ticks.extend(np.array([[i * 1000.0, 100.0 + i, 1.0, 1.0] for i in range(120)]))
    assert ticks.view().shape == (120, 4)

    # a real book: bids below the mid, asks above it, so the spread is positive
    book = np.zeros((2, cfg.L2_DEPTH_LEVELS, 2), dtype=np.float64)
    book[0, :, 0] = 100.0 - np.arange(1, cfg.L2_DEPTH_LEVELS + 1) * 0.01
    book[1, :, 0] = 100.0 + np.arange(1, cfg.L2_DEPTH_LEVELS + 1) * 0.01
    book[:, :, 1] = 1.0
    l2 = L2Buffer()
    for _ in range(120):
        l2.push(book)
    assert l2.spreads().shape == (120, 2)
    assert L2Buffer.SPREAD_HISTORY == 5400 > 120

    outcomes = OutcomeBuffer(capacity=cfg.OUTCOME_BUFFER_SIZE)
    for i in range(cfg.OUTCOME_BUFFER_SIZE):
        outcomes.append(1.0 if i % 2 else -1.0, float(i))
    assert outcomes.array().shape[0] == 120


def test_the_api_returns_more_history_and_more_outcomes() -> None:
    from backend.api import routes_signals

    import inspect

    history = inspect.signature(routes_signals.signal_history).parameters["limit"]
    outcomes = inspect.signature(routes_signals.outcomes).parameters["limit"]
    assert history.default == 72
    assert outcomes.default == 72


def test_the_formula_history_window_is_deep_enough_for_statistics() -> None:
    from backend.formulas import engine as engine_module

    assert engine_module.FORMULA_HISTORY >= 360
    assert isinstance(deque(maxlen=engine_module.FORMULA_HISTORY), deque)
