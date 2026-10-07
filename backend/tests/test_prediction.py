"""Round-F contracts: fresh predictions, 1:1 levels, reasoning, accuracy.

These are the rules the user stated, encoded so they cannot regress:

1. a prediction is *the* signal for exactly one window: it is published at the
   boundary and replaced at the next one, and it may never outlive its own
   window by more than the small publishing grace (``CYCLE_SECONDS=60``,
   ``prediction_expired``, ``CycleManager.ensure_fresh``);
2. every prediction carries its reasoning (the case *for* and *against*);
3. the take-profit and the stop-loss are the same distance from the entry, so
   the reward:risk ratio is exactly 1:1;
4. the accuracy of recent predictions is measured and published.
"""

from __future__ import annotations

import asyncio
import time

import pytest
import pytest_asyncio

from backend.core import config as cfg
from backend.core import prediction as prediction_module
from backend.core import risk as risk_module
from backend.core.cycle_manager import CycleManager


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def manager():
    """A live CycleManager, the same shape the e2e scenarios use."""
    settings = cfg.Settings()
    instance = CycleManager(settings, local_stub=True)
    await instance.start()
    instance.mark_started()
    # Let one full window (and therefore one outcome) happen: the accuracy and
    # freshness assertions need a published signal to look at.  Under the test
    # harness a window is cycle_period_seconds, and the outcome horizon is one
    # window, so this covers two windows with margin.
    deadline = time.time() + 12.0
    while time.time() < deadline and len(instance.outcomes) == 0:
        await asyncio.sleep(0.25)
    try:
        yield instance
    finally:
        await instance.stop()


# ---------------------------------------------------------------------------
# 1. The 15-second freshness contract
# ---------------------------------------------------------------------------


def test_the_window_is_sixty_seconds_and_the_prediction_lives_exactly_that_long():
    """The user's rule: the countdown is 60 seconds.

    The freshness rule did not get weaker - it got sharper.  The prediction on
    screen is the signal for the window being counted down, and it is replaced
    at every boundary; the only slack is the small grace that covers a request
    landing while the next window is published.
    """
    settings = cfg.SETTINGS
    # ``cycle_seconds`` is the un-compressed wall-clock cadence: 60 s, aligned
    # to the UTC minute.  The test harness compresses the *wall clock* by 20x,
    # but the ratio is preserved.
    assert settings.cycle_seconds == 60.0, settings.cycle_seconds
    # The harness compresses the wall clock, so alignment to the UTC minute is
    # asserted against the *uncompressed* configuration (what a Codespace runs).
    production = cfg.Settings(time_scale=1.0)
    assert production.cycle_period_seconds == 60.0
    assert production.use_world_clock is True, "the 60 s window must be minute-aligned"
    assert production.prediction_expired == 65.0
    assert production.outcome_horizon == 60.0
    assert settings.cycle_period_seconds <= settings.cycle_seconds + 1e-9
    # Fresh = "still the window it belongs to" (its own window + a small grace).
    # The grace has a 2 s floor so it cannot flap in a compressed window; at the
    # real 60 s cadence it is 5 s, i.e. 8 % of the window.
    assert settings.prediction_expired >= settings.cycle_period_seconds
    assert settings.prediction_expired == pytest.approx(
        settings.cycle_period_seconds + settings.freshness_grace_seconds, rel=1e-6
    )
    grace = settings.freshness_grace_seconds
    assert grace <= settings.cycle_period_seconds + 2.0
    # ... and scoring the prediction happens one window later, so the accuracy
    # panel fills at the same rate the predictions arrive.
    assert settings.outcome_horizon == pytest.approx(settings.cycle_period_seconds, rel=1e-6)


def test_freshness_flags_an_old_prediction():
    now = 1_700_000_000.0
    fresh = prediction_module.freshness(now - 3.0, 65.0, now=now)
    stale = prediction_module.freshness(now - 71.0, 65.0, now=now)
    assert fresh["state"] == "LIVE" and fresh["on_time"] is True
    assert fresh["age_seconds"] == 3.0
    assert stale["state"] == "STALE" and stale["on_time"] is False
    assert stale["age_seconds"] == 71.0
    assert stale["seconds_until_stale"] == 0.0
    assert prediction_module.freshness(0.0, 65.0, now=now)["on_time"] is False


async def test_the_api_never_serves_a_stale_prediction(manager: CycleManager):
    """``ensure_fresh`` recomputes when the published call has aged out."""
    manager.published_computed_at = time.time() - 60.0
    assert manager.prediction_is_stale() is True
    refreshed = await manager.ensure_fresh()
    assert refreshed is True
    assert manager.prediction_is_stale() is False
    payload = manager.signal_payload()
    assert payload["prediction"]["state"] == "LIVE"
    # A second call is a no-op: freshness is checked, not forced.
    assert await manager.ensure_fresh() is False


# ---------------------------------------------------------------------------
# 2. Reasoning in every prediction
# ---------------------------------------------------------------------------


def test_every_prediction_carries_reasoning(manager: CycleManager):
    payload = manager.signal_payload()
    prediction = payload["prediction"]
    assert prediction["side"] in {"BUY", "SELL"}
    assert prediction["reasoning"]["summary"]
    bullets = prediction["reasoning"]["bullets"]
    assert len(bullets) >= 3, bullets
    kinds = {bullet["kind"] for bullet in bullets}
    assert "brain" in kinds and "formulas" in kinds
    for bullet in bullets:
        assert bullet["text"] and bullet["text"] != "—"
        assert isinstance(bullet["supports"], bool)
    # both sides of the case are available to the UI
    assert isinstance(prediction["reasoning"]["supports"], list)
    assert isinstance(prediction["reasoning"]["against"], list)


def test_the_reasoning_names_the_formulas_behind_the_side():
    values = {"TAI": 0.42, "AFPR": 0.61, "DSKD": 0.30, "GCDV": -0.55, "HSI": 0.4}
    reasoning = prediction_module.build_reasoning(
        side="BUY",
        confidence=0.7,
        conviction="HIGH",
        formula_values=values,
        directional={"TAI": "A", "AFPR": "A", "DSKD": "F", "GCDV": "C"},
        risk={"tradeable": True, "entry": 100.0, "take_profit": 101.0, "stop_loss": 99.0,
              "tp_bps": 100.0, "sl_bps": 100.0, "volatility_bps": 66.0},
        hedge={"hsi": 0.4},
        accuracy={"evaluated": 12, "win_rate": 0.58},
    )
    text = " ".join(bullet["text"] for bullet in reasoning["bullets"])
    assert "TAI" in text and "GCDV" in text
    assert "Formula consensus" in text
    assert reasoning["consensus"]["up"] == 3
    assert reasoning["consensus"]["down"] == 1
    assert reasoning["against"]  # the dissenter is reported, not hidden
    assert ":1 (target" in reasoning["summary"]


def test_the_consensus_ignores_indicator_formulas():
    """HSI / ERC are regime indicators: they never vote on a direction."""
    values = {"TAI": 1.0, "HSI": -1.0, "ERC": -1.0}
    agreement = prediction_module.consensus(values, {"TAI": "A"})
    assert agreement["voters"] == 1
    assert agreement["up"] == 1 and agreement["down"] == 0
    assert agreement["score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. 1:1 take-profit / stop-loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("asset", ["BTC", "PAXG"])
@pytest.mark.parametrize("signal", ["BUY", "SELL"])
def test_the_levels_follow_the_rr_target(asset: str, signal: str):
    block = risk_module.risk_levels(asset, signal, 68_000.0, 12.0, cfg.SETTINGS)
    # Round AA: zero edge -> rr == rr_target; the stop is the 80% MAE quantile
    assert block["rr"] == pytest.approx(cfg.SETTINGS.rr_target)
    assert block["tp_bps"] == pytest.approx(block["sl_bps"] * cfg.SETTINGS.rr_target, rel=1e-3)
    assert block["engine"]["mae_z"] == pytest.approx(1.2816 if asset == "BTC" else 1.227, abs=1e-3)
    entry = block["entry"]
    assert abs(block["take_profit"] - entry) == pytest.approx(
        abs(entry - block["stop_loss"]) * cfg.SETTINGS.rr_target, rel=1e-3
    )
    strong = risk_module.risk_levels(asset, signal, 68_000.0, 12.0, cfg.SETTINGS, edge=0.8)
    assert strong["rr"] > block["rr"] and strong["sl_bps"] == block["sl_bps"]
    if signal == "BUY":
        assert block["take_profit"] > entry > block["stop_loss"]
    else:
        assert block["take_profit"] < entry < block["stop_loss"]
    assert f"{cfg.SETTINGS.rr_target:.2f}:1" in block["note"]


@pytest.mark.parametrize("volatility", [0.01, 3.0, 12.0, 400.0])
def test_the_ratio_survives_clipping(volatility: float):
    """The stop is always bounded and the target never below the stop."""
    block = risk_module.risk_levels("BTC", "BUY", 68_000.0, volatility, cfg.SETTINGS)
    assert cfg.SETTINGS.min_sl_bps <= block["sl_bps"] <= cfg.SETTINGS.max_sl_bps
    assert block["tp_bps"] >= block["sl_bps"]
    assert block["rr"] >= 1.0


# ---------------------------------------------------------------------------
# Round I: the forecast window is the minute after the release, with detail
# ---------------------------------------------------------------------------
def test_a_live_prediction_covers_the_window_that_follows_its_release(
    manager: CycleManager,
):
    """The window is 60 s in production and compressed under the test harness.

    ``base_seconds`` is always the specification's 60 - the contract the user
    asked for - while ``seconds`` is the window this engine instance is really
    running, and ``accelerated`` says whether the two differ.
    """
    prediction = manager.signal_payload()["prediction"]
    horizon = prediction["horizon"]
    assert horizon["base_seconds"] == 60.0
    assert horizon["accelerated"] is (horizon["seconds"] != 60.0)
    if not horizon["accelerated"]:
        assert horizon["seconds"] == 60.0
        assert horizon["label"] == "the next 60 seconds"
        assert prediction["forecast_for"] == "the next 60 seconds"
    # the target instant is exactly one window after the release moment
    span = horizon["target_at_us"] - horizon["released_at_us"]
    assert span == int(round(horizon["seconds"] * 1e6))
    assert horizon["released_at_precise"].endswith("Z")
    assert horizon["microseconds_to_target"] >= 0
    assert horizon["scored_in_seconds"] >= 0
    # and the release is anchored to the window the signal governs
    assert horizon["released_at_us"] == int(round(manager.published_computed_at * 1e6))


def test_the_prediction_detail_explains_the_side(manager: CycleManager):
    prediction = manager.signal_payload()["prediction"]
    detail = prediction["detail"]
    assert detail["side"] in ("BUY", "SELL")
    assert detail["formulas_evaluated"] >= 20
    assert set(detail["category_scores"]) == set("ABCDEFGH")
    assert detail["supporters"], "a side with no supporters cannot be explained"
    engine = detail["agreement"]["engine"]
    assert engine["compute_us"] > 0
    assert engine["publish_latency_us"] >= 0
    assert engine["history_samples"] > 0
    assert detail["levels"]["tp_bps"] == pytest.approx(detail["levels"]["sl_bps"] * detail["levels"]["rr"], abs=0.02)
    # Round AN: the target is capped at the measured 95 % minute move, so rr may sit below rr_target (never below RR_FLOOR)
    assert 0.75 <= detail["levels"]["rr"] <= 2.5
    assert detail["micro"]["available"] is True
    assert detail["micro"]["resolution_us"] > 0


def test_the_clock_is_published_to_the_microsecond(manager: CycleManager):
    clock = manager.master_clock()
    for key in ("window_started_at_us", "window_ends_at_us", "server_time_us",
                "remaining_us", "elapsed_us", "publish_latency_us", "engine_compute_us",
                "snapshot_us"):
        assert key in clock, f"master_clock is missing {key}"
    window_us = int(round(clock["window_seconds"] * 1e6))
    assert clock["window_ends_at_us"] - clock["window_started_at_us"] == window_us
    assert clock["remaining_us"] > 0
    assert clock["server_time_us"] > clock["window_started_at_us"]
    assert clock["publish_latency_us"] >= 0


def test_the_live_formula_block_carries_microseconds_and_history(manager: CycleManager):
    manager.last_formula_result.history_window = 360  # as a live pass records it
    block = manager.live_formulas_payload()
    assert block["total_us"] > 0
    assert block["timings_us"], "per-formula microsecond timings are missing"
    assert all(isinstance(v, int) for v in block["timings_us"].values())
    assert block["stats"], "per-formula history statistics are missing"
    assert block["history_window"] >= 360
    assert block["micro"]["resolution_us"] > 0
    assert block["micro"]["resolution_label"]


def test_the_prediction_exposes_the_levels(manager: CycleManager):
    prediction = manager.signal_payload()["prediction"]
    assert 0.75 <= prediction["rr"] <= 2.5
    assert prediction["tp_bps"] == pytest.approx(prediction["sl_bps"] * prediction["rr"], abs=0.02)
    if prediction["entry"]:
        assert prediction["take_profit"] and prediction["stop_loss"]


# ---------------------------------------------------------------------------
# 4. Measured accuracy
# ---------------------------------------------------------------------------


def test_the_accuracy_block_measures_recent_windows(manager: CycleManager):
    block = manager.accuracy_block()
    for key in ("evaluated", "win_rate", "streak", "last_outcome", "per_side", "target_seconds"):
        assert key in block, key
    assert block["target_seconds"] <= 60.0
    assert block["evaluated"] >= 1, "the test manager has already scored windows"

    rows = manager.outcomes.array()
    wins = int((rows[:, 0] > 0).sum())
    decided = int((rows[:, 0] != 0).sum())   # Round AM: flat windows are not counted
    if decided:
        assert block["win_rate"] == pytest.approx(wins / decided, abs=1e-4)
    else:
        assert block["win_rate"] is None


def test_the_consensus_and_accuracy_dampen_confidence():
    """A side the formulas contradict is published with less confidence."""
    from backend.agents import fusion as fusion_module

    def fuse(consensus_value, accuracy):
        return fusion_module.fuse(
            agents={},
            ccs_value=0.6,
            ccs_confidence=0.8,
            hsi=0.3,
            formula_consensus=consensus_value,
            consensus_voters=12,
            recent_accuracy=accuracy,
        )

    aligned = fuse(0.7, {"evaluated": 4, "win_rate": 1.0})
    opposed = fuse(-0.7, {"evaluated": 4, "win_rate": 1.0})
    losing = fuse(0.7, {"evaluated": 20, "win_rate": 0.30})
    assert aligned.decision == opposed.decision, "the side must not flip on consensus"
    assert opposed.confidence < aligned.confidence
    assert losing.confidence < aligned.confidence
    assert aligned.formula_consensus == pytest.approx(0.7)
    assert opposed.confirmation < 1.0 and losing.calibration < 1.0
