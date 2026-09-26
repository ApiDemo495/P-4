"""Round-F contracts: fresh predictions, 1:1 levels, reasoning, accuracy.

These are the rules the user stated, encoded so they cannot regress:

1. a prediction may never be older than 15 seconds (``CYCLE_SECONDS`` and
   ``PREDICTION_MAX_AGE_SECONDS``, plus ``CycleManager.ensure_fresh``);
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
    # freshness assertions need a published signal to look at.
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


def test_the_window_is_short_enough_to_keep_predictions_fresh():
    settings = cfg.SETTINGS
    # ``cycle_seconds`` is the un-compressed wall-clock cadence: under the test
    # harness the window is scaled down further, which only makes it fresher.
    assert settings.cycle_seconds <= 15.0 + 1e-9, settings.cycle_seconds
    assert settings.prediction_expired <= 15.0 + 1e-9, settings.prediction_expired
    assert settings.prediction_expired <= settings.cycle_seconds * 1.5
    assert settings.cycle_period_seconds <= settings.cycle_seconds + 1e-9
    # ... and the outcome tracking is window-relative, so accuracy fills at the
    # same rate the predictions arrive.
    assert settings.outcome_horizon_seconds <= 60.0


def test_freshness_flags_an_old_prediction():
    now = 1_700_000_000.0
    fresh = prediction_module.freshness(now - 3.0, 15.0, now=now)
    stale = prediction_module.freshness(now - 21.0, 15.0, now=now)
    assert fresh["state"] == "LIVE" and fresh["on_time"] is True
    assert fresh["age_seconds"] == 3.0
    assert stale["state"] == "STALE" and stale["on_time"] is False
    assert stale["age_seconds"] == 21.0
    assert stale["seconds_until_stale"] == 0.0
    assert prediction_module.freshness(0.0, 15.0, now=now)["on_time"] is False


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
    assert "1:1" in reasoning["summary"]


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
def test_the_levels_are_one_to_one(asset: str, signal: str):
    block = risk_module.risk_levels(asset, signal, 68_000.0, 12.0, cfg.SETTINGS)
    assert block["rr"] == pytest.approx(cfg.SETTINGS.rr_target)
    assert block["tp_bps"] == pytest.approx(block["sl_bps"])
    entry = block["entry"]
    assert abs(block["take_profit"] - entry) == pytest.approx(
        abs(entry - block["stop_loss"]), rel=1e-9
    )
    if signal == "BUY":
        assert block["take_profit"] > entry > block["stop_loss"]
    else:
        assert block["take_profit"] < entry < block["stop_loss"]
    assert "1:1" in block["note"]


@pytest.mark.parametrize("volatility", [0.01, 3.0, 12.0, 400.0])
def test_the_ratio_survives_clipping(volatility: float):
    """The clamps must not break the 1:1 geometry at either extreme."""
    block = risk_module.risk_levels("BTC", "BUY", 68_000.0, volatility, cfg.SETTINGS)
    assert block["tp_bps"] == pytest.approx(block["sl_bps"])
    assert block["rr"] == pytest.approx(1.0)


def test_the_prediction_exposes_the_levels(manager: CycleManager):
    prediction = manager.signal_payload()["prediction"]
    assert prediction["rr"] == pytest.approx(1.0)
    assert prediction["tp_bps"] == pytest.approx(prediction["sl_bps"])
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
    assert block["win_rate"] == pytest.approx(wins / rows.shape[0], abs=1e-4)


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
