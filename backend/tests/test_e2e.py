"""Appendix E - the 13 mandatory end-to-end acceptance scenarios.

Each test maps 1:1 to an Appendix E scenario, and each one asserts observable
behaviour (broadcast messages, locked payloads, public status) rather than
private internals:

    E01  cold start      - first cycle locks a signal within one cycle
    E02  immutability    - the locked signal cannot change mid-cycle
    E03  isolation       - one broken formula cannot kill the pass
    E04  evidence gate   - 11+ zero formulas keep the side but drop conviction
    E05  emergency       - critical news forces the exit side (never HOLD)
    E06  trust gate      - only Tier <=2 critical headlines can break a lock
    E07  brain fallback  - committed 80x80 matrix keeps the brain healthy
    E08  degradation     - levels 1-6 map to real component failures
    E09  local model     - GGUF/ONNX validation + stub-mode usability
    E10  agents          - key tests fail safe and never block a cycle
    E11  asset switch    - BTC -> PAXG takes effect at a cycle boundary
    E12  hedge stress    - HSI > 0.8 dampens confidence and conviction
    E13  outcomes        - locked signals are scored and feed DRG

Run with:  PYTHONPATH=. python -m pytest backend/tests -q
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import pytest_asyncio

from backend.agents import fusion as fusion_module
from backend.agents.gemini_agent import GeminiAgent
from backend.agents.github_agent import GitHubAgent
from backend.agents.local_model_agent import LocalModelAgent
from backend.core import config as cfg
from backend.core.cycle_manager import CycleManager
from backend.core.errors import DegradationLevel
from backend.core.signal_lock import FrozenSignal
from backend.formulas import drg as drg_module
from backend.formulas.engine import INPUT_FORMULAS, FormulaEngine, FormulaResult

FORMULA_NAMES = [spec.name for spec in INPUT_FORMULAS]
ALL_KEYS = set(FORMULA_NAMES) | {"KCAE", "CCSv2", "DRG"}


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def manager():
    """One live CycleManager shared by the scenarios that need real cycles."""
    settings = cfg.Settings()
    instance = CycleManager(settings, local_stub=True)
    await instance.start()
    instance.mark_started()
    try:
        yield instance
    finally:
        await instance.stop()


async def wait_for(queue: asyncio.Queue, kind: str, timeout: float = 30.0) -> dict:
    """Pull messages until one of ``kind`` arrives (other types are skipped)."""
    deadline = time.time() + timeout
    skipped: list[str] = []
    while time.time() < deadline:
        try:
            message = await asyncio.wait_for(
                queue.get(), timeout=max(0.05, deadline - time.time())
            )
        except asyncio.TimeoutError:
            break
        if message.get("type") == kind:
            return message
        skipped.append(str(message.get("type")))
    raise AssertionError(f"no {kind} message within {timeout:.0f}s (saw {skipped})")


async def wait_for_signal(queue: asyncio.Queue, asset: str | None = None, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        message = await wait_for(queue, "SIGNAL", timeout=max(0.05, deadline - time.time()))
        if asset is None or message["data"]["asset"] == asset:
            return message["data"]
    raise AssertionError(f"no SIGNAL for {asset} within {timeout:.0f}s")


def fresh_snapshot(manager: CycleManager):
    return manager.market.freeze(
        news_items=manager.news.cache.latest(5),
        drg_outcomes=manager.outcomes.array(),
    )


# ---------------------------------------------------------------------------
# E01 - cold start
# ---------------------------------------------------------------------------


async def test_e01_cold_start_locks_a_complete_signal(manager: CycleManager):
    queue = manager.subscribe()
    try:
        data = await wait_for_signal(queue)
    finally:
        manager.unsubscribe(queue)

    assert data["lock_state"] == "LOCKED"
    assert data["lock_icon"] == "\U0001f512"          # padlock - not the hourglass
    # Round E: the third state is gone - a locked signal is always executable
    assert data["signal"] in {"BUY", "SELL"}
    assert 0.0 <= data["confidence"] <= 0.95
    assert data["cycle_number"] >= 1
    assert data["price"] > 0

    # all 22 formulas + the DRG meta-parameter, none of them silently zeroed
    values = data["formulas"]
    assert ALL_KEYS <= set(values), sorted(ALL_KEYS - set(values))
    assert manager.last_formula_result is not None
    assert not manager.last_formula_result.errors
    assert manager.last_formula_result.zero_count <= 3

    # latency budget: the spec allows ~3 ms for the whole pass
    assert data["total_ms"] < 50.0

    # the payload the UI binds to (Section 11)
    for key in ("hedge", "news", "agents", "conviction_note", "ccs_value", "warnings"):
        assert key in data, key
    assert "hold_warning" not in data
    # ... and the payload explains *which side* won and how much it is trusted
    assert data["conviction"] in {"HIGH", "MEDIUM", "LOW"}
    assert data["direction_source"]


# ---------------------------------------------------------------------------
# E02 - the locked signal is immutable
# ---------------------------------------------------------------------------


async def test_e02_locked_signal_never_changes_mid_cycle(manager: CycleManager):
    locked = manager.lock.get_current()
    queue = manager.subscribe()
    updates = 0
    saw_new_cycle = False
    try:
        deadline = time.time() + 6.0
        while time.time() < deadline and updates < 2 and not saw_new_cycle:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            kind = message.get("type")
            if kind == "FORMULA_UPDATE":
                updates += 1
                # the explorer refreshes with live numbers and says so
                assert "Live formula values only" in message["data"]["note"]
                assert message["data"]["signal"] == locked.signal
                # ... and the frozen signal object is literally the same one
                assert manager.lock.get_current() is locked
                assert manager.lock.current_signal.confidence == locked.confidence
            elif kind == "CYCLE_START":
                saw_new_cycle = True
    finally:
        manager.unsubscribe(queue)

    assert updates >= 1, "the Formula Explorer never refreshed"

    # A late/duplicate computation cannot rewrite the cycle's signal.
    intruder = FrozenSignal(
        cycle_number=locked.cycle_number,
        timestamp=locked.timestamp,
        asset=locked.asset,
        signal="SELL" if locked.signal != "SELL" else "BUY",
        confidence=0.99,
        reasoning="intruder",
        formula_values=(),
        agent_results=(),
        is_emergency_override=False,
    )
    returned = manager.lock.lock(intruder)
    assert returned is locked
    assert manager.lock.get_current().confidence == locked.confidence


# ---------------------------------------------------------------------------
# E03 - formula isolation
# ---------------------------------------------------------------------------


async def test_e03_broken_formula_is_isolated(manager: CycleManager):
    from backend.formulas.category_a_microstructure import tai as tai_module

    engine = FormulaEngine(brain=manager.brain)
    snapshot = fresh_snapshot(manager)
    baseline = engine.run(snapshot, "BTC")
    assert not baseline.errors

    original = tai_module.compute

    def explode(*_args, **_kwargs):
        raise RuntimeError("injected fault")

    tai_module.compute = explode
    try:
        broken = engine.run(snapshot, "BTC")
    finally:
        tai_module.compute = original

    assert "TAI" in broken.errors and "injected fault" in broken.errors["TAI"]
    assert broken.values["TAI"] == 0.0
    # every other formula still produced a number
    assert [name for name in FORMULA_NAMES if name not in ("TAI",)] == [
        name for name in FORMULA_NAMES if name not in ("TAI",) and name not in broken.errors
    ]
    assert broken.total_ms > 0
    assert set(broken.timings_ms) >= set(FORMULA_NAMES)


# ---------------------------------------------------------------------------
# E04 - insufficient evidence keeps the side and drops the conviction
# ---------------------------------------------------------------------------


def test_e04_insufficient_evidence_keeps_side_low_conviction():
    result = FormulaResult(asset="BTC")
    for index, name in enumerate(FORMULA_NAMES):
        result.values[name] = 0.0 if index < 11 else 0.4
    assert result.zero_count == 11
    assert result.insufficient_evidence is True

    healthy = FormulaResult(asset="BTC")
    for name in FORMULA_NAMES:
        healthy.values[name] = 0.4
    assert healthy.insufficient_evidence is False

    forced = fusion_module.fuse(
        agents={},
        ccs_value=0.9,
        ccs_confidence=0.95,
        hsi=0.05,
        settings=cfg.SETTINGS,
        insufficient_evidence=True,
    )
    # The evidence gate no longer silences the engine: it names the side the
    # brain points at, flags it as weak, and reports LOW conviction so the UI
    # can size it down instead of printing nothing.
    assert forced.decision in {"BUY", "SELL"}
    assert forced.decision == "BUY"          # CCSv2 is +0.9 in this case
    assert forced.forced_reason == "Insufficient formula evidence"
    assert forced.direction is not None
    assert forced.direction.conviction == "LOW"
    assert forced.direction.weak is True
    assert "degraded" in forced.direction.source.lower()
    assert 0.0 <= forced.confidence <= 0.95


# ---------------------------------------------------------------------------
# E05 - emergency override
# ---------------------------------------------------------------------------


async def test_e05_critical_event_overrides_lock_to_exit_side(manager: CycleManager):
    manager.lock.clear_emergency()
    before = manager.lock.try_get_current()
    queue = manager.subscribe()
    try:
        payload = await manager.trigger_emergency(
            {
                "headline": "Major exchange reports security breach - withdrawals frozen",
                "severity": "CRITICAL",
                "reason": "critical keyword from Tier 1 source",
                "tier": 1,
                "source": "Reuters",
            }
        )
        message = await wait_for(queue, "EMERGENCY_OVERRIDE", timeout=5)
    finally:
        manager.unsubscribe(queue)

    previous = before.signal if before else None
    # In a two-state system "get flat" is expressed as the *opposite* side of
    # whatever is open; with nothing open the engine keeps its direction.
    expected = {"BUY": "SELL", "SELL": "BUY"}.get(previous, previous or "BUY")
    assert payload["overridden_to"] in {"BUY", "SELL"}
    assert payload["overridden_to"] == expected
    assert payload["previous_signal"] == (previous or "COMPUTING")
    assert message["data"]["headline"] == payload["headline"]

    current = manager.lock.get_current()
    assert current.signal in {"BUY", "SELL"}
    assert current.signal == expected
    assert current.is_emergency_override is True
    assert current.lock_state == "EMERGENCY_OVERRIDE"
    assert current.superseded_by == payload["previous_signal"]
    assert current.confidence == 1.0
    assert current.conviction == "HIGH"
    assert current.direction_reason

    wire = payload["signal"]
    assert wire["lock_icon"] == "\u26a1"
    assert wire["is_emergency_override"] is True
    assert wire["signal"] in {"BUY", "SELL"}
    # the broadcast names the exit side explicitly, so a two-state signal can
    # still say "this closes the position you are in"
    if previous in {"BUY", "SELL"}:
        assert message["data"]["exit_side"] == expected
        assert message["data"]["closes_position"] is True
    else:
        assert message["data"]["closes_position"] is False

    # expiration semantics: the override is bounded, not permanent
    assert manager.lock.emergency_remaining() > 0
    manager.lock.clear_emergency()
    assert manager.lock.emergency_active() is False


# ---------------------------------------------------------------------------
# E06 - only trusted critical headlines break a lock
# ---------------------------------------------------------------------------


def test_e06_only_trusted_critical_headlines_break_the_lock(manager: CycleManager):
    detector = manager.news.detector

    assert detector.check_headline(
        "Bitcoin ETF sees $500M daily inflows as institutional demand accelerates",
        "Reuters",
        tier=1,
    ) is None

    assert detector.check_headline(
        "Analyst sees new all-time high in Q4", "CoinTelegraph", tier=2
    ) is None

    # critical keyword, but an untrusted source: must NOT fire (Section 14.1)
    low_trust = detector.check_headline(
        "Unverified blog claims exchange hack", "SomeBlog", tier=4
    )
    assert low_trust is None

    trusted = detector.check_headline(
        "Major exchange reports security breach after exploit", "Reuters", tier=1
    )
    assert trusted is not None
    assert trusted.severity == "CRITICAL"
    assert "security breach" in trusted.reason or "exploit" in trusted.reason

    # flash-move trigger is price-based and independent of source trust
    flash = detector.check_flash_move("BTC", -6.5, -1.0)
    assert flash is not None and flash.severity == "CRITICAL"
    assert detector.check_flash_move("BTC", -0.4, -1.0) is None

    # nothing above may have disturbed the live lock
    assert manager.lock.emergency_active() is False


# ---------------------------------------------------------------------------
# E07 - brain fallback
# ---------------------------------------------------------------------------


async def test_e07_brain_uses_the_committed_fallback_matrix(manager: CycleManager):
    brain = manager.brain
    assert brain.status.value in ("LIVE", "FALLBACK_CSV", "CACHED")
    assert brain.matrix.shape == (80, 80)
    assert brain.matrix.max() <= 1.0 + 1e-9

    health = await brain.refresh_health(ping=False)
    assert health.healthy, health.message
    assert health.matrix_checksum
    assert health.output_magnitude > 1e-6, "the 3-layer GCN read-out is all zeros"

    # the fallback matrix is a connected, signed graph - not a dummy
    assert int(np.count_nonzero(brain.matrix)) > 200
    assert int((brain.matrix > 0).sum()) > 0
    assert int((brain.matrix < 0).sum()) > 0

    payload = brain.matrix_payload()
    assert payload["stats"]["shape"] == [80, 80]
    assert payload["stats"]["edges"] > 200

    status = brain.status_dict()
    assert status["matrix"]["checksum"]
    assert len(status["matrix"]["shape"]) == 2
    assert status["status"] in ("LIVE", "FALLBACK_CSV", "CACHED")
    assert set(status["node_layout"]) == {"pn", "kenyon_cells", "dans", "mbons", "lateral_horn"}


# ---------------------------------------------------------------------------
# E08 - the six degradation levels
# ---------------------------------------------------------------------------


def test_e08_degradation_ladder(manager: CycleManager):
    levels = list(DegradationLevel)
    assert [int(level) for level in levels] == [1, 2, 3, 4, 5, 6]
    labels = {int(level): level.label for level in levels}
    assert "Full" in labels[1]
    assert "Minimal" in labels[6]

    snapshot = fresh_snapshot(manager)
    baseline = manager._compute_degradation(snapshot)
    assert int(baseline) >= 1

    # no news APIs -> level 2
    original_news = manager.news.degradation_ok
    manager.news.degradation_ok = lambda: False
    try:
        assert manager._compute_degradation(snapshot) >= DegradationLevel.NO_NEWS
    finally:
        manager.news.degradation_ok = original_news

    # no AI agents (Gemini/GitHub unconfigured, local not loaded) -> level 3
    # (`configured` / `ready` are read-only properties, so patch the classes)
    with (
        mock.patch.object(GeminiAgent, "configured", property(lambda self: False)),
        mock.patch.object(GitHubAgent, "configured", property(lambda self: False)),
        mock.patch.object(LocalModelAgent, "ready", property(lambda self: False)),
    ):
        assert manager._compute_degradation(snapshot) >= DegradationLevel.NO_AGENTS
    assert manager.agents.local.ready is True  # the live stub is back

    # no market data at all -> level 6, and the fusion layer still names a side
    class DeadSnapshot:
        def tick_count(self, _asset: str) -> int:
            return 0

    assert manager._compute_degradation(DeadSnapshot()) == DegradationLevel.MINIMAL
    forced = fusion_module.fuse(
        agents={},
        ccs_value=0.7,
        ccs_confidence=0.6,
        hsi=0.1,
        settings=cfg.SETTINGS,
        insufficient_evidence=True,
    )
    assert forced.decision in {"BUY", "SELL"}
    assert forced.direction is not None and forced.direction.weak is True


# ---------------------------------------------------------------------------
# E09 - local model validation
# ---------------------------------------------------------------------------


async def test_e09_local_model_validation_and_stub(tmp_path: Path):
    from backend.agents.local_model_agent import (
        GGUF_MAGIC,
        LocalModelAgent,
        validate_file,
    )

    assert GGUF_MAGIC == b"GGUF"

    missing = validate_file(tmp_path / "absent.gguf")
    assert not missing.ok and "not found" in missing.error.lower()

    wrong_format = tmp_path / "model.txt"
    wrong_format.write_bytes(b"just some text")
    result = validate_file(wrong_format)
    assert not result.ok and "gguf" in result.error.lower()

    bad_magic = tmp_path / "model.gguf"
    bad_magic.write_bytes(b"NOPE" + b"\x00" * 4096)
    result = validate_file(bad_magic)
    assert not result.ok and "magic bytes" in result.error

    truncated = tmp_path / "tiny.gguf"
    truncated.write_bytes(GGUF_MAGIC + b"\x03\x00\x00\x00" + b"\x00" * 512)
    result = validate_file(truncated)
    assert not result.ok and "100 MB" in result.error

    bad_onnx = tmp_path / "model.onnx"
    bad_onnx.write_bytes(b"\x09\x09\x09" + b"\x00" * 512)
    result = validate_file(bad_onnx)
    assert not result.ok

    # the stub is clearly labelled and keeps the agent tier exercisable
    agent = LocalModelAgent()
    agent.stub = True
    assert await agent.ensure_stub() is True
    assert agent.ready is True
    assert agent.status().value == "STUB"

    context = {
        "asset": "BTC",
        "price": 60000.0,
        "timestamp": "2026-01-01T00:00:00Z",
        "formulas": {"CCSv2": 0.5, "TAI": 0.2, "SHRP": 0.1, "MPS": 0.1, "HSI": 0.1},
        "ccs_confidence": 0.3,
        "drg": 0.0,
        "win_rate": 0.5,
        "headline": "test",
        "brain_status": "FALLBACK_CSV",
    }
    decision = await agent.decide(context, 0.2)
    assert decision.status.value == "STUB"
    assert decision.decision in {"BUY", "SELL"}
    assert 0.0 <= float(decision.confidence) <= 1.0
    agent.unload()


# ---------------------------------------------------------------------------
# E10 - agent key tests and failure isolation
# ---------------------------------------------------------------------------


async def test_e10_agent_key_tests_fail_safe(manager: CycleManager):
    gemini = await manager.agents.gemini.test_key("")
    assert gemini["valid"] is False and gemini["error"]

    github = await manager.agents.github.test_token("")
    assert github["valid"] is False and github["error"]

    # agent status is always reported, configured or not
    status = manager.agents.status_payload()
    for name in ("gemini", "local", "github"):
        assert name in status
        assert "status" in status[name]
    weights = status["weights"]
    assert weights["drosophila"] == 0.40
    assert weights["gemini"] == 0.25
    assert weights["local"] == 0.20
    assert weights["github"] == 0.15

    # with Gemini/GitHub unconfigured the engine still completes a full cycle
    queue = manager.subscribe()
    try:
        data = await wait_for_signal(queue)
    finally:
        manager.unsubscribe(queue)
    assert data["agents"]["local"]["status"] == "STUB"
    assert data["agents"]["local"]["decision"] in {"BUY", "SELL", None}
    assert data["agents"]["gemini"]["decision"] is None


# ---------------------------------------------------------------------------
# E11 - queued asset switching
# ---------------------------------------------------------------------------


def test_e11_unsupported_asset_is_rejected(manager: CycleManager):
    with pytest.raises(ValueError):
        manager.switch_asset("DOGE")


async def test_e11_asset_switch_takes_effect_at_the_cycle_boundary(manager: CycleManager):
    queue = manager.subscribe()
    try:
        current = manager.asset
        other = "PAXG" if current == "BTC" else "BTC"

        queued = manager.switch_asset(other)
        assert queued["pending"] == other
        assert queued["effective"] == "next cycle"
        assert manager.asset == current, "the switch must not apply mid-cycle"

        # the very next locked signal belongs to the new asset
        switched = await wait_for_signal(queue, asset=other, timeout=30)
        assert switched["asset"] == other
        assert switched["price"] > 0
        assert manager.asset == other

        # Appendix C is applied per asset
        params = cfg.asset_params(other)
        expected_ticks = cfg.ASSET_PARAMS[other]["tai_ticks"]
        assert params["tai_ticks"] == expected_ticks
        assert cfg.ASSET_PARAMS["BTC"]["tai_ticks"] != cfg.ASSET_PARAMS["PAXG"]["tai_ticks"]

        manager.switch_asset(current)
        back = await wait_for_signal(queue, asset=current, timeout=30)
        assert back["asset"] == current
        assert manager.pending_asset is None
    finally:
        manager.unsubscribe(queue)


# ---------------------------------------------------------------------------
# E12 - hedge stress
# ---------------------------------------------------------------------------


def test_e12_hedge_stress_dampens_confidence_and_conviction():
    calm = fusion_module.fuse(
        agents={}, ccs_value=0.8, ccs_confidence=0.6, hsi=0.10, settings=cfg.SETTINGS
    )
    stressed = fusion_module.fuse(
        agents={}, ccs_value=0.8, ccs_confidence=0.6, hsi=0.95, settings=cfg.SETTINGS
    )

    assert calm.hsi_adjustment == 1.0
    assert calm.decision == "BUY"

    assert stressed.hsi_adjustment == pytest.approx(cfg.SETTINGS.hsi_confidence_floor)
    assert stressed.confidence < calm.confidence
    # The hedge breakdown dampens the *trust* in the signal, not its existence:
    # the side is unchanged and the conviction drops instead of becoming HOLD.
    assert stressed.decision == "BUY"
    assert stressed.direction.conviction != "HIGH"
    assert calm.direction.conviction == "HIGH"
    assert "above the" in stressed.reasoning

    note = fusion_module.conviction_note(stressed.direction, stressed.confidence)
    assert note is not None
    assert note["conviction"] in {"LOW", "MEDIUM"}
    assert note["direction"] == "BUY"
    assert note["text"]


# ---------------------------------------------------------------------------
# E13 - outcome tracking feeds the DRG reward learner
# ---------------------------------------------------------------------------


async def test_e13_outcomes_are_scored_and_feed_drg(manager: CycleManager):
    snapshot = fresh_snapshot(manager)
    entry = snapshot.last_price("BTC")
    assert entry > 0

    queue = manager.subscribe()
    before = len(manager.outcomes)

    async def fake_price(_asset: str, fallback: float) -> float:
        return fallback * 1.002  # +20 bps

    original = manager._price_at
    manager._price_at = fake_price          # type: ignore[assignment]
    try:
        signal = FrozenSignal(
            cycle_number=999_001,
            timestamp="2026-01-01T00:00:00Z",
            asset="BTC",
            signal="BUY",
            confidence=0.9,
            reasoning="appendix E13",
            formula_values=(),
            agent_results=(),
            is_emergency_override=False,
            price=entry,
        )
        await manager._evaluate_outcome(signal, snapshot)
        # the live engine keeps producing its own outcomes in parallel, so
        # wait specifically for ours
        deadline = time.time() + 8
        outcome_message = None
        while outcome_message is None and time.time() < deadline:
            message = await wait_for(queue, "OUTCOME", timeout=5)
            if message["data"]["cycle_number"] == 999_001:
                outcome_message = message
    finally:
        manager._price_at = original       # type: ignore[assignment]
        manager.unsubscribe(queue)

    # concurrent live cycles may have appended their own outcomes too
    assert len(manager.outcomes) >= before + 1
    assert outcome_message is not None
    assert outcome_message["data"]["outcome"] == 1.0
    assert outcome_message["data"]["pnl_bps"] == pytest.approx(20.0, abs=1.0)
    assert 0.0 <= outcome_message["data"]["win_rate"] <= 1.0

    scored_rows = [tuple(row) for row in manager.outcomes.array()]
    assert any(
        abs(row[0] - 1.0) < 1e-9 and abs(row[1] - 20.0) < 1.0 for row in scored_rows
    ), scored_rows

    # a weak fallback signal is still scored (there is no neutral state left)
    manager._schedule_outcome(
        FrozenSignal(
            cycle_number=999_002,
            timestamp="2026-01-01T00:00:00Z",
            asset="BTC",
            signal="SELL",
            conviction="LOW",
            weak=True,
            direction_source="previous window",
            confidence=0.4,
            reasoning="weak sell from the tie-break ladder",
            formula_values=(),
            agent_results=(),
            is_emergency_override=False,
        ),
        snapshot,
    )
    assert len(manager.outcomes) >= before + 2

    # DRG turns the reward history into the dopamine signal the brain gates on
    rewards = np.array([[1.0, 20.0], [1.0, 15.0], [1.0, 25.0]], dtype=np.float64)
    assert drg_module.discounted_value(rewards) > 0
    losses = np.array([[-1.0, 20.0], [-1.0, 15.0]], dtype=np.float64)
    assert drg_module.discounted_value(losses) < 0

    scored = manager.market.freeze(
        news_items=manager.news.cache.latest(5), drg_outcomes=rewards
    )
    engine = FormulaEngine(brain=manager.brain)
    result = engine.run(scored, "BTC")
    assert result.values.get("DRG", 0.0) > 0.0
    assert result.values.get("DRG", 0.0) <= 1.0
