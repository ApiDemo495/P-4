"""The 60-second cycle manager - the heart of DROSOPHILA TRADER v2.0.

One cycle, exactly as specified in Section 1.1 and Section 2.2:

    t=0.000s    freeze the market snapshot, reset the lock to COMPUTING
    t=0-3ms     run all 22 formulas on the frozen snapshot
    t=3-8s      call the AI agents concurrently (7 s budget)
    t~8s        fuse, then LOCK the signal and broadcast it
    t=8-60s     broadcast nothing that changes the signal panel; the Formula
                Explorer refreshes every 15 s with the explicit note that the
                signal remains locked
    t=60s       next cycle begins

The manager also owns:

* the **outcome tracker** - 60 s after a signal fires, the result is evaluated
  and pushed into the DRG reward buffer
* the **emergency path** - critical news events or a flash move override the
  locked signal to the *exit side* (the opposite of the direction that is open)
  for three cycles; the protocol is binary, so "flat" is expressed as a flip
* **queued asset switching** - toggling BTC/PAXG mid-cycle takes effect at the
  next boundary (test case 13)
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field

from backend.agents import fusion as fusion_module
from backend.agents.orchestrator import AgentOrchestrator
from backend.brain.brain import Brain
from backend.core import config as cfg
from backend.core.clock import WorldClock
from backend.core.direction import describe as describe_direction
from backend.core.errors import ComponentStatus, DegradationLevel
from backend.core.prediction import build as build_prediction
from backend.core.emotions import EmotionMonitor, analyze as analyze_emotions, compact as compact_emotions
from backend.core.prediction import detail as prediction_detail
from backend.core.prediction import build_reasoning, consensus
from backend.core.calibration import EvidenceLedger
from backend.core.frozen_snapshot import FrozenMarketSnapshot
from backend.core.redis_bus import Store
from backend.core.timebase import now_us
from backend.core import window_clock
from backend.core.risk import realized_volatility_bps, risk_levels, quoted_spread_bps
from backend.core.signal_lock import FrozenSignal, LockState, SignalLockController
from backend.data.market_hub import MarketDataHub
from backend.data.ring_buffer import OutcomeBuffer
from backend.formulas.engine import FormulaEngine, FormulaResult
from backend.news.critical_event_detector import CriticalEvent
from backend.news.news_engine import NewsEngine

log = logging.getLogger("drosophila.cycle")


_grid_offsets = window_clock.grid_offsets   # kept as a name for the tests


def _iso(timestamp: float) -> str:
    """Wall-clock ISO-8601 (UTC) stamp, second precision - what the UI renders."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


@dataclass
class CycleStats:
    cycle_number: int = 0
    lock_ms: float = 0.0
    formula_ms: float = 0.0
    agent_ms: float = 0.0
    last_cycle_started: float = 0.0
    cycles_completed: int = 0
    forced_fallbacks: int = 0
    """Windows whose side came from the tie-break ladder instead of a real edge."""
    weak_windows: int = 0
    emergency_count: int = 0
    signal_counts: dict = field(default_factory=lambda: {"BUY": 0, "SELL": 0})

    def to_dict(self) -> dict:
        return {
            "cycle_number": self.cycle_number,
            "cycles_completed": self.cycles_completed,
            "lock_ms": round(self.lock_ms, 2),
            "formula_ms": round(self.formula_ms, 2),
            "agent_ms": round(self.agent_ms, 2),
            "forced_fallbacks": self.forced_fallbacks,
            "weak_windows": self.weak_windows,
            "emergency_count": self.emergency_count,
            "signal_counts": dict(self.signal_counts),
        }


def _readings_for(result) -> dict:
    """The plain-words interpretation of each value, or {} if unavailable."""
    if result is None:
        return {}
    try:
        from backend.formulas import logic as logic_module

        return logic_module.readings(result.values)
    except Exception:  # pragma: no cover - defensive
        return {}


class CycleManager:
    """Owns every subsystem and drives the 60-second world clock."""

    def __init__(self, settings=None, clock: WorldClock | None = None, local_stub: bool | None = None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.clock = clock or WorldClock()
        self.store = Store(self.settings.redis_url)

        self.market = MarketDataHub(self.settings)
        self.brain = Brain(self.settings, self.store)
        self.news = NewsEngine(self.settings, on_critical=self._on_critical_event)
        self.agents = AgentOrchestrator(self.settings, local_stub=local_stub)
        self.formulas = FormulaEngine(brain=self.brain)
        # Round T: the thermodynamic capital layer - one more weighted voter.
        from backend.physics.engine import PhysicsEngine

        self.physics = PhysicsEngine()
        self.last_physics: dict | None = None
        self.last_news_impact: dict = {}
        self.lock = SignalLockController()

        self.outcomes = OutcomeBuffer()
        # Round N: which inputs actually predict - learned from scored windows.
        self.ledger = EvidenceLedger(
            Path(os.environ.get("CALIBRATION_PATH") or (cfg.REPO_ROOT / ".run" / "calibration.json"))
        )
        self._last_votes: dict[str, int] = {}
        self._last_learned: dict = {}
        #: (side, outcome) pairs, so the accuracy panel can say which side the
        #: engine has actually been getting right rather than only the total.
        self._side_outcomes: deque[tuple[str, float]] = deque(maxlen=40)
        self.stats = CycleStats()

        self.asset = "BTC"
        self.pending_asset: str | None = None
        self.degradation = DegradationLevel.FULL
        self.warnings: list[str] = []
        self.last_snapshot: FrozenMarketSnapshot | None = None
        #: The formula pass that produced the **locked** signal.  Kept separate
        #: from the prefetch pass so /api/formulas/current and /api/brain/trace
        #: always describe what the user is actually looking at.
        self.last_formula_result: FormulaResult | None = None
        self.last_live_formulas: dict[str, float] = {}
        #: The pass those live values came from, so the explorer can show the
        #: intermediate numbers (traces) behind each one, not just the value.
        self.last_live_result: FormulaResult | None = None
        self.last_fusion: dict = {}
        self.last_error: str = ""
        #: The crowd's emotions, measured continuously on the live tape (the
        #: user's "which emotion is dominant, live" ask).  Sampled on its own
        #: timer so the panel moves between PULSE marks too.
        self.emotions = EmotionMonitor(
            hub=self.market,
            asset=self.asset,
            news_provider=lambda: self.news.cache.latest(30),
            formula_provider=lambda: (self.last_live_formulas, self._directional_map()),
            interval_seconds=self.settings.emotion_interval_seconds,
            history_size=self.settings.emotion_history_size,
        )
        #: The same measurement taken on the *frozen* snapshot at lock time, so
        #: the emotion that shaped a signal cannot change afterwards.
        self.emotion_locked: dict = {}
        #: The "thin edge" note (replaces the Section 10.2 HOLD box).  Present
        #: only when the side came from the tie-break ladder.
        self.conviction_note: dict | None = None

        # --- pipelined publication (Sections 10.4) ----------------------
        self._pending_signal: FrozenSignal | None = None
        self._prefetch_agents: dict = {}
        self._prefetch_context: dict = {}
        #: when the final (fast) freeze of the pending signal happened
        self.final_freeze_at: float = 0.0
        self.final_freeze_ms: float = 0.0
        self._pending_vol_bps: float = 0.0
        self._pending_spread_bps: float = 0.0
        self.pending_formula_result: FormulaResult | None = None
        #: Everything the UI shows *about* the current prediction (fusion
        #: breakdown, crowd at lock, conviction note, warnings, degradation)
        #: is staged here by the prefetch / re-freeze and swapped in only at
        #: the window boundary.  Writing them live was the lock break the
        #: user kept seeing: the side stayed put while its reasoning,
        #: supporters and learned block flipped to the *next* window's.
        self._pending_view: dict | None = None
        self.prefetch_ready: bool = False
        self.prefetch_ms: float = 0.0
        self.prefetch_at: float = 0.0
        #: how long before the boundary the next signal starts being computed
        self.prefetch_lead: float = min(
            max(self.settings.scaled(self.settings.lock_deadline_seconds), 0.2),
            self.settings.cycle_period_seconds * 0.5,
        )
        #: When the *published* signal was computed (it is always one window
        #: before it goes live when the pipeline is on).
        self.published_computed_at: float = 0.0
        self.published_compute_ms: float = 0.0
        #: The same numbers at the resolution the engine runs at.
        self.published_compute_us: int = 0
        self.publish_latency_us: int = 0
        self.snapshot_us: int = 0
        self.window_started_us: int = 0
        self.window_valid_from: float = 0.0
        self.window_valid_until: float = 0.0
        self._origin_wall: float = 0.0

        self._subscribers: set[asyncio.Queue] = set()
        self._tasks: list[asyncio.Task] = []
        self._outcome_tasks: set[asyncio.Task] = set()
        self._stop = asyncio.Event()
        self._running = False
        self._cycle_started = False
        self._started_ts = time.time()

        # --- warm-up bookkeeping ------------------------------------------
        # The HTTP port opens *before* the subsystems are up (brain
        # verification and the market connect can take seconds, and a
        # forwarded Codespaces port that nobody answers returns 502).  The UI
        # reads these fields to say "warming up" instead of showing an error.
        self.warming: bool = False
        self.ready: bool = False
        self.ready_at: float = 0.0
        self.start_error: str | None = None

    # ==================================================================
    # Lifecycle
    # ==================================================================
    async def start(self) -> None:
        log.info("Starting DROSOPHILA TRADER v2 cycle manager")
        await self.store.connect()
        await self.market.start()
        await self.brain.start()
        await self.news.start()
        await self.clock.sync(force=True)

        await self._restore_state()

        self._running = True
        self._tasks = [
            asyncio.create_task(self._cycle_loop(), name="cycle-loop"),
            asyncio.create_task(self.clock.run(), name="clock"),
            asyncio.create_task(self._flash_watch(), name="flash-watch"),
            asyncio.create_task(self._emotion_loop(), name="emotions"),
        ]
        log.info(
            "Cycle manager running: period=%.1fs, world_clock=%s",
            self.settings.cycle_period_seconds,
            self.settings.use_world_clock,
        )

    async def warm_up(self) -> None:
        """Bring the subsystems up **after** the port is already serving.

        ``lifespan`` schedules this instead of awaiting it, so a request that
        arrives during startup gets a page that says "warming up" instead of a
        502 from the port forwarder.  Any failure is recorded in
        ``start_error`` and surfaced by ``/api/health`` and the dashboard.
        """
        self.warming = True
        try:
            await self.start()
        except asyncio.CancelledError:  # shutdown while warming up
            raise
        except Exception as exc:  # noqa: BLE001 - must never take the port down
            self.start_error = f"{type(exc).__name__}: {exc}"
            log.exception("warm-up failed: %s", exc)
        else:
            self.ready = True
            self.ready_at = time.time()
        finally:
            self.warming = False

    async def stop(self) -> None:
        self._stop.set()
        self._running = False
        for task in self._tasks + list(self._outcome_tasks):
            task.cancel()
        await asyncio.gather(*(self._tasks + list(self._outcome_tasks)), return_exceptions=True)
        self._tasks.clear()
        self._outcome_tasks.clear()
        # Shutdown must work even if warm-up never completed, so every step is
        # individually guarded.
        for name, step in (
            ("persist", self._persist_state),
            ("news", self.news.stop),
            ("brain", self.brain.stop),
            ("market", self.market.stop),
        ):
            try:
                await step()
            except Exception as exc:  # noqa: BLE001
                log.debug("stop(%s) failed: %s", name, exc)
        log.info("Cycle manager stopped")

    # ==================================================================
    # Subscription (WebSocket fan-out)
    # ==================================================================
    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    async def broadcast(self, message: dict) -> None:
        dead: list[asyncio.Queue] = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                dead.append(queue)
        for queue in dead:
            self._subscribers.discard(queue)

    # ==================================================================
    # The cycle
    # ==================================================================
    async def _cycle_loop(self) -> None:
        if self.settings.signal_pipeline:
            await self._pipelined_loop()
            return
        while not self._stop.is_set():
            try:
                await self._wait_for_cycle_start()
                if self._stop.is_set():
                    break
                await self._run_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the loop must never die
                log.exception("cycle failed: %s", exc)
                await asyncio.sleep(1.0)

    # ==================================================================
    # Pipelined publication (Section 10.4) - the default mode
    #
    #   window N   :  |--- publish signal N -- compute signal N+1 ---|
    #                       ^ boundary                              ^ prefetch (lead before the
    #                                                                  next boundary so the data
    #                                                                  is as fresh as possible)
    #
    # The signal that governs a countdown is computed *during the previous
    # countdown*, so the panel is never empty, the user never waits, and the
    # engine is always working on the next window while the current one runs.
    # ==================================================================
    async def _pipelined_loop(self) -> None:
        period = self.settings.cycle_period_seconds
        lead = min(max(self.settings.scaled(self.settings.lock_deadline_seconds), 0.2), period * 0.5)

        # --- bootstrap: one immediate window so the panel is never blank ----
        # The signal in this window is computed now (there was no previous
        # window to compute it in) and is flagged ``preview`` so the UI can say
        # so instead of pretending the pipeline is already running.
        await self._wait_for_cycle_start()
        try:
            await self._run_cycle(bootstrap=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("bootstrap cycle failed: %s", exc)

        boundary = self.window_valid_until or (time.time() + period)
        self._origin_wall = boundary
        index = 0
        while not self._stop.is_set():
            try:
                deadline = boundary + index * period
                if self.settings.use_world_clock:
                    # Stay phase-locked to the UTC minute even across restarts.
                    deadline = self._snap_to_minute(deadline)

                # --- compute the NEXT window while this one is still running --
                # Happens at the very end of the countdown (``lead`` seconds
                # before the boundary) so the frozen snapshot is as fresh as it
                # can be while still landing before the lock deadline.
                await self._sleep_until(deadline - lead)
                if self._stop.is_set():
                    break
                await self._prefetch(deadline)

                # --- the final freeze: the fast half of the pipeline again, on
                # a snapshot taken a fraction of a second before the boundary.
                # The agents' votes are kept from the prefetch; the tape, the
                # 22 formulas, the crowd and the fusion are all re-done, so the
                # locked prediction describes the market *at* the lock.
                final_lead = min(max(self.settings.final_lock_lead_seconds, 0.05), lead)
                await self._sleep_until(deadline - final_lead)
                if self._stop.is_set():
                    break
                try:
                    await self._refreeze(deadline)
                except Exception as exc:  # noqa: BLE001 - the prefetch draft still stands
                    log.exception("final freeze failed - publishing the prefetch draft: %s", exc)

                # --- the boundary: publish what was just prepared ------------
                await self._sleep_until(deadline)
                if self._stop.is_set():
                    break
                await self._open_window(deadline)
                index += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("window failed: %s", exc)
                self.last_error = f"window {index + 1}: {type(exc).__name__}: {exc}"
                self.warnings.append(f"window failed: {type(exc).__name__}: {exc}")
                del self.warnings[:-20]
                # Never spin on a boundary that is already in the past: the
                # loop used to retry the same deadline every 0.5 s forever, so
                # one persistent error froze the panel while the countdown
                # kept running.  Skip to the next boundary and publish there,
                # inline if need be (``_open_window`` computes when nothing
                # was prepared).
                if time.time() >= deadline:
                    index += 1
                    self._pending_signal = None
                await asyncio.sleep(0.5)

    # ------------------------------------------------------------------
    @staticmethod
    def _snap_to_minute(timestamp: float) -> float:
        """Round a wall-clock timestamp up to the next UTC minute boundary."""
        return math.ceil((timestamp - 1e-6) / 60.0) * 60.0

    async def _sleep_until(self, deadline_wall: float) -> None:
        """Sleep until an absolute wall-clock deadline, without drift."""
        while not self._stop.is_set():
            remaining = deadline_wall - time.time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.2, remaining) if remaining > 0.05 else remaining)

    async def _open_window(self, deadline_wall: float) -> None:
        """Publish the pre-computed signal for the window that starts now."""
        self.stats.cycle_number += 1
        cycle = self.stats.cycle_number

        # Asset switches are applied here and nowhere else (test case 13).
        if self.pending_asset and self.pending_asset != self.asset:
            log.info("Asset switch applied at window boundary: %s -> %s", self.asset, self.pending_asset)
            self.asset = self.pending_asset
            self.pending_asset = None
            # The crowd belongs to an asset, so the monitor follows the switch -
            # and the new tape starts its own history rather than inheriting the
            # old asset's mood.
            self.emotions.asset = self.asset
            self.emotions.tracker.asset = self.asset
            self.emotions.tracker.history.clear()
            self.emotions.tracker.emotions = {}
            self.emotions.tracker.last_dominant = ""
            self.emotions.tracker.samples = 0
            self.emotions.last = {}
            self.emotion_locked = {}

        self.lock.new_cycle(cycle)
        self.window_valid_from = deadline_wall
        self.window_valid_until = deadline_wall + self.settings.cycle_period_seconds

        pending = self._pending_signal
        self._pending_signal = None
        self.prefetch_ready = False
        # Promote the prefetched formula pass: explain()/the Formula Explorer
        # must describe the window that is now live, not the bootstrap one.
        if self.pending_formula_result is not None:
            self.last_formula_result = self.pending_formula_result
            self.pending_formula_result = None
        self._apply_pending_view()

        if pending is None:
            # The prefetch failed or the first window is late: compute now.
            log.warning("window %d opened without a prepared signal - computing inline", cycle)
            await self._run_cycle(bootstrap=False, window_from=deadline_wall)
            return

        signal = self._stamp_window(pending, cycle, deadline_wall)
        locked = self.lock.lock(signal)
        self.stats.lock_ms = (time.time() - deadline_wall) * 1000.0
        self.stats.cycles_completed += 1
        self.stats.signal_counts[locked.signal] = self.stats.signal_counts.get(locked.signal, 0) + 1

        # The freshness clock starts when the prediction goes *live*, i.e. at
        # the window boundary - not when it was prefetched.  Prefetch happens up
        # to half a window earlier, and charging that time to the prediction's
        # age made a 12-second window look 18 seconds old just before its
        # boundary (which triggered an unnecessary out-of-band refresh).  How
        # long the *computation* took is still reported separately, and the
        # panel still says "computed Xs before this window started".
        self.published_computed_at = deadline_wall
        self.published_compute_ms = self.prefetch_ms
        self.published_compute_us = int(round(self.prefetch_ms * 1000.0))
        self.window_started_us = int(round(deadline_wall * 1e6))
        # How far past the intended boundary the publication actually landed.
        # This is the number that says whether "1 minute" means 60.000000 s.
        self.publish_latency_us = max(0, now_us() - self.window_started_us)
        await self.broadcast(
            {
                "type": "SIGNAL",
                "data": self.snapshot_payload(locked, include_history=True),
            }
        )
        self._schedule_outcome(locked, None)
        asyncio.create_task(self._heartbeat_loop(cycle, deadline_wall, self.window_valid_until))
        log.info(
            "window %d published %s (%.0f%%) computed %.1fs earlier",
            cycle, locked.signal, locked.confidence * 100,
            max(0.0, deadline_wall - self.prefetch_at) if self.prefetch_at else 0.0,
        )

    def _stamp_window(self, draft: FrozenSignal, cycle: int, deadline_wall: float) -> FrozenSignal:
        """Re-anchor a pre-computed draft to the window it now governs.

        The entry price and the take-profit / stop-loss levels are taken *at the
        boundary*, not at prefetch time, because that is the price the user is
        trading from.
        """
        from backend.core.direction import opposite

        period = self.settings.cycle_period_seconds
        price = self.market.last_price(self.asset) or draft.price
        emergency = self.lock.emergency_active()
        exit_side = opposite(draft.signal)
        signal_for_levels = draft.signal
        if emergency and exit_side:
            # An emergency that fires between prefetch and publication: the
            # frozen draft direction is replaced by the side that flattens it.
            signal_for_levels = exit_side
        risk = risk_levels(
            self.asset,
            signal_for_levels,
            price,
            self._pending_vol_bps,
            self.settings,
            horizon_seconds=period,
            conviction="HIGH" if (emergency and exit_side) else draft.conviction,
            emergency_exit=bool(emergency and exit_side),
            edge=float(getattr(draft, "confidence", 0.0) or 0.0),
            spread_bps=float(self._pending_spread_bps or 0.0),
        )
        signal = draft._replace(
            cycle_number=cycle,
            asset=self.asset,
            timestamp=_iso(deadline_wall),
            valid_from=_iso(deadline_wall),
            valid_until=_iso(deadline_wall + period),
            window_seconds=period,
            price=price,
            risk=tuple(sorted(risk.items())),
            preview=False,
        )
        if emergency and not signal.is_emergency_override:
            # The override may have fired while the signal was being prepared.
            # Binary protocol: the exit side flattens the direction that was
            # about to be published (see backend.core.direction).
            previous = signal.signal
            flipped = exit_side or previous
            signal = signal._replace(
                signal=flipped,
                confidence=1.0,
                is_emergency_override=True,
                lock_state=LockState.EMERGENCY_OVERRIDE.value,
                superseded_by=previous,
                conviction="HIGH" if exit_side else "LOW",
                weak=exit_side is None,
                direction_source=(
                    f"emergency exit of the open {previous}" if exit_side
                    else "emergency with nothing open - direction unchanged"
                ),
                direction_reason=(
                    f"{flipped} closes the open {previous}: emergency exit, flat is the only safe state."
                    if exit_side
                    else f"⚡ emergency on a flat book: keep {flipped} but trade it small."
                ),
                closed_signal=previous if exit_side else None,
                emergency_headline=str(
                    (self.lock.emergency_event or {}).get("headline", "")
                ),
            )
        return signal

    def _stage_view(self, fusion, emotion_locked: dict, warnings: list, degradation) -> None:
        """Stage the explanation of the *next* window without touching the one
        on screen (Round R: the lock covers the reasoning too)."""
        self._pending_view = {
            "fusion": fusion.to_dict(),
            "emotion_locked": emotion_locked,
            "conviction_note": fusion_module.conviction_note(fusion.direction, fusion.confidence),
            "warnings": list(warnings),
            "degradation": degradation,
        }

    def _apply_pending_view(self) -> None:
        """Swap the staged explanation in - only ever called at the boundary."""
        view = self._pending_view
        if view is None:
            return
        self._pending_view = None
        self.last_fusion = view["fusion"]
        self.emotion_locked = view["emotion_locked"]
        self.conviction_note = view["conviction_note"]
        self.warnings = view["warnings"]
        self.degradation = view["degradation"]

    def _previous_direction(self) -> str | None:
        """The direction of the window before this one, for the tie-break ladder."""
        current = self.lock.current_signal
        if current is not None and current.signal in ("BUY", "SELL"):
            return current.signal
        if self.lock.history:
            for past in reversed(self.lock.history):
                if past.signal in ("BUY", "SELL"):
                    return past.signal
        return None

    async def _prefetch(self, publish_deadline_wall: float) -> None:
        """Compute the signal for the next window - during this one."""
        started = time.perf_counter()
        snapshot = self.market.freeze(
            news_items=self.news.cache.latest(30),
            drg_outcomes=self.outcomes.array(),
        )
        self._pending_vol_bps = realized_volatility_bps(snapshot, self.asset)
        self._pending_spread_bps = quoted_spread_bps(snapshot, self.asset)
        formula_result = self.formulas.run(snapshot, self.asset)
        self.pending_formula_result = formula_result
        self.last_live_formulas = dict(formula_result.values)
        self.last_live_result = formula_result
        emotion_locked = analyze_emotions(
            snapshot, self.asset,
            formulas=formula_result.values, directional=self._directional_map(),
        ).to_dict()

        context = self._agent_context(snapshot, formula_result)
        agent_results = await self.agents.run_all(context)
        # Kept for the final freeze (the agents are the slow half).
        self._prefetch_agents = agent_results
        self._prefetch_context = context

        warnings = list(snapshot.warnings)
        if formula_result.insufficient_evidence:
            warnings.append("Insufficient formula evidence: 11+ formulas returned zero.")
        for name, result in agent_results.items():
            if result.status.value == "TIMEOUT":
                warnings.append(f"{name} timed out this cycle.")

        fusion = self._fuse(snapshot, formula_result, agent_results, warnings, crowd=emotion_locked)
        self._stage_view(fusion, emotion_locked, warnings, self._compute_degradation(snapshot))

        draft = self._build_frozen_signal(snapshot, formula_result, agent_results, fusion, context)
        draft = draft._replace(
            computed_at=_iso(time.time()),
            valid_from=_iso(publish_deadline_wall),
            valid_until=_iso(publish_deadline_wall + self.settings.cycle_period_seconds),
            window_seconds=self.settings.cycle_period_seconds,
        )
        self._pending_signal = draft
        self.prefetch_ready = True
        self.prefetch_ms = (time.perf_counter() - started) * 1000.0
        self.prefetch_at = time.time()

        # Deliberately does not reveal the direction: only that the next window
        # is armed, so the UI can show readiness without breaking the lock.
        await self.broadcast(
            {
                "type": "NEXT_WINDOW_READY",
                "data": {
                    "cycle_number": self.stats.cycle_number + 1,
                    "prepared": True,
                    "compute_ms": round(self.prefetch_ms, 1),
                    "lead_seconds": round(max(0.0, publish_deadline_wall - time.time()), 1),
                    "lock_state": "LOCKED",
                    "message": (
                        "agents ready - the tape, formulas and crowd are frozen again "
                        "at the boundary"
                    ),
                    "final_lock_lead_seconds": self.settings.final_lock_lead_seconds,
                },
            }
        )

    async def _refreeze(self, publish_deadline_wall: float) -> None:
        """The fast half of the pipeline, again, right before the boundary.

        Everything that costs milliseconds - a fresh frozen snapshot, the 22
        formulas, the crowd reading, the fusion - is recomputed so the locked
        prediction and the locked crowd describe the market at the lock
        instant.  The agents' votes (seconds) come from the prefetch.  Nothing
        is published here; ``_open_window`` publishes at the boundary.
        """
        if self._pending_signal is None:
            return
        started = time.perf_counter()
        snapshot = self.market.freeze(
            news_items=self.news.cache.latest(30),
            drg_outcomes=self.outcomes.array(),
        )
        self._pending_vol_bps = realized_volatility_bps(snapshot, self.asset)
        self._pending_spread_bps = quoted_spread_bps(snapshot, self.asset)
        formula_result = self.formulas.run(snapshot, self.asset)
        self.pending_formula_result = formula_result
        self.last_live_formulas = dict(formula_result.values)
        self.last_live_result = formula_result
        # The crowd at lock time is read with the same formulas it is being
        # locked against, so the two blocks of the payload can be compared.
        emotion_locked = analyze_emotions(
            snapshot, self.asset,
            formulas=formula_result.values, directional=self._directional_map(),
        ).to_dict()

        agent_results = self._prefetch_agents or {}
        context = self._prefetch_context or self._agent_context(snapshot, formula_result)
        warnings = list(snapshot.warnings)
        if formula_result.insufficient_evidence:
            warnings.append("Insufficient formula evidence: 11+ formulas returned zero.")
        for name, result in agent_results.items():
            if result.status.value == "TIMEOUT":
                warnings.append(f"{name} timed out this cycle.")

        fusion = self._fuse(snapshot, formula_result, agent_results, warnings, crowd=emotion_locked)
        self._stage_view(fusion, emotion_locked, warnings, self._compute_degradation(snapshot))

        draft = self._build_frozen_signal(snapshot, formula_result, agent_results, fusion, context)
        draft = draft._replace(
            computed_at=_iso(time.time()),
            valid_from=_iso(publish_deadline_wall),
            valid_until=_iso(publish_deadline_wall + self.settings.cycle_period_seconds),
            window_seconds=self.settings.cycle_period_seconds,
        )
        self._pending_signal = draft
        self.final_freeze_ms = (time.perf_counter() - started) * 1000.0
        self.final_freeze_at = time.time()
        # The "computed X s before it opened" line must describe the freeze
        # the user is looking at, not the agent prefetch.
        self.prefetch_at = self.final_freeze_at
        self.prefetch_ms = self.final_freeze_ms

    async def _wait_for_cycle_start(self) -> None:
        """Sleep until the next boundary.

        With the true time scale this is phase-locked to the UTC minute via the
        NTP-corrected world clock (Section 9).  Compressed time scales simply
        use a fixed period.
        """
        if not self._cycle_started:
            self._cycle_started = True
            return
        if self.settings.use_world_clock:
            await asyncio.sleep(max(0.01, self.clock.seconds_until_next_minute()))
        else:
            await asyncio.sleep(max(0.01, self.settings.cycle_period_seconds))

    async def _run_cycle(self, bootstrap: bool = False, window_from: float | None = None) -> None:
        """Compute and publish the signal for the window that is starting.

        In pipelined mode this is only used for the very first window (so the
        dashboard is never blank) and as the fallback when a prefetch failed;
        every other window is published by :meth:`_open_window` from a signal
        that was computed during the previous countdown.
        """
        started = time.perf_counter()
        self.stats.cycle_number += 1
        self.stats.last_cycle_started = started

        # --- Asset switch takes effect here, never mid-cycle (test 13) ---
        if self.pending_asset and self.pending_asset != self.asset:
            log.info("Asset switch applied at cycle boundary: %s -> %s", self.asset, self.pending_asset)
            self.asset = self.pending_asset
            self.pending_asset = None

        # --- t=0: freeze -------------------------------------------------
        self.lock.new_cycle(self.stats.cycle_number)
        freeze_started_us = now_us()
        snapshot = self.market.freeze(
            news_items=self.news.cache.latest(30),
            drg_outcomes=self.outcomes.array(),
        )
        # How long the immutable copy took: the formulas only ever see this, so
        # it is the point where "analysis" officially begins.
        self.snapshot_us = now_us() - freeze_started_us
        self.last_snapshot = snapshot
        # The crowd, measured on the frozen tape - the same immutable inputs the
        # formulas see, so the reading that dampens this window's confidence is
        # reproducible from the snapshot alone.
        self.emotion_locked = analyze_emotions(
            snapshot, self.asset,
            formulas=self.last_live_formulas, directional=self._directional_map(),
        ).to_dict()
        self.warnings = list(snapshot.warnings)
        self.degradation = self._compute_degradation(snapshot)
        await self.broadcast(
            {
                "type": "CYCLE_START",
                "data": {
                    "cycle_number": self.stats.cycle_number,
                    "asset": self.asset,
                    "timestamp": self.clock.iso(),
                    "lock_state": LockState.COMPUTING.value,
                    "degradation_level": int(self.degradation),
                    "message": "Computing...",
                },
            }
        )

        # --- t=0.001-0.010: formulas -------------------------------------
        formula_result = self.formulas.run(snapshot, self.asset)
        self.last_formula_result = formula_result
        self.last_live_formulas = dict(formula_result.values)
        self.last_live_result = formula_result
        self.stats.formula_ms = formula_result.total_ms

        if formula_result.insufficient_evidence:
            self.warnings.append("Insufficient formula evidence: 11+ formulas returned zero.")

        # --- t=0.010-8.0: agents -----------------------------------------
        context = self._agent_context(snapshot, formula_result)
        agent_results = await self.agents.run_all(context)
        self.stats.agent_ms = self.agents.last_run_ms

        # --- fuse + lock --------------------------------------------------
        from backend.agents import fusion as fusion_module

        for name, result in agent_results.items():
            if result.status.value == "TIMEOUT":
                self.warnings.append(f"{name} timed out this cycle.")

        fusion = self._fuse(snapshot, formula_result, agent_results, self.warnings)
        self.last_fusion = fusion.to_dict()
        self.conviction_note = fusion_module.conviction_note(fusion.direction, fusion.confidence)

        signal = self._build_frozen_signal(snapshot, formula_result, agent_results, fusion, context)
        self._pending_vol_bps = realized_volatility_bps(snapshot, self.asset)
        self._pending_spread_bps = quoted_spread_bps(snapshot, self.asset)

        if not self.settings.signal_pipeline:
            # Classic (literal draft) timing: the panel shows "Computing..." for
            # the first 8 seconds and the lock lands at the deadline.
            await self._hold_until_lock_deadline(started)

        from_wall = window_from or time.time()
        until_wall = from_wall + self.settings.cycle_period_seconds
        if bootstrap and self.settings.use_world_clock:
            # A cold start lands mid-minute: run a short first window and end it
            # on the UTC boundary, so every later countdown is minute-aligned.
            until_wall = math.floor(from_wall / 60.0) * 60.0 + 60.0
            if until_wall - from_wall < 1.0:
                until_wall += 60.0
        elif bootstrap:
            # 15-second cadence: the first window is simply a full period long,
            # so the countdown the user sees is the one they will keep seeing.
            until_wall = from_wall + self.settings.cycle_period_seconds
        self.window_valid_from = from_wall
        self.window_valid_until = until_wall
        signal = self._stamp_window(signal, self.stats.cycle_number, from_wall)
        # Computed inline at the boundary (no previous window existed), so the
        # timestamp is simply now - the UI uses it to prove the pipeline is
        # real rather than to fake a whole second of pipeline.
        signal = signal._replace(computed_at=_iso(time.time()))
        if bootstrap:
            signal = signal._replace(
                preview=True,
                valid_until=_iso(until_wall),
                window_seconds=round(until_wall - from_wall, 3),
            )

        locked = self.lock.lock(signal)
        self.published_computed_at = time.time()
        self.published_compute_ms = formula_result.total_ms
        self.published_compute_us = int(formula_result.total_us)
        self.window_started_us = int(round(self.window_valid_from * 1e6))
        self.publish_latency_us = max(0, now_us() - self.window_started_us)
        self.stats.lock_ms = (time.perf_counter() - started) * 1000.0
        self.stats.cycles_completed += 1
        self.stats.signal_counts[locked.signal] = self.stats.signal_counts.get(locked.signal, 0) + 1
        if fusion.forced_reason:
            self.stats.forced_fallbacks += 1
        if fusion.weak:
            self.stats.weak_windows += 1

        await self.broadcast(
            {
                "type": "SIGNAL",
                "data": self.snapshot_payload(locked, include_history=True),
            }
        )
        self._schedule_outcome(locked, snapshot)

        # --- one heartbeat for the whole dashboard, on the window grid ----
        asyncio.create_task(
            self._heartbeat_loop(
                self.stats.cycle_number, self.window_valid_from, self.window_valid_until
            )
        )

    async def _hold_until_lock_deadline(self, cycle_started: float) -> None:
        """Wait out the "Computing..." window before publishing the signal."""
        deadline = self.settings.scaled(self.settings.lock_deadline_seconds)
        if deadline <= 0:
            return
        target = cycle_started + deadline
        while True:
            if self.lock.emergency_active():
                return
            remaining = target - time.perf_counter()
            if remaining <= 0:
                return
            if self._stop.is_set():
                return
            await asyncio.sleep(min(0.1, remaining))

    # ------------------------------------------------------------------
    def _compute_degradation(self, snapshot: FrozenMarketSnapshot) -> DegradationLevel:
        level = self.market.degradation_level()
        if not self.news.degradation_ok():
            level = max(level, DegradationLevel.NO_NEWS)
        if not any(
            agent.configured for agent in (self.agents.gemini, self.agents.github)
        ) and not self.agents.local.ready:
            level = max(level, DegradationLevel.NO_AGENTS)
        if self.brain.status.value in ("FALLBACK_CSV", "CACHED"):
            level = max(level, DegradationLevel.FALLBACK_BRAIN)
        if snapshot.tick_count("BTC") < 15:
            level = DegradationLevel.MINIMAL
        return level

    def _agent_context(self, snapshot: FrozenMarketSnapshot, formula_result: FormulaResult) -> dict:
        news_items = snapshot.news_items or ()
        headline = news_items[0].headline if news_items else "no headlines available"
        return {
            "asset": self.asset,
            "price": snapshot.last_price(self.asset),
            "timestamp": self.clock.iso(),
            "formulas": {k: round(v, 4) for k, v in formula_result.values.items()},
            "ccs_confidence": formula_result.ccs_confidence,
            "drg": float(formula_result.values.get("DRG", 0.0)),
            "win_rate": self.outcomes.win_rate(),
            "headline": headline,
            "brain_status": self.brain.status.value,
        }

    def _build_frozen_signal(
        self,
        snapshot: FrozenMarketSnapshot,
        formula_result: FormulaResult,
        agent_results: dict,
        fusion,
        context: dict,
    ) -> FrozenSignal:
        f = formula_result.values
        hedge = {
            "hsi": round(float(f.get("HSI", 0.0)), 4),
            "hrdd": round(float(f.get("HRDD", 0.0)), 4),
            "shrp": round(float(f.get("SHRP", 0.0)), 4),
            "gcdv": round(float(f.get("GCDV", 0.0)), 4),
            "stress": "high" if float(f.get("HSI", 0.0)) > 0.8 else "low",
        }
        latest = (snapshot.news_items or (None,))[0]
        news_block = {
            "latest_headline": latest.headline if latest else "No headlines available",
            "source": latest.source if latest else "",
            "tier": latest.tier if latest else 0,
            "niv": round(float(f.get("NIV", 0.0)), 4),
            "smd": round(float(f.get("SMD", 0.0)), 4),
            "last_poll_seconds_ago": round(self.news.cache.seconds_since_poll(), 1)
            if self.news.cache.last_poll
            else None,
        }

        emergency = self.lock.emergency_active()
        emergency_headline = ""
        if emergency and self.lock.emergency_event:
            emergency_headline = str(self.lock.emergency_event.get("headline", ""))

        return FrozenSignal(
            cycle_number=self.stats.cycle_number,
            timestamp=self.clock.iso(),
            asset=self.asset,
            signal=fusion.decision,
            confidence=float(fusion.confidence),
            reasoning=fusion.reasoning,
            formula_values=tuple(sorted((k, float(v)) for k, v in f.items())),
            agent_results=tuple(sorted(self.agents.context_block(agent_results).items())),
            is_emergency_override=emergency,
            ccs_value=float(f.get("CCSv2", 0.0)),
            ccs_confidence=float(formula_result.ccs_confidence),
            conviction=fusion.conviction,
            weak=bool(fusion.weak),
            direction_source=fusion.direction_source,
            direction_reason=describe_direction(fusion.direction, fusion.confidence)
            if fusion.direction
            else "",
            edge=float(fusion.edge),
            closed_signal=fusion.direction.closed_signal if fusion.direction else None,
            hedge=tuple(sorted(hedge.items())),
            news=tuple(sorted(news_block.items())),
            drg=float(f.get("DRG", 0.0)),
            brain_status=self.brain.status.value,
            degradation_level=int(self.degradation),
            warnings=tuple(self.warnings),
            price=snapshot.last_price(self.asset),
            total_ms=formula_result.total_ms,
            emergency_headline=emergency_headline,
            lock_state=LockState.EMERGENCY_OVERRIDE.value
            if emergency
            else LockState.LOCKED.value,
            superseded_by=None,
        )

    # ------------------------------------------------------------------
    # Live formula refresh (15 s) - Rule 2: the signal panel does NOT update
    # ------------------------------------------------------------------
    async def _heartbeat_loop(self, cycle_number: int, started: float, ends: float) -> None:
        """One PULSE per mark on the grid - everything refreshes together.

        This used to be a private 15-second formula timer, and the dashboard ran
        eight more of them (news, agents, brain, readiness, history...).  The
        user's report was exactly that: *"not in parallel with other features, it
        randomly running"*.  So there is now one schedule, owned by the window,
        and one message per mark carrying every feature the dashboard shows.
        A client that receives a PULSE renders the whole page from it in a single
        pass, at the same instant, and the countdown it is running was computed
        from the same grid.
        """
        try:
            for mark in self.tick_grid(started, ends):
                await self._sleep_until(mark["at_wall"])
                if self._stop.is_set() or self.stats.cycle_number != cycle_number:
                    return
                parts = mark["parts"]
                if "news" in parts:
                    # Fresh headlines on the tick, not on a private timer.
                    try:
                        await self.news.poll_all()
                    except Exception as exc:  # noqa: BLE001 - news is optional
                        log.debug("news poll on tick failed: %s", exc)
                if "formulas" in parts:
                    snapshot = self.market.freeze(
                        news_items=self.news.cache.latest(30),
                        drg_outcomes=self.outcomes.array(),
                    )
                    result = self.formulas.run(snapshot, self.asset)
                    self.last_live_result = result
                    self.last_live_formulas = {k: round(v, 6) for k, v in result.values.items()}
                await self.broadcast(
                    {
                        "type": "PULSE",
                        "data": {
                            "cycle_number": cycle_number,
                            "parts": parts,
                            "offset_seconds": mark["offset_seconds"],
                            "timestamp": self.clock.iso(),
                            "clock": self.master_clock(),
                            "window": self.window_status(),
                            "note": "Live values only - the signal stays LOCKED until the boundary.",
                            "lock": self.lock.status(),
                            # NB: deliberately NOT called "signal".  A signal
                            # *payload* uses "signal" for the locked direction,
                            # and clashing the two made the client mistake a
                            # pulse for a signal payload.
                            "locked_side": self.lock.current_signal.signal
                            if self.lock.current_signal
                            else None,
                            "live_formulas": self.live_formulas_payload(
                                note=(
                                    f"live at t+{int(mark['offset_seconds'])}s of window "
                                    f"{cycle_number}; the signal stays LOCKED until the boundary"
                                )
                            ),
                            "agents_status": self.agents_payload(),
                            "news_feed": self.news_payload(limit=30),
                            "emotions": self.emotions_payload(),
                            "accuracy": self.accuracy_block(),
                            "brain_explain": self.brain_explain_payload(),
                            "physics": self.physics_payload(),
                            # Round Y: the live mid, so the client can draw the
                            # realised path over the probability branches.
                            "live_price": float(self.market.last_price(self.asset) or 0.0) or None,
                        },
                    }
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.debug("heartbeat stopped: %s", exc)

    # ==================================================================
    # Crowd emotions (the user's "which emotion is dominant" panel)
    # ==================================================================
    async def _emotion_loop(self) -> None:
        """Score the live tape for emotions, several times a second.

        The cadence is deliberately faster than the PULSE grid: the emotion
        panel is the one reading that should move *between* the marks, because
        the whole point of the module is that a crowd's mood changes on the
        second.  Every sample is cheap (a few hundred microseconds over the
        last 600 ticks) and purely read-only, so it cannot disturb the cycle.
        """
        interval = max(0.1, float(self.settings.emotion_interval_seconds))
        # The 22 formulas cost ~6 ms, so the live pass the crowd is checked
        # against is refreshed every fourth sample (2 s at the default 0.5 s)
        # rather than only on the 15 s PULSE grid.  The heavy ``deep`` block
        # (~8 KB) rides along on the same samples only: the four-per-second
        # bars stay small so the socket never queues behind them - that queue
        # was what made the countdown stutter.
        sample_no = 0
        while not self._stop.is_set():
            try:
                if self.market.tick_count(self.asset) >= 15:
                    sample_no += 1
                    full = sample_no % 4 == 1
                    if full:
                        try:
                            snapshot = self.market.freeze(
                                news_items=self.news.cache.latest(30),
                                drg_outcomes=self.outcomes.array(),
                            )
                            result = self.formulas.run(snapshot, self.asset)
                            self.last_live_result = result
                            self.last_live_formulas = {
                                k: round(v, 6) for k, v in result.values.items()
                            }
                        except Exception as exc:  # noqa: BLE001 - keep the crowd loop alive
                            log.debug("live formula refresh failed: %s", exc)
                    reading = self.emotions.sample()
                    streamed = compact_emotions(reading)
                    if not full:
                        streamed.pop("deep", None)
                    # Streamed as its own small message so the panel moves
                    # between the PULSE marks.  The client has no timer of its
                    # own for this - the backend is still the only schedule.
                    await self.broadcast(
                        {
                            "type": "EMOTION",
                            "data": {
                                "cycle_number": self.stats.cycle_number,
                                "asset": self.asset,
                                "locked_side": (
                                    self.lock.current_signal.signal
                                    if self.lock.current_signal
                                    else None
                                ),
                                "deep_included": full,
                                "emotions": streamed,
                                "live_price": float(self.market.last_price(self.asset) or 0.0) or None,
                                "dampening": {
                                    "applied": (self.last_fusion or {}).get("crowd_adjustment", 1.0),
                                    "note": (self.last_fusion or {}).get("crowd_note", ""),
                                },
                            },
                        }
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - never take the engine down
                self.emotions.errors += 1
                log.debug("emotion sample failed: %s", exc)
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise

    def emotions_payload(self) -> dict:
        """The live emotion reading, plus the frozen one that shaped the signal.

        Both are in every payload because the panel's job is to answer two
        different questions at once: *what is the crowd feeling right now* and
        *what was it feeling when this locked signal was computed*.
        """
        payload = self.emotions.payload()
        payload["locked"] = self.emotion_locked or None
        payload["dampening"] = {
            "threshold": self.settings.emotion_dampen_threshold,
            "max": self.settings.emotion_dampen_max,
            "applied": (self.last_fusion or {}).get("crowd_adjustment", 1.0),
            "note": (self.last_fusion or {}).get("crowd_note", ""),
        }
        payload["asset"] = self.asset
        return payload

    # ==================================================================
    # Emergency handling
    # ==================================================================
    async def _on_critical_event(self, event: CriticalEvent) -> None:
        await self.trigger_emergency(event)

    async def trigger_emergency(
        self, event: CriticalEvent | dict, duration_seconds: float | None = None
    ) -> dict:
        payload = event.to_dict() if isinstance(event, CriticalEvent) else dict(event)
        # The caller may shorten or lengthen the hold (the manual trigger does);
        # a detected critical event always uses the configured default.
        hold = (
            float(duration_seconds)
            if duration_seconds is not None and duration_seconds > 0
            else self.settings.scaled(self.settings.emergency_duration_seconds)
        )
        previous = self.lock.current_signal.signal if self.lock.current_signal else "COMPUTING"
        from backend.core.direction import opposite as _opposite

        exit_side = _opposite(previous)
        overridden = self.lock.emergency_override(
            payload,
            duration_seconds=hold,
            timestamp=self.clock.iso(),
        )
        self.stats.emergency_count += 1
        self.warnings.append(f"EMERGENCY: {payload.get('headline', '')}")
        message = {
            "type": "EMERGENCY_OVERRIDE",
            "data": {
                "cycle_number": self.stats.cycle_number,
                "timestamp": self.clock.iso(),
                "headline": payload.get("headline", ""),
                "severity": payload.get("severity", "CRITICAL"),
                "reason": payload.get("reason", ""),
                "previous_signal": previous,
                "overridden_to": overridden.signal,
                "exit_side": exit_side,
                "closes_position": exit_side is not None,
                "emergency_duration_seconds": hold,
                "remaining_seconds": self.lock.emergency_remaining(),
                "signal": self.signal_payload(overridden),
            },
        }
        await self.broadcast(message)
        log.warning("EMERGENCY broadcast: %s", payload.get("headline", ""))
        return message["data"]

    async def _flash_watch(self) -> None:
        """Triggers 1 & 2 - price-based flash crash / spike detection."""
        while not self._stop.is_set():
            await asyncio.sleep(1.0)
            try:
                if self.lock.emergency_active():
                    continue
                for asset in cfg.ASSETS:
                    move, direction = self.market.recent_flash_move(asset)
                    detector = self.news.detector
                    event = detector.check_flash_move(asset, move * (direction or 1.0), direction)
                    if event is not None:
                        # ``report_price_event`` used to be the documented entry
                        # point for this; it only delegated back to _dispatch, so
                        # the trigger now calls the detector path directly.
                        await self.trigger_emergency(event)
                        break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.debug("flash watch error: %s", exc)

    # ==================================================================
    # Outcome tracking (Appendix B) - feeds the DRG reward signal
    # ==================================================================
    def _schedule_outcome(self, signal: FrozenSignal, snapshot: FrozenMarketSnapshot) -> None:
        if signal.signal not in ("BUY", "SELL"):
            # Cannot happen any more (the protocol is binary), but an unknown
            # direction must never be scored as a win or a loss.
            log.warning("outcome skipped: unknown signal %r", signal.signal)
            return
        if self._last_votes:
            self.ledger.remember(signal.asset, signal.cycle_number, self._last_votes,
                                 float((self._last_learned or {}).get("p_up") or 0.5))
        task = asyncio.create_task(self._evaluate_outcome(signal, snapshot))
        self._outcome_tasks.add(task)
        task.add_done_callback(self._outcome_tasks.discard)

    async def _evaluate_outcome(self, signal: FrozenSignal, snapshot: FrozenMarketSnapshot) -> None:
        horizon = self.settings.outcome_horizon
        await asyncio.sleep(horizon)
        entry = signal.price or (
            snapshot.last_price(signal.asset) if snapshot is not None else 0.0
        )
        try:
            exit_price = await self._price_at(signal.asset, entry)
        except Exception as exc:  # noqa: BLE001
            log.debug("outcome evaluation failed: %s", exc)
            return
        if entry <= 0 or exit_price <= 0:
            return

        change_bps = (exit_price / entry - 1.0) * 10_000.0
        if signal.signal == "BUY":
            outcome = 1.0 if exit_price > entry else -1.0
        else:
            outcome = 1.0 if exit_price < entry else -1.0
        # Signed P&L of the *position*, not the raw move: a SELL into a
        # +20 bps move is -20 bps.  (It used to print "LOSS +198 bps".)
        pnl_bps = change_bps if signal.signal == "BUY" else -change_bps
        self.outcomes.append(outcome, pnl_bps)
        learned_update = self.ledger.score(
            signal.asset, signal.cycle_number,
            None if exit_price == entry else exit_price > entry,
        )
        if learned_update:
            log.info("ledger scored window %s: %s (%d sources right, %d wrong)",
                     signal.cycle_number, learned_update["actual"],
                     learned_update["sources_right"], learned_update["sources_wrong"])
        self._side_outcomes.append((signal.signal, outcome))
        log.info(
            "outcome %s on %s: %+.1f bps (%s)",
            signal.signal,
            signal.asset,
            change_bps,
            "win" if outcome > 0 else "loss",
        )
        await self.broadcast(
            {
                "type": "OUTCOME",
                "data": {
                    "cycle_number": signal.cycle_number,
                    "asset": signal.asset,
                    "signal": signal.signal,
                    "outcome": outcome,
                    "pnl_bps": round(pnl_bps, 2),
                    "move_bps": round(change_bps, 2),
                    "win_rate": round(self.outcomes.win_rate(), 4),
                    "entry": entry,
                    "exit": exit_price,
                    "accuracy": self.accuracy_block(),
                    "horizon_seconds": round(horizon, 1),
                },
            }
        )

    async def _price_at(self, asset: str, fallback: float) -> float:
        """Prefer live data; the CoinGecko/simulator feeds both end up here."""
        price = self.market.last_price(asset)
        return price if price > 0 else fallback

    # ==================================================================
    # Public API surface
    # ==================================================================
    # ------------------------------------------------------------------
    # Feature payloads: one shape, used by both the REST mirrors and the
    # WebSocket snapshot, so a panel can never disagree with its endpoint.
    # ------------------------------------------------------------------
    def news_payload(self, limit: int = 30) -> dict:
        return {
            "items": self.news.latest_items(limit),
            "impact": self._news_impact_safe(),
            "status": self.news.status.to_dict(),
            "niv": round(self.news.current_niv(), 4),
            "cache_size": len(self.news.cache.all()),
            "providers": self.news.cache.providers,
        }

    def agents_payload(self) -> dict:
        return self.agents.status_payload()

    def brain_payload(self) -> dict:
        """Exactly what ``/api/brain/status`` returns, built without HTTP."""
        return self.brain.status_dict()

    def history_payload(self, limit: int = 72) -> dict:
        # The lock keeps 1440 windows (a day at 60 s, x6 data); the clamp
        # matches it instead of silently cutting the request at 240.
        return {"history": self.lock.recent_history(min(max(int(limit), 1), 1440))}

    def outcomes_payload(self, limit: int = 72) -> dict:
        rows = self.outcomes.array()
        recent = rows[-int(limit):] if limit else rows
        return {
            "count": int(rows.shape[0]),
            "returned": int(recent.shape[0]),
            "buffer_size": int(rows.shape[0]),
            "win_rate": round(self.outcomes.win_rate(), 4),
            "rows": [
                {
                    "outcome": float(o),
                    "pnl_bps": float(p),
                    "index": index,
                }
                for index, (o, p) in enumerate(recent, start=max(0, rows.shape[0] - recent.shape[0]))
            ],
        }

    def physics_payload(self) -> dict:
        """The thermodynamic layer's report *as locked with the window on
        screen* (it travels inside the fusion, so the Round R lock covers it)."""
        locked = (self.last_fusion or {}).get("physics") or None
        return {
            "locked": locked is not None,
            "weight": float(getattr(self.settings, "weight_physics", 0.0) or 0.0),
            "report": locked or self.last_physics,
        }

    def brain_explain_payload(self) -> dict:
        """What the brain just did, in plain language - the /api/brain/explain shape."""
        # Imported here: backend.brain.explain imports the manager's module.
        from backend.brain import explain as brain_explain

        return brain_explain.explain(self)

    def live_formulas_payload(self, note: str | None = None) -> dict:
        """The Formula Explorer block: live values, readings, traces, timings.

        Same shape as ``/api/formulas/live`` (which now calls this), so the
        Formula Explorer renders identically whether the values arrived in a
        PULSE or over REST.
        """
        result = self.last_live_result
        payload = {
            "cycle_number": self.stats.cycle_number,
            "timestamp": self.clock.iso(),
            "note": note or "Live formula values only. Signal remains LOCKED.",
            "signal": self.lock.current_signal.signal if self.lock.current_signal else None,
            "formulas": dict(self.last_live_formulas),
            "readings": _readings_for(result) if result is not None else {},
            "traces": result.traces if result is not None else {},
            "checks": result.checks if result is not None else {},
            "double_check": result.check_summary() if result is not None else {},
            "timings_ms": (
                {k: round(float(v), 4) for k, v in result.timings_ms.items()}
                if result is not None
                else {}
            ),
            "total_ms": round(result.total_ms, 4) if result is not None else 0.0,
            # Round I: the same pass at microsecond resolution, plus the history
            # statistics and the tape measurements the explorer prints next to
            # each value.  The Formulae Explorer used to show a value and an ms
            # timing; now it can say how normal that value is, how long the
            # formula took in µs, and what the tape itself was doing.
            "timings_us": (
                {k: int(v) for k, v in result.timings_us.items()} if result is not None else {}
            ),
            "total_us": int(result.total_us) if result is not None else 0,
            "stats": dict(result.stats) if result is not None else {},
            "history_window": getattr(result, "history_window", 0) if result is not None else 0,
            "micro": dict(result.micro) if result is not None else {},
            # Round Z: &b (what fed this pass) and ¶gn (where in the window it sits).
            "provenance": dict(getattr(result, "provenance", {}) or {}) if result is not None else {},
            "phase": dict(getattr(result, "phase", {}) or {}) if result is not None else {},
            "feed_status": result.feed_status() if result is not None else {},
        }
        return payload

    def snapshot_payload(self, signal: FrozenSignal | None = None, *, include_history: bool = False) -> dict:
        """Everything the dashboard renders, in one message.

        This is what makes the panels update *in parallel*: the client applies
        one snapshot in a single render pass, so the signal, the formulas, the
        news list, the agents and the accuracy panel all change on the same
        frame instead of on five private timers.
        """
        payload = self.signal_payload(signal)
        payload["clock"] = self.master_clock()
        payload["live_formulas"] = self.live_formulas_payload()
        payload["agents_status"] = self.agents_payload()
        payload["news_feed"] = self.news_payload(limit=30)
        payload["emotions"] = self.emotions_payload()
        payload["accuracy"] = self.accuracy_block()
        payload["brain_explain"] = self.brain_explain_payload()
        payload["physics"] = self.physics_payload()
        if include_history:
            payload["history"] = self.history_payload(limit=72)
            payload["outcomes"] = self.outcomes_payload()
        return payload

    def signal_payload(self, signal: FrozenSignal | None = None) -> dict:
        """The message the dashboard and the Flutter client both render.

        Every payload carries the window block, so a client can always answer
        "which second of which window am I looking at, and is the next signal
        ready?" without making a second request.
        """
        window = self.window_status()
        current = signal or self.lock.current_signal
        if current is None:
            # Cold start: nothing has been locked yet.  A sentinel keeps the
            # layout intact and tells the user what is happening, instead of a
            # null payload that leaves the panel blank or half-rendered.
            return {
                "lock_state": LockState.COMPUTING.value,
                "lock_icon": LockState.COMPUTING.icon,
                "signal": None,
                "prediction": self.prediction_payload(None),
                "cycle_number": self.stats.cycle_number,
                "asset": self.asset,
                "confidence": 0.0,
                "reasoning": "preparing the first window",
                "preview": True,
                "risk": {},
                "computed_at": "",
                "valid_from": window["valid_from"],
                "valid_until": window["valid_until"],
                "window_seconds": window["window_seconds"],
                "seconds_remaining": window["seconds_remaining"],
                "conviction_note": None,
                "fusion": self.last_fusion,
                "pending_asset": self.pending_asset,
                "window": window,
            }
        payload = current.to_dict()
        payload["conviction_note"] = self.conviction_note
        # The prediction block is what the widget panel renders: the side, the
        # 1:1 levels, how old the call is, and why it was made.
        prediction = self.prediction_payload(current)
        payload["prediction"] = prediction
        payload["prediction_age_seconds"] = prediction["age_seconds"]
        payload["prediction_stale"] = prediction["state"] == "STALE"
        # The per-formula provenance of the *locked* values: what each number
        # means (reading) and the intermediate arithmetic behind it (trace).
        # `last_formula_result` is documented as the pass behind the locked
        # signal, so the explorer can show it without a second computation.
        locked_result = self.last_formula_result
        if locked_result is not None:
            payload["readings"] = _readings_for(locked_result)
            payload["traces"] = locked_result.traces
        payload["fusion"] = self.last_fusion
        payload["pending_asset"] = self.pending_asset
        payload["window"] = window
        payload.setdefault("lock_icon", LockState.LOCKED.icon)
        return payload

    def switch_asset(self, asset: str) -> dict:
        asset = (asset or "").upper()
        if asset not in cfg.ASSETS:
            raise ValueError(f"unsupported asset {asset!r}")
        if asset == self.asset:
            self.pending_asset = None
            return {"asset": self.asset, "pending": None, "effective": "current cycle"}
        self.pending_asset = asset
        log.info("Asset switch queued: %s -> %s at next cycle", self.asset, asset)
        return {
            "asset": self.asset,
            "pending": asset,
            "effective": "next cycle",
            "message": f"Switching to {asset} at next cycle...",
        }

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "warming_up": self.warming,
            "start_error": self.start_error,
            "uptime_seconds": round(time.time() - self._started_ts, 1),
            "ready_seconds": round(time.time() - self.ready_at, 1) if self.ready_at else None,
            "asset": self.asset,
            "pending_asset": self.pending_asset,
            "cycle": self.stats.to_dict(),
            "lock": self.lock.status(),
            "degradation_level": int(self.degradation),
            "degradation_label": self.degradation.label,
            "warnings": self.warnings,
            "clock": {
                "utc": self.clock.iso(),
                "ntp_synced": self.clock.ntp_synced,
                "seconds_into_minute": round(self.clock.seconds_into_minute(), 3),
                "cycle_period_seconds": self.settings.cycle_period_seconds,
                "time_scale": self.settings.time_scale,
            },
            "window": self.window_status(),
            "pipeline": self.settings.signal_pipeline,
            # Round Z: the tape's provenance, live - never a start-up constant.
            "tape": {
                "source": self.market.active_source,
                "simulated": self.market.tape_is_simulated,
                "btc_ticks": self.market.tick_count("BTC"),
                "rejected_writes": dict(self.market.rejected),
                "source_age_seconds": round(time.time() - self.market.source_changed_at, 1),
                "feeds": self.market.feeds_report().get("feeds", {}),
            },
            "infrastructure": {
                "redis": self.store.backend,
                "outcomes": len(self.outcomes),
                "win_rate": round(self.outcomes.win_rate(), 4),
            },
        }

    # ==================================================================
    # Prediction block (freshness + reasoning + 1:1 levels)
    # ==================================================================
    def _news_impact_safe(self) -> dict:
        try:
            impact = self.news.impact()
            self.last_news_impact = impact
            return impact
        except Exception as exc:  # noqa: BLE001
            log.debug("news impact failed: %s", exc)
            return {}

    def _fuse(self, snapshot, formula_result, agent_results: dict, warnings: list,
              crowd: dict | None = None) -> fusion_module.FusionResult:
        """The spec recipe, then the evidence ledger's verdict on top of it.

        Pass 1 is the specification's fusion (brain 0.40 + agents); its score
        is itself one of the ledger's sources.  Pass 2 hands the ledger's
        weighted verdict to ``fuse`` which, once the ledger is active, lets it
        decide the side and cap the confidence at what has been earned.
        """
        agreement = consensus(formula_result.values, self._directional_map())
        physics_report = None
        if float(getattr(self.settings, "weight_physics", 0.0) or 0.0) > 0 and snapshot is not None:
            try:
                physics_report = self.physics.compute(snapshot, self.asset)
                self.last_physics = physics_report
            except Exception as exc:  # noqa: BLE001
                log.warning("thermodynamic layer failed this pass: %s", exc)
                warnings.append(f"thermodynamic layer skipped: {exc}")
        common = dict(
            agents=agent_results,
            ccs_value=float(formula_result.values.get("CCSv2", 0.0)),
            ccs_confidence=formula_result.ccs_confidence,
            hsi=float(formula_result.values.get("HSI", 0.0)),
            settings=self.settings,
            warnings=warnings,
            insufficient_evidence=formula_result.insufficient_evidence,
            emergency=self.lock.emergency_active(),
            previous_signal=self._previous_direction(),
            formula_consensus=agreement["score"],
            consensus_voters=agreement["voters"],
            recent_accuracy=self.accuracy_block(),
            crowd=crowd if crowd is not None else self.emotion_locked,
            physics=physics_report,
            news_impact=self._news_impact_safe(),
            asset=self.asset,
            ledger_sources=self._ledger_sources_safe(),
        )
        crowd = common["crowd"]
        spec = fusion_module.fuse(**common)
        try:
            niv = float(self.news.current_niv())
        except Exception:  # noqa: BLE001
            niv = 0.0
        crowd_tone = 0.0
        try:
            crowd_tone = float((crowd or {}).get("tone") or
                               ((crowd or {}).get("formula_agreement") or {}).get("crowd_tone") or 0.0)
        except (TypeError, ValueError):
            crowd_tone = 0.0
        votes = EvidenceLedger.votes_from(
            formula_values=formula_result.values,
            directional=self._directional_map(),
            ccs_value=float(formula_result.values.get("CCSv2", 0.0)),
            agents=agent_results,
            spec_score=spec.score,
            niv=niv,
            crowd_tone=crowd_tone,
            candles=snapshot.candles(self.asset) if snapshot is not None else None,
            ticks=snapshot.ticks(self.asset) if snapshot is not None else None,
            physics_vote=float((physics_report or {}).get("vote") or 0.0),
        )
        learned = self.ledger.evaluate(self.asset, votes)
        self._last_votes = votes
        self._last_learned = learned
        if not learned.get("active"):
            spec.learned = {"active": False, "scored": learned.get("scored", 0),
                            "min_samples": learned.get("min_samples"), "p_up": learned.get("p_up"),
                            "voters": learned.get("voters", 0), "watching": len(votes)}
            return spec
        return fusion_module.fuse(**common, learned=learned)

    def _ledger_sources_safe(self) -> list:
        """Ledger source rows for the lock-weight table (never raises)."""
        try:
            return list((self.ledger.report(self.asset).get("assets") or {}).get(self.asset, {}).get("sources") or [])
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def _directional_map() -> dict[str, str]:
        """name -> category, for the formulas that carry a direction.

        HSI and ERC are regime indicators: they modulate the others and never
        vote, so they must not be counted as dissent in the consensus.
        """
        from backend.formulas.engine import ALL_FORMULAS

        return {
            spec.name: spec.category
            for spec in ALL_FORMULAS
            if getattr(spec, "directional", True)
        }

    def accuracy_block(self) -> dict:
        """Measured form of recent windows, overall and per side."""
        rows = self.outcomes.array()
        evaluated = int(rows.shape[0])
        wins = int((rows[:, 0] > 0).sum()) if evaluated else 0
        streak = 0
        for outcome in reversed(rows[:, 0].tolist() if evaluated else []):
            if outcome == 0:
                break
            sign = 1 if outcome > 0 else -1
            if streak == 0:
                streak = sign
            elif (streak > 0) == (sign > 0):
                streak += sign
            else:
                break
        per_side = {}
        for side in ("BUY", "SELL"):
            history = [row for row in self._side_outcomes if row[0] == side]
            if history:
                hits = sum(1 for _, outcome in history if outcome > 0)
                per_side[side] = {
                    "evaluated": len(history),
                    "win_rate": round(hits / len(history), 4),
                }
        return {
            "evaluated": evaluated,
            "win_rate": round(wins / evaluated, 4) if evaluated else None,
            "streak": streak,
            "last_outcome": (
                None
                if not evaluated
                else ("WIN" if rows[-1, 0] > 0 else "LOSS" if rows[-1, 0] < 0 else "FLAT")
            ),
            "last_pnl_bps": None if not evaluated else round(float(rows[-1, 1]), 2),
            "target_seconds": round(self.settings.outcome_horizon, 1),
            "per_side": per_side,
        }

    def _reasoning_for(self, signal: FrozenSignal | None, formula_result: FormulaResult | None) -> dict:
        """The plain-English case for the current side."""
        fusion = self.last_fusion or {}
        hedge = dict(signal.hedge) if signal else {}
        news = dict(signal.news) if signal else {}
        risk = dict(signal.risk) if signal else {}
        values = dict(formula_result.values) if formula_result is not None else {}
        brain = {}
        if signal is not None:
            brain = {"ccs": signal.ccs_value}
        extra_against: list[str] = []
        if signal is not None and signal.weak:
            extra_against.append(
                f"The side came from the tie-break ladder ({signal.direction_reason or 'thin edge'}) "
                "rather than from a strong score — treat the conviction, not the side, as the signal."
            )
        if formula_result is not None and formula_result.insufficient_evidence:
            extra_against.append(
                "Evidence is thin: 11 or more of the 22 formulas returned zero this window."
            )
        if signal is not None and signal.is_emergency_override:
            extra_against.append(
                "An emergency override is active: the level is an exit, not a fresh entry."
            )
        return build_reasoning(
            side=(signal.signal if signal else "BUY"),
            confidence=(signal.confidence if signal else 0.0),
            conviction=(signal.conviction if signal else "LOW"),
            fusion=fusion,
            formula_values=values,
            directional=self._directional_map(),
            hedge=hedge,
            news=news,
            risk=risk,
            brain=brain,
            accuracy=self.accuracy_block(),
            window_seconds=self.settings.cycle_period_seconds,
            extra_against=extra_against,
            crowd=self.emotion_locked,
        )

    def prediction_payload(self, signal: FrozenSignal | None = None,
                           formula_result: FormulaResult | None = None,
                           now: float | None = None) -> dict:
        """The prediction block: side, levels, freshness, reasoning, detail.

        The forecast window is the 60-second window that starts when the
        prediction is released, and its result is scored one window later; both
        are stated explicitly in the ``horizon`` block, in microseconds.
        """
        current = signal or self.lock.try_get_current()
        result = formula_result if formula_result is not None else self.last_formula_result
        risk = dict(current.risk) if current else {}
        computed_wall = self._computed_wall(current)
        reasoning = self._reasoning_for(current, result)
        return build_prediction(
            side=(current.signal if current else "·  ·  ·"),
            confidence=(current.confidence if current else 0.0),
            conviction=(current.conviction if current else "LOW"),
            computed_wall=computed_wall,
            max_age=self.settings.prediction_expired,
            risk=risk,
            reasoning=reasoning,
            accuracy=self.accuracy_block(),
            window_seconds=self.settings.cycle_period_seconds,
            horizon_seconds=self.settings.cycle_period_seconds,
            scoring_seconds=self.settings.outcome_horizon,
            weak=bool(current.weak) if current else False,
            emergency=bool(current.is_emergency_override) if current else False,
            now=now,
            detail_block=self.prediction_detail(
                current, result, risk=risk, reasoning=reasoning
            ),
        )

    @staticmethod
    def _crowd_deep_summary(deep: dict) -> dict:
        """The deep-reasoning block of a locked crowd reading, trimmed for the
        prediction detail: what the Bayesian filter believed, the verdict of
        each microstructure formula and the full chain."""
        if not deep or not deep.get("available"):
            return {"available": False}
        post = deep.get("posterior") or {}
        flow = deep.get("flow") or {}
        bands = deep.get("bands") or {}
        seconds = bands.get("seconds") or {}
        return {
            "available": True,
            "belief": post.get("argmax"),
            "belief_probability": post.get("argmax_probability"),
            "certainty": post.get("certainty"),
            "surprise_kl": post.get("surprise_kl"),
            "regime": (deep.get("regime") or {}).get("label"),
            "branching_ratio": (deep.get("hawkes") or {}).get("branching_ratio"),
            "vpin": flow.get("vpin"),
            "kyle_lambda_bps": flow.get("kyle_lambda_bps"),
            "kyle_r2": flow.get("kyle_r2"),
            "variance_ratio": seconds.get("variance_ratio"),
            "hurst": seconds.get("hurst"),
            "entropy": seconds.get("entropy"),
            "sign_memory": flow.get("sign_memory"),
            "manipulation": dict(deep.get("manipulation") or {}),
            "compute_us": deep.get("compute_us"),
            "chain": [
                {k: step.get(k) for k in ("step", "name", "formula", "value", "unit", "reads", "timescale", "feeds")}
                for step in deep.get("chain") or []
            ],
        }

    def prediction_detail(
        self,
        signal: FrozenSignal | None,
        result: FormulaResult | None,
        *,
        risk: dict | None = None,
        reasoning: dict | None = None,
    ) -> dict:
        """Every number behind the side, grouped the way a trader would ask.

        Category scores, the ten strongest supporters and opponents with their
        values and categories, the arithmetic of the confidence, the level
        geometry in bps and in currency, the microsecond picture of the tape,
        the per-formula history statistics, the agent spread and the brain
        read-out.  This is the "increase details" ask, made concrete.
        """
        from backend.formulas.engine import ALL_FORMULAS

        values = dict(result.values) if result is not None else {}
        stats = dict(result.stats) if result is not None else {}
        micro = dict(result.micro) if result is not None else {}
        directional = self._directional_map()
        consensus = (reasoning or {}).get("consensus") or {}
        supporters = list(consensus.get("up_names") or [])
        opponents = list(consensus.get("down_names") or [])

        def row(name: str) -> dict:
            entry = stats.get(name, {})
            spec = next((s for s in ALL_FORMULAS if s.name == name), None)
            return {
                "name": name,
                "category": (spec.category if spec else ""),
                "value": round(float(values.get(name, 0.0)), 6),
                "zscore": entry.get("zscore"),
                "percentile": entry.get("percentile"),
                "mean": entry.get("mean"),
                "samples": entry.get("samples"),
            }

        # Category scores: the mean of the signed values in each category, so a
        # reader can see *where* the agreement comes from.
        category_scores: dict[str, dict] = {}
        for spec in ALL_FORMULAS:
            value = float(values.get(spec.name, 0.0))
            bucket = category_scores.setdefault(
                spec.category, {"category": spec.category, "count": 0, "sum": 0.0, "mean": 0.0, "directional": spec.directional}
            )
            bucket["count"] += 1
            bucket["sum"] += value
        for bucket in category_scores.values():
            bucket["mean"] = round(bucket["sum"] / max(1, bucket["count"]), 6)
            bucket["sum"] = round(bucket["sum"], 6)

        # The arithmetic of the confidence, read straight off the fusion result.
        contributions = (self.last_fusion or {}).get("contributions") or {}
        crowd = self.emotion_locked or {}
        crowd_emotions = crowd.get("emotions") or []
        crowd_dominant = crowd.get("dominant") or {}
        crowd_manipulation = crowd.get("manipulation") or {}
        parts = {
            "fusion_confidence": round(float((signal.confidence if signal else 0.0)), 6),
            "formula_consensus": consensus.get("score"),
            "consensus_multiplier": (self.last_fusion or {}).get("consensus_multiplier"),
            "calibration_multiplier": (self.last_fusion or {}).get("calibration_multiplier"),
            "hedge_dampening": (self.last_fusion or {}).get("hedge_dampening"),
            "crowd_dampening": (self.last_fusion or {}).get("crowd_adjustment", 1.0),
            "crowd_note": (self.last_fusion or {}).get("crowd_note", ""),
            "degradation_level": int(self.degradation),
            "contributions": {
                key: value.get("score") if isinstance(value, dict) else value
                for key, value in contributions.items()
            },
        }

        levels = {
            "entry": (risk or {}).get("entry"),
            "take_profit": (risk or {}).get("take_profit"),
            "stop_loss": (risk or {}).get("stop_loss"),
            "tp_bps": (risk or {}).get("tp_bps"),
            "sl_bps": (risk or {}).get("sl_bps"),
            "rr": (risk or {}).get("rr"),
            "volatility_bps": (risk or {}).get("volatility_bps"),
            "distance_price": (
                round(abs(float((risk or {}).get("take_profit", 0.0)) - float((risk or {}).get("entry", 0.0))), 6)
                if (risk or {}).get("take_profit")
                else None
            ),
            "note": (risk or {}).get("note"),
        }

        engine = {
            "compute_us": int(result.total_us) if result is not None else 0,
            "compute_ms": round(result.total_ms, 3) if result is not None else 0.0,
            "per_formula_us": dict(result.timings_us) if result is not None else {},
            "publish_latency_us": self.publish_latency_us,
            "tick_interval_us": micro.get("mean_interval_us"),
            "resolution_us": micro.get("resolution_us"),
            "history_samples": max((s.get("samples", 0) for s in stats.values()), default=0),
        }

        agents = {}
        if signal is not None:
            # ``agent_results`` is the frozen tuple of (name, payload) pairs; the
            # signal object exposes it as ``agent_dict()``.
            for name, payload in dict(signal.agent_dict()).items():
                if isinstance(payload, dict):
                    agents[name] = {
                        "decision": payload.get("decision"),
                        "confidence": payload.get("confidence"),
                        "status": payload.get("status"),
                    }

        return prediction_detail(
            side=(signal.signal if signal else "·  ·  ·"),
            category_scores=category_scores,
            supporters=[row(name) for name in supporters[:10]],
            opponents=[row(name) for name in opponents[:10]],
            confidence_parts=parts,
            levels=levels,
            micro=micro,
            stats={name: stats.get(name, {}) for name in sorted(stats)},
            agreement={
                "score": consensus.get("score"),
                "voters": consensus.get("voters"),
                "up": consensus.get("up"),
                "down": consensus.get("down"),
                "directional_formulas": len(directional),
                "engine": engine,
            },
            agents=agents,
            brain=self.brain_explain_payload(),
            learned=dict((self.last_fusion or {}).get("learned") or {}),
            crowd={
                "available": bool(crowd.get("available")),
                "dominant": crowd_dominant.get("label"),
                "percent": crowd_dominant.get("percent"),
                "timescale": crowd_dominant.get("dominant_timescale"),
                "tone_bias": crowd.get("tone_bias"),
                "read": crowd.get("read"),
                "manipulation": crowd_manipulation.get("score"),
                "manipulation_kind": crowd_manipulation.get("kind"),
                "manipulation_note": crowd_manipulation.get("note"),
                "emotions": [
                    {
                        "name": item.get("name"),
                        "label": item.get("label"),
                        "percent": item.get("percent"),
                        "timescale": item.get("dominant_timescale"),
                    }
                    for item in crowd_emotions
                ],
                "drivers": crowd_dominant.get("drivers") or [],
                # Round L: the locked crowd against the lock-time formulas, and
                # the rule that the formulas keep the vote.
                "formula_agreement": dict(crowd.get("formula_agreement") or {}),
                # The deep layer at lock time: the filter's belief, the
                # microstructure verdicts and the reasoning chain (round K).
                "deep": self._crowd_deep_summary(crowd.get("deep") or {}),
            },
            formula_count=sum(1 for spec in ALL_FORMULAS if spec.name in values),
        )

    def _computed_wall(self, signal: FrozenSignal | None) -> float:
        """When the published signal was computed, as a unix timestamp."""
        if self.published_computed_at:
            return self.published_computed_at
        stamp = getattr(signal, "computed_at", "") or ""
        if not stamp:
            return 0.0
        try:
            import calendar

            return float(calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")))
        except Exception:  # noqa: BLE001
            return 0.0

    def prediction_is_stale(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        computed = self._computed_wall(self.lock.try_get_current())
        if computed <= 0:
            return True
        return (now - computed) > self.settings.prediction_expired

    async def ensure_fresh(self, now: float | None = None) -> bool:
        """Recompute and republish if the live prediction has aged out.

        The cycle loop already refreshes every window, so this is the belt to
        that pair of braces: if a window was ever missed (a stalled event loop,
        a suspended process), the API calls this before it answers, and the
        user never sees a prediction older than the contract.  Returns True
        when a refresh was performed.
        """
        if not self.prediction_is_stale(now=now):
            return False
        snapshot = self.market.freeze(
            news_items=self.news.cache.latest(30),
            drg_outcomes=self.outcomes.array(),
        )
        result = self.formulas.run(snapshot, self.asset)
        self.last_live_result = result
        self.last_live_formulas = {k: round(v, 6) for k, v in result.values.items()}
        self.last_formula_result = result
        self.published_computed_at = time.time()
        self.published_compute_ms = float(result.total_ms)
        self.published_compute_us = int(result.total_us)
        self._pending_vol_bps = realized_volatility_bps(snapshot, self.asset)
        self._pending_spread_bps = quoted_spread_bps(snapshot, self.asset)
        await self.broadcast(
            {
                "type": "PREDICTION_REFRESH",
                "data": {
                    "cycle_number": self.stats.cycle_number,
                    "reason": f"prediction older than {self.settings.prediction_expired:.0f}s",
                    "computed_at": _iso(self.published_computed_at),
                },
            }
        )
        log.warning(
            "prediction refreshed out of band: the last window was older than %.0fs",
            self.settings.prediction_expired,
        )
        return True

    # ------------------------------------------------------------------
    # One clock, one tick (the countdown contract)
    # ------------------------------------------------------------------
    def tick_grid(self, started: float, ends: float) -> list[dict]:
        """The refresh marks inside one window, on a fixed grid.

        Everything that refreshes mid-window does it *here*, at the same
        instants, so the whole dashboard updates in parallel instead of each
        panel running a private timer:

            t+0    the signal for this window (published at the boundary)
            t+15   formulas + news + agents + brain + accuracy   (one PULSE)
            t+30   the same, plus a fresh news poll
            t+45   the same

        The marks are derived from the window start, never from "now", so two
        clients - and the backend - always agree on when the next one is.
        """
        return window_clock.grid_marks(
            started, ends,
            self.settings.scaled(self.settings.formula_refresh_seconds),
            self.settings.scaled(self.settings.news_poll_seconds),
        )

    def master_clock(self, now: float | None = None) -> dict:
        """The one clock every countdown in both frontends is rendered from.

        It is deliberately absolute: the client is told when the window started
        and when it ends (as epoch milliseconds), plus the server's own time.
        The client measures its offset against ``server_time_ms`` once and then
        counts down to a fixed instant, so the number on screen cannot drift,
        jump or restart when an unrelated message arrives.
        """
        moment = time.time() if now is None else now
        period = self.settings.cycle_period_seconds
        started = self.window_valid_from or moment
        ends = self.window_valid_until or (started + period)
        return window_clock.describe(
            started, ends, now=moment,
            period_seconds=period,
            cycle_id=self.stats.cycle_number,
            marks=self.tick_grid(started, ends),
            minute_aligned=bool(self.settings.use_world_clock),
            freshness_max_age_seconds=self.settings.prediction_expired,
            scoring_horizon_seconds=self.settings.outcome_horizon,
            publish_latency_us=int(self.publish_latency_us),
            engine_compute_us=int(self.published_compute_us),
            snapshot_us=int(self.snapshot_us),
        )

    def window_status(self) -> dict:
        """The countdown the UI renders, plus what the engine is doing in it.

        With the pipeline on, the signal governing the window was computed
        *during the previous window* (``computed_at`` is in the past by
        definition) and the next one is being prepared right now
        (``prefetch_ready``).
        """
        now = time.time()
        period = self.settings.cycle_period_seconds
        started = self.window_valid_from or now
        ends = self.window_valid_until or (started + period)
        signal = self.lock.try_get_current()
        return {
            "valid_from": _iso(started),
            "valid_until": _iso(ends),
            "seconds_remaining": round(max(0.0, ends - now), 1),
            "window_seconds": period,
            # The authoritative clock block: every countdown in the dashboard and
            # in the Flutter client is rendered from this, and from nothing else.
            "clock": self.master_clock(now),
            "computed_at": getattr(signal, "computed_at", "") if signal else "",
            "computed_seconds_ago": (
                round(max(0.0, now - self.published_computed_at), 1)
                if self.published_computed_at
                else None
            ),
            "compute_ms": round(self.published_compute_ms, 1),
            # Round L: the two-stage lock.  Agents are prepared ``lock_lead``
            # seconds early; the tape, the 22 formulas, the crowd and the fusion
            # are frozen again ``final_lock_lead`` seconds before the boundary.
            "lock": {
                "stages": 2,
                "agents_lead_seconds": round(self.prefetch_lead, 1),
                "final_lock_lead_seconds": self.settings.final_lock_lead_seconds,
                "final_freeze_ms": round(self.final_freeze_ms, 1),
                "data_age_at_open_seconds": (
                    round(max(0.0, started - self.final_freeze_at), 2)
                    if self.final_freeze_at and started >= self.final_freeze_at
                    else None
                ),
                "rule": (
                    "the side, confidence, crowd and formulas of a window are frozen "
                    "at its boundary and never change until it ends"
                ),
            },
            "prefetch_ready": self.prefetch_ready,
            "prefetch_ms": round(self.prefetch_ms, 1),
            "next_window": self.stats.cycle_number + 1 if self.prefetch_ready else None,
            "pipeline": self.settings.signal_pipeline,
            "compute_starts_in": (
                round(max(0.0, ends - now - self.prefetch_lead), 1)
                if self.settings.signal_pipeline and not self.prefetch_ready
                else 0.0
            ),
            "compute_progress": (
                1.0 if self.prefetch_ready else
                round(min(1.0, max(0.0, (self.prefetch_lead - (ends - now)) / self.prefetch_lead)), 3)
                if self.settings.signal_pipeline and self.prefetch_lead > 0
                else 0.0
            ),
            "phase": (
                "NEXT_READY" if self.prefetch_ready else
                "PREPARING_NEXT" if self.settings.signal_pipeline else "LOCKED"
            ),
        }

    def health(self) -> dict:
        brain_health = self.brain.last_health
        try:
            agent_status = self.agents.status_payload()
        except Exception:  # pragma: no cover - only during warm-up
            agent_status = {
                name: {"status": "STARTING", "detail": "warming up"}
                for name in ("gemini", "local", "github")
            }
        components = {
            "market_data": self.market.status(),
            "news": self.news.status_payload(),
            "brain": ComponentStatus(
                name="brain",
                healthy=bool(brain_health and brain_health.healthy),
                detail=brain_health.message if brain_health else "not checked",
                mode=self.brain.status.value,
                extra={"checksum": brain_health.matrix_checksum if brain_health else ""},
            ),
            "agents": ComponentStatus(
                name="agents",
                healthy=any(
                    agent_status[name]["status"] in ("ACTIVE", "STUB")
                    for name in ("gemini", "local", "github")
                ),
                detail=", ".join(
                    f"{name}={agent_status[name]['status']}"
                    for name in ("gemini", "local", "github")
                ),
                mode="fusion",
            ),
        }
        # Market data, news and brain must all be healthy for an overall pass;
        # the agent tier is optional by design (degradation level 3 is valid).
        required = ("market_data", "news", "brain")
        overall = all(components[key].healthy for key in required)
        # A loop that has not opened a window for more than one period plus a
        # grace is stalled, whatever the components say: the supervisor
        # (run.sh --supervise) restarts the process on this flag.
        period = float(self.settings.cycle_period_seconds)
        since_window = (time.time() - self.window_valid_from) if self.window_valid_from else 0.0
        stalled = bool(self.ready and self._running and since_window > period + 45.0)
        if stalled:
            overall = False
        return {
            "ready": self.ready,
            "warming_up": self.warming,
            "start_error": self.start_error,
            "cycle_manager": "STALLED" if stalled else ("RUNNING" if self._running else ("WARMING_UP" if self.warming else "IDLE")),
            "stalled": stalled,
            "feeds": self.market.feeds_report(),
            "seconds_since_window": round(since_window, 1),
            "last_error": self.last_error,
            "healthy": overall,
            "degradation_level": int(self.degradation),
            "degradation_label": self.degradation.label,
            "components": {key: value.to_dict() for key, value in components.items()},
            "agents": agent_status,
            "brain": self.brain.status_dict(),
            "cycle": self.stats.to_dict(),
            "uptime_seconds": round(time.time() - self._started_at(), 1),
        }

    def _started_at(self) -> float:
        return getattr(self, "_started_ts", time.time())

    def mark_started(self) -> None:
        self._started_ts = time.time()

    # ==================================================================
    # Persistence of formula state across restarts
    # ==================================================================
    async def _persist_state(self) -> None:
        try:
            await self.store.set_pickle(
                "formulas:state", self.formulas.export_state(), ttl=86_400
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("could not persist formula state: %s", exc)

    async def _restore_state(self) -> None:
        try:
            payload = await self.store.get_pickle("formulas:state")
            if payload:
                restored = self.formulas.import_state(payload)
                log.info("Restored %d formula states from cache", restored)
        except Exception as exc:  # noqa: BLE001
            log.debug("could not restore formula state: %s", exc)


