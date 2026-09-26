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
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from backend.agents.orchestrator import AgentOrchestrator
from backend.brain.brain import Brain
from backend.core import config as cfg
from backend.core.clock import WorldClock
from backend.core.direction import describe as describe_direction
from backend.core.errors import ComponentStatus, DegradationLevel
from backend.core.prediction import build as build_prediction
from backend.core.prediction import build_reasoning, consensus
from backend.core.frozen_snapshot import FrozenMarketSnapshot
from backend.core.redis_bus import Store
from backend.core.risk import realized_volatility_bps, risk_levels
from backend.core.signal_lock import FrozenSignal, LockState, SignalLockController
from backend.data.market_hub import MarketDataHub
from backend.data.ring_buffer import OutcomeBuffer
from backend.formulas.engine import FormulaEngine, FormulaResult
from backend.news.critical_event_detector import CriticalEvent
from backend.news.news_engine import NewsEngine

log = logging.getLogger("drosophila.cycle")


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
        self.lock = SignalLockController()

        self.outcomes = OutcomeBuffer()
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
        #: The "thin edge" note (replaces the Section 10.2 HOLD box).  Present
        #: only when the side came from the tie-break ladder.
        self.conviction_note: dict | None = None

        # --- pipelined publication (Sections 10.4) ----------------------
        self._pending_signal: FrozenSignal | None = None
        self._pending_vol_bps: float = 0.0
        self.pending_formula_result: FormulaResult | None = None
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

        self.published_computed_at = self.prefetch_at or deadline_wall
        self.published_compute_ms = self.prefetch_ms
        await self.broadcast({"type": "SIGNAL", "data": self.signal_payload(locked)})
        self._schedule_outcome(locked, None)
        asyncio.create_task(self._live_refresh_loop(cycle, time.perf_counter()))
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
            news_items=self.news.cache.latest(5),
            drg_outcomes=self.outcomes.array(),
        )
        self._pending_vol_bps = realized_volatility_bps(snapshot, self.asset)
        formula_result = self.formulas.run(snapshot, self.asset)
        self.pending_formula_result = formula_result
        self.last_live_formulas = dict(formula_result.values)
        self.last_live_result = formula_result

        context = self._agent_context(snapshot, formula_result)
        agent_results = await self.agents.run_all(context)

        warnings = list(snapshot.warnings)
        if formula_result.insufficient_evidence:
            warnings.append("Insufficient formula evidence: 11+ formulas returned zero.")
        for name, result in agent_results.items():
            if result.status.value == "TIMEOUT":
                warnings.append(f"{name} timed out this cycle.")

        from backend.agents import fusion as fusion_module

        agreement = consensus(formula_result.values, self._directional_map())
        fusion = fusion_module.fuse(
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
        )
        self.last_fusion = fusion.to_dict()
        self.conviction_note = fusion_module.conviction_note(fusion.direction, fusion.confidence)
        self.warnings = warnings
        self.degradation = self._compute_degradation(snapshot)

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
                    "message": "next signal computed and held until the countdown ends",
                },
            }
        )

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
        snapshot = self.market.freeze(
            news_items=self.news.cache.latest(5),
            drg_outcomes=self.outcomes.array(),
        )
        self.last_snapshot = snapshot
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

        agreement = consensus(formula_result.values, self._directional_map())
        fusion = fusion_module.fuse(
            agents=agent_results,
            ccs_value=float(formula_result.values.get("CCSv2", 0.0)),
            ccs_confidence=formula_result.ccs_confidence,
            hsi=float(formula_result.values.get("HSI", 0.0)),
            settings=self.settings,
            warnings=self.warnings,
            insufficient_evidence=formula_result.insufficient_evidence,
            emergency=self.lock.emergency_active(),
            previous_signal=self._previous_direction(),
            formula_consensus=agreement["score"],
            consensus_voters=agreement["voters"],
            recent_accuracy=self.accuracy_block(),
        )
        self.last_fusion = fusion.to_dict()
        self.conviction_note = fusion_module.conviction_note(fusion.direction, fusion.confidence)

        signal = self._build_frozen_signal(snapshot, formula_result, agent_results, fusion, context)
        self._pending_vol_bps = realized_volatility_bps(snapshot, self.asset)

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
        self.stats.lock_ms = (time.perf_counter() - started) * 1000.0
        self.stats.cycles_completed += 1
        self.stats.signal_counts[locked.signal] = self.stats.signal_counts.get(locked.signal, 0) + 1
        if fusion.forced_reason:
            self.stats.forced_fallbacks += 1
        if fusion.weak:
            self.stats.weak_windows += 1

        await self.broadcast({"type": "SIGNAL", "data": self.signal_payload(locked)})
        self._schedule_outcome(locked, snapshot)

        # --- 15-second live formula refresh (never touches the signal) ----
        asyncio.create_task(self._live_refresh_loop(self.stats.cycle_number, started))

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
    async def _live_refresh_loop(self, cycle_number: int, cycle_started: float) -> None:
        # The Formula Explorer must refresh *inside* the window it belongs to.
        # With the 15-second cadence the nominal 15-second refresh would land
        # exactly on the next boundary, so the interval is capped at a third of
        # the window: the panel always shows numbers from the current window.
        cadence = max(
            0.2,
            min(
                self.settings.scaled(self.settings.formula_refresh_seconds),
                self.settings.cycle_period_seconds / 3.0,
            ),
        )
        try:
            while True:
                await asyncio.sleep(cadence)
                if self._stop.is_set() or self.stats.cycle_number != cycle_number:
                    return
                snapshot = self.market.freeze(
                    news_items=self.news.cache.latest(5),
                    drg_outcomes=self.outcomes.array(),
                )
                result = self.formulas.run(snapshot, self.asset)
                self.last_live_formulas = {k: round(v, 6) for k, v in result.values.items()}
                await self.broadcast(
                    {
                        "type": "FORMULA_UPDATE",
                        "data": {
                            "cycle_number": cycle_number,
                            "timestamp": self.clock.iso(),
                            "note": "Live formula values only. Signal remains LOCKED.",
                            "signal": self.lock.current_signal.signal
                            if self.lock.current_signal
                            else None,
                            "formulas": self.last_live_formulas,
                            "readings": _readings_for(self.last_live_result),
                            "traces": self.last_live_result.traces
                            if self.last_live_result is not None
                            else {},
                        },
                    }
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.debug("live formula refresh stopped: %s", exc)

    # ==================================================================
    # Emergency handling
    # ==================================================================
    async def _on_critical_event(self, event: CriticalEvent) -> None:
        await self.trigger_emergency(event)

    async def trigger_emergency(self, event: CriticalEvent | dict) -> dict:
        payload = event.to_dict() if isinstance(event, CriticalEvent) else dict(event)
        previous = self.lock.current_signal.signal if self.lock.current_signal else "COMPUTING"
        from backend.core.direction import opposite as _opposite

        exit_side = _opposite(previous)
        overridden = self.lock.emergency_override(
            payload,
            duration_seconds=self.settings.scaled(self.settings.emergency_duration_seconds),
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
                "emergency_duration_seconds": self.settings.scaled(
                    self.settings.emergency_duration_seconds
                ),
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
        task = asyncio.create_task(self._evaluate_outcome(signal, snapshot))
        self._outcome_tasks.add(task)
        task.add_done_callback(self._outcome_tasks.discard)

    async def _evaluate_outcome(self, signal: FrozenSignal, snapshot: FrozenMarketSnapshot) -> None:
        horizon = self.settings.scaled(self.settings.outcome_horizon_seconds)
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
        self.outcomes.append(outcome, abs(change_bps))
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
                    "pnl_bps": round(change_bps, 2),
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
            "infrastructure": {
                "redis": self.store.backend,
                "outcomes": len(self.outcomes),
                "win_rate": round(self.outcomes.win_rate(), 4),
            },
        }

    # ==================================================================
    # Prediction block (freshness + reasoning + 1:1 levels)
    # ==================================================================
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
            "target_seconds": round(self.settings.outcome_horizon_seconds, 1),
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
        )

    def prediction_payload(self, signal: FrozenSignal | None = None,
                           formula_result: FormulaResult | None = None,
                           now: float | None = None) -> dict:
        """The prediction block: side, levels, freshness, reasoning, accuracy."""
        current = signal or self.lock.try_get_current()
        result = formula_result if formula_result is not None else self.last_formula_result
        risk = dict(current.risk) if current else {}
        computed_wall = self._computed_wall(current)
        return build_prediction(
            side=(current.signal if current else "·  ·  ·"),
            confidence=(current.confidence if current else 0.0),
            conviction=(current.conviction if current else "LOW"),
            computed_wall=computed_wall,
            max_age=self.settings.prediction_expired,
            risk=risk,
            reasoning=self._reasoning_for(current, result),
            accuracy=self.accuracy_block(),
            window_seconds=self.settings.cycle_period_seconds,
            weak=bool(current.weak) if current else False,
            emergency=bool(current.is_emergency_override) if current else False,
            now=now,
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
            news_items=self.news.cache.latest(5),
            drg_outcomes=self.outcomes.array(),
        )
        result = self.formulas.run(snapshot, self.asset)
        self.last_live_result = result
        self.last_live_formulas = {k: round(v, 6) for k, v in result.values.items()}
        self.last_formula_result = result
        self.published_computed_at = time.time()
        self.published_compute_ms = float(result.total_ms)
        self._pending_vol_bps = realized_volatility_bps(snapshot, self.asset)
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
            "computed_at": getattr(signal, "computed_at", "") if signal else "",
            "computed_seconds_ago": (
                round(max(0.0, now - self.published_computed_at), 1)
                if self.published_computed_at
                else None
            ),
            "compute_ms": round(self.published_compute_ms, 1),
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
        return {
            "ready": self.ready,
            "warming_up": self.warming,
            "start_error": self.start_error,
            "cycle_manager": "RUNNING" if self._running else ("WARMING_UP" if self.warming else "IDLE"),
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


async def run_cycle_manager(manager: CycleManager) -> None:
    """Convenience coroutine for tests and scripts."""
    await manager.start()
    manager.mark_started()
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await manager.stop()
