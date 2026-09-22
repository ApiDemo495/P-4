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
  locked signal to HOLD for three cycles
* **queued asset switching** - toggling BTC/PAXG mid-cycle takes effect at the
  next boundary (test case 13)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from backend.agents.orchestrator import AgentOrchestrator
from backend.brain.brain import Brain
from backend.core import config as cfg
from backend.core.clock import WorldClock
from backend.core.errors import ComponentStatus, DegradationLevel
from backend.core.frozen_snapshot import FrozenMarketSnapshot
from backend.core.redis_bus import Store
from backend.core.signal_lock import FrozenSignal, LockState, SignalLockController
from backend.data.market_hub import MarketDataHub
from backend.data.ring_buffer import OutcomeBuffer
from backend.formulas.engine import FormulaEngine, FormulaResult
from backend.news.critical_event_detector import CriticalEvent
from backend.news.news_engine import NewsEngine

log = logging.getLogger("drosophila.cycle")


@dataclass
class CycleStats:
    cycle_number: int = 0
    lock_ms: float = 0.0
    formula_ms: float = 0.0
    agent_ms: float = 0.0
    last_cycle_started: float = 0.0
    cycles_completed: int = 0
    forced_holds: int = 0
    emergency_count: int = 0
    signal_counts: dict = field(default_factory=lambda: {"BUY": 0, "SELL": 0, "HOLD": 0})

    def to_dict(self) -> dict:
        return {
            "cycle_number": self.cycle_number,
            "cycles_completed": self.cycles_completed,
            "lock_ms": round(self.lock_ms, 2),
            "formula_ms": round(self.formula_ms, 2),
            "agent_ms": round(self.agent_ms, 2),
            "forced_holds": self.forced_holds,
            "emergency_count": self.emergency_count,
            "signal_counts": dict(self.signal_counts),
        }


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
        self.stats = CycleStats()

        self.asset = "BTC"
        self.pending_asset: str | None = None
        self.degradation = DegradationLevel.FULL
        self.warnings: list[str] = []
        self.last_snapshot: FrozenMarketSnapshot | None = None
        self.last_formula_result: FormulaResult | None = None
        self.last_live_formulas: dict[str, float] = {}
        self.last_fusion: dict = {}
        self.hold_warning: dict | None = None

        self._subscribers: set[asyncio.Queue] = set()
        self._tasks: list[asyncio.Task] = []
        self._outcome_tasks: set[asyncio.Task] = set()
        self._stop = asyncio.Event()
        self._running = False
        self._cycle_started = False

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

    async def stop(self) -> None:
        self._stop.set()
        self._running = False
        for task in self._tasks + list(self._outcome_tasks):
            task.cancel()
        await asyncio.gather(*(self._tasks + list(self._outcome_tasks)), return_exceptions=True)
        self._tasks.clear()
        self._outcome_tasks.clear()
        await self._persist_state()
        await self.news.stop()
        await self.brain.stop()
        await self.market.stop()
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

    async def _run_cycle(self) -> None:
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

        fusion = fusion_module.fuse(
            agents=agent_results,
            ccs_value=float(formula_result.values.get("CCSv2", 0.0)),
            ccs_confidence=formula_result.ccs_confidence,
            hsi=float(formula_result.values.get("HSI", 0.0)),
            settings=self.settings,
            warnings=self.warnings,
            insufficient_evidence=formula_result.insufficient_evidence,
            emergency=self.lock.emergency_active(),
        )
        self.last_fusion = fusion.to_dict()

        if fusion.decision == "HOLD":
            self.hold_warning = fusion_module.hold_warning(fusion.score, fusion.lean, fusion.confidence)
        else:
            self.hold_warning = None

        signal = self._build_frozen_signal(snapshot, formula_result, agent_results, fusion, context)

        # Section 10.3: the UI shows "Computing..." for the first 8 seconds and
        # the lock is published at the end of that window.  Computation itself
        # is usually far quicker (formulas ~3 ms, agents when configured up to
        # their 7 s timeout), so we hold the publication until the deadline -
        # unless an emergency fires, in which case we lock immediately.
        await self._hold_until_lock_deadline(started)

        locked = self.lock.lock(signal)
        self.stats.lock_ms = (time.perf_counter() - started) * 1000.0
        self.stats.cycles_completed += 1
        self.stats.signal_counts[locked.signal] = self.stats.signal_counts.get(locked.signal, 0) + 1
        if fusion.forced_reason:
            self.stats.forced_holds += 1

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
            hold_lean=fusion.lean,
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
        try:
            while True:
                await asyncio.sleep(self.settings.scaled(self.settings.formula_refresh_seconds))
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
                "overridden_to": "HOLD",
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
        if signal.signal == "HOLD":
            self.outcomes.append(0.0, 0.0)
            return
        task = asyncio.create_task(self._evaluate_outcome(signal, snapshot))
        self._outcome_tasks.add(task)
        task.add_done_callback(self._outcome_tasks.discard)

    async def _evaluate_outcome(self, signal: FrozenSignal, snapshot: FrozenMarketSnapshot) -> None:
        horizon = self.settings.scaled(self.settings.outcome_horizon_seconds)
        await asyncio.sleep(horizon)
        entry = snapshot.last_price(signal.asset) or signal.price
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
        current = signal or self.lock.current_signal
        if current is None:
            return {"lock_state": LockState.COMPUTING.value, "signal": None}
        payload = current.to_dict()
        payload["hold_warning"] = self.hold_warning if current.signal == "HOLD" else None
        payload["fusion"] = self.last_fusion
        payload["pending_asset"] = self.pending_asset
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
            "infrastructure": {
                "redis": self.store.backend,
                "outcomes": len(self.outcomes),
                "win_rate": round(self.outcomes.win_rate(), 4),
            },
        }

    def health(self) -> dict:
        brain_health = self.brain.last_health
        agent_status = self.agents.status_payload()
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
