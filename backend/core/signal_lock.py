"""The Signal Lock Protocol (Section 2).

v1.0 recomputed the signal every 15 seconds, so a user could start entering a BUY
at t=0 and watch it flip to SELL at t=15.  v2.0 makes that structurally
impossible:

     COMPUTING --lock()--> LOCKED --emergency_override()--> EMERGENCY_OVERRIDE
         ^                                                          |
         +---------------------- new_cycle() ---------------------+

``lock()`` can be called **at most once per cycle**.  A second call is a no-op
that returns the existing frozen signal, which means a late or duplicate
computation can never rewrite what the user is looking at.

The single exception is a Critical News Impact Event, and even then the override
is always to HOLD - never to BUY or SELL.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, NamedTuple

from backend.core.errors import SignalNotReady

log = logging.getLogger("drosophila.signal_lock")


class LockState(str, Enum):
    COMPUTING = "COMPUTING"
    LOCKED = "LOCKED"
    EMERGENCY_OVERRIDE = "EMERGENCY_OVERRIDE"

    @property
    def icon(self) -> str:
        return {"COMPUTING": "\u23f3", "LOCKED": "\U0001f512", "EMERGENCY_OVERRIDE": "\u26a1"}[
            self.value
        ]


class FrozenSignal(NamedTuple):
    """Immutable signal for one cycle.  A NamedTuple cannot be mutated."""

    cycle_number: int
    timestamp: str
    asset: str
    signal: str  # "BUY" | "SELL" | "HOLD"
    confidence: float
    reasoning: str
    formula_values: tuple  # ((name, value), ...)
    agent_results: tuple  # ((agent, {...}), ...)
    is_emergency_override: bool

    # -- v2.0 UI extras -------------------------------------------------
    ccs_value: float = 0.0
    ccs_confidence: float = 0.0
    hold_lean: str | None = None
    hedge: tuple = ()
    news: tuple = ()
    drg: float = 0.0
    brain_status: str = ""
    degradation_level: int = 1
    warnings: tuple = ()
    price: float = 0.0
    total_ms: float = 0.0
    emergency_headline: str = ""
    lock_state: str = LockState.LOCKED.value
    superseded_by: str | None = None
    """The original directional signal, if an emergency override replaced it."""

    # ------------------------------------------------------------------
    def formula_dict(self) -> dict[str, float]:
        return {name: value for name, value in self.formula_values}

    def agent_dict(self) -> dict[str, Any]:
        return {name: payload for name, payload in self.agent_results}

    def to_dict(self) -> dict:
        return {
            "cycle_number": self.cycle_number,
            "timestamp": self.timestamp,
            "asset": self.asset,
            "signal": self.signal,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "is_locked": True,
            "is_emergency_override": self.is_emergency_override,
            "lock_state": self.lock_state,
            "lock_icon": LockState(self.lock_state).icon,
            "formulas": {k: round(v, 6) for k, v in self.formula_values},
            "agents": self.agent_dict(),
            "hedge": dict(self.hedge),
            "news": dict(self.news),
            "hold_lean": self.hold_lean,
            "drg": round(self.drg, 4),
            "ccs_value": round(self.ccs_value, 4),
            "ccs_confidence": round(self.ccs_confidence, 4),
            "brain_status": self.brain_status,
            "degradation_level": self.degradation_level,
            "warnings": list(self.warnings),
            "price": round(self.price, 4),
            "total_ms": round(self.total_ms, 3),
            "emergency_headline": self.emergency_headline,
            "superseded_by": self.superseded_by,
        }


class SignalLockController:
    """Manages the immutable signal for each 60-second cycle."""

    def __init__(self) -> None:
        self.state: LockState = LockState.COMPUTING
        self.current_signal: FrozenSignal | None = None
        self.cycle_number: int = 0
        self.lock_timestamp: float = 0.0
        self.computed_at: float = 0.0
        self.emergency_until: float = 0.0
        self.emergency_event: dict | None = None
        self.history: list[FrozenSignal] = []

    # ------------------------------------------------------------------
    # Cycle control
    # ------------------------------------------------------------------
    def new_cycle(self, cycle_number: int) -> None:
        """Reset to COMPUTING at t=0.  Clears the previous signal entirely."""
        self.cycle_number = cycle_number
        self.current_signal = None
        self.state = LockState.COMPUTING
        self.lock_timestamp = 0.0
        self.computed_at = 0.0

    def lock(self, signal: FrozenSignal) -> FrozenSignal:
        """Freeze the signal for this cycle - callable once per cycle."""
        if self.state in (LockState.LOCKED, LockState.EMERGENCY_OVERRIDE) and self.current_signal:
            log.warning(
                "lock() ignored: cycle %d is already %s", self.cycle_number, self.state.value
            )
            return self.current_signal

        frozen = signal._replace(lock_state=LockState.LOCKED.value)
        self.current_signal = frozen
        self.state = LockState.LOCKED
        self.lock_timestamp = time.time()
        self.computed_at = self.lock_timestamp
        self.history.append(frozen)
        if len(self.history) > 240:
            del self.history[:-240]
        log.info(
            "cycle %d signal LOCKED: %s (%.0f%%)",
            frozen.cycle_number,
            frozen.signal,
            frozen.confidence * 100,
        )
        return frozen

    # ------------------------------------------------------------------
    # Emergency override
    # ------------------------------------------------------------------
    def emergency_override(
        self,
        event: dict,
        duration_seconds: float = 180.0,
        timestamp: str | None = None,
    ) -> FrozenSignal:
        """Break the lock for a Critical News Impact Event -> forced HOLD."""
        base = self.current_signal
        previous_signal = base.signal if base else "COMPUTING"
        headline = str(event.get("headline") or "Critical news event")
        now = time.time()
        self.emergency_until = max(self.emergency_until, now + duration_seconds)
        self.emergency_event = {**event, "remaining_seconds": self.emergency_until - now}

        overridden = FrozenSignal(
            cycle_number=self.cycle_number,
            timestamp=timestamp or _iso(now),
            asset=base.asset if base else "BTC",
            signal="HOLD",
            confidence=1.0,
            reasoning=(
                f"EMERGENCY OVERRIDE: {headline}. {event.get('reason', '')}. "
                "Signal overridden to HOLD - exit any open positions."
            ),
            formula_values=base.formula_values if base else (),
            agent_results=base.agent_results if base else (),
            is_emergency_override=True,
            ccs_value=base.ccs_value if base else 0.0,
            ccs_confidence=base.ccs_confidence if base else 0.0,
            hold_lean=previous_signal if previous_signal in ("BUY", "SELL") else None,
            hedge=base.hedge if base else (),
            news=base.news if base else (),
            drg=base.drg if base else 0.0,
            brain_status=base.brain_status if base else "",
            degradation_level=base.degradation_level if base else 1,
            warnings=base.warnings if base else (),
            price=base.price if base else 0.0,
            total_ms=base.total_ms if base else 0.0,
            emergency_headline=headline,
            lock_state=LockState.EMERGENCY_OVERRIDE.value,
            superseded_by=previous_signal,
        )
        self.current_signal = overridden
        self.state = LockState.EMERGENCY_OVERRIDE
        self.history.append(overridden)
        log.warning("EMERGENCY OVERRIDE on cycle %d: %s", self.cycle_number, headline)
        return overridden

    # ------------------------------------------------------------------
    def emergency_active(self) -> bool:
        return self.emergency_until > time.time()

    def emergency_remaining(self) -> float:
        return max(0.0, self.emergency_until - time.time())

    def clear_emergency(self) -> None:
        self.emergency_until = 0.0
        self.emergency_event = None
        if self.state is LockState.EMERGENCY_OVERRIDE:
            self.state = LockState.LOCKED

    # ------------------------------------------------------------------
    def get_current(self) -> FrozenSignal:
        """The locked signal.  Raises while the cycle is still COMPUTING."""
        if self.state is LockState.COMPUTING or self.current_signal is None:
            raise SignalNotReady(
                f"cycle {self.cycle_number} is still COMPUTING - no signal available yet"
            )
        return self.current_signal

    def try_get_current(self) -> FrozenSignal | None:
        return None if self.state is LockState.COMPUTING else self.current_signal

    def is_locked(self) -> bool:
        return self.state is not LockState.COMPUTING and self.current_signal is not None

    def status(self) -> dict:
        return {
            "state": self.state.value,
            "icon": self.state.icon,
            "cycle_number": self.cycle_number,
            "locked": self.is_locked(),
            "lock_timestamp": self.lock_timestamp,
            "seconds_since_lock": round(time.time() - self.lock_timestamp, 2)
            if self.lock_timestamp
            else None,
            "emergency_active": self.emergency_active(),
            "emergency_remaining": round(self.emergency_remaining(), 1),
            "emergency_event": self.emergency_event,
            "history_length": len(self.history),
        }

    def recent_history(self, limit: int = 20) -> list[dict]:
        return [s.to_dict() for s in self.history[-limit:]][::-1]


def _iso(ts: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
