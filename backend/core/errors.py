"""Shared error/status types.

Every subsystem in DROSOPHILA TRADER v2.0 must be able to describe its own
health, because Section 13.2 defines six explicit degradation levels and the
UI renders the current one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DegradationLevel(int, Enum):
    """Section 13.2 - graceful degradation hierarchy."""

    FULL = 1  # everything connected: 22 formulas, 4 agents, live brain, news
    NO_NEWS = 2  # NIV/SMD output 0, price-only emergency detection
    NO_AGENTS = 3  # CCSv2 is the sole decision maker
    FALLBACK_BRAIN = 4  # CCSv2 runs on the committed CSV matrix
    COINGECKO = 5  # reduced tick rate: DGW/LCS/BAR disabled
    MINIMAL = 6  # price only: the direction is kept but flagged as weak

    @property
    def label(self) -> str:
        return {
            1: "Full functionality",
            2: "No news APIs - price-only analysis",
            3: "No AI agents - Drosophila brain only",
            4: "Fallback brain matrix",
            5: "CoinGecko fallback - order book disabled",
            6: "Minimal mode - insufficient data",
        }[int(self)]


@dataclass
class ComponentStatus:
    name: str
    healthy: bool
    detail: str = ""
    mode: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            "name": self.name,
            "healthy": self.healthy,
            "detail": self.detail,
            "mode": self.mode,
        }
        if self.extra:
            payload["extra"] = self.extra
        return payload


class TraderError(Exception):
    """Base class for recoverable subsystem errors."""


class DataUnavailable(TraderError):
    """Raised when no market data source can produce a usable snapshot."""


class BrainUnavailable(TraderError):
    """Raised when even the committed fallback matrix cannot be loaded."""


class SignalNotReady(TraderError):
    """Raised by SignalLockController.get_current() while COMPUTING."""
