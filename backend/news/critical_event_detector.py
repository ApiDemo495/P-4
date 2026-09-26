"""Critical News Event detection - the ONLY mechanism that breaks a signal lock.

Four independent triggers (Section 4.3):

1. flash crash   - an asset drops more than 5% inside 30 seconds
2. flash spike   - an asset rises more than 5% inside 30 seconds
3. extreme keyword + Tier 1/2 source ("hack", "exploit", "rug pull", "de-peg",
   "bank run", "emergency", "war", "sanction")
4. extreme sentiment swing - NIV flips from > +0.5 to < -0.5 (or vice versa)
   within a single 30-second poll cycle

When a trigger fires, the signal is overridden to **HOLD** - never to BUY or
SELL.  During an emergency the only safe action is to stop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from backend.core import config as cfg
from backend.news import sentiment_lexicon as lexicon

log = logging.getLogger("drosophila.news.critical")


@dataclass
class CriticalEvent:
    headline: str
    severity: str  # CRITICAL | HIGH
    reason: str
    asset: str = ""
    magnitude: float = 0.0
    detected_at: float = field(default_factory=time.time)
    source: str = ""
    tier: int = 4

    def to_dict(self) -> dict:
        return {
            "headline": self.headline,
            "severity": self.severity,
            "reason": self.reason,
            "asset": self.asset,
            "magnitude": round(self.magnitude, 4),
            "detected_at": self.detected_at,
            "source": self.source,
            "tier": self.tier,
        }


class CriticalEventDetector:
    """Stateful detector - holds the previous NIV so it can measure swings."""

    def __init__(self, settings=None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.previous_niv: float | None = None
        self.previous_poll_ts: float = 0.0
        self.recent_events: list[CriticalEvent] = []

    # ------------------------------------------------------------------
    # Trigger 4: extreme sentiment swing
    # ------------------------------------------------------------------
    def check_sentiment_swing(self, niv: float, now: float | None = None) -> CriticalEvent | None:
        now = now or time.time()
        previous = self.previous_niv
        self.previous_niv = niv
        self.previous_poll_ts = now

        if previous is None:
            return None
        if (previous > 0.5 and niv < -0.5) or (previous < -0.5 and niv > 0.5):
            direction = "bullish to bearish" if previous > 0 else "bearish to bullish"
            event = CriticalEvent(
                headline=f"Sentiment regime flip: {previous:+.2f} -> {niv:+.2f}",
                severity="HIGH",
                reason=f"Extreme sentiment swing ({direction}) in a single poll cycle",
                magnitude=abs(niv - previous),
            )
            self._record(event)
            return event
        return None

    # ------------------------------------------------------------------
    # Trigger 3: extreme keyword in a credible source
    # ------------------------------------------------------------------
    def check_headline(self, headline: str, source: str = "", tier: int | None = None) -> CriticalEvent | None:
        keywords = lexicon.critical_keywords_in(headline)
        if not keywords:
            return None
        resolved_tier = lexicon.source_tier(source) if tier is None else int(tier)
        if resolved_tier > 2:
            return None  # only Tier 1 / Tier 2 headlines can break the lock
        event = CriticalEvent(
            headline=headline,
            severity="CRITICAL",
            reason=f"Extreme keyword ({', '.join(keywords)}) from a Tier {resolved_tier} source",
            source=source,
            tier=resolved_tier,
        )
        self._record(event)
        return event

    # ------------------------------------------------------------------
    # Triggers 1 & 2: price-based flash moves
    # ------------------------------------------------------------------
    def check_flash_move(self, asset: str, pct_move: float, direction: float = 0.0) -> CriticalEvent | None:
        threshold = self.settings.flash_move_pct
        if abs(pct_move) < threshold:
            return None
        move = "crash" if (direction or pct_move) < 0 else "spike"
        event = CriticalEvent(
            headline=f"{asset} flash {move}: {pct_move:+.2f}% in under "
            f"{self.settings.flash_move_window_seconds:.0f}s",
            severity="CRITICAL",
            reason=f"Price-based detection: |{pct_move:.2f}%| >= {threshold:.1f}% inside "
            f"{self.settings.flash_move_window_seconds:.0f}s",
            asset=asset,
            magnitude=abs(pct_move),
        )
        self._record(event)
        return event

    def _record(self, event: CriticalEvent) -> None:
        self.recent_events.append(event)
        if len(self.recent_events) > 50:
            del self.recent_events[:-50]
        log.warning("CRITICAL EVENT [%s]: %s (%s)", event.severity, event.headline, event.reason)

    def history(self, limit: int = 20) -> list[dict]:
        return [e.to_dict() for e in self.recent_events[-limit:]][::-1]
