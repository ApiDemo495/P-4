"""The continuous news sentiment engine (Section 4).

Runs three pollers on independent cadences and merges everything into one cache:

    CryptoPanic  every 30 s   (key required)
    NewsAPI      every 60 s   (key required)
    RSS feeds    every 60 s   (no key)

Polls update the cache but **never** change a locked signal.  The only path from
news to the signal panel is an emergency override, and that is handled by
``CriticalEventDetector`` + ``SignalLockController``.  Degradation is explicit:
with no keys at all the engine still runs on RSS, NIV/SMD keep working, and the
UI shows "News feed: Limited".
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

from backend.core import config as cfg
from backend.core.errors import ComponentStatus
from backend.data.ring_buffer import NewsCache, NewsItem

# A deterministic offline headline pack so the news bar, NIV/SMD and the
# emergency path are all exercisable without network access.
from backend.news import cryptopanic_source, newsapi_source, rss_source
from backend.news.critical_event_detector import CriticalEvent, CriticalEventDetector

log = logging.getLogger("drosophila.news")

SYNTHETIC_HEADLINES: tuple[tuple[str, str, float], ...] = (
    ("Bitcoin ETF sees $500M daily inflows as institutional demand accelerates", "Reuters", 0.72),
    ("Gold holds steady as traders weigh rate path and safe-haven demand", "Bloomberg", 0.15),
    ("BTC rallies past key resistance after options expiry, dealers reposition", "CoinDesk", 0.44),
    ("Analysts warn of thin liquidity into the weekend session", "The Block", -0.18),
    ("Tokenised gold volumes climb as investors seek inflation hedge", "CoinTelegraph", 0.36),
    ("Regulatory review of crypto spot markets continues, no timeline given", "Reuters", -0.22),
    ("Miner outflows tick higher, historically a short-term headwind", "Decrypt", -0.31),
    ("Stablecoin supply expands for a third week, dry powder building", "Blockworks", 0.41),
)


@dataclass
class NewsStatus:
    cryptopanic: str = "not_configured"
    newsapi: str = "not_configured"
    rss: str = "unknown"
    last_poll: float = 0.0
    items: int = 0
    niv: float = 0.0
    coverage: str = "limited"

    def to_dict(self) -> dict:
        return {
            "cryptopanic": self.cryptopanic,
            "newsapi": self.newsapi,
            "rss": self.rss,
            "last_poll": self.last_poll,
            "seconds_since_poll": round(time.time() - self.last_poll, 1) if self.last_poll else None,
            "items": self.items,
            "niv": round(self.niv, 4),
            "coverage": self.coverage,
        }


class NewsEngine:
    """Owns the cache, the pollers and the critical-event detector."""

    def __init__(self, settings=None, on_critical=None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.cache = NewsCache()
        self.detector = CriticalEventDetector(self.settings)
        self.status = NewsStatus()
        self.on_critical = on_critical  # async callback -> SignalLockController
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()
        self._client: httpx.AsyncClient | None = None
        self._last_cryptopanic = 0.0
        self._last_newsapi = 0.0
        self._last_rss = 0.0
        self._synthetic_index = 0
        self.last_error: str = ""

    # ------------------------------------------------------------------
    async def start(self) -> None:
        if not self.settings.news_enabled:
            log.info("News engine disabled by configuration")
            self.status.coverage = "disabled"
            return
        self._client = httpx.AsyncClient(
            headers={"accept": "application/json", "user-agent": "DrosophilaTrader/2.0"}
        )
        # Seed the cache immediately so the very first cycle has news.
        await self.poll_all(force=True)
        self._tasks.append(asyncio.create_task(self._runner(), name="news-engine"))

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    async def _runner(self) -> None:
        """Poll each provider on its own cadence (Section 4.4)."""
        while not self._stop.is_set():
            try:
                await self.poll_all()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                log.warning("news poll failed: %s", exc)
            await asyncio.sleep(max(5.0, self.settings.news_poll_seconds))

    async def poll_all(self, force: bool = False) -> dict:
        now = time.time()
        polled: dict[str, int] = {}

        # --- CryptoPanic: every 30 s -----------------------------------
        if self.settings.cryptopanic_key and (
            force or (now - self._last_cryptopanic) >= self.settings.news_poll_seconds
        ):
            polled["cryptopanic"] = await self._poll_cryptopanic()
            self._last_cryptopanic = now

        # --- NewsAPI: every 60 s ---------------------------------------
        if self.settings.newsapi_key and (
            force or (now - self._last_newsapi) >= self.settings.newsapi_poll_seconds
        ):
            polled["newsapi"] = await self._poll_newsapi()
            self._last_newsapi = now

        # --- RSS: every 60 s -------------------------------------------
        if force or (now - self._last_rss) >= self.settings.rss_poll_seconds:
            polled["rss"] = await self._poll_rss()
            self._last_rss = now

        # --- Last resort: keep the pipeline alive offline ---------------
        if self.cache.all() == () or (now - self.cache.last_poll) > 300:
            polled["synthetic"] = self._seed_synthetic()

        self._update_status()
        await self._scan_for_critical()
        return polled

    # ------------------------------------------------------------------
    async def _poll_cryptopanic(self) -> int:
        assert self._client is not None
        try:
            items = await cryptopanic_source.fetch(self._client, self.settings)
            self.cache.providers["cryptopanic"] = f"ok ({len(items)})"
            self.status.cryptopanic = "ok"
            return self.cache.add(items)
        except Exception as exc:  # noqa: BLE001
            self.status.cryptopanic = "error"
            self.cache.providers["cryptopanic"] = str(exc)
            log.info("CryptoPanic unavailable: %s", exc)
            return 0

    async def _poll_newsapi(self) -> int:
        assert self._client is not None
        try:
            items = await newsapi_source.fetch(self._client, self.settings)
            self.status.newsapi = "ok"
            self.cache.providers["newsapi"] = f"ok ({len(items)})"
            return self.cache.add(items)
        except Exception as exc:  # noqa: BLE001
            self.status.newsapi = "error"
            self.cache.providers["newsapi"] = str(exc)
            log.info("NewsAPI unavailable: %s", exc)
            return 0

    async def _poll_rss(self) -> int:
        try:
            items = await rss_source.fetch_all(self.settings)
            self.status.rss = "active" if items else "empty"
            self.cache.providers["rss"] = f"ok ({len(items)})"
            return self.cache.add(items)
        except Exception as exc:  # noqa: BLE001
            self.status.rss = "error"
            self.cache.providers["rss"] = str(exc)
            log.info("RSS feeds unavailable: %s", exc)
            return 0

    def _seed_synthetic(self) -> int:
        """Offline headline pack (also used to demo the emergency path)."""
        headline, source, sentiment = SYNTHETIC_HEADLINES[
            self._synthetic_index % len(SYNTHETIC_HEADLINES)
        ]
        self._synthetic_index += 1
        item = NewsItem(
            headline=headline,
            source=source,
            tier=1 if source in ("Reuters", "Bloomberg") else 2,
            sentiment=sentiment,
            published_at=time.time(),
            provider="OfflinePack",
        )
        return self.cache.add([item])

    # ------------------------------------------------------------------
    async def _scan_for_critical(self) -> None:
        """Run triggers 3 and 4 against the freshest items."""
        for item in self.cache.latest(8):
            if item.provider == "OfflinePack":
                continue
            event = self.detector.check_headline(item.headline, item.source, item.tier)
            if event is not None:
                await self._dispatch(event)
                return

        niv = self.current_niv()
        event = self.detector.check_sentiment_swing(niv)
        if event is not None:
            await self._dispatch(event)

    async def report_price_event(self, event: CriticalEvent) -> None:
        """Entry point for the price-based triggers (flash crash / spike)."""
        await self._dispatch(event)

    async def _dispatch(self, event: CriticalEvent) -> None:
        if self.on_critical is None:
            return
        try:
            await self.on_critical(event)
        except Exception as exc:  # noqa: BLE001
            log.warning("critical event dispatch failed: %s", exc)

    # ------------------------------------------------------------------
    def current_niv(self) -> float:
        """Quick NIV over the newest five items (mirrors Formula 19)."""
        items = self.cache.latest(5)
        if not items:
            return 0.0
        now = time.time()
        num = den = 0.0
        for item in items:
            weight = item.credibility * pow(2.718281828, -item.age_seconds(now) / 300.0)
            num += item.sentiment * weight
            den += weight
        return float(num / den) if den > 0 else 0.0

    def latest_items(self, limit: int = 8) -> list[dict]:
        return [item.to_dict() for item in self.cache.latest(limit)]

    def _update_status(self) -> None:
        self.status.last_poll = self.cache.last_poll
        self.status.items = len(self.cache.all())
        self.status.niv = self.current_niv()
        sources = sum(
            1
            for flag in (
                self.settings.cryptopanic_key,
                self.settings.newsapi_key,
                bool(self.settings.rss_feeds),
            )
            if flag
        )
        if self.status.items == 0:
            self.status.coverage = "unavailable"
        elif sources >= 3:
            self.status.coverage = "full"
        else:
            self.status.coverage = "limited"

    def degradation_ok(self) -> bool:
        """True when at least one provider produced items."""
        return self.status.items > 0

    def status_payload(self) -> ComponentStatus:
        healthy = self.degradation_ok()
        return ComponentStatus(
            name="news",
            healthy=healthy,
            detail={
                "full": "All sources active",
                "limited": "Limited (add API keys for full coverage)",
                "unavailable": "Unavailable - using price-only analysis",
                "disabled": "Disabled in settings",
            }.get(self.status.coverage, self.status.coverage),
            mode=self.status.coverage,
            extra=self.status.to_dict(),
        )
