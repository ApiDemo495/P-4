"""RSS feeds - tertiary source and no-key backup (Section 4.2, source 3).

Defaults: CoinDesk, CoinTelegraph and The Block.  No API key, no rate limit, so
this is what keeps NIV/SMD alive when both paid providers are unconfigured.
"""

from __future__ import annotations

import asyncio
import calendar
import logging
import time

import httpx

from backend.core import config as cfg
from backend.data.ring_buffer import NewsItem
from backend.news import sentiment_lexicon as lexicon

log = logging.getLogger("drosophila.news.rss")

PROVIDER = "RSS"
USER_AGENT = "DrosophilaTrader/2.0 (+https://github.com/ApiDemo495/P-4)"


async def fetch_all(settings=None, per_feed: int = 10) -> list[NewsItem]:
    settings = settings or cfg.SETTINGS
    feeds = settings.rss_feeds
    if not feeds:
        return []

    results = await asyncio.gather(
        *(fetch_feed(url, per_feed) for url in feeds), return_exceptions=True
    )
    items: list[NewsItem] = []
    for url, result in zip(feeds, results):
        if isinstance(result, Exception):
            log.debug("RSS feed failed (%s): %s", url, result)
            continue
        items.extend(result)
    return items


async def fetch_feed(url: str, limit: int = 10) -> list[NewsItem]:
    import feedparser

    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        response = await client.get(url, timeout=8.0)
        response.raise_for_status()
        body = response.content

    parsed = feedparser.parse(body)
    feed_title = str(parsed.feed.get("title") or _host(url))
    items: list[NewsItem] = []
    for entry in parsed.entries[:limit]:
        title = str(entry.get("title") or "").strip()
        if not title:
            continue
        items.append(
            NewsItem(
                headline=title,
                source=feed_title,
                tier=lexicon.source_tier(feed_title),
                sentiment=lexicon.score_headline(title),
                published_at=_entry_time(entry),
                url=str(entry.get("link") or ""),
                provider=f"{PROVIDER}:{_host(url)}",
            )
        )
    return items


def _entry_time(entry) -> float:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            try:
                return float(calendar.timegm(parsed))
            except (TypeError, ValueError):
                continue
    return time.time()


def _host(url: str) -> str:
    return url.split("//")[-1].split("/")[0] or url


async def check_feeds(urls: list[str]) -> list[dict]:
    """Health of each configured feed, for the Settings screen."""
    async def one(url: str) -> dict:
        try:
            items = await fetch_feed(url, limit=1)
            return {"url": url, "ok": True, "items": len(items)}
        except Exception as exc:  # noqa: BLE001
            return {"url": url, "ok": False, "error": str(exc)}

    return list(await asyncio.gather(*(one(u) for u in urls)))
