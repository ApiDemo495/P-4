"""CryptoPanic - primary news source (Section 4.2, source 1).

    GET https://cryptopanic.com/api/v1/posts/?auth_token={KEY}
        &kind=news&filter=hot&currencies=BTC,PAXG

Polled every 30 seconds (120 calls/hour, inside the free tier's 5/minute).
Sentiment comes from the community votes rather than a lexicon.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import httpx

from backend.core import config as cfg
from backend.data.ring_buffer import NewsItem
from backend.news import sentiment_lexicon as lexicon

log = logging.getLogger("drosophila.news.cryptopanic")

API_URL = "https://cryptopanic.com/api/v1/posts/"
PROVIDER = "CryptoPanic"


async def fetch(client: httpx.AsyncClient, settings=None, limit: int = 20) -> list[NewsItem]:
    settings = settings or cfg.SETTINGS
    if not settings.cryptopanic_key:
        raise RuntimeError("no CryptoPanic API key configured")

    params = {
        "auth_token": settings.cryptopanic_key,
        "kind": "news",
        "filter": "hot",
        "currencies": "BTC,PAXG",
        "public": "true",
    }
    response = await client.get(API_URL, params=params, timeout=8.0)
    if response.status_code == 401:
        raise PermissionError("CryptoPanic rejected the API key")
    if response.status_code == 429:
        raise RuntimeError("CryptoPanic rate limit reached")
    response.raise_for_status()

    payload = response.json()
    items: list[NewsItem] = []
    for post in (payload.get("results") or [])[:limit]:
        title = str(post.get("title") or "").strip()
        if not title:
            continue
        votes = post.get("votes") or {}
        sentiment = lexicon.votes_to_sentiment(votes.get("positive", 0), votes.get("negative", 0))
        if sentiment == 0.0:
            # No community votes yet: fall back to the lexicon so a brand-new
            # critical headline is never scored as neutral.
            sentiment = lexicon.score_headline(title)
        source_name = str((post.get("source") or {}).get("title") or "CryptoPanic")
        items.append(
            NewsItem(
                headline=title,
                source=source_name,
                tier=lexicon.source_tier(source_name),
                sentiment=sentiment,
                published_at=_parse_time(post.get("published_at")),
                url=str(post.get("url") or ""),
                provider=PROVIDER,
            )
        )
    return items


def _parse_time(raw) -> float:
    if not raw:
        return time.time()
    try:
        text = str(raw).replace("Z", "+00:00")
        return datetime.fromisoformat(text).astimezone(timezone.utc).timestamp()
    except ValueError:
        return time.time()


async def test_key(key: str) -> dict:
    """Validate a key from the Settings screen without storing it yet."""
    if not key:
        return {"valid": False, "error": "No key entered"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                API_URL,
                params={"auth_token": key, "kind": "news", "public": "true"},
                timeout=8.0,
            )
        if response.status_code == 200:
            count = len((response.json() or {}).get("results") or [])
            return {"valid": True, "detail": f"{count} posts returned"}
        if response.status_code in (401, 403):
            return {"valid": False, "error": "Invalid CryptoPanic token"}
        if response.status_code == 429:
            return {"valid": True, "detail": "Rate limited, but the key was accepted"}
        return {"valid": False, "error": f"HTTP {response.status_code}"}
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "error": str(exc)}
