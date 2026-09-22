"""NewsAPI - secondary news source (Section 4.2, source 2).

    GET https://newsapi.org/v2/everything?q=bitcoin+OR+gold+crypto
        &sortBy=publishedAt&apiKey={KEY}

Polled every 60 seconds to stay inside the free tier (100 requests/day).
NewsAPI exposes no community votes, so headlines are scored with the lexicon.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import httpx

from backend.core import config as cfg
from backend.data.ring_buffer import NewsItem
from backend.news import sentiment_lexicon as lexicon

log = logging.getLogger("drosophila.news.newsapi")

API_URL = "https://newsapi.org/v2/everything"
PROVIDER = "NewsAPI"
QUERY = "bitcoin OR ethereum OR crypto OR gold OR paxg"


async def fetch(client: httpx.AsyncClient, settings=None, limit: int = 20) -> list[NewsItem]:
    settings = settings or cfg.SETTINGS
    if not settings.newsapi_key:
        raise RuntimeError("no NewsAPI key configured")

    params = {
        "q": QUERY,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": limit,
        "apiKey": settings.newsapi_key,
    }
    response = await client.get(API_URL, params=params, timeout=8.0)
    if response.status_code == 401:
        raise PermissionError("NewsAPI rejected the API key")
    if response.status_code == 429:
        raise RuntimeError("NewsAPI rate limit reached")
    response.raise_for_status()

    payload = response.json()
    if payload.get("status") == "error":
        raise RuntimeError(str(payload.get("message") or "NewsAPI error"))

    items: list[NewsItem] = []
    for article in (payload.get("articles") or [])[:limit]:
        title = str(article.get("title") or "").strip()
        if not title or title == "[Removed]":
            continue
        source_name = str((article.get("source") or {}).get("name") or "NewsAPI")
        items.append(
            NewsItem(
                headline=title,
                source=source_name,
                tier=lexicon.source_tier(source_name),
                sentiment=lexicon.score_headline(title),
                published_at=_parse_time(article.get("publishedAt")),
                url=str(article.get("url") or ""),
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
    if not key:
        return {"valid": False, "error": "No key entered"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                API_URL,
                params={"q": "bitcoin", "pageSize": 1, "apiKey": key},
                timeout=8.0,
            )
        if response.status_code == 200:
            return {"valid": True, "detail": "Key accepted"}
        if response.status_code == 401:
            return {"valid": False, "error": "Invalid NewsAPI key"}
        return {"valid": False, "error": f"HTTP {response.status_code}"}
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "error": str(exc)}
