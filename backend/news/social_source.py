"""Social media as a crowd sensor (Round AP).

What is wired, and honestly what is not
---------------------------------------
* **Reddit** - public JSON listings, no key: ``r/Bitcoin``, ``r/CryptoCurrency``,
  ``r/Gold`` and ``r/wallstreetbets`` (new + hot).  Each post becomes a tier-4
  item with its up-vote score and comment count as the buzz weight.
* **StockTwits** - public symbol streams, no key: ``BTC.X`` and ``GLD``.  Posts
  carry the author's own *Bullish* / *Bearish* label, which is a cleaner
  sentiment than any lexicon; it is used when present.
* **X (Twitter)** - the v2 *recent search* endpoint, **key required**
  (``X_BEARER_TOKEN``; X's free tier is write-only, the $100/month Basic tier
  is the first that can read).  Query: bitcoin / btc / gold / paxg, English,
  no retweets.  Without a token the row says "no token" and nothing else.
* **Instagram / Facebook** - Meta's Graph API offers no public search or
  hashtag firehose any more (deprecated 2018-2024; hashtag search needs an
  approved business app and returns only the business's own media).  There is
  no honest way to read them from here, so they are listed as *not available*
  rather than faked.

Everything social is capped: tier 4 credibility (0.2), relevance 0.6, and
the social share of the news-impact index is bounded in ``news_engine`` so
a Reddit thread never outranks a Tier-1 wire.
"""
from __future__ import annotations

import logging
import time

import httpx

from backend.core import config as cfg
from backend.data.ring_buffer import NewsItem
from backend.news import sentiment_lexicon as lexicon

log = logging.getLogger("drosophila.news.social")

PROVIDER = "Social"
USER_AGENT = "DrosophilaTrader/2.0 social sensor (+https://github.com/ApiDemo495/P-4)"
REDDIT_SUBS = ("Bitcoin", "CryptoCurrency", "Gold", "wallstreetbets")
STOCKTWITS_SYMBOLS = ("BTC.X", "GLD")
X_QUERY = "(bitcoin OR btc OR #bitcoin OR \"gold price\" OR paxg OR xau) lang:en -is:retweet"

#: the networks we cannot read, and why - surfaced in the UI as-is
UNAVAILABLE = {
    "instagram": "Meta Graph API has no public search / hashtag firehose (needs an approved business app; "
                 "returns only your own media)",
    "facebook": "same Graph API limits - public post search was removed in 2018",
}


class RateLimited(RuntimeError):
    pass


def _item(headline: str, source: str, sentiment: float, published_at: float, url: str,
          provider: str, buzz: float) -> NewsItem:
    return NewsItem(
        headline=headline.strip()[:200], source=source, tier=4,
        sentiment=max(-1.0, min(1.0, float(sentiment))), published_at=float(published_at),
        url=url, provider=provider, relevance=0.6, relevance_tag="social",
    )


# ----------------------------------------------------------------- Reddit
async def fetch_reddit(client: httpx.AsyncClient, limit_per_sub: int = 8) -> list[NewsItem]:
    items: list[NewsItem] = []
    for sub in REDDIT_SUBS:
        try:
            r = await client.get(f"https://www.reddit.com/r/{sub}/new.json", params={"limit": limit_per_sub},
                                 headers={"User-Agent": USER_AGENT}, timeout=8.0)
            if r.status_code == 429:
                raise RateLimited("reddit 429")
            r.raise_for_status()
            for child in ((r.json() or {}).get("data") or {}).get("children") or []:
                d = child.get("data") or {}
                title = str(d.get("title") or "")
                if not title:
                    continue
                buzz = float(d.get("score") or 0) + 2.0 * float(d.get("num_comments") or 0)
                items.append(_item(title, f"r/{sub}", lexicon.score_headline(title),
                                   float(d.get("created_utc") or time.time()),
                                   "https://www.reddit.com" + str(d.get("permalink") or ""),
                                   f"{PROVIDER}:reddit", buzz))
        except RateLimited:
            raise
        except Exception as exc:  # noqa: BLE001 - one subreddit down is not the sensor down
            log.debug("reddit r/%s failed: %s", sub, exc)
    return items


# ------------------------------------------------------------- StockTwits
async def fetch_stocktwits(client: httpx.AsyncClient, limit: int = 15) -> list[NewsItem]:
    items: list[NewsItem] = []
    for symbol in STOCKTWITS_SYMBOLS:
        try:
            r = await client.get(f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json",
                                 headers={"User-Agent": USER_AGENT}, timeout=8.0)
            if r.status_code == 429:
                raise RateLimited("stocktwits 429")
            r.raise_for_status()
            for msg in ((r.json() or {}).get("messages") or [])[:limit]:
                body = str(msg.get("body") or "")
                if not body:
                    continue
                label = (((msg.get("entities") or {}).get("sentiment") or {}).get("basic") or "").lower()
                sentiment = 0.6 if label == "bullish" else -0.6 if label == "bearish" else lexicon.score_headline(body)
                created = msg.get("created_at") or ""
                try:
                    published = time.mktime(time.strptime(created, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
                except Exception:  # noqa: BLE001
                    published = time.time()
                user = (msg.get("user") or {}).get("username") or "stocktwits"
                items.append(_item(body, f"stocktwits ${symbol} @{user}", sentiment, published,
                                   f"https://stocktwits.com/symbol/{symbol}", f"{PROVIDER}:stocktwits",
                                   float((msg.get("likes") or {}).get("total") or 0)))
        except RateLimited:
            raise
        except Exception as exc:  # noqa: BLE001
            log.debug("stocktwits %s failed: %s", symbol, exc)
    return items


# ---------------------------------------------------------------------- X
async def fetch_x(client: httpx.AsyncClient, token: str, limit: int = 25) -> list[NewsItem]:
    if not token:
        return []
    r = await client.get("https://api.twitter.com/2/tweets/search/recent",
                         params={"query": X_QUERY, "max_results": max(10, min(limit, 100)),
                                 "tweet.fields": "created_at,public_metrics,lang"},
                         headers={"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}, timeout=8.0)
    if r.status_code in (401, 403):
        raise PermissionError("X rejected the bearer token (reading needs the paid Basic tier)")
    if r.status_code == 429:
        raise RateLimited("X 429")
    r.raise_for_status()
    items: list[NewsItem] = []
    for tw in (r.json() or {}).get("data") or []:
        text = str(tw.get("text") or "")
        if not text:
            continue
        created = tw.get("created_at") or ""
        try:
            published = time.mktime(time.strptime(created[:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
        except Exception:  # noqa: BLE001
            published = time.time()
        pm = tw.get("public_metrics") or {}
        buzz = float(pm.get("like_count") or 0) + 3.0 * float(pm.get("retweet_count") or 0)
        items.append(_item(text, "x.com", lexicon.score_headline(text), published,
                           f"https://x.com/i/web/status/{tw.get('id')}", f"{PROVIDER}:x", buzz))
    return items


async def fetch(client: httpx.AsyncClient, settings=None) -> tuple[list[NewsItem], dict]:
    """All social sensors at once; returns ``(items, per_network_status)``."""
    settings = settings or cfg.SETTINGS
    status: dict[str, str] = {}
    items: list[NewsItem] = []
    for name, coro in (("reddit", fetch_reddit(client)), ("stocktwits", fetch_stocktwits(client))):
        try:
            got = await coro
            items.extend(got)
            status[name] = f"ok ({len(got)})" if got else "reachable, nothing new"
        except RateLimited as exc:
            status[name] = f"rate limited ({exc})"
        except Exception as exc:  # noqa: BLE001
            status[name] = f"error: {exc}"
    token = str(getattr(settings, "x_bearer_token", "") or "")
    if token:
        try:
            got = await fetch_x(client, token)
            items.extend(got)
            status["x"] = f"ok ({len(got)})"
        except Exception as exc:  # noqa: BLE001
            status["x"] = f"error: {exc}"
    else:
        status["x"] = "no token (X_BEARER_TOKEN) - X's read API is paid; add a Basic-tier bearer token on /settings"
    for name, why in UNAVAILABLE.items():
        status[name] = f"not available: {why}"
    return items, status


async def test_key(token: str) -> dict:
    """The /settings test button for the X bearer token."""
    if not token:
        return {"valid": False, "error": "No token entered"}
    try:
        async with httpx.AsyncClient() as client:
            got = await fetch_x(client, token, limit=10)
        return {"valid": True, "detail": f"X recent search answered with {len(got)} posts"}
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "error": str(exc)}
