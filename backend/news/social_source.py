"""Social media as a crowd sensor (Round AP, rebuilt in Round BP).

What is wired - all keyless unless stated, all fetched **concurrently** with
one deadline, each network with its own back-off so one rate limit never
stalls the others:

* **Reddit** - public JSON, ``new`` + ``rising`` of r/Bitcoin, r/BitcoinMarkets,
  r/CryptoCurrency, r/CryptoMarkets, r/Gold, r/wallstreetbets, plus a
  site-wide ``search.json`` for *bitcoin OR btc OR gold price* sorted by new.
  Up-votes and comments are the engagement weight.
* **StockTwits** - public symbol streams ``BTC.X`` and ``GLD`` and the
  *trending* stream; the author's own Bullish / Bearish label is used when
  present (cleaner than any lexicon).
* **Bluesky** - the public AppView ``searchPosts`` endpoint (latest), no key.
* **Mastodon** - public hashtag timelines on mastodon.social and
  fosstodon.org (#bitcoin, #btc, #gold).
* **Hacker News** - Algolia search by date, bitcoin / crypto / gold stories.
* **Fear & Greed** - alternative.me's crypto index, one number for the whole
  crowd's mood, read as sentiment ``(value - 50) / 50``.
* **X (Twitter)** - v2 recent search, **key required** (``X_BEARER_TOKEN``;
  reading needs the paid Basic tier).  Without a token the row says so.
* **Instagram / Facebook** - no public search / hashtag firehose any more;
  listed as *not available* rather than faked.

Efficiency
----------
* every network runs in its own task under ``asyncio.gather`` with a single
  ``DEADLINE_S`` budget - the sensor returns in ~2 s instead of the old
  sequential 20–40 s worst case;
* a network that answers 429 / 5xx backs off exponentially (90 s → 20 min)
  and is skipped, not retried, until its back-off ends;
* posts are de-duplicated by URL and by normalised text across networks;
* **engagement-weighted sentiment** and a **pulse** (posts in the last
  10 min against the trailing hour's rate) are returned beside the items,
  so the dashboard and the crowd layer see *how loud* the crowd is, not just
  what it says.

Everything social is capped: tier 4 credibility (0.2), relevance 0.6, and
the social share of the news-impact index is bounded in ``news_engine`` so
a Reddit thread never outranks a Tier-1 wire.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import time

import httpx

from backend.core import config as cfg
from backend.data.ring_buffer import NewsItem
from backend.news import sentiment_lexicon as lexicon

log = logging.getLogger("drosophila.news.social")

PROVIDER = "Social"
USER_AGENT = "DrosophilaTrader/2.0 social sensor (+https://github.com/ApiDemo495/P-4)"
DEADLINE_S = 9.0
REDDIT_SUBS = ("Bitcoin", "BitcoinMarkets", "CryptoCurrency", "CryptoMarkets", "Gold", "wallstreetbets")
REDDIT_SEARCH = "bitcoin OR btc OR \"gold price\" OR xau OR paxg"
STOCKTWITS_SYMBOLS = ("BTC.X", "GLD")
BLUESKY_QUERIES = ("bitcoin", "btc price", "gold price")
MASTODON_HOSTS = ("mastodon.social", "fosstodon.org")
MASTODON_TAGS = ("bitcoin", "btc", "gold")
HN_QUERY = "bitcoin OR btc OR crypto OR \"gold price\""
X_QUERY = "(bitcoin OR btc OR #bitcoin OR \"gold price\" OR paxg OR xau) lang:en -is:retweet"
BACKOFF_MIN_S, BACKOFF_MAX_S = 90.0, 1200.0
KEYWORDS = re.compile(r"\b(bitcoin|btc|crypto|gold|xau|paxg|satoshi|sats|halving|etf|fed|rate|inflation|war|tariff)\b", re.I)

#: the networks we cannot read, and why - surfaced in the UI as-is
UNAVAILABLE = {
    "instagram": "Meta Graph API has no public search / hashtag firehose (needs an approved business app; "
                 "returns only your own media)",
    "facebook": "same Graph API limits - public post search was removed in 2018",
}

#: per-network back-off state: name -> (until, seconds)
_backoff: dict[str, tuple[float, float]] = {}
#: recent social posts (published_at) for the pulse, across polls
_pulse_times: list[float] = []


class RateLimited(RuntimeError):
    pass


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").replace("&amp;", "&").replace("&#39;", "'").replace("&quot;", '"')


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()[:120]


def _parse_iso(created: str) -> float:
    try:
        return time.mktime(time.strptime(created[:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
    except Exception:  # noqa: BLE001
        return time.time()


def _item(headline: str, source: str, sentiment: float, published_at: float, url: str,
          provider: str, buzz: float) -> NewsItem:
    # engagement nudges relevance inside the social band [0.5, 0.75]: a post
    # a thousand people pushed up is more "the crowd" than one nobody saw
    rel = 0.5 + 0.25 * min(1.0, math.log1p(max(buzz, 0.0)) / math.log1p(1000.0))
    return NewsItem(
        headline=_strip_html(headline).strip()[:200], source=source, tier=4,
        sentiment=max(-1.0, min(1.0, float(sentiment))), published_at=float(published_at),
        url=url, provider=provider, relevance=round(rel, 3), relevance_tag="social",
    )


def _relevant(text: str) -> bool:
    return bool(KEYWORDS.search(text or ""))


async def _get(client: httpx.AsyncClient, url: str, **kw) -> httpx.Response:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    headers.update(kw.pop("headers", {}) or {})
    r = await client.get(url, headers=headers, timeout=kw.pop("timeout", 6.0), **kw)
    if r.status_code == 429 or r.status_code >= 500:
        raise RateLimited(f"{r.status_code}")
    r.raise_for_status()
    return r


def _collect(results: list, network: str) -> list[NewsItem]:
    """Merge one network's parallel calls.  A call that failed is not the
    network down - but when *every* call failed the network is reported as
    down (rate limited if any call was), never as "nothing new"."""
    items: list[NewsItem] = []
    limited = 0
    errors: list[BaseException] = []
    for res in results:
        if isinstance(res, RateLimited):
            limited += 1
            errors.append(res)
        elif isinstance(res, BaseException):
            errors.append(res)
            log.debug("%s call failed: %s", network, res)
        else:
            items.extend(res)
    if not items and errors:
        if limited:
            raise RateLimited(f"{network} {errors[0]}")
        raise RuntimeError(str(errors[0]) or type(errors[0]).__name__)
    return items


# ----------------------------------------------------------------- Reddit
async def fetch_reddit(client: httpx.AsyncClient, limit_per_sub: int = 10) -> list[NewsItem]:
    async def one(url: str, params: dict, label: str) -> list[NewsItem]:
        out: list[NewsItem] = []
        r = await _get(client, url, params=params)
        for child in ((r.json() or {}).get("data") or {}).get("children") or []:
            d = child.get("data") or {}
            title = str(d.get("title") or "")
            if not title:
                continue
            buzz = float(d.get("score") or 0) + 2.0 * float(d.get("num_comments") or 0)
            sub = str(d.get("subreddit") or label)
            out.append(_item(title, f"r/{sub}", lexicon.score_headline(title),
                             float(d.get("created_utc") or time.time()),
                             "https://www.reddit.com" + str(d.get("permalink") or ""), f"{PROVIDER}:reddit", buzz))
        return out
    jobs = []
    for sub in REDDIT_SUBS:
        jobs.append(one(f"https://www.reddit.com/r/{sub}/new.json", {"limit": limit_per_sub}, sub))
        jobs.append(one(f"https://www.reddit.com/r/{sub}/rising.json", {"limit": 6}, sub))
    jobs.append(one("https://www.reddit.com/search.json", {"q": REDDIT_SEARCH, "sort": "new", "limit": 15, "t": "hour"}, "search"))
    results = await asyncio.gather(*jobs, return_exceptions=True)
    return _collect(results, "reddit")


# ------------------------------------------------------------- StockTwits
async def fetch_stocktwits(client: httpx.AsyncClient, limit: int = 20) -> list[NewsItem]:
    def parse(payload: dict, symbol: str) -> list[NewsItem]:
        out: list[NewsItem] = []
        for msg in (payload.get("messages") or [])[:limit]:
            body = str(msg.get("body") or "")
            if not body or (symbol == "trending" and not _relevant(body)):
                continue
            label = (((msg.get("entities") or {}).get("sentiment") or {}).get("basic") or "").lower()
            sentiment = 0.6 if label == "bullish" else -0.6 if label == "bearish" else lexicon.score_headline(body)
            user = (msg.get("user") or {}).get("username") or "stocktwits"
            out.append(_item(body, f"stocktwits ${symbol} @{user}", sentiment, _parse_iso(str(msg.get("created_at") or "")),
                             f"https://stocktwits.com/symbol/{symbol}" if symbol != "trending" else "https://stocktwits.com/rankings/trending",
                             f"{PROVIDER}:stocktwits", float((msg.get("likes") or {}).get("total") or 0)))
        return out
    urls = [(f"https://api.stocktwits.com/api/2/streams/symbol/{s}.json", s) for s in STOCKTWITS_SYMBOLS]
    urls.append(("https://api.stocktwits.com/api/2/streams/trending.json", "trending"))
    results = await asyncio.gather(*[_get(client, u) for u, _ in urls], return_exceptions=True)
    parsed = [res if isinstance(res, BaseException) else parse(res.json() or {}, symbol) for (_, symbol), res in zip(urls, results)]
    return _collect(parsed, "stocktwits")


# ---------------------------------------------------------------- Bluesky
async def fetch_bluesky(client: httpx.AsyncClient, limit: int = 25) -> list[NewsItem]:
    async def one(q: str) -> list[NewsItem]:
        r = await _get(client, "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts",
                       params={"q": q, "sort": "latest", "limit": limit, "lang": "en"})
        out: list[NewsItem] = []
        for post in (r.json() or {}).get("posts") or []:
            rec = post.get("record") or {}
            text = str(rec.get("text") or "")
            if not text or not _relevant(text):
                continue
            handle = (post.get("author") or {}).get("handle") or "bsky"
            buzz = float(post.get("likeCount") or 0) + 3.0 * float(post.get("repostCount") or 0) + float(post.get("replyCount") or 0)
            uri = str(post.get("uri") or "")
            rkey = uri.rsplit("/", 1)[-1] if uri else ""
            out.append(_item(text, f"bsky @{handle}", lexicon.score_headline(text), _parse_iso(str(rec.get("createdAt") or "")),
                             f"https://bsky.app/profile/{handle}/post/{rkey}" if rkey else "https://bsky.app", f"{PROVIDER}:bluesky", buzz))
        return out
    results = await asyncio.gather(*[one(q) for q in BLUESKY_QUERIES], return_exceptions=True)
    return _collect(results, "bluesky")


# --------------------------------------------------------------- Mastodon
async def fetch_mastodon(client: httpx.AsyncClient, limit: int = 20) -> list[NewsItem]:
    async def one(host: str, tag: str) -> list[NewsItem]:
        r = await _get(client, f"https://{host}/api/v1/timelines/tag/{tag}", params={"limit": limit})
        out: list[NewsItem] = []
        for st in r.json() or []:
            text = _strip_html(str(st.get("content") or ""))
            if not text.strip():
                continue
            acct = (st.get("account") or {}).get("acct") or host
            buzz = float(st.get("favourites_count") or 0) + 3.0 * float(st.get("reblogs_count") or 0) + float(st.get("replies_count") or 0)
            out.append(_item(text, f"{host} @{acct} #{tag}", lexicon.score_headline(text), _parse_iso(str(st.get("created_at") or "")),
                             str(st.get("url") or f"https://{host}/tags/{tag}"), f"{PROVIDER}:mastodon", buzz))
        return out
    results = await asyncio.gather(*[one(h, t) for h in MASTODON_HOSTS for t in MASTODON_TAGS], return_exceptions=True)
    return _collect(results, "mastodon")


# ------------------------------------------------------------ Hacker News
async def fetch_hackernews(client: httpx.AsyncClient, limit: int = 20) -> list[NewsItem]:
    r = await _get(client, "https://hn.algolia.com/api/v1/search_by_date",
                   params={"query": HN_QUERY, "tags": "story", "hitsPerPage": limit})
    items: list[NewsItem] = []
    for hit in (r.json() or {}).get("hits") or []:
        title = str(hit.get("title") or "")
        if not title or not _relevant(title):
            continue
        buzz = float(hit.get("points") or 0) + 2.0 * float(hit.get("num_comments") or 0)
        items.append(_item(title, "news.ycombinator.com", lexicon.score_headline(title), float(hit.get("created_at_i") or time.time()),
                           f"https://news.ycombinator.com/item?id={hit.get('objectID')}", f"{PROVIDER}:hackernews", buzz))
    return items


# ----------------------------------------------------------- Fear & Greed
async def fetch_fear_greed(client: httpx.AsyncClient) -> list[NewsItem]:
    r = await _get(client, "https://api.alternative.me/fng/", params={"limit": 2})
    rows = (r.json() or {}).get("data") or []
    if not rows:
        return []
    cur = rows[0]
    value = float(cur.get("value") or 50)
    prev = float(rows[1].get("value") or value) if len(rows) > 1 else value
    label = str(cur.get("value_classification") or "")
    sentiment = (value - 50.0) / 50.0
    head = f"Crypto Fear & Greed index {value:.0f} ({label}), {value - prev:+.0f} vs yesterday"
    return [_item(head, "alternative.me fear & greed", sentiment, float(cur.get("timestamp") or time.time()),
                  "https://alternative.me/crypto/fear-and-greed-index/", f"{PROVIDER}:feargreed", 1000.0)]


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
        pm = tw.get("public_metrics") or {}
        buzz = float(pm.get("like_count") or 0) + 3.0 * float(pm.get("retweet_count") or 0)
        items.append(_item(text, "x.com", lexicon.score_headline(text), _parse_iso(str(tw.get("created_at") or "")),
                           f"https://x.com/i/web/status/{tw.get('id')}", f"{PROVIDER}:x", buzz))
    return items


# ------------------------------------------------------------- the sensor
def _dedupe(items: list[NewsItem]) -> list[NewsItem]:
    seen_url: set[str] = set()
    seen_text: set[str] = set()
    out: list[NewsItem] = []
    for it in items:
        key_t = _norm(it.headline)
        if (it.url and it.url in seen_url) or (key_t and key_t in seen_text):
            continue
        if it.url:
            seen_url.add(it.url)
        if key_t:
            seen_text.add(key_t)
        out.append(it)
    return out


def pulse(items: list[NewsItem], now: float | None = None) -> dict:
    """How loud the crowd is: posts in the last 10 min against the trailing
    hour's per-10-min rate, engagement-weighted sentiment, loudest network."""
    now = float(now if now is not None else time.time())
    global _pulse_times
    _pulse_times.extend(float(i.published_at) for i in items if now - float(i.published_at) < 3600.0)
    _pulse_times = sorted(set(t for t in _pulse_times if now - t < 3600.0))[-2000:]
    last10 = sum(1 for t in _pulse_times if now - t < 600.0)
    hour = len(_pulse_times)
    base = max(1.0, hour / 6.0)
    velocity = last10 / base
    if items:
        w = [max(0.1, float(i.relevance) - 0.5 + 0.1) for i in items]
        wsent = float(sum(wi * float(i.sentiment) for wi, i in zip(w, items)) / max(1e-9, sum(w)))
    else:
        wsent = 0.0
    by_net: dict[str, int] = {}
    for i in items:
        by_net[i.provider.split(":")[-1]] = by_net.get(i.provider.split(":")[-1], 0) + 1
    loudest = max(by_net, key=by_net.get) if by_net else ""
    return {"posts_10m": int(last10), "posts_1h": int(hour), "velocity": round(velocity, 3),
            "weighted_sentiment": round(wsent, 4), "momentum": round(wsent * min(3.0, velocity), 4),
            "loudest": loudest, "by_network": by_net,
            "reads": (f"{last10} posts in the last 10 min vs {base:.1f}/10 min over the hour = x{velocity:.2f}; "
                      f"engagement-weighted tone {wsent:+.2f}; loudest {loudest or '-'}")}


async def fetch(client: httpx.AsyncClient, settings=None) -> tuple[list[NewsItem], dict]:
    """All social sensors at once, one deadline; returns ``(items, status)``
    where ``status`` has a line per network plus ``_pulse`` and ``_ms``."""
    settings = settings or cfg.SETTINGS
    status: dict = {}
    now = time.time()
    token = str(getattr(settings, "x_bearer_token", "") or "")
    networks: list[tuple[str, object]] = [
        ("reddit", fetch_reddit(client)), ("stocktwits", fetch_stocktwits(client)), ("bluesky", fetch_bluesky(client)),
        ("mastodon", fetch_mastodon(client)), ("hackernews", fetch_hackernews(client)), ("feargreed", fetch_fear_greed(client)),
    ]
    if token:
        networks.append(("x", fetch_x(client, token)))
    live: list[tuple[str, object]] = []
    for name, coro in networks:
        until, _ = _backoff.get(name, (0.0, 0.0))
        if until > now:
            status[name] = f"backing off {until - now:.0f} s after a rate limit"
            coro.close()  # type: ignore[union-attr]
            continue
        live.append((name, coro))
    started = time.perf_counter()
    results = await asyncio.gather(*[asyncio.wait_for(c, timeout=DEADLINE_S) for _, c in live], return_exceptions=True)  # type: ignore[arg-type]
    items: list[NewsItem] = []
    for (name, _), res in zip(live, results):
        if isinstance(res, RateLimited):
            _, prev = _backoff.get(name, (0.0, 0.0))
            wait = min(BACKOFF_MAX_S, max(BACKOFF_MIN_S, prev * 2.0))
            _backoff[name] = (now + wait, wait)
            status[name] = f"rate limited ({res}) - backing off {wait:.0f} s"
        elif isinstance(res, asyncio.TimeoutError):
            status[name] = f"timed out after {DEADLINE_S:.0f} s"
        elif isinstance(res, BaseException):
            status[name] = f"error: {res}"
        else:
            _backoff.pop(name, None)
            items.extend(res)
            status[name] = f"ok ({len(res)})" if res else "reachable, nothing new"
    if not token:
        status["x"] = "no token (X_BEARER_TOKEN) - X's read API is paid; add a Basic-tier bearer token on /settings"
    for name, why in UNAVAILABLE.items():
        status[name] = f"not available: {why}"
    items = _dedupe(items)
    status["_pulse"] = pulse(items, now)
    status["_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
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
