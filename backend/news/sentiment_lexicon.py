"""Lexicon-based headline sentiment (Section 4.2, sources 2 and 3).

CryptoPanic gives community votes, so it needs no lexicon.  NewsAPI and the RSS
feeds do not, so their headlines are scored with the exact keyword lists from the
specification:

    s_j = tanh((bullish_hits - bearish_hits) / 3)

Dividing by three means a single keyword cannot max out the score - it takes
three net-bullish terms to reach tanh(1) = 0.76.
"""

from __future__ import annotations

import re

import numpy as np

BULLISH_KEYWORDS: tuple[str, ...] = (
    "surge",
    "rally",
    "breakout",
    "approval",
    "adoption",
    "bullish",
    "soars",
    "gains",
    "record high",
    "accumulation",
    "inflow",
    "etf approved",
    "all-time high",
    "institutional",
    "upgrade",
)

BEARISH_KEYWORDS: tuple[str, ...] = (
    "crash",
    "plunge",
    "hack",
    "ban",
    "regulation",
    "crackdown",
    "bearish",
    "dumps",
    "outflow",
    "liquidation",
    "fraud",
    "investigation",
    "lawsuit",
    "delisting",
    "exploit",
)

#: Section 4.4 - keywords that mark a headline as potentially critical.
CRITICAL_KEYWORDS: tuple[str, ...] = (
    "hack",
    "exploit",
    "rug pull",
    "de-peg",
    "depeg",
    "bank run",
    "emergency",
    "war",
    "sanction",
)

DIVISOR = 3.0


def _count(text: str, keywords: tuple[str, ...]) -> int:
    hits = 0
    for keyword in keywords:
        # word-ish match so "gains" does not fire inside "bargains"
        pattern = r"(?<![a-z])" + re.escape(keyword) + r"(?![a-z])"
        hits += len(re.findall(pattern, text))
    return hits


#: Round AE - the spec lists are crypto-desk vocabulary; a real world wire
#: (BBC / Reuters / AP) is written in other words.  "Israel strikes Gaza",
#: "tariffs on Chinese goods", "drone attack hits power grid", "stocks fall as
#: yields climb" all scored exactly 0.0, so NIV read 0.00 on every real
#: Codespace while the fixtures looked fine.  These are the risk-on / risk-off
#: words a macro desk actually reads, scored with the same tanh(·/3) rule.
RISK_ON_KEYWORDS: tuple[str, ...] = (
    "ceasefire", "truce", "peace deal", "peace talks", "agreement", "deal reached", "deal signed",
    "rebound", "rebounds", "recover", "recovers", "recovery", "climbs", "rises", "jumps", "advances",
    "optimism", "relief", "eases", "easing", "cools", "cooling", "beats expectations", "better than expected",
    "stimulus", "rate cut", "cuts rates", "dovish", "demand accelerates", "record inflows",
)
RISK_OFF_KEYWORDS: tuple[str, ...] = (
    "war", "strikes", "strike", "airstrike", "missile", "missiles", "attack", "attacks", "attacked", "invasion",
    "invades", "troops", "escalation", "escalates", "shelling", "bombing", "explosion", "killed", "kills",
    "casualties", "hostage", "terror", "nuclear", "sanction", "sanctions", "tariff", "tariffs", "trade war",
    "embargo", "blockade", "falls", "fall", "drops", "slides", "slump", "slumps", "tumbles", "sell-off",
    "selloff", "default", "recession", "layoffs", "bankruptcy", "collapse", "collapses", "crisis", "turmoil",
    "shutdown", "hawkish", "rate hike", "hikes rates", "yields climb", "yields surge", "inflation rises",
    "hotter than expected", "worse than expected", "warning", "warns", "threatens", "threat", "unrest",
    "coup", "protests", "riots", "outage", "breach", "shortage", "stall", "stalls", "stalled",
)


def score_headline(headline: str) -> float:
    """Headline sentiment in [-1, +1] for a risk asset.

    s = tanh( (bull_hits - bear_hits) / 3 ) over the spec lists **plus** the
    macro risk-on / risk-off lists, blended with the signed BTC impact of the
    headline's theme (``impact.classify``) so a war headline that uses none of
    the crypto words still moves the needle.
    """
    text = (headline or "").lower()
    if not text:
        return 0.0
    bullish = _count(text, BULLISH_KEYWORDS) + _count(text, RISK_ON_KEYWORDS)
    bearish = _count(text, BEARISH_KEYWORDS) + _count(text, RISK_OFF_KEYWORDS)
    lexical = float(np.tanh((bullish - bearish) / DIVISOR))
    try:
        from backend.news import impact as _impact  # local import: impact imports nothing from here

        theme = _impact.classify(headline)
        thematic = float(theme["btc"]) * float(theme["magnitude"])
    except Exception:  # noqa: BLE001 - scoring must never fail a poll
        thematic = 0.0
    if thematic == 0.0:
        return lexical
    # the theme carries the direction a desk would trade; the words carry the
    # intensity.  Equal blend, then squashed back into [-1, 1].
    return float(np.tanh(0.5 * lexical * DIVISOR / 2.0 + 0.5 * thematic * 2.0))


#: A critical headline must also be *about the markets this engine trades*.
#: A Tier-1 wire runs dozens of "war" / "sanction" stories a day about
#: elections and trade policy; without this gate every one of them flattened
#: the open position (the user saw a locked SELL turn into "BUY 100 %" ten
#: seconds into a window, several times an hour).
MARKET_TERMS: tuple[str, ...] = (
    "bitcoin", "btc", "crypto", "cryptocurrency", "stablecoin", "tether", "usdt", "usdc",
    "exchange", "binance", "coinbase", "defi", "blockchain", "token", "gold", "paxg",
    "bullion", "fed", "treasury", "market", "markets", "wall street", "bank", "banks",
    "dollar", "liquidity", "etf",
)


def critical_keywords_in(headline: str) -> list[str]:
    """Whole-word critical keywords (``war`` must not match ``warning``/``award``)."""
    import re

    text = (headline or "").lower()
    found = []
    for kw in CRITICAL_KEYWORDS:
        pattern = r"(?<![a-z0-9])" + re.escape(kw).replace(r"\ ", r"[\s-]+") + r"(?:s|es|ed|ing)?(?![a-z0-9])"
        if re.search(pattern, text):
            found.append(kw)
    return found


def is_market_headline(headline: str) -> bool:
    import re

    text = (headline or "").lower()
    return any(re.search(r"(?<![a-z0-9])" + re.escape(term) + r"s?(?![a-z0-9])", text) for term in MARKET_TERMS)


def votes_to_sentiment(positive: int, negative: int) -> float:
    """CryptoPanic community votes -> sentiment (Section 4.2, source 1)."""
    positive = max(0, int(positive or 0))
    negative = max(0, int(negative or 0))
    return float((positive - negative) / (positive + negative + 1))


# ---------------------------------------------------------------------------
# Source credibility tiers (Section 4.2)
# ---------------------------------------------------------------------------

TIER_1 = (
    "reuters",
    "bloomberg",
    "coindesk",
    "the block",
    "theblock",
    "financial times",
    "wsj",
    "wall street journal",
    "cnbc",
    "associated press",
    "ap news",
    "ft.com",
)
TIER_2 = (
    "decrypt",
    "cointelegraph",
    "cointelegraph.com",
    "blockworks",
    "bitcoin magazine",
    "the defiant",
    "dl news",
    "crypto slate",
    "cryptoslate",
)
TIER_3 = (
    "beincrypto",
    "ambcrypto",
    "newsbtc",
    "bitcoin.com",
    "coinjournal",
    "u.today",
    "cryptonews",
    "coingape",
    "zycrypto",
)


def source_tier(source: str) -> int:
    """Map a source name onto the 1-4 credibility tier table."""
    name = (source or "").strip().lower()
    if not name:
        return 4
    for needle in TIER_1:
        if needle in name:
            return 1
    for needle in TIER_2:
        if needle in name:
            return 2
    for needle in TIER_3:
        if needle in name:
            return 3
    return 4
