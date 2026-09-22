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


def score_headline(headline: str) -> float:
    """Lexicon sentiment in [-1, +1]."""
    text = (headline or "").lower()
    if not text:
        return 0.0
    bullish = _count(text, BULLISH_KEYWORDS)
    bearish = _count(text, BEARISH_KEYWORDS)
    return float(np.tanh((bullish - bearish) / DIVISOR))


def critical_keywords_in(headline: str) -> list[str]:
    text = (headline or "").lower()
    return [kw for kw in CRITICAL_KEYWORDS if kw in text]


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
