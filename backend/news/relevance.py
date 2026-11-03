"""Market relevance of a headline (Round AP).

The world feeds (BBC World, Al Jazeera, Google News) were added so wars,
sanctions and tariffs reach the engine - but they also carried immigration
rows, sport and local politics into the news wire, and every one of them
counted towards the news-impact index.  Every headline now gets a relevance
score *before* it enters the cache:

* ``direct``  - names the assets or their market (bitcoin, btc, crypto, gold,
  PAXG, ETF, exchange, stablecoin, miner ...)                       -> 1.0
* ``macro``   - moves the assets through rates / dollar / inflation (Fed,
  ECB, CPI, treasury yields, recession, bank failure ...)          -> 0.7
* ``geo``     - risk-off shocks that move gold and bitcoin (war, missile,
  strike, attack, sanctions, tariff, oil shock, blockade ...)      -> 0.6
* anything else                                                      -> 0.0

Only ``score > 0`` is kept; the rest is counted (``filtered``) and shown in
the provider line so a quiet wire is explained, never hidden.  A relevance of
0.6-0.7 also scales the item's contribution to the news-impact index so a
tariff headline never outweighs a Bitcoin ETF headline.
"""
from __future__ import annotations

import re

DIRECT = (
    "bitcoin", "btc", "crypto", "cryptocurrenc", "blockchain", "stablecoin", "tether", "usdt", "usdc",
    "ethereum", "eth ", "solana", "binance", "coinbase", "kraken", "gemini", "bitfinex", "okx", "bybit",
    "etf", "spot etf", "miner", "mining", "hashrate", "halving", "satoshi", "defi", "token", "altcoin",
    "gold", "xau", "bullion", "paxg", "pax gold", "precious metal", "silver", "comex", "lbma",
    "microstrategy", "strategy inc", "blackrock", "grayscale", "sec ", "cftc", "digital asset",
)
MACRO = (
    "federal reserve", "the fed", "fed ", "fomc", "powell", "rate cut", "rate hike", "interest rate",
    "ecb", "bank of england", "boj", "bank of japan", "pboc", "rbi", "central bank", "treasury", "yield",
    "bond", "inflation", "cpi", "pce", "payroll", "jobs report", "unemployment", "gdp", "recession",
    "dollar", "dxy", "yuan", "yen", "debt ceiling", "default", "bank failure", "bank run", "liquidity",
    "stock market", "wall street", "s&p", "nasdaq", "dow jones", "nikkei", "hang seng", "sensex",
    "oil price", "brent", "wti", "opec", "commodit", "stimulus", "quantitative", "qt ", "qe ",
)
GEO = (
    "war", "invasion", "invade", "missile", "drone strike", "airstrike", "air strike", "attack", "bomb",
    "explosion", "nuclear", "ceasefire", "troops", "military", "escalat", "sanction", "tariff", "trade war",
    "embargo", "blockade", "strait of hormuz", "red sea", "taiwan", "ukraine", "russia", "iran", "israel",
    "gaza", "north korea", "china", "coup", "martial law", "cyberattack", "hack", "exploit", "breach",
    "pipeline", "shipping", "election", "impeach", "shutdown", "assassinat", "terror",
)
NOISE = (
    "immigration", "migrant", "visa", "deport", "asylum", "football", "soccer", "cricket", "tennis", "nba",
    "nfl", "premier league", "olympic", "celebrity", "actor", "actress", "film", "movie", "music", "album",
    "recipe", "royal family", "prince", "princess", "wedding", "obituary", "weather", "horoscope",
)

_WORD = re.compile(r"[a-z0-9&$ ]+")


def score(headline: str) -> tuple[float, str]:
    """Return ``(relevance, tag)`` for a headline."""
    text = " " + " ".join(_WORD.findall((headline or "").lower())) + " "
    direct = any(k in text for k in DIRECT)
    macro = any(k in text for k in MACRO)
    geo = any(k in text for k in GEO)
    noise = any(k in text for k in NOISE)
    if direct:
        return 1.0, "direct"
    if macro:
        return 0.7, "macro"
    if geo and not noise:
        return 0.6, "geo"
    return 0.0, "noise" if noise else "unrelated"


def keep(headline: str) -> bool:
    return score(headline)[0] > 0.0
