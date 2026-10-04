"""News impact - what a headline does to BTC and to PAXG (Round Z).

The sentiment lexicon says whether a headline is *good or bad*; it does not say
*for whom*.  A war is bad for a risk asset and good for tokenised gold; a hot
inflation print is bad for both; an ETF approval is good for BTC and nothing
for PAXG.  This module maps every headline - crypto or not - onto a theme with
a signed impact per asset, then aggregates the live wire into one number per
asset the fusion can vote with, decayed by age and weighted by source tier.

    impact(asset) in [-1, +1]  =  tanh( sum_i  w_tier(i) * decay(age_i) * impact_i(asset) )

    decay(age) = 0.5 ** (age / HALF_LIFE_S)         (half-life 20 min)
    w_tier     = 1.0 / 0.7 / 0.45 for tier 1 / 2 / 3

Nothing here is a prediction; it is the sign and size of the shock the
headline describes, so the engine can *take it into consideration* instead of
printing the headline next to a signal that ignored it.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

HALF_LIFE_S = 20 * 60.0
TIER_WEIGHT = {1: 1.0, 2: 0.7, 3: 0.45}


@dataclass(frozen=True)
class Theme:
    name: str
    label: str
    keywords: tuple[str, ...]
    btc: float      # signed impact on BTC when the theme fires
    paxg: float     # signed impact on PAXG (tokenised gold)
    magnitude: float  # how big a shock this theme usually is, 0..1


#: Order matters: the first theme that matches wins, so the most specific
#: (and most consequential) themes come first.
THEMES: tuple[Theme, ...] = (
    Theme("war", "war / military escalation",
          ("war", "invasion", "invades", "airstrike", "air strike", "missile", "troops", "military strike",
           "shelling", "offensive", "ceasefire collapses", "declares war", "nuclear"),
          btc=-0.6, paxg=+0.8, magnitude=0.9),
    Theme("attack", "terror / cyber attack",
          ("terror attack", "terrorist", "bombing", "explosion", "cyberattack", "cyber attack", "ransomware",
           "assassination", "hostage"),
          btc=-0.5, paxg=+0.5, magnitude=0.7),
    Theme("hack", "exchange / protocol hack",
          ("hack", "hacked", "exploit", "drained", "stolen funds", "rug pull", "bridge exploit"),
          btc=-0.7, paxg=+0.1, magnitude=0.8),
    Theme("depeg", "stablecoin / peg stress",
          ("depeg", "de-peg", "loses peg", "bank run", "insolvent", "insolvency", "halts withdrawals"),
          btc=-0.8, paxg=+0.4, magnitude=0.9),
    Theme("sanctions", "sanctions / trade war / tariffs",
          ("sanction", "sanctions", "tariff", "tariffs", "trade war", "export ban", "embargo", "blacklist"),
          btc=-0.35, paxg=+0.5, magnitude=0.6),
    Theme("hawkish", "central bank hawkish / inflation hot",
          ("rate hike", "raises rates", "hikes rates", "hawkish", "inflation rises", "inflation jumps",
           "hotter than expected", "cpi rises", "yields surge", "higher for longer"),
          btc=-0.5, paxg=-0.4, magnitude=0.7),
    Theme("dovish", "central bank dovish / rate cut",
          ("rate cut", "cuts rates", "cut rates", "dovish", "quantitative easing", "stimulus", "liquidity injection",
           "inflation cools", "cooler than expected", "pivot"),
          btc=+0.5, paxg=+0.4, magnitude=0.7),
    Theme("dollar", "dollar strength",
          ("dollar surges", "dollar strengthens", "dxy", "strong dollar", "dollar rally"),
          btc=-0.25, paxg=-0.45, magnitude=0.5),
    Theme("recession", "recession / growth scare",
          ("recession", "contraction", "layoffs", "default", "debt ceiling", "credit crunch", "bankruptcy",
           "bank failure", "bank collapse"),
          btc=-0.4, paxg=+0.5, magnitude=0.6),
    Theme("regulation", "crypto regulation / enforcement",
          ("ban", "crackdown", "lawsuit", "sec charges", "sec sues", "investigation", "delisting", "fine",
           "regulators", "enforcement"),
          btc=-0.45, paxg=0.0, magnitude=0.5),
    Theme("adoption", "adoption / ETF / institutional",
          ("etf approved", "etf approval", "spot etf", "institutional", "adoption", "treasury buys", "adds bitcoin",
           "legal tender", "inflows", "record inflow", "allocates", "custody"),
          btc=+0.55, paxg=+0.05, magnitude=0.6),
    Theme("mining", "hashrate / mining / supply",
          ("hashrate", "halving", "miners capitulate", "mining ban", "difficulty"),
          btc=+0.15, paxg=0.0, magnitude=0.3),
    Theme("gold", "gold market",
          ("gold price", "gold hits", "gold rises", "gold falls", "bullion", "central bank gold", "gold reserves",
           "gold demand"),
          btc=0.0, paxg=+0.5, magnitude=0.5),
    Theme("rally", "momentum / price action",
          ("surge", "rally", "soars", "breakout", "all-time high", "record high", "jumps"),
          btc=+0.3, paxg=+0.1, magnitude=0.35),
    Theme("selloff", "momentum / price action",
          ("crash", "plunge", "plunges", "dumps", "tumbles", "liquidations", "sell-off", "selloff", "slump"),
          btc=-0.3, paxg=-0.05, magnitude=0.35),
)

_PATTERNS = {t.name: re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(k) for k in t.keywords) + r")(?![a-z])")
             for t in THEMES}


def classify(headline: str) -> dict:
    """Theme + signed per-asset impact for one headline (``theme`` is
    ``"neutral"`` with zero impacts when nothing matches)."""
    text = (headline or "").lower()
    for theme in THEMES:
        hit = _PATTERNS[theme.name].search(text)
        if hit:
            return {
                "theme": theme.name,
                "label": theme.label,
                "matched": hit.group(0),
                "btc": theme.btc,
                "paxg": theme.paxg,
                "magnitude": theme.magnitude,
                "scope": "world" if theme.name in ("war", "attack", "sanctions", "hawkish", "dovish", "dollar",
                                                   "recession", "gold") else "crypto",
            }
    return {"theme": "neutral", "label": "no market theme", "matched": "", "btc": 0.0, "paxg": 0.0,
            "magnitude": 0.0, "scope": "none"}


def aggregate(items, now: float | None = None, limit_drivers: int = 4) -> dict:
    """Decayed, tier-weighted impact of the live wire on each asset.

    ``items`` are ``NewsItem``-like objects (``headline``, ``tier``,
    ``published_at``).  Returns ``{"BTC": x, "PAXG": y, "drivers": {...},
    "classified": n, "world_items": m}`` - the numbers the fusion votes with
    and the headlines that moved them, so the panel can say *why*.
    """
    now = time.time() if now is None else float(now)
    total = {"BTC": 0.0, "PAXG": 0.0}
    weight_sum = 0.0
    drivers: dict[str, list] = {"BTC": [], "PAXG": []}
    classified = 0
    world = 0
    for item in items or ():
        info = classify(getattr(item, "headline", ""))
        if info["theme"] == "neutral":
            continue
        classified += 1
        if info["scope"] == "world":
            world += 1
        age = max(0.0, now - float(getattr(item, "published_at", now) or now))
        decay = 0.5 ** (age / HALF_LIFE_S)
        w = TIER_WEIGHT.get(int(getattr(item, "tier", 3) or 3), 0.45) * decay * info["magnitude"]
        weight_sum += w
        for asset, key in (("BTC", "btc"), ("PAXG", "paxg")):
            contribution = w * float(info[key])
            total[asset] += contribution
            if abs(contribution) > 1e-6:
                drivers[asset].append({
                    "headline": getattr(item, "headline", "")[:140],
                    "theme": info["label"],
                    "impact": round(contribution, 4),
                    "age_seconds": round(age, 0),
                })
    out: dict = {"classified": classified, "world_items": world, "weight": round(weight_sum, 4), "drivers": {}}
    for asset in ("BTC", "PAXG"):
        out[asset] = round(math.tanh(total[asset]), 4)
        out["drivers"][asset] = sorted(drivers[asset], key=lambda d: -abs(d["impact"]))[:limit_drivers]
    return out
