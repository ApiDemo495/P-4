"""News impact - what a headline does to BTC and to PAXG (Round Z).

The sentiment lexicon says whether a headline is *good or bad*; it does not say
*for whom*.  A war is bad for a risk asset and good for tokenised gold; a hot
inflation print is bad for both; an ETF approval is good for BTC and nothing
for PAXG.  This module maps every headline - crypto or not - onto a theme with
a signed impact per asset, then aggregates the live wire into one number per
asset the fusion can vote with, decayed by age and weighted by source tier.

    impact(asset) in [-1, +1]  =  tanh( sum_i  w_tier(i) * decay(age_i) * novelty_i * impact_i(asset) )

    decay(age) = 0.5 ** (age / HALF_LIFE_S)         (half-life 20 min)
    w_tier     = 1.0 / 0.7 / 0.45 for tier 1 / 2 / 3

Round AI - "not all the time the same news affects the market again and
again".  Three corrections make a headline's weight depend on how *new* it is:

* **duplicates** - two headlines whose word sets overlap by more than 60 %
  are the same story told twice; the second one weighs nothing;
* **saturation** - the k-th distinct headline on the same theme in the window
  weighs 1/k: ten "war" headlines are not ten shocks, they are one shock
  with ten reporters;
* **habituation** - a theme that has been continuously on the wire for hours
  is already in the price.  The module remembers when each theme first
  appeared without a gap; its weight halves every ``HABITUATION_HALF_LIFE_S``
  (2 h) of continuous presence and recovers after a ``THEME_GAP_RESET_S``
  (90 min) absence.  A *new* theme always enters at full weight.

The learning layer completes this: every theme that drives the wire also
votes in the evidence ledger (``news:theme:<name>``), so a theme whose
headlines repeatedly fail to move this asset is faded by the record.

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
DUPLICATE_JACCARD = 0.6
HABITUATION_HALF_LIFE_S = 2 * 3600.0
THEME_GAP_RESET_S = 90 * 60.0
_STOP = frozenset("a an the of to in on for and or as at by from with is are was were be been it its this that after over "
                  "amid into vs says said say will would could new us u.s".split())

#: theme -> (first_seen_continuous, last_seen) - module state, reset on restart.
_THEME_PRESENCE: dict[str, list[float]] = {}


def _tokens(headline: str) -> frozenset:
    words = re.findall(r"[a-z0-9][a-z0-9\-\.]+", (headline or "").lower())
    return frozenset(w for w in words if w not in _STOP and len(w) > 2)


def _is_duplicate(tokens: frozenset, seen: list[frozenset]) -> bool:
    if not tokens:
        return False
    for other in seen:
        if not other:
            continue
        inter = len(tokens & other)
        union = len(tokens | other)
        if union and inter / union >= DUPLICATE_JACCARD:
            return True
    return False


def habituation(theme: str, now: float, present: bool = True) -> float:
    """Weight multiplier for a theme given how long it has been continuously
    on the wire: 1.0 when new, 0.5 after two hours, 0.25 after four."""
    rec = _THEME_PRESENCE.get(theme)
    if present:
        if rec is None or now - rec[1] > THEME_GAP_RESET_S:
            rec = [now, now]
            _THEME_PRESENCE[theme] = rec
        rec[1] = max(rec[1], now)
    if rec is None:
        return 1.0
    continuous = max(0.0, now - rec[0])
    return 0.5 ** (continuous / HABITUATION_HALF_LIFE_S)


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
           "shelling", "offensive", "ceasefire collapses", "declares war", "nuclear", "strikes on", "strikes in",
           "army strikes", "israel strikes", "russia strikes", "drone attack", "drone strike", "attack on",
           "under attack", "escalation", "escalates"),
          btc=-0.6, paxg=+0.8, magnitude=0.9),
    Theme("attack", "terror / cyber attack",
          ("terror attack", "terrorist", "bombing", "explosion", "cyberattack", "cyber attack", "ransomware",
           "assassination", "hostage", "attack hits", "attack kills", "attacked", "gunman", "shooting"),
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
    duplicates = 0
    seen_tokens: list[frozenset] = []
    theme_count: dict[str, int] = {}
    theme_habit: dict[str, float] = {}
    # newest first so the first telling of a story is the one that counts
    ordered = sorted(items or (), key=lambda i: -float(getattr(i, "published_at", 0.0) or 0.0))
    for item in ordered:
        headline = getattr(item, "headline", "")
        info = classify(headline)
        if info["theme"] == "neutral":
            continue
        classified += 1
        if info["scope"] == "world":
            world += 1
        age = max(0.0, now - float(getattr(item, "published_at", now) or now))
        if age > 12 * HALF_LIFE_S:
            continue                                   # four hours: < 0.03 % weight left
        tokens = _tokens(headline)
        if _is_duplicate(tokens, seen_tokens):
            duplicates += 1
            continue
        seen_tokens.append(tokens)
        theme = info["theme"]
        k = theme_count.get(theme, 0) + 1
        theme_count[theme] = k
        if theme not in theme_habit:
            theme_habit[theme] = habituation(theme, now)
        novelty = (1.0 / k) * theme_habit[theme]
        decay = 0.5 ** (age / HALF_LIFE_S)
        w = TIER_WEIGHT.get(int(getattr(item, "tier", 3) or 3), 0.45) * decay * info["magnitude"] * novelty
        weight_sum += w
        for asset, key in (("BTC", "btc"), ("PAXG", "paxg")):
            contribution = w * float(info[key])
            total[asset] += contribution
            if abs(contribution) > 1e-6:
                drivers[asset].append({
                    "headline": headline[:140],
                    "theme": info["label"],
                    "theme_key": theme,
                    "impact": round(contribution, 4),
                    "age_seconds": round(age, 0),
                    "novelty": round(novelty, 3),
                    "nth_on_theme": k,
                    "habituation": round(theme_habit[theme], 3),
                })
    out: dict = {"classified": classified, "world_items": world, "duplicates": duplicates,
                 "themes": {t: {"headlines": n, "habituation": round(theme_habit.get(t, 1.0), 3)} for t, n in theme_count.items()},
                 "weight": round(weight_sum, 4), "drivers": {}}
    for asset in ("BTC", "PAXG"):
        out[asset] = round(math.tanh(total[asset]), 4)
        out["drivers"][asset] = sorted(drivers[asset], key=lambda d: -abs(d["impact"]))[:limit_drivers]
    return out
