# doc11 — Component-by-component transfer, written for an agent that has to *plant* it

This file is different from `ADVICE_FOR_A_MULTI_ASSET_INTRADAY_APP.md`. That one
is the map. This one is the soil: for each component it says exactly what we
built, in which file, with which data structures, which numbers, which
mistakes we made and how each was fixed, what to show on the screen, what to
test, and what to do differently on a 22 / 44 / 130 / 247-minute clock with
several assets. Where a decision depends on an optional key (NewsAPI,
CryptoPanic, X bearer, Gemini) it spells out **what changes with and without
the key**, so the agent can stop guessing.

Chapters are added one component at a time, in the order the sister app is
stuck on them.

1. **News** — this chapter
2. Emotions (next)
3. Physics
4. Brain
5. Genesis
6. Agents and keys (Gemini, local, decision models)
7. UI, toggles, settings
8. Lock, clock, ledger, operations

---

# Chapter 1 — The news component

## 1.0 What "continuous" honestly means, and the 4-second question

Your app wants "a continuous stream of news that updates every 4 seconds".
Separate three things that are usually mixed up, because mixing them is why it
"can't figure it out":

| layer | what changes | real cadence | who drives it |
|---|---|---|---|
| **ingest** — fetching headlines from sources | new items arrive | RSS 60–120 s, CryptoPanic 30 s, NewsAPI 60 s (paid) / ≥ 15 min (free), Reddit/StockTwits 30–60 s, X filtered stream = push (paid) | one asyncio task per source, each with its own interval and backoff |
| **evaluate** — re-weighting what you already have | every item's age, decay, novelty, habituation, the NIV per asset per horizon | **every 4 s is fine and cheap** (pure arithmetic over ≤ 200 cached items) | one 4 s ticker in the engine |
| **publish** — pushing the result to the UI | the news card, the NIV chips, the drivers list | every 4 s over the existing WebSocket (`type: "NEWS"`) plus a REST mirror | the same ticker |

No public news source gives you a new headline every 4 seconds; the wires
themselves don't produce that. What *can* and *should* update every 4 seconds
is the **evaluation**: the weight of each headline decays with age, duplicates
and saturated themes lose weight, the per-asset impact index moves, and the UI
sees a living stream. The moment you implement "4 s" as "call the source every
4 s" you get rate-limited (NewsAPI free = 100 requests/day = one call per
14.4 min), banned from RSS hosts, or served cached pages — and the stream
looks dead anyway because nothing new arrived.

So the design is: **slow, polite ingest per source; a fast, pure evaluation
tick; one publish channel.** That is exactly how ours works
(`backend/news/news_engine.py`: `_runner` → `poll_all` with per-source
"last polled" timestamps; the evaluation is stateless arithmetic in
`impact.aggregate` and the NIV formula and runs whenever anybody asks — the
live-refresh loop asks every 2 s, the lock asks at the boundary).

## 1.1 Files and responsibilities (ours)

| file | job |
|---|---|
| `backend/news/news_engine.py` | owns the cache, the pollers, the key rings, the status, the critical-event scan; `start()` seeds immediately (`poll_all(force=True)`) so the first window has news; `_runner` loops every `max(5, news_poll_seconds)` and each source decides inside `poll_all` whether its own interval has elapsed |
| `backend/news/rss_source.py` | keyless; `feedparser` over a list of feeds, `gather(return_exceptions=True)` so one feed failing never blocks the rest; 10 items per feed |
| `backend/news/cryptopanic_source.py` | keyed; community votes → sentiment; 30 s |
| `backend/news/newsapi_source.py` | keyed; `/v2/everything` with a **title-scoped** query; 60 s |
| `backend/news/social_source.py` | Reddit + StockTwits keyless JSON, X with a paid bearer only; adaptive interval (30 s when velocity ≥ 1.5× baseline, else 60 s) |
| `backend/news/relevance.py` | gate + scale: direct 1.0 / macro 0.7 / geo 0.6 / else 0.0 (dropped and counted) |
| `backend/news/sentiment_lexicon.py` | headline sentiment, source tiers, critical keywords, market terms |
| `backend/news/impact.py` | theme classification → **signed impact per asset**, decay, duplicates (Jaccard), saturation (1/k), habituation (2 h half-life), aggregation to one number per asset with drivers |
| `backend/news/critical_event_detector.py` | three triggers (headline, sentiment swing, flash move) → emergency re-lock with duration |
| `backend/data/ring_buffer.py::NewsCache` | deque of `NewsItem`, dedupe by headline, `providers` status strings, `filtered` counters |
| `backend/formulas/category_g_news/niv.py`, `smd.py` | the two news formulas that enter the formula engine, the brain and fusion |
| `backend/api/routes_news.py` | `/api/news/feed`, `/api/news/status`, `/api/news/poll`, `/api/news/emergency`, `/api/news/impact` |

### The item (`NewsItem`)
```
headline: str          published_at: float (epoch s)   source: str
sentiment: float ∈[-1,1]   credibility: float ∈(0,1]   tier: int 1|2|3
relevance: float {1.0, 0.7, 0.6}   url: str   provider: "rss"|"newsapi"|"cryptopanic"|"social"
fingerprint: str (we dedupe on the normalised headline; use title+domain)
```
Add for multi-asset: `impacts: dict[asset, float]` filled by `impact.classify`
at ingest, so evaluation never re-parses text.

## 1.2 The ingest loop, the way that does not die

```python
class NewsEngine:
    async def start(self):
        self._client = httpx.AsyncClient(headers={"user-agent": "YourApp/1.0"})
        await self.poll_all(force=True)                 # seed before the first window
        self._tasks.append(asyncio.create_task(self._runner()))
        self._tasks.append(asyncio.create_task(self._ticker()))   # NEW: 4 s evaluation

    async def _runner(self):
        while not self._stop.is_set():
            try:
                await self.poll_all()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = str(exc); log.warning("news poll failed: %s", exc)
            await asyncio.sleep(5.0)                    # the loop ticks; sources decide

    async def poll_all(self, force=False):
        now = time.time(); polled = {}
        for src in self.sources:                        # each has interval, last, backoff_until
            if force or (now >= src.backoff_until and now - src.last >= src.interval):
                polled[src.name] = await self._poll_one(src)
                src.last = now
        if not self.cache.all() or now - self.cache.last_poll > 300:
            polled["synthetic"] = self._seed_synthetic()   # labelled, never scored as real
        self._update_status(); await self._scan_for_critical()
        return polled

    async def _poll_one(self, src):
        try:
            items = await src.fetch(self._client, self.settings)   # raises on HTTP errors
        except src.RateLimited as exc:
            src.backoff_until = time.time() + min(900, src.backoff * 2); src.backoff *= 2
            src.state = f"rate limited, retry in {src.backoff_until - time.time():.0f}s"; return 0
        except Exception as exc:
            src.backoff_until = time.time() + min(300, src.backoff * 2); src.backoff *= 2
            src.state = f"error: {str(exc)[:80]}"; return 0
        src.backoff = 5.0; src.state = f"ok ({len(items)})"; src.last_ok = time.time()
        return self.cache.add(items)                     # dedupe + relevance gate inside

    async def _ticker(self):                             # the "every 4 seconds" part
        while not self._stop.is_set():
            payload = self.stream_payload()              # pure arithmetic, < 1 ms
            await self.broadcast({"type": "NEWS", "data": payload})
            await asyncio.sleep(4.0)
```

Rules baked into that skeleton, each one a bug we had:
- the loop never awaits one source's failure; each source has its own
  `interval`, `last`, `backoff`, `state`, `last_ok` (error N-1 below);
- keyed sources go through a **key ring** (`_poll_with_ring`): on 401 the key
  is marked rejected, on 429 it cools for the provider's window and the next
  key is tried, on success the slot is reported healthy; the status string
  says which slot answered;
- the cache seeds synthetic, clearly labelled headlines only when the wire is
  empty for 5 minutes, so the pipeline (and UI) is exercisable offline, and
  the status says `coverage: "synthetic"`;
- `start()` seeds synchronously so the **first** window is never newsless.

## 1.3 Sources: what each gives you, with and without keys

### RSS (keyless) — your backbone
Ours (`config.rss_feeds`): CoinDesk, CoinTelegraph, The Block, Decrypt (crypto
desk); BBC World, Al Jazeera (world); Federal Reserve press releases; CNBC
markets; **Google News RSS search** with a query
(`news.google.com/rss/search?q=war OR sanctions OR tariffs OR "rate cut" OR
"rate hike" OR gold price`). For a multi-asset app add per asset: the
exchange/issuer status page feed, the central bank of the currency, Google News
RSS searches per asset name and ticker, and a sector wire (oil → OPEC/EIA,
equities → company IR feeds).
- Poll 60 s (fast desks) to 120 s (world). Honour `ETag`/`Last-Modified`
  (feedparser passes `etag=`/`modified=`) — most RSS hosts return 304 and you
  stay welcome.
- RSS publishes within 1–10 minutes of the story on the desks that matter.
  For a 22-minute window that is fine; for 70 s it was the main latency.
- Set a real `User-Agent` with a URL; anonymous agents get 403 from several
  hosts (we did).

### CryptoPanic (free key, optional)
Aggregates crypto desks with **community votes** (bullish/bearish/important),
30 s polling inside the free tier (5/min). What you gain: a second opinion on
sentiment that is not your lexicon (`votes_to_sentiment`), currency tags per
item (BTC, ETH…) that are a free relevance signal, and "important"/"hot"
filters. Without it: RSS covers the same desks 1–5 minutes later with lexicon
sentiment only. Crypto-only — useless for gold/FX/equities.

### NewsAPI (optional key) — what actually changes
With a key you get `GET /v2/everything?q=<query>&searchIn=title,description&sortBy=publishedAt&language=en&pageSize=20`
over ~150 k sources. What that buys you, concretely:
1. **Query-scoped recall** across sources you don't have an RSS for (regional
   business press, sector trades). For multi-asset this is the real value:
   one query per asset (`"Reliance Industries" OR RELIANCE`, `"crude oil" OR
   brent OR opec`) instead of hunting RSS URLs.
2. **Structured fields**: `source.name` (→ tier), `publishedAt` (ISO, reliable),
   `description` (→ better lexicon hits than titles alone), `url` (→
   fingerprint by domain).
3. **Exclusions** in the query (`NOT (immigration OR football OR celebrity)`)
   that killed our "gold" false positives before they reached the cache.

What it does **not** buy you, and you must plan for:
- The free *Developer* plan is **100 requests per day** and NewsAPI documents
  a delay on new articles for that plan (check their current pricing page;
  historically 24 h for the free tier). So on a free key NewsAPI is a
  **backfill and relevance** source — poll it every 15 minutes, use it to
  enrich tiers/descriptions and to seed history at boot — not a breaking-news
  feed. On a paid plan (no delay, higher quota) poll every 60 s like we do and
  it becomes a first-class source.
- `pageSize` caps at 100; sort by `publishedAt` and stop at the first item
  you've already seen.
- Errors: 401 → key rejected (ring marks it), 426 → plan upgrade needed (treat
  as "configured but limited", show it), 429 → rate limited (cool the slot for
  the day, not 60 s — the quota is daily).

Decision table for your settings page:

| key present | ingest | what the panel says |
|---|---|---|
| none | RSS + Google News RSS + Reddit/StockTwits | `coverage: limited — RSS only · add CryptoPanic/NewsAPI for faster desks and query recall` |
| CryptoPanic | + votes, currency tags, 30 s | `coverage: good (crypto)` |
| NewsAPI free | + 15-min query backfill, descriptions, tiers | `coverage: good · NewsAPI free tier is delayed/backfill` |
| NewsAPI paid | + 60 s query-scoped breaking | `coverage: full` |
| X bearer (paid) | + push stream of selected accounts/keywords | `social: streaming` |

Put that table in the UI next to the key fields. The user's "what would be
the difference" question must be answered by the app itself.

### Social (keyless Reddit/StockTwits, keyed X)
Reddit `r/<sub>/new.json` and StockTwits `streams/symbol/<TICKER>.json` need no
key; respect their user-agent rules; derive `social_buzz` (posts/min vs a
baseline) and `social_sentiment` (StockTwits carries bullish/bearish tags).
X needs a paid bearer for the filtered stream; without it do not pretend
(we removed every fake Instagram/Facebook "sensor" after the user called it
out). Social drives FOMO/EUPHORIA (positive buzz) and FEAR/PANIC (negative
buzz) in the emotion band, never the direction on its own.

### Others worth adding for multi-asset
- **GDELT 2.1** (keyless, 15-minute update files, global events with tone) —
  excellent for geo themes across all assets.
- Exchange **status pages** (status.binance.com RSS, NSE/BSE circulars) —
  the only source for "trading halted", which is a critical event.
- **Central bank calendars** (ECB/Fed/RBI release times) — not news, but
  schedule the half-life shorter around releases.

## 1.4 Turning text into numbers (the part your app "can't use")

Every item gets, at ingest, in this order:

1. **Relevance** `r ∈ {1.0, 0.7, 0.6, 0}` (`relevance.py`): direct asset/market
   terms → 1.0; macro (Fed, CPI, yields, dollar, recession) → 0.7; geo (war,
   strike, sanctions, tariff, oil shock) → 0.6; else 0 → **dropped and
   counted** (`cache.filtered`, three examples kept for the panel). For
   multi-asset make `DIRECT` a per-asset dict and store `relevance_by_asset`.
2. **Sentiment** `s ∈ [−1, 1]` (`score_headline`):
   `lexical = tanh((bull_hits − bear_hits)/3)` over the crypto lists **plus**
   macro risk-on/risk-off lists (the crypto lists alone scored "Israel strikes
   Gaza" and "tariffs on Chinese goods" at exactly 0.0 — error N-2), blended
   50/50 with the theme's signed impact for the asset, squashed with `tanh`.
   CryptoPanic votes override the lexicon when present.
3. **Credibility / tier** (`source_tier`): Tier 1 = Reuters, Bloomberg,
   CoinDesk, The Block, FT, WSJ, CNBC, AP → weight 1.0; Tier 2 = Decrypt,
   CoinTelegraph, Blockworks… → 0.7; Tier 3 = everything else → 0.45.
4. **Theme and per-asset impact** (`impact.classify`): an ordered list of
   themes, most consequential first, each with keywords and a **signed impact
   per asset** and a magnitude:

   | theme | BTC | gold | magnitude | keywords (excerpt) |
   |---|---|---|---|---|
   | war / escalation | −0.6 | +0.8 | 0.9 | invasion, airstrike, missile, troops, drone attack, escalates |
   | terror / cyber attack | −0.5 | +0.5 | 0.7 | bombing, explosion, cyberattack, ransomware |
   | exchange / protocol hack | −0.7 | +0.1 | 0.8 | hack, exploit, drained, bridge exploit |
   | de-peg / insolvency | −0.8 | +0.4 | 0.9 | depeg, bank run, halts withdrawals |
   | sanctions / tariffs | −0.35 | +0.5 | 0.6 | sanction, tariff, trade war, embargo |
   | hawkish / hot inflation | −0.5 | −0.4 | 0.7 | rate hike, hotter than expected, yields surge |
   | dovish / cut | +0.5 | +0.4 | 0.7 | rate cut, stimulus, inflation cools, pivot |
   | dollar strength | −0.25 | −0.45 | 0.5 | dollar surges, dxy |
   | recession | −0.4 | +0.5 | 0.6 | layoffs, default, bank failure |
   | regulation | −0.45 | 0 | 0.5 | ban, crackdown, sec sues, delisting |
   | adoption / ETF | +0.6 | 0 | 0.7 | etf approval, inflows, treasury buys |

   For your assets extend the matrix by column (equity index, single stock,
   FX pair, oil) — this matrix **is** the weighting you asked for: a theme's
   sign per asset times its magnitude. Start with desk intuition, then let
   the learned layer (step 7) fade the themes that don't move an asset.
5. **Novelty** (`impact.py`, Round AJ, "not all the time the same news
   affects the market again and again"):
   - duplicates: token-set Jaccard ≥ 0.6 with an already-counted headline →
     weight 0 (same story, second reporter);
   - saturation: the k-th distinct headline on the same theme in the window
     weighs `1/k`;
   - habituation: a theme continuously on the wire halves every 2 h of
     presence and resets after a 90-minute gap; a new theme enters at full
     weight.
6. **Decay**: `decay = 0.5^(age / half_life)`; items older than 12 half-lives
   are skipped. Ours is 20 min for the impact index and `exp(−age/300 s)` for
   the five-headline NIV formula. **Per horizon** use `half_life ≈ W/2`
   (22 min → 11 min, 247 min → 2 h) and keep one index per horizon.
7. **Aggregate per asset per horizon**:
   `impact_h(asset) = tanh( Σ_i tier_i · decay_h(age_i) · novelty_i · magnitude_i · impact_i(asset) )`
   plus the **drivers** (top 4 headlines by |contribution| with theme, age,
   novelty, nth-on-theme, habituation) so the panel can say *why*.
8. **NIV** (formula 19, enters the formula engine, the brain as a PN channel,
   fusion and hindsight):
   `NIV = Σ s_j c_j r_j e^(−age_j/τ) / (Σ c_j r_j e^(−age_j/τ) + W₀)`, five
   newest items, `W₀ = 0.5` resting evidence. The `W₀` is not cosmetic: with a
   bare `eps` the decay cancels between numerator and denominator and one
   stale headline holds NIV at +0.72 forever (error N-3). **SMD** =
   `tanh(NIV − TAI)` compares the news with the tape's acceleration.

## 1.5 How the numbers reach the prediction (the weightage)

There are five paths, and all five are printed:

1. **Formula engine**: NIV and SMD are two of the 31 formulas, directional,
   with traces and double checks; they get the same per-formula hindsight
   multiplier as everything else (`news:NIV` in the hourly review).
2. **Brain**: NIV is a projection-neuron channel ("Johnston's organ" in our
   mapping) with an ON/OFF pair, so news reaches the circuit like any sensory
   input.
3. **Fusion vote**: `impact(asset)` for the locked asset is a signed voter with
   prior weight 0.10, shown as "news impact −0.31 (war / military escalation,
   2 headlines, newest 6 min)".
4. **Learned layers**: every theme that drove the wire this window is an input
   `news:theme:<name>` to the hindsight review, the meta-combiner and the
   unified evidence ledger, so a theme whose headlines repeatedly fail to
   move *this asset* is faded by the record. This is how "weightage" stops
   being a constant and becomes measured.
5. **Emergency path** (the only mid-window change allowed): see 1.6.

Weighting rule of thumb that survived: news is a **confidence and context
modifier** most of the time and a **direction setter** only through the
emergency path or when the learned layer has earned it for that theme. In a
70 s contest news was worth ~2 % of decisions; at 22–247 min it matters more
(a Fed decision inside a 247-min window *is* the window), which is exactly
why the half-life and the per-horizon index must scale with W.

## 1.6 Critical events — the mid-window exception, done carefully

`critical_event_detector.py` has three triggers:
- **headline**: whole-word critical keyword (`\bwar\b`, not substring — "war"
  matched "warning", "forward", "award"; "hack" matched "hackathon": error
  N-4) **and** a market term in the headline **and** published ≤ 10 min ago
  **and** not fired before (fingerprint set);
- **sentiment swing**: |ΔNIV| over 90 s above a threshold;
- **flash move**: tape move beyond k·σ inside the window (this one is the
  tape's, but it rides the same dispatcher).

An event → `on_critical` → the lock controller issues an **emergency
re-lock** with `computed_at`, keeps `valid_from/valid_until`, carries the
headline, and the chip reads "⚡ emergency re-lock HH:MM:SSZ" in amber. The
dashboard watchdog counts it as the one documented mid-window change and
keeps checking the *new* text. `POST /api/news/emergency` accepts
`duration_seconds` 10–900 (and honours it — it didn't, error N-7) for drills.

For 22–247-minute windows: keep the same gates, raise the recency window to
~W/10, and require Tier 1–2 for a headline trigger; at these horizons a
false emergency costs a whole window.

## 1.7 What the UI must show (and ours does)

- News card header: `coverage` (full / good / limited / synthetic / disabled),
  `seconds_since_poll`, items in cache, and **one line per source**:
  `rss ok (41) · cryptopanic not configured · newsapi key 2 rate limited (cooling 812 s) · social reddit ok / x no bearer`.
- The per-asset impact chips with the sign, the theme label, headline count,
  newest age; click → drivers list (headline, theme, age, novelty, k-th on
  theme, habituation).
- The NIV trace rows (headlines weighted, weighted sum, weight sum, W₀) under
  the NIV formula in the explorer; "why 0.00" when the wire is quiet.
- `relevance gate dropped n` with three examples — a quiet wire is explained,
  never hidden.
- Emergency history (`/api/news/status` → `detector.history`).
- A 4-second heartbeat dot on the card driven by the `NEWS` WS message, and a
  "last new headline HH:MM:SS" stamp so the user can tell "stream alive, no
  new story" from "stream dead".

## 1.8 Errors we had in the news component, cause → fix → detection

- **N-1 One dead source stalled all news.** Sequential awaits, an exception
  escaped `poll_all`. → Per-source try/except + backoff, `gather(...,
  return_exceptions=True)` for RSS, the runner catches and logs. *Detect:*
  `seconds_since_poll` grows while one source shows `error`.
- **N-2 NIV / impact 0.00 on a real wire for weeks** (fine on fixtures). Crypto
  keyword lists; world headlines scored 0. → Macro risk-on/off lists with the
  phrasings wires use, theme impact blended into sentiment. *Detect:* share of
  cached items with `sentiment == 0.0` above ~60 %.
- **N-3 One stale bullish headline held NIV at +0.72 forever.** `/(Σw + eps)`
  cancels the decay. → `/(Σw + W₀)`, `W₀ = 0.5`. *Detect:* NIV constant across
  windows while the newest item ages.
- **N-4 Lock broken eight times an hour.** Substring critical keywords,
  re-fired every 30 s scan, hours-old RSS backlog counted as breaking. →
  Whole-word + market term + ≤ 10 min + fire-once; amber re-lock chip.
  *Detect:* emergencies per hour > 1.
- **N-5 "News irrelevant": immigration, football, celebrity rows** entered
  through world feeds and Google News. → Relevance gate (direct/macro/geo),
  title-scoped NewsAPI query with `NOT (...)`, dropped count shown.
- **N-6 Same story ten times = ten shocks.** → Jaccard duplicates, 1/k
  saturation, 2 h habituation with 90-min reset.
- **N-7 Emergency drill ignored its duration.** → `duration_seconds` 10–900
  honoured; drill events labelled.
- **N-8 403 from RSS hosts.** Anonymous user-agent. → Named UA with a URL,
  `follow_redirects=True`, conditional GET.
- **N-9 NewsAPI returned nothing useful / burned the daily quota by 09:00.**
  60 s polling on the free tier. → Free tier → 15-min backfill role; 429 →
  cool for the day; plan state shown on settings.
- **N-10 Keyed source "configured" but silently failing.** Key never tested.
  → Key ring with per-slot state from the real HTTP result; `/api/agents/
  {provider}/test` style probe on the settings page.
- **N-11 Social sensors that weren't.** Fake Instagram/Facebook connectors. →
  Removed; keyless Reddit/StockTwits + keyed X only; status says "no bearer".
- **N-12 First window had no news.** Pollers started after the first lock. →
  `start()` seeds with `poll_all(force=True)` before the engine's first
  window; synthetic labelled pack if the network is dead.
- **N-13 News engine fine, UI said "waiting"** — a slow WS subscriber was
  dropped on `QueueFull` and never re-added (broadcast bug, not news). →
  Per-client queue with drop-oldest.
- **N-14 Themes fired on the wrong asset** (a crypto hack headline lifting
  gold's index). → Per-asset signed impact in the theme table; `scope`
  (world/crypto) carried on the item.

## 1.9 Tests to copy (`backend/tests/test_round_ae.py`, `test_round_aj.py`, `test_news*.py`)
- A fixture wire of 30 real-world headlines (wars, tariffs, CPI, ETF, hack,
  sport, immigration): relevance keeps exactly the market ones; sentiment of
  the war headline is negative for BTC and positive for gold; the sport row is
  counted in `filtered`.
- NIV with one 25-minute-old headline reads < 0.05; five fresh agreeing
  Tier-1 headlines read ≥ 0.85 of their sentiment.
- Duplicates: two tellings of one story → one contribution; ten headlines on
  one theme → weight ≤ Σ 1/k; habituation halves after 2 h continuous
  presence and resets after a 90-min gap.
- Critical detector: "warning" does not fire, "war" without a market term
  does not fire, a 2-hour-old "war" headline does not fire, the same headline
  fires once.
- `poll_all` with one source raising keeps the others' items and records the
  error string on that source only.
- Free-tier NewsAPI: a 429 cools the slot for ≥ 1 h, not 60 s.
- The `NEWS` ticker publishes every 4 s with unchanged items but changing
  ages/weights (pure arithmetic, no network).

## 1.10 Checklist for the sister app

1. One task per source with its own interval/backoff/state; never call a
   source faster than its documented quota.
2. A 4 s evaluation ticker that re-weights the cache and publishes `NEWS`
   over the existing WebSocket; the UI shows a heartbeat and "last new
   headline".
3. `NewsItem` carries relevance, sentiment, tier, theme, `impacts[asset]` at
   ingest.
4. Theme × asset impact matrix as data (JSON), editable, learned-faded.
5. Per-horizon half-lives (`W/2`) and per-horizon indices; NIV with `W₀`.
6. Five printed paths into the prediction; constant weights only as priors.
7. Critical events whole-word, market-term, recency, fire-once; amber
   re-lock chip; drills with duration.
8. Settings page with the key decision table (with/without each key) and a
   real test button per key.
9. Every "0.00" and every quiet wire has a reason on screen.
10. Tests from 1.9 before anything is called "connected".
