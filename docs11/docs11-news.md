# PROMPT — Rebuild the news component

Paste everything below this line to the agent working on the multi-asset
prediction app (22 / 44 / 130 / 247-minute windows). It is written as
instructions to that agent. Every number in it was measured or decided in a
working engine (DROSOPHILA TRADER v2.0); where your app differs, the rule for
adapting is given.

---

You are rebuilding the **news component** of a multi-asset prediction app
with four window lengths: 22, 44, 130 and 247 minutes. The current component
does not deliver a continuous stream, does not turn headlines into numbers
the predictor can use, does not weight them, and cannot explain what an
optional NewsAPI key changes. Do the following, in this order, and do not
report "connected" until every acceptance test in section 9 passes.

## 1. Separate ingest, evaluation and publishing — then make "every 4 seconds" true for the right layer

Build three layers. Never collapse them.

| layer | what changes | cadence you must use |
|---|---|---|
| **ingest** — fetching headlines | new items arrive | per source: RSS 60–120 s, CryptoPanic 30 s, NewsAPI 60 s on a paid plan / 900 s on the free plan, Reddit/StockTwits 30–60 s, X filtered stream = push (paid bearer only) |
| **evaluation** — re-weighting what is already cached | every item's age decay, novelty, habituation, the per-asset per-horizon impact index, NIV | **every 4 seconds**, pure arithmetic over ≤ 200 cached items, no network |
| **publish** — pushing to the UI | the news card, impact chips, drivers | every 4 seconds over the app's single WebSocket as `{"type":"NEWS","data":…}` plus a REST mirror `/api/news/feed` |

Rationale you must keep in the code comments: no public source emits a new
headline every 4 s; polling a source every 4 s burns NewsAPI's free quota
(100 requests/day = one call per 14.4 minutes) before 09:00, gets you 403/429
from RSS hosts, and still shows a dead stream because nothing new arrived.
The thing that is alive every 4 seconds is the **evaluation**.

## 2. Implement the engine with this exact structure

```python
class NewsEngine:
    async def start(self):
        self._client = httpx.AsyncClient(headers={"user-agent": "<AppName>/1.0 (+<repo url>)"})
        await self.poll_all(force=True)                      # seed BEFORE the first window
        self._tasks += [asyncio.create_task(self._runner()),
                        asyncio.create_task(self._ticker())]  # 4 s evaluation + publish

    async def _runner(self):
        while not self._stop.is_set():
            try:
                await self.poll_all()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = str(exc); log.warning("news poll failed: %s", exc)
            await asyncio.sleep(5.0)                         # loop ticks; each source decides

    async def poll_all(self, force=False):
        now = time.time(); polled = {}
        for src in self.sources:                             # interval, last, backoff, backoff_until, state, last_ok
            if force or (now >= src.backoff_until and now - src.last >= src.interval):
                polled[src.name] = await self._poll_one(src)
                src.last = now
        if not self.cache.all() or now - self.cache.last_poll > 300:
            polled["synthetic"] = self._seed_synthetic()     # labelled; never scored as real
        self._update_status(); await self._scan_for_critical()
        return polled

    async def _poll_one(self, src):
        try:
            items = await src.fetch(self._client, self.settings)
        except src.RateLimited:
            src.backoff = min(src.quota_window, src.backoff * 2)        # NewsAPI free: quota_window = 86400
            src.backoff_until = time.time() + src.backoff
            src.state = f"rate limited · retry in {src.backoff:.0f}s"; return 0
        except Exception as exc:
            src.backoff = min(300, src.backoff * 2); src.backoff_until = time.time() + src.backoff
            src.state = f"error: {str(exc)[:80]}"; return 0
        src.backoff = 5.0; src.last_ok = time.time(); src.state = f"ok ({len(items)})"
        return self.cache.add(items)                         # dedupe + relevance gate inside

    async def _ticker(self):
        while not self._stop.is_set():
            await self.broadcast({"type": "NEWS", "data": self.stream_payload()})
            await asyncio.sleep(4.0)
```

Requirements on that structure:
- One `Source` object per feed with its own `interval`, `last`, `backoff`,
  `backoff_until`, `state`, `last_ok`, `items_per_hour`. One source failing
  must never delay another (RSS feeds themselves are fetched with
  `asyncio.gather(..., return_exceptions=True)`).
- Keyed sources go through a **key ring** with up to 3 slots per provider:
  401 → slot `rejected`; 429 → slot cools for the provider's quota window
  (NewsAPI free: the rest of the day, not 60 s) and the next slot is tried;
  success → slot `ok`; the status string names the slot that answered.
- RSS fetches send a named `User-Agent` with a URL, follow redirects, and use
  conditional GET (`etag`/`modified`) so hosts answer 304 instead of banning
  you.
- `_seed_synthetic` inserts a fixed, clearly labelled headline pack only when
  the cache is empty or nothing arrived for 5 minutes; status then reads
  `coverage: "synthetic"` in red. Synthetic items must never enter a scored
  prediction.

## 3. Sources — implement these, and make the app itself explain the keys

### Keyless, always on
- RSS: the crypto desks (CoinDesk `/arc/outboundfeeds/rss/`, CoinTelegraph
  `/rss`, The Block `/rss.xml`, Decrypt), world (BBC World, Al Jazeera), the
  central banks of your assets' currencies (Fed `press_all.xml`, ECB, RBI),
  CNBC markets, and **Google News RSS search** per asset and per theme
  (`https://news.google.com/rss/search?q=<query>&hl=en-US&gl=US&ceid=US:en`).
  For every asset add: issuer/exchange status feed, a Google News query on
  the name and the ticker, one sector wire.
- Reddit `r/<sub>/new.json` and StockTwits `streams/symbol/<TICKER>.json`
  (buzz per minute vs a 24 h baseline; StockTwits bullish/bearish tags).
- GDELT 2.1 15-minute update files for global geo themes (optional but
  keyless and multi-asset).

### CryptoPanic (optional free key)
Community votes → sentiment, currency tags → relevance, 30 s inside the
5/min free tier. Crypto assets only.

### NewsAPI (optional key) — you must implement *both* behaviours and show the difference
Call `GET https://newsapi.org/v2/everything` with
`q=<per-asset query with NOT (immigration OR football OR celebrity)>`,
`searchIn=title,description`, `sortBy=publishedAt`, `language=en`,
`pageSize=20`. Use `source.name` for the tier, `publishedAt` for the
timestamp, `description` for the lexicon, `url` domain for the fingerprint.

- **Free (Developer) plan**: 100 requests/day and NewsAPI documents a delay on
  new articles for this plan (read their current pricing page at build time
  and print the figure you find in the status line). Therefore: poll every
  900 s, use it as a **backfill and relevance** source (history at boot,
  better tiers/descriptions, query recall for assets without RSS), never as
  the breaking feed. HTTP 426 = plan limit → state `configured · plan-limited`.
- **Paid plan**: poll every 60 s; it is a first-class breaking source.
- Auto-detect: if a call returns 426, or the newest `publishedAt` is older
  than 2 h while RSS has items from the last 10 min, mark the plan as
  `free/delayed` and switch to the 900 s role.

Render this table on the settings page next to the key fields, and compute
the `coverage` string from the live states, not from "key present":

| keys | ingest | coverage string |
|---|---|---|
| none | RSS + Google News RSS + Reddit/StockTwits | `limited — RSS only · add CryptoPanic / NewsAPI for faster desks and query recall` |
| CryptoPanic | + votes, currency tags, 30 s | `good (crypto)` |
| NewsAPI free | + 15-min backfill, descriptions, tiers | `good · NewsAPI free tier delayed — backfill role` |
| NewsAPI paid | + 60 s query-scoped breaking | `full` |
| X bearer | + push stream | `social: streaming` |

Never implement a social source you cannot actually connect (no Instagram/
Facebook stand-ins). Without an X bearer the status must read `x: no bearer`.

## 4. Turn every headline into numbers at ingest — this is the "weightage"

The item record:

```
headline, published_at (epoch s), source, url, provider,
fingerprint = normalised(title) + domain,
relevance_by_asset: {asset: 1.0 | 0.7 | 0.6 | 0.0},
sentiment ∈ [-1, 1], tier ∈ {1, 2, 3}, credibility = {1: 1.0, 2: 0.7, 3: 0.45}[tier],
theme, magnitude ∈ [0, 1], impacts: {asset: signed float}, scope ∈ {"world", "sector", "asset"}
```

Compute in this order:

1. **Relevance** per asset: direct terms (asset name, ticker, its market,
   its exchange, its instruments) → 1.0; macro (central bank, CPI, yields,
   dollar, recession, oil) → 0.7; geo (war, strike, missile, sanctions,
   tariff, blockade) → 0.6; else 0.0. An item with 0.0 for every asset is
   **dropped and counted** (`filtered`, keep 3 examples for the panel).
2. **Sentiment**: `lexical = tanh((bull_hits − bear_hits) / 3)` over a bullish/
   bearish lexicon **plus** macro risk-on / risk-off lists written in the words
   wires use ("strikes", "tariffs on", "yields climb", "drone attack hits",
   "cooler than expected"). Blend 50/50 with the theme impact for the asset
   and squash with `tanh`. CryptoPanic votes override the lexicon when
   present.
3. **Tier**: 1 = Reuters, Bloomberg, AP, FT, WSJ, CNBC, CoinDesk, The Block;
   2 = Decrypt, CoinTelegraph, Blockworks, major national business press;
   3 = everything else.
4. **Theme → signed impact per asset.** Keep this as a JSON data file, not
   code. Seed it with:

   | theme | keywords (whole-word, ordered most specific first) | magnitude | BTC | gold | equity index | USD |
   |---|---|---|---|---|---|---|
   | war / escalation | invasion, airstrike, missile, troops, drone attack, escalates | 0.9 | −0.6 | +0.8 | −0.6 | +0.4 |
   | terror / cyber attack | bombing, explosion, cyberattack, ransomware | 0.7 | −0.5 | +0.5 | −0.4 | +0.3 |
   | exchange / protocol hack | hack, exploit, drained, bridge exploit | 0.8 | −0.7 | +0.1 | 0 | 0 |
   | de-peg / insolvency | depeg, bank run, halts withdrawals, insolvent | 0.9 | −0.8 | +0.4 | −0.3 | +0.2 |
   | sanctions / tariffs | sanction(s), tariff(s), trade war, embargo | 0.6 | −0.35 | +0.5 | −0.5 | +0.3 |
   | hawkish / hot inflation | rate hike, hotter than expected, yields surge | 0.7 | −0.5 | −0.4 | −0.5 | +0.5 |
   | dovish / cut | rate cut, stimulus, inflation cools, pivot | 0.7 | +0.5 | +0.4 | +0.5 | −0.5 |
   | dollar strength | dollar surges, dxy | 0.5 | −0.25 | −0.45 | −0.2 | +0.6 |
   | recession | layoffs, default, bank failure, credit crunch | 0.6 | −0.4 | +0.5 | −0.6 | +0.2 |
   | regulation | ban, crackdown, sues, delisting | 0.5 | −0.45 | 0 | −0.1 | 0 |
   | adoption / ETF / inflows | etf approval, inflows, treasury buys | 0.7 | +0.6 | 0 | +0.1 | 0 |

   Add a column per asset you trade and a row per sector theme (earnings,
   guidance, OPEC, crop report…). These numbers are the **prior** weight;
   step 7 of section 5 makes them learned.
5. **Novelty**: duplicates (token-set Jaccard ≥ 0.6 against already-counted
   headlines) weigh 0; the k-th distinct headline on the same theme in the
   lookback weighs `1/k`; a theme continuously on the wire halves every 2 h
   and resets after a 90-minute gap; a new theme enters at full weight.
6. **Decay per horizon**: `decay_h = 0.5^(age / (W_h / 2))` → half-lives
   11 min (22), 22 min (44), 65 min (130), 123 min (247). Skip items older
   than 12 half-lives of the horizon. Keep **one index per asset per
   horizon**.
7. **Aggregate**:
   `impact_h(asset) = tanh( Σ_i credibility_i · decay_h(age_i) · novelty_i · magnitude_i · impacts_i[asset] · relevance_i[asset] )`
   and keep the top-4 **drivers** per asset per horizon (headline, theme,
   age, novelty, k-th on theme, habituation, contribution).
8. **NIV** (the formula that enters the formula engine, the brain input and
   fusion), five newest relevant items for the asset:
   `NIV = Σ s_j c_j r_j e^(−age_j/τ_h) / ( Σ c_j r_j e^(−age_j/τ_h) + W₀ )`,
   `τ_h = W_h / 4`, `W₀ = 0.5`. The `W₀` is mandatory: with a bare epsilon
   the decay cancels and one stale headline holds NIV forever.
   Also compute `SMD = tanh(NIV − tape_acceleration)` as the news-vs-price
   divergence formula.

## 5. Wire the numbers into the prediction through five printed paths

1. **Formulas**: NIV and SMD are directional formulas with traces (headlines
   weighted, weighted sum, weight sum, W₀, τ) and a double-check that
   re-derives the value from the trace. They get the same per-formula
   hindsight multiplier as every other formula.
2. **Brain input**: NIV is one sensory channel (ON/OFF pair) like any other
   directional input.
3. **Fusion vote**: `impact_h(asset)` is a signed voter with prior weight
   0.10, printed as `news impact −0.31 · war / military escalation · 2
   headlines · newest 6 min`.
4. **Learned inputs**: every theme present this window is an input
   `news:theme:<name>` (value = its signed contribution) to the hindsight
   review, the meta-combiner and the evidence ledger. A theme whose headlines
   do not move an asset is faded by its own record. This is where constant
   weights stop being constants.
5. **Emergency path** (section 6) — the only mid-window change allowed.

Policy: news is a confidence/context modifier by default and sets the
direction only through the emergency path or when the learned layer has
earned it for that theme on that asset. At 130/247 minutes a scheduled
central-bank decision inside the window is the window: shorten `τ_h` to
`W_h / 8` for the 30 minutes around a scheduled release (keep a release
calendar as data).

## 6. Critical events — careful, or you will break your own lock

Triggers:
- **headline**: a critical keyword matched **whole-word** (`\bwar\b`, never
  substring — "war" ⊂ "warning"/"forward"/"award", "hack" ⊂ "hackathon"),
  **and** a market term for one of your assets in the same headline, **and**
  `published_at` within `W_h / 10` (minimum 10 min), **and** Tier 1–2,
  **and** not fired before (fingerprint set).
- **sentiment swing**: |ΔNIV| over the last 90 s above 0.5.
- **flash move**: tape move beyond k·σ inside the window (the tape raises it,
  the same dispatcher handles it).

On an event issue an **emergency re-lock** that carries `computed_at`, keeps
`valid_from/valid_until`, carries the headline and the trigger, and shows as
`⚡ emergency re-lock HH:MM:SSZ` in amber — never as a normal lock. Count
emergencies per hour on the panel; more than one is a bug until proven
otherwise. Provide `POST /api/news/emergency {duration_seconds: 10–900}` for
drills, honour the duration, label drill events.

## 7. API and UI — exactly this, nothing hidden

Endpoints: `/api/news/feed` (latest items with all computed fields),
`/api/news/status` (per-source state, coverage, seconds_since_poll, filtered
counts with examples, detector history), `/api/news/impact?asset=&horizon=`
(indices + drivers), `POST /api/news/poll` (manual kick), `POST
/api/news/emergency`.

News card must show:
- header: `coverage`, `seconds_since_poll`, items cached, `last new headline
  HH:MM:SS`, a heartbeat dot driven by the 4 s `NEWS` message;
- one line per source: `rss ok (41) · cryptopanic not configured · newsapi
  slot 2 rate limited (cooling 812 s) · reddit ok · x no bearer`;
- per asset: four impact chips (22/44/130/247) with sign, theme label,
  headline count, newest age → click opens the drivers;
- `relevance gate dropped n` with three examples;
- emergency history with trigger and headline;
- NIV trace rows under the NIV formula in the formula explorer, and a
  `why 0.00` reason when the wire is quiet.

## 8. Errors we had — do not repeat them

| # | symptom | cause | fix |
|---|---|---|---|
| N-1 | one dead source stalled all news | sequential awaits, exception escaped | per-source try/except + backoff; `gather(return_exceptions=True)` |
| N-2 | impact/NIV 0.00 on a real wire for weeks, fine on fixtures | crypto-only keyword lists; world headlines scored 0 | macro risk-on/off lexicon in wire phrasing; theme impact blended in |
| N-3 | one stale bullish headline held NIV at +0.72 forever | `/(Σw + eps)` cancels the decay | `/(Σw + W₀)`, `W₀ = 0.5` |
| N-4 | lock broken 8× an hour | substring critical keywords, re-fired each scan, hours-old backlog | whole-word + market term + recency + fire-once; amber re-lock chip |
| N-5 | immigration/football/celebrity rows in the wire | world feeds without a gate | relevance gate; NewsAPI `NOT (...)`; dropped count shown |
| N-6 | ten headlines on one story = ten shocks | no novelty model | Jaccard duplicates, 1/k saturation, 2 h habituation, 90-min reset |
| N-7 | emergency drill ignored its duration | parameter unused | honour 10–900 s; label drills |
| N-8 | 403 from RSS hosts | anonymous user-agent | named UA with URL, redirects, conditional GET |
| N-9 | NewsAPI quota gone by 09:00; "nothing useful" | 60 s polling on the free tier; delayed articles | 900 s backfill role on free; 429 cools for the day; plan auto-detected and shown |
| N-10 | keyed source "configured" but silently failing | key never tested | key ring with per-slot state from real HTTP results; test button on settings |
| N-11 | "social sensors" that were not connected to anything | fake connectors | removed; keyless Reddit/StockTwits + keyed X only |
| N-12 | first window had no news | pollers started after the first lock | `poll_all(force=True)` inside `start()`; labelled synthetic pack offline |
| N-13 | engine healthy, UI said "waiting for news" | slow WS subscriber dropped on `QueueFull`, never re-added | per-client queue with drop-oldest |
| N-14 | a crypto hack headline lifted gold's index | one impact per theme | signed impact **per asset** in the theme table; `scope` on the item |

## 9. Acceptance tests — write these first, pass them before saying "connected"

1. Fixture wire of 30 real-world headlines (wars, tariffs, CPI, ETF, hack,
   earnings, sport, immigration): the relevance gate keeps exactly the market
   ones per asset; sport/immigration are counted in `filtered`.
2. The war headline scores negative for BTC and the equity index, positive
   for gold and USD; the ETF headline positive for BTC and 0 for gold.
3. NIV with one 25-minute-old headline at W = 22 min reads < 0.05; five fresh
   agreeing Tier-1 headlines read ≥ 0.85 of their mean sentiment.
4. Two tellings of one story contribute once; ten headlines on one theme
   weigh ≤ Σ 1/k; habituation halves after 2 h continuous presence and
   resets after a 90-minute gap.
5. Decay: the same item has four different weights at the four horizons,
   ordered 22 < 44 < 130 < 247.
6. Critical detector: "warning" does not fire; "war" without a market term
   does not fire; a "war" headline older than `W/10` does not fire; the same
   headline fires once; a Tier-3 headline does not fire.
7. `poll_all` with one source raising keeps the others' items and records
   the error on that source only; `seconds_since_poll` keeps updating.
8. NewsAPI 429 cools the slot ≥ 1 h; 426 sets `plan-limited`; the role
   switches to 900 s.
9. The `NEWS` ticker publishes every 4 s with unchanged items but changing
   ages/weights, with no network call in the tick.
10. `start()` seeds before the first lock: the first `FrozenSnapshot` has
    `news_items` non-empty (synthetic allowed, labelled).
11. Synthetic items never appear in a scored prediction's drivers.
12. The settings page renders the key decision table and a test button per
    key that performs a real request and shows the resulting state.

## 10. Report back in this form

When done, report: per-source intervals and states observed over one hour;
items/hour per source; `filtered` share; the four impact indices per asset
at one instant with their drivers; NIV trace for one window; emergencies per
hour; and the output of the twelve tests. Do not report a win-rate effect —
that is measured by the ledger over weeks, not claimed by the component.
