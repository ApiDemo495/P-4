# Advice for a multi-asset, long-horizon prediction app (22 / 44 / 130 / 247 min windows)

Written from DROSOPHILA TRADER v2.0 (this repo) for a sister app that is new and
"feels useless". Everything below is what we actually did, why it worked, what
broke first, and how to carry it to a slower clock with several assets. Hand it
to the other agent as-is. Paths in `code` point into this repository so it can
read the real implementation.

The one rule above all others: **the app is only as good as its honesty.** No
0.00 placeholders, no faked predictions, no "online" badges on a dead feed, no
win rate you did not measure strict-sign on the real tape. Every problem you
list below (brain, news, emotions, physics, genesis, UI, Gemini) became fixable
only after the panel started printing *why* it was wrong.

---

## 0. First: define the clock, the entry, the settlement — then everything else

Your windows are 22, 44, 130 and 247 minutes. Before writing a single formula:

1. **One clock object owns time** (`backend/core/window_clock.py`,
   `timebase.py`). Windows start on a fixed world-clock grid
   (`floor(now / W) * W`), never "now + W". With four horizons, keep four grids
   and a pipelined scheduler that pre-computes each pick ~2 % of the window
   before its boundary (we use 10.5 s lead for 70 s; for 22 min use ~25 s,
   for 247 min ~2 min) and locks it exactly at :00.
2. **Entry and settlement prices are definitions, not estimates.** For us:
   Binance last trade at :00 and at window end, tie = loss. Write the same
   sentence for every asset in your app and settle *only* on that. If two
   assets settle on different venues, say so per asset.
3. **Signal Lock**: once locked, the prediction never changes mid-window.
   Refresh *live values* (formulas, physics, emotions) for the panel, but the
   locked signal, its confidence, its reasoning and its emotion are frozen
   (`backend/core/signal_lock.py`; see the "lock-safe" notes in
   `cycle_manager.py`: prefetch/refreeze may never write `last_fusion`,
   `emotion_locked`, `conviction_note`, `warnings`, `degradation`).
4. **Scoring** lives in one `CompetitionLedger`-like object
   (`backend/core/competition.py`): strict-sign win rate, streak, best streak,
   ties counted as losses, per-asset and per-horizon. Everything else in the
   app is judged against this ledger. If a feature doesn't move it, say so.

### Scaling constants from 70 s to minutes

Almost every constant in our code is a *window fraction* or a *tape span*.
Map them, don't copy them:

| thing | ours (70 s) | rule | 22 min | 247 min |
|---|---|---|---|---|
| prefetch lead | 10.5 s | ~1.5–2 % of W, ≥ agent timeout + 1.5 s | 25 s | 120 s |
| agent timeout | 9 s | ≤ lead − 1.5 s | 20 s | 90 s |
| live refresh | 0.2–15 s | W / 300 … W / 5 | 5–60 s | 60–300 s |
| tape memory | 600 ticks ≈ 10+ windows | ≥ 10 windows of 1 s/1 m bars | 1 m bars × 220 | 1 m bars × 2470 (or 5 m × 500) |
| formula history for z-scores | 134 windows | ≥ 100 settled windows | 100 | 100 (that is 17 days — persist it!) |
| noise floor of the fused score | 0.5·median\|score\| of last 60 | same | same | same |
| takeover record for learned combiners | 40 graded windows ≥ 53 % | same count, so it takes days — persist | same | same |
| news decay | minutes | ~W/2 half-life per item, repeats decay 3× faster | 11 min | 2 h |

At 130/247 min your sample sizes are tiny (≈ 11 and 6 windows a day). Three
consequences: **persist everything** (state vault, `backend/core/state_vault.py`),
**pool across assets** where the physics is asset-agnostic (z-scored inputs,
sign statistics), and **never claim learning before the counts exist** —
print "n = 14 windows, not enough to say" instead of a percentage.

---

## 1. The brain: how to import a real connectome and actually use it

What "real brain" meant for us: the FlyWire FAFB v783 *Drosophila* connectome,
138,639 neurons / 15,091,983 connections, with the Janelia hemibrain as a
second source, plus a baked 80×80 fallback graph for first boot.

### 1.1 Sources and files (`backend/brain/connectome_data.py`)
- Primary: `philshiu/Drosophila_brain_model` → `Connectivity_783.parquet`
  (pre/post root ids, synapse counts, neurotransmitter). Expect ≥ 90 MB.
- Annotations: `flyconnectome/flywire_annotations` Supplemental file 1
  (cell types, super-classes, sides). Expect ≥ 25 MB TSV.
- neuPrint (hemibrain `hemibrain:v1.2.1`, `male-cns:v0.9`) via its REST API
  with a user token. **Test the token against `/api/dbmeta/datasets`**, not
  `/api/databaseInfo` (gone). Never write the user's token to disk.
- Download with retries, verify **size and magic bytes** (parquet starts and
  ends with `PAR1`), cache under a data dir that is *not* in git, and fall
  back to the baked matrix with a red "FALLBACK" badge — never a silent swap.

### 1.2 Build a usable graph (`matrix_builder.py`, `whole_brain.py`)
- Threshold edges at ≥ 5 synapses (standard in the FlyWire papers).
- Keep sparse matrices (SciPy CSR); 15 M edges fit in ~250 MB.
- Group neurons by annotation super-class so you can address *circuits*
  (sensory → projection → Kenyon cells → MBONs → descending). Our "readout"
  is the valence of the mushroom-body output neurons (MBONs) after **3
  propagation hops** — reading at 6 hops diluted everything to zero
  (that was a real bug, Round AQ).
- Normalise by in-degree; sign edges by neurotransmitter (GABA/glutamate
  inhibitory, ACh excitatory, dopamine as modulator → learning rate).

### 1.3 Feed it market state, not raw prices
Map your inputs onto sensory populations: price velocity, volatility,
order-flow imbalance, news valence, cross-asset spread → distinct input
groups. Run a GCN-style propagation (`graph_convolution.py`), read MBON
valence as the brain's vote in [−1, 1] with a confidence = |valence| scaled
by activation coverage. **Dopamine** (`backend/core/dopamine.py`) is the
learning signal: reward prediction error from settled windows adjusts the
KC→MBON weights only — the connectome itself is frozen.

### 1.4 Verify at startup, every time (`startup_verification.py`, `health_check.py`)
Five steps, each printed on the panel with a tick or a red reason:
file present → bytes/magic ok → graph loads with the expected counts → a
synthetic impulse propagates (non-zero MBON output) → a BUY-biased and a
SELL-biased probe produce opposite signs. If step 5 fails you have a sign
bug; don't ship a vote from it. Expose `/api/brain/status` and
`/api/brain/reconnect`.

### 1.5 In the direction ladder the brain is a **tie-breaker**, not the boss
Our ladder (Round BR): fused formula score beyond a *measured* noise floor →
horizon-matched window formulas → brain vote only if confidence ≥ 25 % →
previous side → BUY. The brain is a slow, nonlinear, learned integrator;
on its own it was ~50 %. Put it where it helps: breaking ties and
modulating confidence.

---

## 2. News: continuous flow, not a poll that dies

What went wrong for us first: a single source, a 1-minute poll that silently
stopped on the first HTTP error, no de-duplication, no decay, "war/attack"
headlines treated like earnings.

### 2.1 Architecture (`backend/news/news_engine.py`)
- **Many sources behind one interface** (`rss_source.py`, `newsapi_source.py`,
  `cryptopanic_source.py`, `social_source.py`). Each source: its own poll
  interval, its own backoff (exponential, capped, with jitter), its own
  "last ok / last error" shown on the panel. One dead source must never
  stall the engine → `asyncio.gather(..., return_exceptions=True)`.
- Poll every 30 s (fast sources) and 60–120 s (RSS); for 247-min windows you
  still poll every 30–60 s — news arrives when it arrives.
- **Fingerprint** each item (normalised title + domain) → dedupe. A repeated
  story keeps its first timestamp and gets a *decayed* weight
  (`impact.py`: repeats decay 3× faster than fresh items).
- **Relevance per asset** (`relevance.py`): keyword/entity sets per asset
  (BTC, gold, equities, FX…) plus a general-macro set that hits every asset
  (rates, war, sanctions, exchange outage). Score = lexicon sentiment ×
  relevance × source trust × time decay.
- **Critical events** (`critical_event_detector.py`): wars, attacks, hacks,
  de-pegs, exchange halts, regulatory bans. These bypass the normal weight and
  raise a 6-level degradation / emergency state that the fusion sees. Print
  them in red with the source link.
- **Decay**: `w(t) = w₀ · 2^(−age / half_life)` with half-life ≈ W/2 per
  horizon. One item can have four different weights, one per horizon.
- Expose `/api/news/poll` (manual kick), `/api/news/feed`, and show on the
  panel: items/hour per source, last success time, queue depth. A news
  engine with no visible heartbeat is assumed dead by the user — correctly.

### 2.2 What to feed downstream
A single **News Impact Vector (NIV)** per asset per horizon in [−1, 1] with
its own confidence (count-based), plus the top-3 headlines and their weights
for the panel. Everything downstream (fusion, emotions, brain) reads NIV; no
one parses headlines twice.

---

## 3. Emotions: measured from the tape, locked with the window

What was broken for us: emotions were derived from the signal (circular),
flipped mid-window, read "complacency" during violent moves, and showed 0.00.

### 3.1 Design (`backend/core/emotions.py`)
Nine emotions, each a **continuous score from measurable tape features**:
FEAR, PANIC, CAPITULATION, DENIAL (negative), HOPE, EUPHORIA, FOMO (positive),
COMPLACENCY, PARALYSIS (neutral). Features per asset: realised volatility vs
its own history, drawdown from window high, volume climax (recent volume /
baseline median), aggressor skew, run lengths of same-side prints, tick rate
vs baseline, price staying at one point (PARALYSIS), spread widening, news
NIV. Every mapping is a **logistic ramp** between two measured anchors
(`_ramp(x, lo, hi)`), never a hard threshold → no dead zeros, no flicker.

### 3.2 Rules that mattered
- **Quote prints** (rows with qty 0 / side 0, if your feed emits mid updates
  between trades) must be excluded from volume and run statistics or the
  crowd reads "flat" forever (our Round BT bug — see
  `_herding` and `FrozenMarketSnapshot.trades()`).
- COMPLACENCY requires *low* volatility relative to history **and** low
  volume climax; gate it by `1 − PANIC` so it cannot coexist with violence.
- The dominant emotion is **locked at the window boundary** along with the
  signal (`emotion_locked`); the live panel shows the live mix separately,
  labelled "live" vs "locked".
- Emotions are **valued below formulas** in fusion (a modulator on
  confidence, ±15 %), never the direction.
- Show on the panel: each emotion's score bar, the feature values that
  produced it (volatility z, climax ratio, skew, run length), and the
  sample size (`ticks`, `trade_prints`, `quote_prints`).

### 3.3 Long horizons
At 22–247 min, compute the features on 1-minute bars (OHLCV + buy volume if
your venue provides taker-buy volume; Binance klines do) over the last
N = 3·W minutes, and the baselines over 30·W. The same nine ramps work; the
anchors must be re-measured per asset (gold's "panic" is BTC's "calm").

---

## 4. Physics layer: real mechanisms, each with its own validity gate

Ours (`backend/physics/`): eleven mechanisms — OU mean reversion, VPIN
(order-flow toxicity), Hawkes self-excitation, kinetic energy of price,
entropy of the sign series, diffusion (variance ratio), pendulum (oscillation
phase), temperature (volatility of volatility), Avellaneda–Stoikov spread,
venue fragmentation, PAXG peg — blended by a **ridge-Kelly** weight on their
trailing signed edges (`unified.py`), with a thermodynamic band that clamps
weight changes per window. Round BU added four estimators that each print
their derivation: Kramers–Moyal drift/diffusion, Ising chain on aggressor
signs, fluctuation–dissipation (Kyle λ vs price variance), maximum-entropy
window distribution (`backend/core/physics_estimators.py`).

Rules:
- Every mechanism returns `{covered, score, logic[], why}`; `covered=False`
  with a reason when the tape can't support it. The panel lists active and
  inactive mechanisms *with the reason*. "Physics 0.00" with no reason was
  the single most-reported complaint.
- Scales come from the tape, not from constants: normalise by the asset's
  own realised σ over the horizon, by its own flow variance, by its own
  spread. A constant floor that was tuned on BTC/70 s will pin gold/247 min
  at zero (our `SelfScale` floor bug: floors must be code constants *and*
  sized to the data; never restore a floor from saved state).
- For long horizons use the 1-minute bar series: OU half-life in minutes,
  Hawkes on bar-level signed volume, VPIN with volume buckets of ~W/50.
- The layer's single output is one signed vote with a weight that the
  learned combiner (section 8) earns, plus the per-mechanism table for the UI.

---

## 5. Genesis layer: generated formulas under a hard fitness regime

Ours (`backend/genesis/`): ten mathematical domains (signatures, homology,
information geometry, optimal transport, fractional calculus, tropical
algebra, quantum-inspired, p-adic, category-theoretic, algorithmic) each
producing candidate formulas from a typed feature set, scored on seven
metrics (strict-sign hit rate, information coefficient, stability across
regimes, turnover, correlation to existing formulas, latency, sample size)
with a lifecycle: candidate → probation → active → retired, plus an
**autopsy** for every retirement. NumPy only — say so.

Rules that kept it honest:
- Fitness is **out-of-sample**: generate on the first 60 % of history, score
  on the last 40 %, then walk forward. Anything scored in-sample is a toy.
- A candidate must beat the *median existing formula* on IC and be
  < 0.6 correlated with everything active, else it adds noise.
- Cap active genesis formulas (we use 12) and give the whole layer one vote
  in fusion whose weight is earned (section 8).
- Run generation in a **worker thread with low priority** and never on the
  lock path; the Python GIL made our first version stall the countdown.
- Show: pool counts per state, next generation time, the top formula's
  definition, its 7 metrics, and its history. `/api/genesis/status`,
  `/api/genesis/formulas`, `/api/genesis/domains`.

---

## 6. UI: the rules that made ours usable (and the toggle you can't get right)

- **One page, one port, one WebSocket, exactly one `setInterval`.** All
  periodic work hangs off that one tick (we even guard it with a test).
  Multiple timers = drift, double renders, and the "glitchy countdown".
- **Countdown from the server clock**, not the browser's: the WS sends
  `server_now_ms` and `window_end_ms`; the client renders
  `window_end − (server_now + (Date.now() − received_at))`. Re-anchor on
  every message. Never count down from a locally computed start.
- **Fixed layout**: every widget has a fixed slot and fixed height; content
  changes inside it. Nothing may cover the signal/price strip ("never cover
  the screen"). No layout shifts when data arrives.
- **Every number has a logic line.** Each formula card has value, reading,
  the traced intermediates, "computed in µs", the data span used, its own
  history z-score, and a double-check that re-derives the value from the
  trace. Expandable, collapsed by default.
- **Status honesty**: a feed chip reads `live · 9.1 ticks/s · binance` or
  `stale 14 s` or `down: <error>`; never "online" by default. Same for each
  agent, the brain, news sources, Gemini keys.
- **Toggles** (the thing you said it can't do):
  1. a toggle is state on the server (`/api/settings`), not in the DOM;
  2. the client renders from server state *after* the POST returns, with the
     control disabled during the round trip;
  3. optimistic flip only if you also revert on error and show the error;
  4. the toggle's effect is printed next to it ("persist: ON — vault writes
     every 30 s", "simulator: OFF — real feed required") so the user sees it
     did something;
  5. keep the `checked` attribute and the label in sync from one render
     function; never set `checked` in two places;
  6. toggles that change clocks (horizon selection) apply **at the next
     boundary**, and the UI says "applies at :00".
  Our `/api/settings/effects` + `refreshEffects()` pattern does exactly this.
- **Haptics/animation**: tiny, under 200 ms, only on lock and on settle.
- Horizons side by side: one row per horizon (22/44/130/247) for every
  prediction, each with its own countdown, pick, confidence and strict-sign
  record. A multi-asset app needs an asset selector that changes only the
  *displayed* asset; all assets keep computing.
- Keep a **premium but calm** style: dark, two accent colours (up/down),
  monospace numbers, tabular figures, no gradients over text.

---

## 7. Gemini (and any model API): connect properly, use it for your own work

What we got wrong first: prefix-checking keys ("AIza…" is not a validity
test), hard-coding a model id that the key could not use, one key for
everything, no timeouts, blocking the lock on the HTTP call.

### 7.1 Keys (`backend/agents/keyring.py`)
- A **key ring** per provider: up to N slots, each with its own state
  (`untested / ok / rate-limited / invalid / error`), last error text, cooldown.
- Test a key only by calling the API: `GET /v1beta/models` (ListModels). The
  returned list is also **the only truth about which models this key may
  use** — pick from it (`choose_model`: pinned id if available, else the best
  text-decision model by a ranked family list; never `-vision`, `-embedding`,
  `-tts`, `-image`).
- Rotate on 429/403, cool down the slot, keep going on the next slot.
- Keys live in the user's env/secrets and in memory. **Never write a user key
  into a file in the repo**, never echo it in logs, never show more than the
  last 4 chars.
- Expose `/api/agents/{name}/test` that returns the chosen model and the
  round-trip time; the settings page shows it.

### 7.2 Calls
- One structured prompt: the frozen state (prices, formula values with
  readings, physics votes, emotions, top news with weights, horizons), the
  exact question ("higher at window end than at start, yes/no, confidence,
  three reasons"), and the required JSON schema. Ask for `responseMimeType:
  application/json`.
- `asyncio.wait_for` with a deadline that ends **before** the lock; a late
  answer is discarded and logged as "late by X s", not applied.
- Cache per (asset, horizon, window) — never call twice for the same lock.
- Treat the answer as **one agent vote** with the weight it has earned
  (section 8). An LLM's raw directional accuracy on price is ~50 %; its value
  is reading news/regime context into a confidence modifier.

### 7.3 Using models "for its own use"
Let the app use the same key ring for its own chores: summarising the
hour's hindsight into a sentence, explaining why a formula was damped,
classifying a headline when the lexicon is unsure. Same ring, same tests,
same timeouts, lowest priority.

### 7.4 Local models
If you ship a local model (we bake Qwen2.5-0.5B/1.5B GGUF and the jevos
decision model): validate uploads by **magic bytes** (`GGUF`, ONNX protobuf
header), size, and a test prompt; run the server out of process with
`nice 19` / `ionice -c 3`; keep thread count = cores − 1; never put
`llama-cpp-python` on the critical install path (sdist-only on PyPI).

---

## 8. Fusion, learning and honesty — the part that makes the rest matter

- **Fused score** = weighted sum of the directional formulas' values; HSI/ERC-
  style regime indicators modulate but never vote. Weights: brain 0.40 /
  decision model 0.30 / Gemini 0.15 / local 0.15 as a *prior*; the measured
  record overrides it.
- **Noise floor**: the sign of a fused score inside
  `max(0.02, 0.5·median|score| of the last 60 windows)` is noise → go down
  the ladder (section 1.5).
- **Hourly hindsight** (`backend/core/hindsight.py`): every formula, agent,
  feature and system is re-graded on the hour that just happened
  (z of its signed edge); multipliers `clip(1 + 0.25 z, 0.25, 1.5)`,
  inversion only at z ≤ −3 with ≥ 16 decided windows. Changes apply at the
  next boundary and are printed ("damped CCS ×0.6: 9/22 last hour").
- **Meta-combiner** (`meta.py`): ridge logistic over all inputs, walk-forward,
  exponentially weighted (half-life 120 windows), regime experts (BOCPD
  change-point age, session, volatility) used only where their record beats
  the global fit; takes the side only with a 40-window record ≥ 53 % and
  ≥ the recipe. Platt-calibrate the probability (slope only).
- **Unified evidence ledger** (`everything.py`): every source as evidence in
  nats, correlation shrinkage `k_eff = k/(1+(k−1)ρ̄)`, predictability
  `1 − H₂(P)` printed live so the user sees when a window carries no
  information.
- **Forward expectancy** per formula per horizon: the table of "when this
  formula read x, the window went up y % of the time" — this is also how you
  make a formula designed for 70 s speak at 22 min: you don't rescale the
  formula, you re-learn its forward table at that horizon.
- **Temporal twin** (`temporal_arb.py`): price a window as one leg vs
  segmented legs and show the volatility-decay spread; at 247 min a
  130 + 117 split is a legitimate experiment — but settle *only* the real
  contest leg.
- Win-rate honesty: strict sign, ties = losses, counted from the ledger only,
  per asset and per horizon, with n printed. Realistic range for any of this
  at short horizons is 51–56 %; longer horizons add drift and news but your
  n collapses. Never build toward a target percentage.

---

## 9. Operations: the things that made it "permanently run and really learn"

- **State vault** (`state_vault.py`): atomic JSON writes every 30 s of
  formula states, hindsight rows, meta/ledger fits, outcomes, horizons,
  temporal twin; restore on boot with max-age checks and walk-forward
  record rebuild. Reconnects must never lose history.
- **Watchdog**: a loop that misses 2.5 periods dumps stacks and restarts the
  cycle loop; a second stall within 10 periods exits non-zero so the
  supervisor restarts the process.
- **Broadcast queue** per client with drop-oldest; a slow browser must not
  stall the engine (our `QueueFull` bug).
- **Degradation levels** (6): feed stale → agents late → brain fallback →
  news dead → simulator → emergency. Each level is printed and each limits
  what may claim to be "live".
- Simulator / stub agents / synthetic self-tests are **core**, not shameful:
  keep them for first boot and tests, label them loudly, and never let
  synthetic data touch a scored prediction.
- Zero-command start: the dev container provisions, downloads requirements
  with retries, starts the engine under a lock (`flock`), self-updates
  fast-forward only, and prints every URL. README: every command in its own
  copyable box, no comment lines inside boxes.

---

## 10. A concrete migration plan for your app (in this order)

1. Clock + lock + ledger (section 0). Nothing else until the strict-sign
   record for each horizon is real and visible.
2. Tape: 1-minute bars per asset (plus trades if the venue streams them),
   quote prints separated from trades, feed chips honest.
3. Ten core formulas per family (microstructure, trend/persistence, regime,
   cross-asset, news), each with trace + double-check + forward table per
   horizon. Re-measure every anchor/floor per asset from the data.
4. Emotions from tape features (section 3), locked with the window.
5. Physics mechanisms with validity gates (section 4).
6. News engine with multiple sources, dedupe, decay, critical events
   (section 2).
7. Brain import with the five-step verification; brain as tie-breaker
   (section 1).
8. Agents behind a key ring with real tests, timeouts and caching
   (section 7).
9. Hindsight → meta-combiner → unified ledger, each earning its takeover
   (section 8).
10. Genesis last — it only helps once 1–9 produce honest history to score
    against (section 5).

And at every step the UI prints the sample size, the reason for every
"not available", and the logic behind every number. That is the whole
difference between an app that "feels useless" and one you can trust at
second 0.
