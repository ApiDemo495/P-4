# Advice for a multi-asset, long-horizon prediction app (22 / 44 / 130 / 247 min)

Written from DROSOPHILA TRADER v2.0 (this repository) for a sister app that is
new and "feels useless". It is a complete transfer of what we built, how we
built it, what broke — every error we hit from the first boot to today, with
its cause and its fix — and how to carry all of it to a slower clock with
several assets. Hand it to the other agent verbatim. Paths in `code` point into
this repository so it can read the real implementation instead of guessing.

The one rule above every other rule: **the app is only as good as its honesty.**
No 0.00 placeholders, no faked predictions, no "online" badge on a dead feed,
no win rate you did not measure strict-sign on the real tape, no "learning"
claimed before the sample exists. Every problem you listed (brain, news,
emotions, physics, genesis, UI, Gemini, toggles) became fixable for us only
after the panel started printing *why* it was wrong. Build the "why" first.

Contents

- Part A — Architecture, in the order you must build it (sections 0–10)
- Part B — The error catalogue: every failure we had, cause, fix, how to detect it in your app (section 11)
- Part C — Checklists, test guards and the migration plan (sections 12–14)

---

# Part A — Architecture

## 0. The clock, the entry, the settlement, the lock, the ledger

Nothing else matters until these five are right. We lost weeks because
formulas were tuned while the clock was wrong.

### 0.1 One clock object owns time
`backend/core/window_clock.py`, `backend/core/timebase.py`,
`backend/core/clock.py`.

- Boundaries are **world-clock grid points**: `start = floor(now / W) · W`
  (epoch seconds), never "now + W". Every restart lands on the same grid, so
  a window that was open before a crash is the same window after it.
- Keep **one grid per horizon** (22, 44, 130, 247 min). They do not share
  boundaries except at their LCM, and that is fine — each horizon is its own
  contest. Do not try to make the shorter window "roll into" the longer one
  (we tested overlapping lanes and rejected them: the user wants exactly one
  pick per boundary per horizon).
- NTP offset: measure once at boot and every 10 min (`ntplib`), apply it to
  the server clock you broadcast. A Codespace clock can be 1–3 s off; at 70 s
  that is visible, at 22 min it is the difference between locking at :00 and
  locking in the previous window.
- The **pipelined scheduler** (`cycle_manager._pipelined_loop`) prepares the
  next pick during the current window:
  - `prefetch` at `boundary − lead` (we use 10.5 s at 70 s): freeze a market
    snapshot, run formulas, physics, emotions, call agents with a deadline;
  - `refreeze` at `boundary − 0.4 s`: re-snapshot the tape (the last seconds
    matter), re-run only the cheap stages;
  - `open_window` at `boundary`: lock, publish, start the settlement timer.
  - **Nothing the prefetch computes may touch the screen** until the
    boundary (see error B-3).
- Rule for leads on your clock: `lead ≈ agent_timeout + 1.5 s + formula pass`,
  and never more than ~2 % of W:

| | ours (70 s) | 22 min | 44 min | 130 min | 247 min |
|---|---|---|---|---|---|
| prefetch lead | 10.5 s | 25 s | 40 s | 90 s | 120 s |
| agent timeout | 9 s | 20 s | 30 s | 75 s | 90 s |
| refreeze lead | 0.4 s | 2 s | 2 s | 5 s | 5 s |
| live-value refresh | 0.2–15 s | 5–60 s | 10–60 s | 30–120 s | 60–300 s |
| countdown pulse | 1 s | 1 s | 1 s | 1 s | 1 s |

### 0.2 Entry and settlement are definitions
For us: *entry = Binance BTC/USDT last trade at :00, settlement = last trade at
window end, tie = loss*. That sentence is in the code
(`backend/core/competition.py`), in the README and on the panel.

For your app write the same sentence **per asset**: which venue, which field
(last trade, mid, mark price, index), which timestamp rule (first print at or
after :00). If an asset's venue is different from the one your tape comes
from, say so on the panel; otherwise the user's scoreboard and yours will
disagree and the app "feels wrong" for a reason nobody can find.

### 0.3 The Signal Lock Protocol (`backend/core/signal_lock.py`)
- A `FrozenSignal` is immutable: side, confidence, price at lock, horizon
  predictions, levels, reasoning text, locked emotion, the frozen snapshot id,
  `computed_at`, `valid_from`, `valid_until`.
- The lock holds for the whole window. Live panels may refresh *live values*
  (formula readings, emotions, physics) but must label them "live" and keep
  the locked block separate.
- The only documented mid-window change is an **emergency re-lock** (critical
  event), and it is shown as "⚡ emergency re-lock HH:MM:SSZ" in amber, never
  as a normal lock. We got this wrong twice (errors B-1, B-2).
- Persist the current lock (`state_vault`) so a restart inside a window
  restores the same pick instead of issuing a new one.

### 0.4 The ledger (`backend/core/competition.py`)
- Strict sign: `win = sign(settle − entry) == side`, `tie → loss`.
- Per asset × per horizon: wins, losses, ties, strict-sign rate with `n`,
  current streak, best streak, milestone ladder if your contest has one.
- Everything the app claims about accuracy comes **from this object only**.
  The hindsight reviewer, the combiners and the UI read it; nothing writes a
  rate anywhere else. We had three different "accuracies" on screen before
  that rule (error G-1).

### 0.5 Sample-size reality at your horizons
Per asset per day: 22 min ≈ 65 windows, 44 min ≈ 33, 130 min ≈ 11,
247 min ≈ 6. Our learners want 40 decided windows before a first fit and
100+ before z-scores mean anything. At 247 min that is a week and a month.
Consequences:
1. persist everything, always (section 9);
2. pool across assets where the maths is asset-agnostic (z-scored inputs,
   sign statistics, calibration slopes) — keep a per-asset table *and* a
   pooled table, and let the pooled one decide until the per-asset `n` exists;
3. print `n` next to every learned number; print "not enough windows to
   say" instead of a percentage below `n = 20`.

---

## 1. The tape: what the formulas actually see

Everything downstream is only as honest as the tape.

### 1.1 Feeds and the hub (`backend/data/market_hub.py`, `binance_ws.py`, `kraken_ws.py`, `gemini_ws.py`, `coingecko.py`, `simulator.py`)
- One `MarketDataHub` per process; **all data enters through
  `hub.ingest(source, asset, row)`**. A write from a feed that is not the
  active source is rejected and *counted* (shown in `/api/feeds/diagnose`).
- Source order and reconciliation: primary WS → secondary WS → REST poller →
  simulator. A real feed gets a grace period (45 s for us; use 2–3 min at
  your horizons) before the simulator may start; **the simulator is cancelled
  the instant a real feed is healthy**; any source change **flushes the tape**
  (error D-1 — two feeds writing one tape was our worst early bug).
- Evidence beats state: if a source is delivering, adopt it even if the state
  machine says "none" (error D-2).
- Record `last_delivery[source]`, ticks/s, rejected writes, reconnect count;
  show them in a feed chip: `live · binance · 9.1 ticks/s`, `stale 14 s`,
  `SIMULATED TAPE` (red), `connecting…`. Never a bare "online".
- Binance is geo-blocked from US regions where Codespaces run: rotate to
  `data-stream.binance.vision` on refusal (error D-3). Kraken v2 and Gemini are
  reachable from the US.

### 1.2 Rows and buffers (`backend/data/ring_buffer.py`, `backend/core/frozen_snapshot.py`)
- Tick row = `[time_ms, price, qty, side]` (side ∈ {+1, −1}, aggressor).
- **Quote prints** (mid updates when no trade arrives) are stored as
  `qty = 0, side = 0`. They are needed for price paths on thin tapes but they
  **must be excluded** from every volume/aggressor statistic
  (`snapshot.trades(asset)`, `_util.real_trades`). Our Round BT: on gold
  (PAXG) every venue is thin, quote prints were 60–80 % of rows, and every
  hedge formula read 0 for weeks (error F-9).
- For 22–247 min windows your primary series is **1-minute bars**
  (OHLCV + taker-buy volume — Binance klines carry `taker_buy_base_volume`,
  which gives you an aggressor imbalance without trades). Keep
  `30 · W` minutes of bars per asset (247 min → ~5 days → 7,400 bars) and
  trades for the last `2 · W` if the venue streams them.
- `FrozenMarketSnapshot` is a `NamedTuple` built at prefetch/refreeze: ticks
  per asset, books (current + previous for absorption), candles, spreads,
  news items, past outcomes, source, warnings. Formulas read the snapshot,
  never the live hub, so a pass is reproducible.

### 1.3 Candles roll from the tape (error F-2)
Only the Binance kline stream filled our candle buffer; every other feed left
it at warm-up contents forever, so volatility and levels froze. Every feed must
roll its own bars from the tape (`MarketDataHub._roll_candle`), with the
venue's own klines authoritative when present.

### 1.4 Bootstraps
On a fresh start you have no history: pull `W · 30` minutes of 1-minute bars
from the venue REST (`/api/v3/klines`, Kraken `OHLC`, Gemini `v2/candles`),
and feed them through the same formula path so the first window already has
z-scores. We call it the warm start (`backend/core/warm_start.py`) and an
**opening audit** that replays the last hour candle by candle before the first
lock and prints where the app would have been wrong.

---

## 2. Formulas: families, traces, double checks, forward tables

We run 31 core formulas in 9 categories plus a learned-outcome gate (DRG),
each ≤ 1 ms, NumPy only. The exact list is in `backend/formulas/engine.py`
and `README.md §9`; the categories matter more than the names:

| category | ours | what it measures | at 22–247 min use |
|---|---|---|---|
| A microstructure | VSD, SED, AFPR, TAI, BAR | volume shock, spread elasticity, aggressor persistence, tape acceleration, book absorption | per-bar taker-buy share, volume vs 30·W median, spread from best bid/ask snapshots |
| B flow / information | VPIN-style, Hawkes, entropy, LCS | toxicity, self-excitation, order of the sign series | VPIN with volume buckets ≈ W/50, Hawkes on bar-level signed volume |
| C hedge / cross-asset | HSI, HRDD, SHRP, GCDV | BTC–gold stress, beta break, co-movement, divergence | your asset pairs: index vs stock, BTC vs ETH, gold vs USD |
| D regime | RSV (Hurst velocity), VSS, ERC | persistence change, volatility regime, efficiency | DFA on `W/1 min` returns, scales 4–64 |
| E momentum / persistence | MPS, TWRS, MCPE | autocorrelation × efficiency, recency-weighted skew | lag 1–3 bar autocorrelation (expect negative = bounce) |
| F filters | DSKD (fast/slow Kalman) | level shift vs window move | q per bar; score against 3σ of the divergence (error F-11) |
| G news | NIV, DGW | impact vector, geopolitical weight | same, with decay per horizon |
| H brain / learned | CCSv2, KCAE, DRG | circuit consensus, Kenyon-cell activity, dopamine gate | same |
| I window-designed | PIN, MPL, RBB, RAR, WDR, PRT, FIH, LAI, ANA | tie risk, path length, range breakout, run, drift, conditional lean, information half-life, last-impulse, analogue | these are *horizon-matched*: re-implement them on your horizon's bar series, they decide ties |

Rules we learned the hard way:

1. **Every formula returns a continuous score in [−1, 1]** through
   `tanh(x / scale)` where `scale` is **measured from the asset's own data**
   (`SelfScale`: EMA of |x| with a floor). Hard thresholds produced dead
   zeros and flicker (errors F-5, F-6).
2. Floors are code constants, sized to the data, never restored from saved
   state (error F-10). A floor tuned on BTC/70 s pins gold/247 min at zero.
3. **Trace everything** (`trace(ctx, label, value, unit)`): inputs seen,
   intermediates, the scale used, the materiality gate, the reading, time in
   µs, data span, own-history z-score, trend over the last 20 windows. The UI
   shows it under each formula. Debugging was impossible before traces.
4. **Double check**: each formula declares a formula string and a
   re-derivation from its traced intermediates
   (`DOUBLE_CHECK = "value = tanh(z/3) x sign x gate"`, `double_check(trace)`),
   plus a replay on the same snapshot and a range check. "✓✓ verified" or a
   red reason.
5. **Self-test tapes** (`backend/formulas/selftest.py`): BULL, BEAR, FLAT,
   STRESS, IMPULSE_UP/DOWN, CYCLE_UP/DOWN synthetic tapes spanning exactly
   one window; each formula states its claim ("positive on IMPULSE_UP,
   |value| ≤ 0.35 on FLAT") and the suite enforces it. This caught sign bugs
   and the DSKD saturation in minutes.
6. **Zero reasons**: when a formula cannot compute it returns 0 *with a
   reason* (`FormulaResult.zero_reasons`: "needs 40 trades, had 12") and the
   panel shows an amber chip. "Computed, genuinely flat" is also a reason.
7. **Forward expectancy tables** (`backend/core/forward.py`): per formula ×
   per horizon, the empirical P(up | reading bucket) with `n`, settled
   strict-sign. **This is how a formula built for one horizon speaks at
   another**: you do not rescale the formula, you re-learn its forward table
   at 22, 44, 130 and 247 min, and the fused score uses the forward
   P(up) − ½, not the raw value, once `n ≥ 30` per bucket.
8. **Superposition**: for each formula show its current value *and* its
   forward-looking value at every horizon side by side (four columns on your
   UI). The user asked for this explicitly and it is the clearest way to see
   what each horizon is trading on.
9. **Flat watch** (`cycle_manager.flat_watch`, Round BT): share of the last 50
   locks each formula spent inside ±0.02, with the trace line that explains
   it and the tape's quote-print counts. "Most of them are flat" becomes a
   list with reasons. Exclude probabilities (tie risk) whose honest reading
   is near zero.
10. Non-directional indicators (stress, efficiency, tie probability) **never
    vote**; they modulate confidence and route regimes.

---

## 3. The brain: importing a real connectome and using it honestly

"Real brain" for us = the FlyWire FAFB v783 *Drosophila* connectome
(138,639 neurons, 15,091,983 connections) with Janelia hemibrain as second
source and a baked 80×80 fallback for first boot.

### 3.1 Sources, download, validation (`backend/brain/connectome_data.py`)
- `philshiu/Drosophila_brain_model` → `Connectivity_783.parquet`
  (pre/post root id, synapse count, neurotransmitter). Must be ≥ 90 MB.
  Primary URL raw.githubusercontent, secondary the GitHub contents API
  (returns base64 for files under 100 MB — handle both).
- `flyconnectome/flywire_annotations` Supplemental file 1 → cell type,
  super-class, hemisphere. Must be ≥ 25 MB.
- neuPrint hemibrain (`hemibrain:v1.2.1`, `male-cns:v0.9`) via REST with a
  user token. **Token test = `GET /api/dbmeta/datasets`** with
  `Authorization: Bearer`. `/api/databaseInfo` no longer exists (error A-5).
- Validate by **bytes and magic** (parquet starts and ends with `PAR1`,
  TSV header contains the expected column names), retry 3× with backoff,
  cache under a data directory that is git-ignored, and on failure load the
  fallback with a **red FALLBACK badge** and the reason — never silently.
- Never store the user's token in the repo, `.env`, or logs; keep it in
  memory and in the platform's secrets.

### 3.2 Building the graph (`matrix_builder.py`, `whole_brain.py`, `spectral_cluster.py`)
- Edge threshold ≥ 5 synapses (the FlyWire papers' convention). ~15 M edges
  fit in a SciPy CSR in ~250 MB; load lazily in a worker thread so the port
  opens before the brain is warm (error E-4).
- Row-normalise by in-degree; sign edges by neurotransmitter (ACh +, GABA −,
  glutamate −, dopamine/octopamine as modulators). Max-abs normalise so the
  convolution gain is stable; **calibrate the gain once per matrix** so a
  unit input gives a unit-scale readout (deviations 22-A/22-C in
  `docs/SPEC_NOTES.md`).
- Group neurons by super-class into addressable populations: sensory →
  projection neurons (PN) → Kenyon cells (KC) → mushroom-body output neurons
  (MBON) → descending; lateral horn for innate valence.
- **Readout at hop 3**, the MBON layer. Our first whole-brain readout
  averaged all six hops and diluted everything to zero (error A-1).

### 3.3 Feeding it market state (`graph_convolution.py`, `brain.py`)
- Map each directional formula to **one PN channel with an ON/OFF
  glomerular pair** (positive goes into ON, negative into OFF). Rectified
  fan-in otherwise hears only the bullish half (error A-2).
- **Calibrate polarity per channel** at boot: probe each channel with +1
  alone and orient it so positive = approach. Nine of our twenty channels
  were wired with the wrong valence (error A-3).
- **Push-pull readout**: propagate `x` and `−x`, `approach = (approach(x)
  + avoid(−x))/2`, so the score is an odd function of the input and the
  resting offset vanishes (deviation 22-D).
- Non-directional inputs enter at zero magnitude (error A-4).
- The vote `CCSv2 ∈ [−1, 1]`, confidence = |vote| scaled by activation
  coverage; `KCAE` (pre-ReLU Kenyon-cell drive) is a separate "how much is
  the circuit excited" gauge.
- **Dopamine** (`backend/core/dopamine.py`): reward-prediction error from
  settled windows updates only KC→MBON weights (the only plastic synapses in
  the biology too); the connectome is frozen. Learning rate is tiny and the
  update is applied at boundaries.

### 3.4 Startup verification, every boot (`startup_verification.py`, `health_check.py`)
Five steps on the panel with ✓ or a red reason: files present → bytes/magic
ok → graph loads with expected counts → synthetic impulse propagates
(non-zero MBON output) → a BUY-biased and a SELL-biased probe give opposite
signs. Expose `/api/brain/status`, `/api/brain/explain`, `/api/brain/wiring`
(formula → channel), `/api/brain/reconnect`.

### 3.5 Where the brain sits in the decision
A slow, nonlinear, learned integrator is ~50 % on its own. In our ladder
(section 8.2) it is a **tie-breaker when it is ≥ 25 % confident** and a
confidence modulator. Do not make it the decider; do show its reasoning
(top-activated circuits, which channels drove the MBON valence) because the
user wants to *see* the wiring — a brain nobody can see is assumed fake.

---

## 4. News: a continuous flow that is visibly alive

### 4.1 Engine (`backend/news/news_engine.py`)
- Sources behind one interface: RSS (a dozen wires: Reuters/AP/FT world,
  CoinDesk, The Block, central banks, exchange status pages), NewsAPI,
  CryptoPanic, social (Reddit/StockTwits keyless; X only with a paid bearer).
  Each source has its own interval, backoff (exponential, capped, jittered),
  last-ok / last-error, items/hour — **all shown on the panel**.
- `poll_all` runs sources concurrently with `return_exceptions=True`; one dead
  source never stalls the rest (error C-1).
- Poll every 30 s for fast sources, 60–120 s for RSS, regardless of horizon.
- **Fingerprint** = normalised title + domain → dedupe; repeats keep the first
  timestamp and decay 3× faster (`impact.py`, "news novelty").
- **Relevance per asset** (`relevance.py`): entity/keyword sets per asset plus
  a macro set (rates, war, sanctions, exchange outage, regulation) that hits
  everything; `score == 0` items are counted as "relevance gate dropped n"
  and shown, not hidden (error C-4).
- **Sentiment**: lexicon (`sentiment_lexicon.py`) with **macro risk-on /
  risk-off phrasings wires actually use** ("strikes", "tariffs", "yields
  climb", "drone attack") — the crypto-desk word list alone scored a real
  world wire at exactly 0.0 for weeks (error C-2). Blend in the signed impact
  of the headline's theme (`impact.classify`).
- **Decay**: `w = w₀ · 2^(−age/half_life)`, half-life ≈ W/2 per horizon; the
  same item has four weights, one per horizon. NIV denominator is
  `Σw + W₀` with resting evidence `W₀ = 0.5`, otherwise one stale headline
  holds NIV at +0.7 forever because decay cancels between numerator and
  denominator (error C-3).
- **Critical events** (`critical_event_detector.py`): whole-word match, the
  headline must also name a market term, published within the last 10 min,
  fires once per fingerprint; flash-move and NIV-swing triggers separate.
  Substring matching made "war" fire on "warning"/"forward"/"award" and broke
  the lock eight times in an hour (error B-2).
- Output: **NIV per asset per horizon** in [−1, 1] with count-based
  confidence, top-3 headlines with weights, critical flag.

### 4.2 Visibility
`/api/news/feed`, `/api/news/poll` (manual kick), `/api/news/status`
(per-source heartbeat). On the card: items/hour per source, last success
time, dropped-by-relevance count, the current NIV and the three headlines
that produced it. A news engine without a visible heartbeat is assumed dead
by the user — and usually is.

---

## 5. Emotions: measured from the tape, locked with the window

### 5.1 Design (`backend/core/emotions.py`)
Nine emotions: FEAR, PANIC, CAPITULATION, DENIAL (negative); HOPE, EUPHORIA,
FOMO (positive); COMPLACENCY, PARALYSIS (neutral). Each is a continuous score
from measured features:

| feature | how | feeds |
|---|---|---|
| volatility z | realised σ over the window vs its own 30·W history | FEAR, PANIC, COMPLACENCY (inverse) |
| drawdown / run-up from window extreme | bps | CAPITULATION, EUPHORIA |
| volume climax | recent volume / baseline median (trades only) | PANIC, FOMO |
| aggressor skew | qty-weighted side mean (trades only) | HOPE/FEAR direction |
| herding | longest same-side run, side skew (side ≠ 0 only) | FOMO, PANIC |
| tick/trade rate vs baseline | trades per second over median | activity gates |
| price at one point | share of prints at the modal price | PARALYSIS |
| spread widening | spread / median spread | FEAR |
| news NIV, social buzz | from the news engine | all, as a band |

- Every mapping is a **logistic ramp** between two measured anchors
  (`_ramp(x, lo, hi)`); never a hard threshold → no dead zeros, no flicker
  (error F-5).
- **COMPLACENCY is gated by `1 − PANIC`** and requires low σ *and* low climax,
  so it cannot show in a violent market (error G-4).
- Exclude quote prints from every volume/side statistic (error F-9); report
  `trade_prints` and `quote_prints` separately.
- The emotion mix is sampled every 2 s (ours) from the tape, **not** derived
  from the signal (that was circular, error G-3), and the **dominant emotion
  is locked at the boundary** with the signal. The card shows "locked" and
  "live" side by side.
- Fusion weight: a confidence modulator of ±15 %, never the direction
  ("formulas over emotions", a user rule after emotions out-voted the tape).
- A geometric layer (`backend/core/geometry.py`) places the nine on a
  valence/arousal plane and reports the trajectory (where the crowd is
  heading), which the UI animates.

### 5.2 For your horizons
Compute the same features on 1-minute bars over the last `3 · W` minutes with
baselines over `30 · W`; aggressor skew from taker-buy share. Re-measure the
ramp anchors **per asset** from a week of data (gold's "panic" σ is BTC's
"calm"). Lock the dominant emotion per horizon.

---

## 6. Physics layer: real mechanisms, each with a validity gate

### 6.1 Mechanisms (`backend/physics/`)
Eleven on the tape: OU mean reversion (half-life, distance in σ), VPIN
(volume-bucket toxicity), Hawkes self-excitation (branching ratio, projected
excess flow), kinetic energy of price, entropy of the sign series, diffusion
(variance ratio), pendulum (oscillation phase), temperature (vol of vol),
Avellaneda–Stoikov spread, venue fragmentation, PAXG peg. Blended by
**ridge-Kelly** weights on the mechanisms' trailing signed edges with a
thermodynamic band that clamps the per-window weight change (`unified.py`).

Round BU added four estimators that print their derivation
(`backend/core/physics_estimators.py`):
- **Kramers–Moyal**: D1(x), D2(x) as conditional moments of 1 s increments in
  five state bins; Péclet number `D1·T / √(2·D2·T)` over the window; the
  linear slope of D1 tells restoring vs self-reinforcing.
- **Ising chain** on aggressor signs: `K = atanh ρ₁`, susceptibility
  `(1+ρ₁)/(1−ρ₁)`, expected mean sign of the next N trades.
- **Fluctuation–dissipation**: Kyle λ = cov(Δp, f)/var(f), Θ = R², flow
  persistence, expected flow-driven move against the diffusion width, gated by
  `0.05 + 0.95 Θ` (never exactly zero).
- **Maximum entropy**: Jaynes distribution of the next window return on the
  observed support under mean (lag-1 regression forecast) and variance
  constraints, Newton-solved; `P(up) = Σ p[r > 0]`.

### 6.2 Rules
- Every mechanism returns `{covered, score, logic[], why}`; the panel lists
  active and inactive mechanisms **with the reason** ("one venue only", "no
  gold spot", "PAXG tape too thin"). "Physics 0.00" with no reason was the
  most-reported complaint (error F-7).
- Scales from the asset's own tape: own realised σ over the horizon, own
  flow variance, own spread. No cross-asset constants.
- The layer's single output is one signed vote plus the per-mechanism table.
  Its fusion weight is a prior (0.10) that the learned combiners override.
- Run it off the lock path when expensive; the Genesis re-score starving the
  loop froze the countdown once (error E-2).

### 6.3 For your horizons
Use the bar series: OU half-life in minutes, Hawkes on bar-level signed
volume (taker-buy − taker-sell), VPIN buckets ≈ total volume / 50 per window,
Kramers–Moyal on 1-minute log-price increments with state = deviation from
the `W`-minute mean, Ising on the sign of bar-level imbalance.

---

## 7. Genesis layer: generated formulas under a hard fitness regime

`backend/genesis/`: ten mathematical domains (`domains/d01…d10`: signatures,
homology, information geometry, optimal transport, fractional calculus,
tropical algebra, quantum-inspired, p-adic, category-theoretic,
algorithmic) each generating candidate formulas from a typed feature set
(`features.py`), scored on **seven metrics** (`fitness.py`: strict-sign hit
rate, information coefficient, regime stability, turnover, correlation to
the active set, latency, sample size) with a lifecycle (`lifecycle.py`:
candidate → probation → active → retired) and an **autopsy** for every
retirement. NumPy only — say so on the card.

Rules:
- Out-of-sample only: generate on the first 60 % of the candle store, score
  on the last 40 %, then walk forward on new windows. In-sample scoring made
  the pool look brilliant and trade like noise.
- Admission: IC above the median existing formula **and** |corr| < 0.6 with
  every active formula.
- Cap the active set (12 for us); the whole layer is one fusion vote whose
  weight is earned.
- Generation and re-scoring run in a low-priority worker thread, chunked,
  never on the lock path.
- Card: pool counts per state, next generation time, top formula's
  definition + 7 metrics + history; `/api/genesis/status`, `/formulas`,
  `/formulas/{id}`, `/domains` (with the maths of each domain).
- Genesis is the **last** thing to build; it only helps once there is honest
  settled history to score against.

---

## 8. Fusion, direction, learning

### 8.1 Fusion (`backend/agents/fusion.py`)
- Fused score = weighted sum of directional formula values (forward P(up)−½
  where the table has `n ≥ 30`), the brain vote, the AI agents, the physics
  vote, Genesis, news NIV, with weights as **priors** (ours: brain 0.40 /
  decision model 0.30 / Gemini 0.15 / local 0.15 among agents; formulas 0.30
  as a voter in their own right; physics 0.10; forward 0.35). The hourly
  hindsight scales them.
- Confidence = magnitude of the fused score + brain term + coherence of the
  formula consensus; *not* agent agreement alone (deviation 8-A).
- Print the consensus as **weighted mass for and against**, and the four to
  six strongest supporters and opponents with their values.
- A **tape veto** before the lock: if the last 20 s (for you: the last 2–3
  bars) contradict the side by more than the spread and the recipe is below
  62 % confidence, flip or flatten — printed as `fusion.tape_veto`.

### 8.2 Direction ladder (`backend/core/direction.py`) — always returns a side
1. fused score beyond the **measured noise floor**
   `max(0.02, 0.5 · median |score| of the last 60 windows)`;
2. the horizon-matched window formulas' lean (magnitude-weighted, tie-risk
   excluded);
3. the brain's vote if confidence ≥ 25 %;
4. the previous window's side;
5. BUY.
The source text is printed with the pick ("fused score −0.026 leans down,
|score| > noise floor 0.020"). HOLD does not exist (the contest forces a
side every window).

### 8.3 Hourly hindsight (`backend/core/hindsight.py`)
Every formula, agent, feature and system is re-graded on the last hour:
hits/misses/silent, won/lost bps, bias, z of its signed edge. Multiplier
`clip(1 + 0.25 z, 0.25, 1.5)`, inversion only at `z ≤ −3` with ≥ 16 decided
windows, `MIN_DECIDED = 8`. Changes apply at the next boundary and the card
prints them ("damped CCSv2 ×0.6: 9/22 last hour"). At your horizons make it
a **daily** review for 130/247 and 4-hourly for 22/44, with the same z rule.

### 8.4 Meta-combiner (`backend/core/meta.py`)
Ridge logistic over all inputs (standardised), walk-forward, exponential
weighting (half-life 120 windows), refit every window off-thread; regime
experts on three routes (BOCPD change-point age, session, volatility) used
only where their own record beats the global fit (`EXPERT_MIN_GRADED = 15`);
Platt slope-only calibration (`a ∈ [0.05, 5]`, `CALIB_MIN = 40`). **Takes the
side only with a 40-window strict-sign record ≥ 53 % and ≥ the recipe**; the
start-up record is built walk-forward in blocks of 20 from stored history.

### 8.5 Unified evidence ledger (`backend/core/everything.py`)
Every source as evidence in nats `eᵢ = aᵢ zᵢ` (1-D ridge logistic slope per
source), correlation shrinkage `k_eff = k / (1 + (k−1) ρ̄)`,
`L = Σeᵢ · √(k_eff/k)`, `P(up) = σ(b·L)`, **predictability `1 − H₂(P)`** and
per-source mutual information printed live. Same takeover rule, plus "not
below the meta-combiner". The predictability gauge is the honest answer to
"why is the app only 52 %": most windows carry < 0.02 bit.

### 8.6 Other learners we keep
- **Forward tables** (section 2.7).
- **BOCPD** regime clock (hazard 1/14 windows) and **Hawkes** cascade at the
  boundary as features.
- **Temporal twin** (`temporal_arb.py`): price the window as one 70 s leg vs
  a 60 + 10 s segmented pair, `var70 = var60 + var10 + 2ρσ60σ10`, show the
  volatility-decay spread. At 247 min a 130 + 117 split is a legitimate
  experiment — but settle only the real contest leg.
- **Edge guard** (`edge_guard.py`): detect price sitting at one point and
  flat windows; the tie-risk formula (PIN) prices them.

### 8.7 Honesty on accuracy
Strict sign, ties = losses, from the ledger only, per asset × horizon with
`n`. Short horizons are 51–56 % for any model; longer horizons add drift and
news but your `n` collapses. Never build toward a target percentage — when
the user asked for 77 % and then 90 %, the right answer was the streak math
(p⁷² at 60 % ≈ 1e-16) and a predictability gauge, not a tuned metric.

---

## 9. Operations: permanent running, real memory, self-start

- **State vault** (`backend/core/state_vault.py`): atomic JSON/pickle writes
  every 30 s of formula states, hindsight rows, meta/ledger fits, outcomes,
  horizons, twin, competition ledger, dopamine weights; restore on boot with
  max-age checks; rebuild walk-forward records from stored rows. Without it
  every restart forgot the streak and the learned weights (error E-7).
- **Watchdog** (`cycle_manager`, Round BM): no window for 2.5 periods → stack
  dump + loop restart; second stall within 10 periods → exit code 75 so the
  supervisor relaunches. `/api/health` reports `stalled`, `seconds_since_window`.
- **Supervisor** (`run.sh --supervise`): health-checks every 10 s; six
  consecutive failures → TERM, KILL, relaunch.
- **Broadcast**: a per-client queue with drop-oldest; `QueueFull` must not
  drop the subscriber forever (error E-3).
- **Six degradation levels**: feed stale → agents late → brain fallback →
  news dead → simulator → emergency; each printed; each limits what may call
  itself "live".
- **Priorities on small machines**: anything heavy (local LLM, Genesis,
  Flutter builds, model downloads) runs `nice -n 19` / `ionice -c 3`, out of
  process where possible, threads = cores − 1 (errors E-5, E-6).
- **Zero-command start** (`.devcontainer`, `run.sh`): provisioning and
  engine start under `flock`; long-lived children started with `9>&-` so
  they don't inherit the lock (error E-8); `sudo -n`,
  `DEBIAN_FRONTEND=noninteractive`, `</dev/null` everywhere (error E-9); a
  stdlib placeholder page answers the port while the venv installs (error
  E-10); self-update is fast-forward only, never checkout/stash; a pre-built
  dev-container image holds requirements, models and the connectome so a
  fresh machine downloads nothing (error E-11).
- README: every command in its own copyable fenced box, no comment lines
  inside boxes, the download of requirements covered, "what to do when the
  download fails" covered.
- Simulator, stub agent, synthetic self-tests, 80×80 fallback brain are
  **core** (first boot, tests, offline demos). Label them loudly, never let
  synthetic data touch a scored prediction, never remove them.

---

## 10. UI: the rules that made ours usable

- **One page, one port, one WebSocket, exactly one `setInterval`** — a test
  counts it. Multiple timers were the "glitchy countdown".
- **Countdown from the server clock**: the WS carries `server_now_ms` and
  `window_end_ms`; the client renders
  `window_end − (server_now + (Date.now() − received_at))`, re-anchored on
  every message. Silent-but-open sockets are closed after 15 s and
  reconnected (error H-6); a 5 s REST safety net keeps the header honest
  ("engine silent").
- **Fixed layout**: every widget has a fixed slot and height; content changes
  inside it; nothing covers the price/signal strip; the emergency notice is
  inline, not an overlay.
- **Every number has a logic line** (formula traces, physics logic, ledger
  evidence, emotion features), collapsed by default, one click away.
- **Status honesty** per chip: feed, each agent, brain, each news source,
  each key slot — state + reason + age. Never a default green.
- **Locked vs live**: the locked block shows `🔒 locked HH:MM:SSZ`, the
  reasoning at the lock, the tape at the lock; live values live elsewhere
  and say "live". Repainting "updated 3 s ago" inside the prediction cell
  made users believe the prediction changed (error H-1).
- **Horizons side by side**: one row per horizon for every prediction, each
  with countdown, pick, confidence, strict-sign record, `n`.
- **Asset selector** changes only what is displayed; all assets keep
  computing; the ledger shows all of them.
- **Toggles that work** (your explicit complaint):
  1. state lives on the server (`GET/POST /api/settings`), never in the DOM;
  2. the control is disabled during the round trip; render from the server's
     response, not from the click;
  3. optimistic flip only if you revert on error and show the error text;
  4. print the toggle's effect next to it (`/api/settings/effects` →
     "persist: ON — vault writes every 30 s", "simulator: OFF — a real feed is
     required, the tape chip will go red without one");
  5. one render function sets `checked`, the label and the effect; never two
     places;
  6. anything that changes clocks or horizons applies at the next boundary
     and the control says "applies at :00";
  7. keep a `data-setting` attribute, not an id per toggle, so a generic
     handler covers all of them (`refreshEffects()` in `app.js`).
- **Settings page** separate from the dashboard: key slots with masked
  tails, test buttons that call the real API and show model + latency,
  local model upload/fetch with validation output, Codespaces notes.
- Minor haptics/animation only on lock and settle, < 200 ms; a win-rate
  ring and streak flame are enough drama.
- Dark, two accent colours (up/down), tabular numerals, no gradients over
  text; `escapeHtml` everything from the server.

---

## 11. Agents: Gemini, local models, decision models

### 11.1 Key ring (`backend/agents/keyring.py`)
- N slots per provider; each slot `untested / ok / rate-limited / invalid /
  error`, last error text, cooldown until, calls, failures.
- **Test = call the API.** For Gemini: `GET /v1beta/models?key=…`
  (ListModels). Never prefix-check (`AIza…` says nothing; error I-1).
- The ListModels response is the **only truth about which models the key may
  call**: filter `supportedGenerationMethods ∋ generateContent`, exclude
  vision/embedding/tts/image/audio ids, rank by family and generation
  (`choose_model`), honour a pinned id only if the key can use it. Hard-coding
  a model id broke every time Google renamed one (error I-2).
- Skip models that answer 429 with quota 0 or that reject `responseSchema`
  (fall back to schema-less JSON); remember bad models per key.
- Rotate slots on 429/403, cool down, continue.
- Keys live in the platform's secrets and in memory; **never write a user key
  to any file in the repo**; show the last 4 characters only; staleness of
  the "key not set" badge was a client bug — re-read state after save
  (error I-4).

### 11.2 Calls (`gemini_agent.py`)
- One structured prompt: frozen state (prices, every formula's value and
  reading, physics votes, emotions, top news with weights, the four horizons),
  the exact question ("higher at window end than at :00? side, confidence,
  three reasons"), the JSON schema; `responseMimeType: application/json`.
- `asyncio.wait_for` with a deadline that ends **before** the refreeze; a late
  answer is discarded and logged "late by X s", never applied mid-window.
- Cache per (asset, horizon, window); never two calls for one lock.
- Weight it as **one agent vote** whose weight is earned. An LLM's raw
  directional accuracy on price is ~50 %; its value is turning news/regime
  context into a confidence modifier and a readable reason.

### 11.3 Using the models for the app's own chores
Same ring, lowest priority, strict budget: summarise the hourly hindsight
in one sentence, classify a headline when the lexicon is unsure, explain a
damped formula. Never on the lock path.

### 11.4 Local and decision models
- Baked GGUF (Qwen2.5-0.5B/1.5B by core count) served out of process with
  `llama-server` at `nice 19`; the engine talks HTTP with a deadline. Never
  put `llama-cpp-python` on the install path (sdist-only; error I-6).
- Uploads validated by **magic bytes** (`GGUF`, ONNX protobuf header), size
  bounds, header metadata, then a test prompt; a baked model goes through
  the identical validate/load/test path.
- Fetch large files server-side (`/api/agents/local/fetch`), never through
  the browser (Codespaces' proxy 5xx's on ~1 GB uploads; error I-7).
- A dedicated yes/no decision model (feder-cr/jev "jevos") answers
  `P(up)` with rules in the question; weighted like any agent. Run it with
  threads = cores − 1 and the installer out of process.
- Thread stacking: two GGUF servers on a 2-core box starved the event loop;
  one pool, one server, measured with `py-spy` before deciding (error E-6).

---

# Part B — The error catalogue

Every failure we hit, in the order the subsystems appear. Format: **symptom →
cause → fix → how to detect it in your app.** Letters group them: A brain,
B lock, C news, D feeds, E runtime/ops, F formulas/physics, G honesty,
H UI, I agents/keys, J tooling.

### A. Brain
- **A-1 Whole-brain vote always ≈ 0.** Readout averaged all six propagation
  hops; signal diluted. → Read MBON valence at hop 3. *Detect:* probe with a
  unit input and print the magnitude per hop.
- **A-2 Brain heard only bullish formulas.** Rectified PN→KC fan-in: +0.7 read
  +0.38, −0.7 read −0.02. → Push-pull readout (propagate x and −x; odd
  function). *Detect:* `score(x) + score(−x)` must be ≈ 0.
- **A-3 Nine of twenty channels had the wrong valence** (bullish set could vote
  SELL). → `calibrate_polarity()` at boot: probe each channel alone with +1
  and orient it. *Detect:* per-channel +1 probe must read approach.
- **A-4 Non-directional indicators pushed the side.** Stress/efficiency
  entered as positive magnitudes. → They enter at zero; stress reaches the
  circuit through a modulator node. *Detect:* flat tape must give ≈ 0.
- **A-5 neuPrint token test failed on a valid token.** Endpoint
  `/api/databaseInfo` removed. → `GET /api/dbmeta/datasets`. *Detect:* test
  endpoint returns 404 for everyone.
- **A-6 Brain "❌" rows on a healthy fresh machine.** Lazy load not finished
  when the health page rendered. → Status rows show "loading… (step 3/5)"
  until verification completes; `/api/brain/reconnect` re-runs it.
- **A-7 Dopamine gate pinned at +1.** `newest_magnitude × newest_outcome` used
  the *signed* P&L, so a −40 bp loss scored +40. → `|pnl| × outcome`.
  *Detect:* gate must go negative after a loss.
- **A-8 Baked matrix checksum mismatches across machines.** Float
  serialisation. → Checksums computed on rounded weights.

### B. The lock
- **B-1 "Prediction changes every few seconds."** Server payload was
  identical; the client repainted "updated 3 s ago", the horizon line and the
  live tape *inside* the prediction cell. → Locked chip shows the lock
  timestamp; the tape at the lock is separate from the live tape. *Detect:* a
  headless DOM run (`tools/lock_watch.js`) sampling every element in the cell
  4×/s.
- **B-2 "Lock broken 8×".** Emergency override fired on "war" ⊂ "warning",
  "hack" ⊂ "hackathon", re-fired every scan, counted hours-old RSS as
  breaking. → Whole-word match + market term + ≤ 10 min old + fire once;
  override shown as "⚡ emergency re-lock" in amber. *Detect:* count overrides
  per hour; > 1 is suspicious.
- **B-3 Reasoning flipped 15 s before the boundary.** Prefetch/refreeze wrote
  `last_fusion`, `emotion_locked`, `conviction_note`, `warnings`,
  `degradation` directly. → They **stage** into `_pending_view`; swapped in
  at `open_window`. A source-guard test asserts they never write those
  fields. *Detect:* poll `/api/signal/current` every 2 s for a window; one
  rendering per window.
- **B-4 Blank 8 s at the boundary.** Lock computed after the boundary. →
  Pipelined two-stage lock, reveal at zero.
- **B-5 Window failed → loop retried the same past deadline forever** (only
  the countdown moved). → Record `last_error`, skip to the next boundary,
  publish inline; `/api/health.stalled`. *Detect:* `seconds_since_window >
  period + 45`.

### C. News
- **C-1 One dead source stalled all news.** Sequential polling, exception
  escaped. → `gather(return_exceptions=True)`, per-source backoff.
- **C-2 NIV = 0.00 on a real wire.** Crypto-desk keyword list; world
  headlines scored 0. → Macro risk-on/off lexicon with wire phrasings +
  theme impact classification. *Detect:* share of items scoring exactly 0.
- **C-3 One stale headline held NIV at +0.72 forever.** Decay cancelled
  between numerator and denominator. → `/(Σw + W₀)` with resting evidence
  `W₀ = 0.5`.
- **C-4 "News irrelevant."** No relevance gate. → Per-asset relevance
  scores; dropped items counted and shown.
- **C-5 Repeats dominated.** No dedupe. → Fingerprint; repeats decay 3×
  faster.
- **C-6 Emergency endpoint ignored duration.** → `duration_seconds` 10–900
  honoured.
- **C-7 Social "connected" with fake sources.** → Keyless Reddit/StockTwits
  only; X only with a real bearer; no Instagram/Facebook pretending.

### D. Feeds / tape
- **D-1 Same TP/SL, same signal, stuck after 1–2 predictions — only on a fresh
  machine with a real connection.** The seeded simulator started 2 s after
  boot because Binance hadn't connected *yet*, and never stopped; both wrote
  one tape (≈65k alternating with the real price). → All data through
  `hub.ingest`; non-active sources rejected and counted; any source change
  flushes the tape; real feed grace period; simulator cancelled when a real
  feed is healthy; header shows the tape source. *Detect:* alternating price
  levels in the tape; `rejected_writes` counter.
- **D-2 Hub stuck on "none" while Kraken/CoinGecko delivered** (every write
  rejected, &b read "offline", hedge 0.00). → `last_delivery[source]`; adopt
  a delivering real feed when neither socket is healthy.
- **D-3 Binance refused from US regions.** → Rotate to
  `data-stream.binance.vision`; Kraken v2 as second tape.
- **D-4 Candles never updated on non-Binance feeds** → volatility frozen,
  levels on the floor. → Every feed rolls its own 1-minute bars.
- **D-5 PAXG tape too thin (one print per ~15 min on Gemini).** → Bootstrap
  from Kraken/Gemini 1-minute candles; quote prints for price paths; hedge
  maths falls back to the last ≤ 121 closes; cross-feed status "derived",
  not "warming".
- **D-6 Gemini WS "walk"**: subscribing to a symbol the venue doesn't carry
  looped reconnects. → Per-symbol capability table; back off per symbol.
- **D-7 Quote prints diluted every trade statistic** (Round BT: "most of
  them are flat"). → `snapshot.trades()`, `real_trades()`; emotions and
  formulas use trades for volume/side. *Detect:* `quote_prints / ticks` on
  the panel; a volume median of 0.
- **D-8 Ticks are not time.** Volatility scaled per-tick returns by
  `√(horizon/span)`: a 60 bp/min tape read 1.7 bp. → Resample on a time grid,
  scale by `√time`.

### E. Runtime / operations
- **E-1 Freeze after the second prediction, only the countdown running.**
  Client-side countdown kept running while the engine was dead (B-5), the
  Flutter web build was saturating 2 cores. → Watchdog + supervisor + silent-
  socket reconnect + `nice 19` for builds.
- **E-2 Genesis re-score starved the loop** (GIL). → Worker thread, niced,
  chunked; never on the lock path.
- **E-3 "Crowd emotion: waiting for the first sample" forever.** `broadcast`
  dropped a slow subscriber on `QueueFull` and never re-added it. → Per-client
  queue with drop-oldest. *Detect:* `/api/system/health` per-stage ages.
- **E-4 Port opened only after the brain warmed** → platform showed a 502.
  → Port first, brain in a worker, placeholder page from a stdlib script.
- **E-5 Local 1.5B model took minutes per vote on 2 cores.** → Out-of-process
  server, `nice 19`, threads = cores − 1, 9 s deadline, 0.5B on 2 cores.
- **E-6 GGUF thread stacking** (two servers). → One pool, one server;
  measured with `py-spy`.
- **E-7 Every restart forgot streak, weights, outcomes.** In-memory store
  without Redis. → State vault.
- **E-8 Autostart "hung" 15 min.** Supervisor inherited the `flock` fd. →
  Start children with `9>&-`; hooks serialised under `flock -n 9`.
- **E-9 Provisioning blocked on a keyboard.** Plain `sudo`, interactive apt,
  a "[y/N]" prompt. → `sudo -n`, `DEBIAN_FRONTEND=noninteractive`,
  `</dev/null`, unattended skips.
- **E-10 Tab opened before the engine listened** → "connection refused". →
  Placeholder page at hook start; `/api/*` answers 503 so health probes
  aren't fooled.
- **E-11 "Setting up remote connection" 10–30 min.** That's the platform
  building the container. → Pre-built image (GHCR) with requirements,
  models, connectome baked; prebuilds enabled.
- **E-12 Provisioning declared done with half the requirements.** Import
  check covered 7 modules. → Check every requirement; three passes; clean
  venv on the second.
- **E-13 Self-update switched branches / stashed user state.** → Fast-forward
  only; never checkout/switch/reset/stash; guarded by a test.
- **E-14 Redis absent → silent in-memory store.** → Log it once, show it on
  health, vault compensates.
- **E-15 Window length drifted between subsystems** (53 s here, 60 s there,
  "1-minute" strings). → One `window_seconds()` read everywhere; a grep test
  for stale strings; legitimate "1-minute" (candle resolution, poll
  intervals) stays.
- **E-16 `emotion_locked["dominant"]` was `None` on a very sparse tape** →
  crash at settlement. → `(x.get("dominant") or {}).get("name")`.

### F. Formulas / physics
- **F-1 Random numbers in formulas** (early stub). → Removed; every formula
  derives from the tape and is traced; a test asserts no `random` import in
  formulas.
- **F-2 TP/SL never moved** (see D-4, D-8). → Volatility-scaled levels sized
  from the measured one-minute excursions; reward:risk reported, not
  targeted.
- **F-3 Levels too wide** → median one-sided excursion as target, 80 % as
  stop, caps 25/30 bp.
- **F-4 Hedge status 0.00 / "partial" forever** → candle fallback;
  outcomes system replaced the status tiles.
- **F-5 Dead zeros from hard thresholds.** → Logistic ramps and tanh with
  measured scales; **a test asserting exact 0.0/1.0 at ramp ends will break
  — write tolerances.**
- **F-6 Everything flat on a thin tape.** → Materiality gates with soft
  edges; zero reasons; flat watch.
- **F-7 Physics "inactive" with no reason.** → Each mechanism prints its
  gate reason; quote prints feed thin tapes.
- **F-8 "11+ formulas returned zero" degraded state on a healthy tape.** →
  Count only genuine computation failures, not honest near-zero readings.
- **F-9 Volume medians of 0 → VSD/SED/AFPR flat** (quote prints). → Trades-
  only accessor.
- **F-10 Self-scale floors restored from the vault** pinned formulas to an
  old build's floor; SED's floor was 400× above any tape. → Floors are code
  constants passed at restore; SED 0.05 → 1e-4, RSV 0.005 → 0.0005.
- **F-11 DSKD compared a Kalman gap with 0.75 × the window's move** → 0.000
  on every tape. → `max(3σ_divergence, floor, 0.4 × realised)`; self-test
  FLAT tolerance kept.
- **F-12 Category scores rendered `ANaN`.** Dicts rendered as numbers. →
  Render `mean × count`.
- **F-13 "24 formulas evaluated"** counted helper keys (`_hsi`). → Count the
  formula registry only.
- **F-14 Sign convention of a cross-asset formula inverted** (LCS). →
  Documented convention + self-test claim.
- **F-15 HSI calibrated against zero instead of the uncorrelated-neutral
  baseline** → always "stressed". → Baseline at the neutral correlation.
- **F-16 Synthetic test tapes didn't span one window** → formulas abstained
  in tests. → 600 ticks × `TICK_STEP_MS` = exactly W.
- **F-17 Forward tables settled on the wrong direction field.** → One
  `_last_window_dir()`; strict sign, ties counted.

### G. Honesty
- **G-1 Three different accuracies on screen.** → The ledger is the only
  writer; everything reads it.
- **G-2 Confidence 56 % from an 18 % brain readout.** → Confidence from
  magnitude + brain term + consensus coherence; single digits when inputs
  are weak.
- **G-3 Emotions derived from the signal** (circular, always agreed). →
  Emotions from tape features only.
- **G-4 "Complacency" during a crash.** → Gated by `1 − PANIC` and low
  climax.
- **G-5 "Right about the past, not the future."** Formulas described the
  window that just closed. → Forward tables, superposition, the opening
  audit and a lag audit per formula (lead/lag vs the next window's sign).
- **G-6 Asked for 77 % / 90 % accuracy.** → Refused to tune to a target;
  printed the streak math and added the predictability gauge; the honest
  range (51–56 %) stays on the README.
- **G-7 "Key not set" after saving a key.** Client staleness. → Re-fetch
  state after POST.
- **G-8 Sandbox-only network failures presented as "API down".** → Real
  integrations written and tested against the API shape; sandbox blocks are
  documented as sandbox blocks.

### H. UI
- **H-1 Repaints inside the prediction cell** (B-1).
- **H-2 Countdown drift / glitch.** Several timers, local start time. → One
  `setInterval`, server-anchored deadline, pipelined pulses at fixed marks.
- **H-3 Overlay covered the screen on emergency.** → Inline notice; a layout
  guard test forbids the overlay class.
- **H-4 Widgets jumped when data arrived.** → Fixed slots and heights.
- **H-5 "Page is not working" (HTTP 502) in Codespaces.** → Port first,
  placeholder page, README explanation of the forwarder.
- **H-6 Open-but-silent WebSocket** (forwarder keeps the client alive after
  the backend died). → Close after 15 s silence, reconnect, replay HELLO.
- **H-7 Template literal `word (${…})` parsed as a call by a guard test.** →
  Avoid `name (` patterns in JS strings (tooling, but it cost an hour).
- **H-8 Toggle flipped visually but did nothing / reverted on refresh.** →
  Section 10's seven toggle rules.
- **H-9 Two frontends (Flutter + web) diverged.** → One web dashboard; the
  Flutter build also starved the machine.
- **H-10 Stale strings** ("22 formulas", removed agents, old weights) across
  prompt/UI/README. → A sweep + grep tests for the retired strings.

### I. Agents / keys
- **I-1 Keys rejected by a prefix check.** → API-call tests only.
- **I-2 Hard-coded model id** 404'd after a rename; quota-0 models chosen. →
  ListModels-driven `choose_model`, skip quota-0 and schema-rejecting
  models, remember bad models.
- **I-3 Agent call after the boundary** mutated the on-screen prediction. →
  Deadline before refreeze; late = discarded + logged.
- **I-4 Key saved but badge "not set"** (G-7).
- **I-5 User keys written to `.env`/repo.** → Never; memory + platform
  secrets.
- **I-6 `llama-cpp-python` on the install path** (sdist-only build, hours or
  failure). → `llama-server` binary out of process; never on the critical
  path.
- **I-7 ~1 GB model upload through the browser** → proxy 5xx. → Server-side
  fetch with progress.
- **I-8 GitHub Models agent** — a Git server repo was mistaken for a model;
  removed entirely; a real decision model (jevos) replaced it.
- **I-9 Stub local agent indistinguishable from a real one.** → `LOCAL_AGENT_STUB`
  is labelled on the panel and in `/api/agents`.
- **I-10 Baked model skipped validation.** → Same validate/load/test path as
  an upload.

### J. Tooling / development (so your agent doesn't repeat them)
- `pkill -f` / `kill $(pgrep …)` inside a tool shell kills the tool itself —
  start/stop test engines with `setsid nohup` scripts.
- Entry point is `python -m backend.api.main` (reads `PORT`), not bare
  uvicorn.
- `a && b <<EOF` skips the heredoc when `a` fails; patch scripts must assert
  the anchor text exists; `sed` with `/` in the replacement — use `|`.
- Certifi/httpx fail in some sandboxes where urllib works; don't conclude
  "API down".
- Sandbox resets wipe `/tmp`, venv and the git index mid-task: commit and
  push after every step; recover with `fetch` + `reset` (never force-push).
- Tests that pass with `-k` in isolation and fail in a file (or vice versa)
  usually share a module-level singleton — reset it in a fixture.
- Keep a `tools/dead_code.py` (0 unused definitions) and run it before each
  commit; a guard test for "exactly one `setInterval`"; grep guards for
  retired strings; source guards for lock-safety.

---

# Part C — Checklists and the plan

## 12. What the panel must show for each subsystem (acceptance list)

| subsystem | must show |
|---|---|
| tape | source, ticks/s, trade vs quote prints, last delivery age, rejected writes, bootstrap span |
| lock | locked side, confidence, lock timestamp, entry price, reasoning at lock, locked emotion, per-horizon rows |
| ledger | per asset × horizon: W/L/T, strict-sign %, `n`, streak, best, last 20 results |
| formulas | value, reading, trace, double-check, µs, span, own z, flat share, forward P(up) per horizon |
| physics | each mechanism: active/inactive + reason, score, logic lines; blend weights |
| emotions | nine bars, features behind them, locked vs live, sample counts |
| news | per-source heartbeat, items/hour, dropped by relevance, NIV per horizon, top-3 headlines with weights, critical flag |
| brain | 5-step verification, counts, readout hop, top circuits, channel polarity table, dopamine gate |
| agents | each: state, model chosen, latency, last answer, deadline misses, weight earned |
| learners | hindsight multipliers with reasons, combiner record (`n`, rate, takeover state), predictability, reliability table |
| ops | degradation level, vault last write/restore, watchdog, uptime, build id vs remote |

## 13. Test guards to copy
- Formula self-tests on 8 synthetic tapes with per-formula claims.
- Double-check re-derivation from traced intermediates (keep trace keys
  stable).
- Lock safety: prefetch/refreeze never write the five on-screen fields;
  preparing the next window changes nothing on screen.
- Countdown: final lock lead < lock deadline; pulse marks fixed.
- Exactly one `setInterval`; no `eval(`; no `location.href = "/"`; no overlay
  class; required ids present.
- Autostart: `flock -n 9`, `9>&-`, `sudo -n`, `DEBIAN_FRONTEND`, waitFor hook.
- Self-update: no `checkout|switch|reset|stash|main`.
- Registry sizes (formula count, window-formula count, emotions = 9).
- No `random` in formulas; dead-code scan = 0.
- Retired strings grep (old agents, old counts, old weights, old window
  length).
- Quote-heavy tape → volume formulas non-zero; SelfScale floor from code.

## 14. Migration plan for your app, in this order

1. Clock, lock, ledger, per-asset settlement sentence (section 0). Ship it
   with a plain "last-bar momentum" pick and watch the strict-sign record
   per horizon for two days. Everything after this is judged against it.
2. Tape: bars + trades per asset, quote/trade separation, hub with source
   reconciliation, feed chips, bootstrap (section 1).
3. Formulas: 8–10 per family with traces, double checks, self-tests, zero
   reasons, forward tables per horizon, flat watch (section 2).
4. Emotions from tape features, locked per horizon (section 5).
5. Physics mechanisms with gates and logic lines (section 6).
6. News: sources, heartbeat, dedupe, decay per horizon, relevance, critical
   events (section 4).
7. Brain import with five-step verification, polarity calibration, push-pull
   readout, hop-3 MBON vote as a tie-breaker (section 3).
8. Agents behind a key ring with API tests, ListModels-driven model choice,
   deadlines, caching, the settings page (section 11).
9. Learners: hourly/daily hindsight → meta-combiner → unified ledger, each
   earning its takeover; predictability on the panel (section 8).
10. Genesis last (section 7).
11. Operations throughout: vault from step 1, watchdog and supervisor from
    step 2, zero-command start before the first user opens it (section 9).

At every step the UI prints the sample size, the reason for every "not
available", and the logic behind every number. That — not a bigger model or
one more formula — is the difference between an app that "feels useless" and
one you can trust at second 0.
