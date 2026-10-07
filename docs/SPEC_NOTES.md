# SPEC NOTES — deviations, decisions and escape hatches

`docs/SPECIFICATION_v2.md` is the normative description of what the code does.
This file records every place where the implementation **deliberately differs**
from the original v2.0 draft, because a specification that silently disagrees
with its implementation is worse than no specification at all.

Each note has the same shape: *symptom → cause → decision → verification →
escape hatch*. Deviations are numbered `<section>-<letter>`.

---

## 11-A — HSI is calibrated against the uncorrelated-neutral baseline

**Symptom.** With real BTC/PAXG data the Hedge Stress Index sat at ≈0.86–0.92 on
every cycle. Because `HSI > 0.80` triggers the stress rule (Section 8.2), the
fused confidence was permanently multiplied by 0.20 and the engine could only
ever emit HOLD. The product was dead on arrival — while the raw formula looked
"correct".

**Cause.** The literal formula `HSI = tanh(2·(1 − HE))` measures the hedge's
diversification benefit against a *zero-correlation* ideal, not against what a
BTC/PAXG pair can physically deliver. Even a perfectly diversified equal-weight
pair of two independent volatile legs has a residual variance floor of
`σ_hedge = sqrt(0.5²σ_B² + 0.5²σ_G²)`, i.e. `HE ≈ 0.29` and therefore
`HSI ≈ tanh(1.41) ≈ 0.89` — above the stress threshold *by construction*. The
formula's own "hedge works" region was unreachable.

**Decision.** Keep the spec formula as the *raw* statistic and calibrate the
operative value against the neutral baseline:

```
σ_neutral = sqrt(w²·σ_B² + (1−w)²·σ_G²)      # uncorrelated equal-weight legs
base      = σ_neutral / σ_avg
HSI       = clip( (σ_hedge/σ_avg − base) / (1 − base), 0, 1 )
```

`HSI = 0` now means "the hedge is doing exactly what an uncorrelated pair would
do"; `HSI = 1` means "the pair is more volatile than its own legs". The regime
semantics of the spec (0 = working, 1 = breaking down) are preserved, the
threshold 0.80 is finally reachable, and observability is preserved: the raw
value stays in `State.last_raw` and is reported in the formula's diagnostics.

**Verification.** `test_e12` asserts the dampening; live cycles report
`HSI = 0.04 … 0.11` on the simulator (raw ≈ 0.9), i.e. the stress rule fires only
when the pair genuinely breaks.

**Escape hatch.** The HSI state exposes `use_raw_calibration = True`, which
restores the literal spec value without touching anything else.

---

## 21-A — KCAE is measured on the pre-ReLU Kenyon-Cell drive

**Symptom.** For net-bearish ensembles KCAE was exactly `0.000` on every cycle,
so the brain's own confidence term was zero and CCSv2's confidence could not
clear the 0.55 entry gate. Bullish ensembles were unaffected.

**Cause.** The KC code is created by `a1 = ReLU(A·a0)`. When the weighted input
to the Kenyon Cells is negative — which is what a coherent *sell* ensemble looks
like, since the signed PN→KC weights are negative on that path — the rectified
code is identically zero. Formula 21 is defined as `p_j = |a_j|²/Σ|a_k|²`, and
`0/0` was being guarded to 0. The entropy of an all-zero code is undefined, not
"zero confidence".

**Decision.** Measure the drive **before** rectification (`kc_drive = A·a0`,
preserved in the activation trace). The `|a|²` notation in the spec is
sign-symmetric, and the sparsity of a representation is a property of the
pattern, not of its sign. The rectified code is still what propagates through
layers 2 and 3, so the circuit itself is unchanged.

**Verification.** The brain probe shows `kcae_post = 0.000` for an all-`−1` input
(rectified) versus a healthy `kcae = 0.589` on the drive; the engine reports
KCAE 0.17–0.65 on live cycles with both bullish and bearish ensembles.

**Escape hatch.** `ccsv2.prepare()` still records the rectified activations as
`_kc_activations`; a consumer that wants the literal post-ReLU statistic can read
it from the trace.

---

## 22-A — Adjacency weights are signed and max-abs normalised

**Symptom.** Early builds produced all-zero activations: the network was
provably dead regardless of input.

**Cause.** Two independent problems. (1) The CSV stores `M[source, target]`, but
the convolution normalised `A` rather than `Aᵀ`, so every "connection" pointed
backwards. (2) Weights were originally clamped to `[0, 1]`, which deletes every
inhibitory edge — and the mushroom body is defined by its excitatory/inhibitory
balance (approach versus avoidance).

**Decision.** Normalise `Aᵀ` (row-normalised), keep signs, and scale the whole
matrix by `1/max|w|` so the strongest connection is 1.0. The committed fallback
matrix therefore contains 285 excitatory and 71 inhibitory edges (356 total).

**Verification.** `test_e07` asserts the matrix has both signs and a non-zero
three-layer read-out; the probe table in the brain module shows `+1 → CCSv2
+0.756` and `−1 → −0.859`.

---

## 22-B — The PN→MBON-α3 pathway is signed (no unsigned shortcut)

**Symptom.** The approach read-out responded to *every* strong input, so
bullish and bearish ensembles produced the same `LH_approach`.

**Cause.** The direct PN→MBON-α3 pathway was initialised with absolute values
"because it is a fast feed-forward shortcut". Unsigned weights make the shortcut
direction-blind, which destroys the sign of CCSv2.

**Decision.** The shortcut is signed and fan-out scaled exactly like the
KC→MBON pathway. Combined with 22-A this makes `CCSv2 = tanh(LH_appr − LH_avoid)`
monotone in the direction of the ensemble (probe: +0.756 / −0.859).

---

## 22-C — The convolution gain is calibrated once per matrix

**Symptom.** A saturated input produced a read-out of ~0.1 instead of ~0.76;
CCSv2 occupied a fraction of its intended dynamic range and never crossed the
0.25 signal threshold.

**Cause.** Row normalisation spreads each node's influence over all of its
out-edges; three layers multiply that attenuation. The draft assumed unit gain
per layer, which is only true for a single saturated path.

**Decision.** `calibrate_gain(target=1.0, probes=16, seed=7)` fits
`gain = (target/peak)^(1/3)` over 16 deterministic probe vectors *including the
all-ones probe* (without it the fit is biased by the mean of the probe set).
For the committed matrix the calibrated gain is **3.2802**. The gain is stored
with the matrix and shown on `/matrix` and `/api/brain/status`.

**Verification.** Probe table: all `+1` → +0.756, all `−1` → −0.859, all `+0.5`
→ +0.452, realistic mixed → −0.016 (i.e. neutral input stays neutral).

---

## 8-A — Confidence is magnitude + brain term, not agent consensus alone

**Symptom.** The fused confidence could not exceed ~0.35 even for a unanimous,
high-conviction ensemble, so `min_fusion_confidence = 0.55` made BUY/SELL
unreachable. Simultaneously, a *non-directional* cycle could report high
confidence.

**Cause.** The draft defined confidence as the weighted mean of the contributors'
own confidences. Agent confidences are small by nature (the Drosophila arm
reports `ccs_confidence` in the 0.1–0.3 band by design — it is a sparsity
measure, not a probability), so the mean had a low ceiling. It also ignored how
far the ensemble was from neutral.

**Decision.** Two terms with explicit roles (Section 8.3):

```
magnitude   = min(1, |score| / (2·signal_threshold))          # conviction
brain_term  = 0.6·ccs_confidence + 0.4·mean(agent confidences)
confidence  = (0.65·magnitude + 0.35·brain_term) · (0.85 + 0.15·agreement) · hsi_adj
```

With no agent answering, `brain_term = ccs_confidence` alone, so degradation
level 3 is correctly *less* confident than level 1.

**Verification.** Live cycles report confidence 0.72–0.82 for BUY and 0.23–0.60
for HOLD; `test_e12` asserts HSI dampening is applied *after* the rework.

---

## 10-A — The lock is published at the end of the 8-second window

**Symptom.** In a compressed-time demo the "⏳ Computing…" state flashed for
~3 ms, so the protocol's most visible state was effectively invisible.

**Cause.** The draft specified computation *at* `t=0` and a lock *at* `t≈8 s`,
but the implementation published as soon as the computation finished.

**Decision.** `_hold_until_lock_deadline()` waits out the remainder of the
window before calling `lock()`, returning immediately if an emergency fires. The
computation itself is unchanged and still finishes in milliseconds; only the
publication time is fixed. This makes the timeline a property of the protocol
rather than of machine speed.

**Verification.** `test_e01` (first locked payload after the window) and
`test_e02` (the lock cannot move afterwards).

---

## 5-A — The simulator is a supported offline mode, not an error

**Symptom.** In an offline environment (CI, a sandbox, a plane) the engine
reported `market_data: unhealthy` and dropped straight to level 6/HOLD, making
the whole product undemonstrable without internet.

**Cause.** The draft treated "no exchange feed" as "no data".

**Decision.** The built-in simulator (`SIMULATOR_SEED`) is a supported mode:
`MARKET_ALLOW_SIMULATOR=1` (default) keeps every formula computable and the UI
fully alive, while the tape is *always* flagged — `warnings` carries *"Live
exchange feed unavailable — simulated tape in use."* and the component status
says *"built-in simulator — simulated tape, not live market data"*. The real
Binance/CoinGecko clients remain the primary path and are used whenever the
network allows.

**Escape hatch.** `MARKET_ALLOW_SIMULATOR=0` restores strict spec behaviour: no
feed ⇒ level 6 ⇒ forced HOLD.

---

## 7-A — `LOCAL_AGENT_STUB` exists for testing, and says so

Loading a real GGUF model requires a multi-gigabyte download and (for
llama-cpp-python) a C++ toolchain. `LOCAL_AGENT_STUB=1` loads a deterministic
stand-in that reads the same numbers from the same prompt and applies a
transparent rule (`0.4·CCSv2 + 0.2·TAI + 0.2·SHRP + 0.2·MPS`).

It is **never** presented as real inference: the agent status is `STUB`, the
dashboard shows a yellow `STUB` chip, and `/api/agents/local/status` reports it
explicitly. Its purpose is to make the fusion weights, the degradation ladder and
the UI testable in CI (`test_e09`, `test_e10`, `test_e13`).

---

## 22-D — Matrix checksums are computed on rounded weights

**Symptom.** The checksum changed after a save/load round-trip of an identical
matrix, so a cache hit looked like a different circuit.

**Cause.** The CSV stores six decimals; hashing raw float64 bytes distinguishes
values that are identical at the file's precision (and at any precision that
matters for a connectome estimate).

**Decision.** Round to 6 decimals before hashing. The checksum is now stable
across neuPrint → memory → Redis → CSV, which is what makes the
"matrix_checksum matches what was loaded" health check meaningful.

---

## Formula 6 (LCS) sign convention

The draft left the asymmetry sign ambiguous. Resolved as:

```
LCS = tanh( (|C_ask| − |C_bid|) / (|C_ask| + |C_bid| + eps) )
```

A deeper ask-side cliff means there is nothing above the offer to absorb a
buyer, so a modest push travels further: **ask cliff deeper ⇒ bullish.**

---

## 10-B — The countdown is pipelined (supersedes the "blank 8 seconds" reading)

**Symptom.** With the draft timeline the dashboard spends the first eight seconds
of every minute on "⏳ Computing…": there is nothing to show, and the user is
asked to wait for a signal that will not change again after t = 8 s anyway.

**Decision.** The default engine (`SIGNAL_PIPELINE=1`) publishes the signal for
window *N* from a computation performed during window *N−1*, in the last
`lead = min(scaled lock deadline, 30 s)` seconds of that window — late enough
that the frozen snapshot is still fresh, early enough to beat the lock deadline.
Consequences that are now part of the interface:

* `COMPUTING` is only reachable at a cold start; the client renders a labelled
  sentinel instead of an empty panel;
* every payload carries `computed_at`, `valid_from`, `valid_until`, `preview`,
  `age_seconds`, `seconds_remaining`, `risk` and a `window` block, so a client
  can *prove* the pipeline rather than trust a claim;
* the bootstrap window is flagged `preview: true` and, with the world clock, runs
  only to the next UTC minute so every later window is minute-aligned;
* `NEXT_WINDOW_READY` announces readiness **without the direction**, so the lock
  cannot be read through the side channel.

**Escape hatch.** `SIGNAL_PIPELINE=0` restores the literal draft timeline, and
the Appendix E tests exercise that path.

---

## 10-C — Take-profit / stop-loss are volatility-scaled

**Why.** The draft requires a TP/SL with every signal but does not define one.
Fixed pip targets are wrong across a 2 bps and a 90 bps tape, so the levels are
derived from the realised 1-minute volatility and clamped
(`tp = clip(1.6·σ, 4, 150) bps`, `sl = clip(1.0·σ, 3, 120) bps`; PAXG uses
1.4/1.0). A HOLD carries **no** levels and the UI says "no position while the
signal is HOLD" rather than inventing a trade the engine did not recommend.

---

## 11-A — The emergency notice is inline, not an overlay

**Correction from use.** A full-screen red overlay hides the price, the levels
and the news at exactly the moment a trader needs them. The override is therefore
announced by (a) a small glittering chip immediately under the prediction and
(b) the HOLD box turning red and glittering harder, both inside the page. The
Flutter client follows the same rule and adds haptics.

**Why the overlay came back anyway (round 3).** The overlay *markup* was removed
but two things were left behind: the class name and a CSS rule.

```css
.emergency { position: fixed; inset: 0; z-index: 100; }   /* deleted */
```

```js
holdBox.classList.add("emergency");                        /* -> "override" */
```

The dashboard still painted that class onto the HOLD widget whenever an override
arrived, so the leftover rule lifted the widget out of its grid cell and stretched
it over the viewport — a red, glittering screen with no app in it. The override
now uses a *scoped* class (`hold-box.override`, `timer.override`), the bare
`.emergency` / `.shade` / `.emergency-card` rules are deleted, the HOLD box is
pinned to the flow with `position: static` and a `max-height` so a long headline
scrolls instead of growing, and `backend/tests/test_ui_layout_guard.py` fails the
build if any class the client applies ever picks up `position: fixed` again
(dialogs such as `.modal` are the only sanctioned full-screen layers). The static
assets are loaded with `?v=` stamps so a cached stylesheet cannot hide the fix.

---

## 9-A — The port opens before the engine is warm (and a supervisor keeps it open)

**Symptom.** GitHub Codespaces (and any port forwarder) answers **HTTP 502** when
nothing is listening on the forwarded port. A crash, a container that was asleep,
or a warm-up that takes ten seconds all look identical to the user: "this page
isn't working".

**Decision.** Three structural changes:

1. `lifespan` no longer awaits the engine. It creates the `CycleManager`, starts
   warm-up as a background task, and yields immediately, so uvicorn binds the
   socket at once. `manager.ready`, `manager.warming` and `manager.start_error`
   are reported by `/api/health`, and the dashboard shows a "warming up" banner
   (with the error text, if there was one) instead of an error page.
2. `bash run.sh --bg` starts the engine under a supervisor
   (`run.sh --supervise`) that restarts it ~2 s after an unexpected exit, records
   its own pid in `.run/supervisor.pid` (so `--stop` is exact), and is idempotent:
   starting twice never creates a second server. The start is verified with a real
   HTTP request against `/api/health` before a URL is printed.
3. A forwarded port must be bound to `0.0.0.0`. If `HOST` is left as a loopback
   address while `CODESPACE_NAME` is set, the app overrides it and logs why —
   loopback binding is a 502 that curl inside the container cannot reproduce.

**Escape hatches.** `bash run.sh --status` reports the three layers separately
(supervisor, listener, HTTP). `bash run.sh --clean` removes stray
Flutter/Dart dev servers that open random ports, and
`bash frontend/run_web.sh --dev` is pinned to 8081 instead of letting Flutter
choose one. The Flutter client is optional: the dashboard on the same port
carries every feature.

---

## What was *not* changed

For the avoidance of doubt, the following spec behaviour is implemented exactly
as written: the 0.40/0.25/0.20/0.15 fusion weights, the 0.25 signal threshold,
the 0.55 confidence gate, 11-of-20 zeros ⇒ HOLD, the 60 s outcome horizon, the
30 s CryptoPanic poll, the Tier ≤2 emergency rule, the 8 s lock deadline, the
60 s world clock, the HOLD warning text (verbatim), the 180 s emergency
duration, and the six degradation levels.


---

## Round F - fresh predictions, reasoning, 1:1 levels

The user's rules, and where they live in the code:

| Rule | Implementation |
| --- | --- |
| "Predictions can't be older than 15 seconds" | `Settings.cycle_seconds` (default 12 s) + `prediction_max_age_seconds` (hard cap 15 s). The prediction is computed during the previous countdown and re-stamped at the boundary. `CycleManager.prediction_payload()` publishes `age_seconds`, `state` (LIVE / STALE), `expires_at`; `CycleManager.ensure_fresh()` is called by `GET /api/signal/current` and recomputes out of band if the window was ever missed. The dashboard chip (`#w-fresh`) ticks the age locally and turns red if the contract breaks. |
| "Add reasoning in predictions" | `backend/core/prediction.py::build_reasoning()` composes a summary plus bullets ordered the way the decision was made: brain read-out (with its fusion weight), formula consensus, named supporters, named dissenters, agents, hedge state, rotation, news, level geometry, measured hit rate. Served as `prediction.reasoning` and rendered by both clients. |
| "Add 1:1 tp:sl" | `backend/core/risk.py::risk_levels()` computes one volatility-derived distance, clamps it once and applies it to both sides, so `tp_bps == sl_bps` and `rr == RR_TARGET == 1.0`. Verified across the clipping extremes by `backend/tests/test_prediction.py`. |

### Accuracy work in the same round

* **Formula consensus confirmation.** `backend/core/prediction.py::consensus()`
  weights the directional formulas (regime indicators HSI / ERC excluded) by
  category. `fusion.fuse()` takes the result and scales confidence by
  0.90-1.00 when the tape agrees and down to 0.65 when it disagrees. The *side*
  is never flipped by the consensus - the direction ladder stays the spec's.
* **Realised-form calibration.** With 8 or more scored windows, a hit rate
  below 50 % scales confidence down (`calibration`), because a high confidence
  the engine has not been earning is a lie the panel should not print.
* **Denser scoring.** Every window is scored: `OUTCOME_HORIZON_SECONDS=0` means
  "one window after the prediction fired", so the accuracy panel fills at the
  rate predictions arrive. `accuracy_block()` adds per-side hit rates.
* **Two formulas that read a permanent zero on live data** were fixed as part of
  this round: `HRDD` used a hard dead zone (now a soft 1-to-3-sigma ramp) and
  `MPS` silenced every mean-reverting tape (now the spec's signed reading, so a
  choppy tape says "fade it" instead of 0.000).

### The "page is not working" bug, and the guard that now catches it

`app.js` kept calling `renderHoldBox()` after Round E renamed it to
`renderConvictionBox()`. `node --check` passed (a call to a missing name is
valid syntax), every Python test passed, and the page rendered nothing:
`renderAll` threw `ReferenceError` on boot.

`backend/tests/test_dashboard_scripts.py` now fails the build if any script
calls a function it never defines, or looks up an element id its page never
renders, or fails `node --check`. The jsdom walkthrough
(`tools/ui_override_check.js`) additionally boots the live page against a
running server and asserts the freshness chip and the reasoning list render.

---

## G — One clock, one tick: the 60-second countdown

**Symptom.** *"Countdown is not 60 seconds and it is too glitchy and not in
parallel with other features it randomly running."* Three complaints, one
architecture: the engine ran a 12-second window, the dashboard ran eleven
private `setInterval`s (news 15 s, agents 5 s, brain 20 s, readiness 2 s,
history 30 s, outcomes 15 s, brain-explain 10 s, emergency 1 s, formulas 15 s,
status 20 s, countdown 100 ms), and the countdown re-anchored from
`seconds_remaining` on every one of them. Each panel changed at its own moment,
and any message arriving mid-second could nudge the number on screen.

**Cause.**
* `CYCLE_SECONDS` had been lowered to 12 s (a Round-F freshness decision that
  traded the spec's minute for headroom under a 15 s cap), so the "1..60"
  widget could only ever count 12.
* The countdown was derived from a *duration* (`seconds_remaining`, rounded to
  0.1 s on the server, then advanced locally with `performance.now()` from the
  moment the message happened to arrive). Round trip time and re-anchoring both
  leaked into the digits.
* Nothing owned the schedule: every feature and every panel fetched when its own
  timer told it to.

**Decision.**
* **The window is 60 seconds again**, phase-locked to the UTC minute
  (`CYCLE_SECONDS=60` → `use_world_clock`, `_snap_to_minute`). Freshness is now
  *window-relative*: `PREDICTION_MAX_AGE_SECONDS=0` means "as long as its own
  window" (60 s + a 5 s publishing grace) and `OUTCOME_HORIZON_SECONDS=0` means
  "one window later". The Round-F rule ("what is on screen is always the current
  call, never an old one") is unchanged - only the window it is measured against.
* **One authoritative clock.** `CycleManager.master_clock()` publishes the window
  as absolute instants (`window_started_at_ms`, `window_ends_at_ms`),
  `server_time_ms`, `cycle_id`, `seconds_remaining`, and the window's refresh
  marks. The client measures the offset once, nudges it by at most 120 ms per
  message, and counts down to a fixed instant. **Within a window the deadline
  can never move**: only a new `cycle_id` re-anchors it.
* **One tick.** `tick_grid()` derives the in-window marks from the window start
  (t+15 / t+30 / t+45 for a 60-second window, news polled on the t+30 mark), and
  `_heartbeat_loop` broadcasts **one `PULSE` per mark** carrying the live
  formulas, the news feed, the agent status, the brain read-out and the accuracy
  block. The boundary `SIGNAL` is the same snapshot plus history and outcomes.
  Both clients apply one message in one render pass.
* **The clients lost their timers.** The web client has exactly one
  `setInterval` left (a safety net that returns immediately while the socket is
  open) plus one `requestAnimationFrame` loop for the countdown; the Flutter
  client has one 100 ms tick and one 5 s safety net. Nothing polls when the
  socket is healthy.

**Verification.**
* `backend/tests/test_countdown.py` - the window is 60 s and minute-aligned, the
  clock counts down monotonically inside a window and never exceeds the window,
  the grid is derived from the window (15/30/45), a pulse carries every panel
  and is not a formula-only message, the client has one timer, and the
  anti-glitch rule (`re-anchor only on a new cycle id`) is asserted in the source.
* `tools/clock_check.js` - boots the live dashboard in a real DOM and watches a
  full boundary: it asserts the digits reach 60, walk down one second at a time,
  never jump upward mid-window, agree with the ring, and that the panels repaint
  in **four batches 15 seconds apart** (the grid), each one a single synchronous
  pass.
* `backend/tests/test_flutter_contract.py` + `tools/dart_balance.py` - the same
  contract in the Flutter client, plus a structural check of the Dart sources
  (there is no Dart toolchain in this environment).

**A bug this round found and fixed.** The first cut of the heartbeat named the
locked side `signal` in the pulse payload. A *signal payload* uses `signal` for
the direction ("BUY"), so the client's shape test mistook the pulse for a signal
payload and the dashboard never adopted the locked signal - the countdown worked
while the signal panel stayed empty. The pulse now says `locked_side`, the client
recognises a signal payload by `lock_state` + `risk` + `cycle_number`, and
`test_countdown.py::test_only_a_signal_payload_is_called_signal` locks the rule
in.

**Escape hatch.** `CYCLE_SECONDS=12` (or any value) restores a shorter window;
`TIME_SCALE` compresses the whole schedule for demos and tests without touching
the ratios. `SIGNAL_PIPELINE=0` restores the literal draft timing.

## I — Microseconds, a one-minute horizon, and six times the data

**Asked for, verbatim.** *"Currently out system is analysing 1 minute prediction
in seconds, it would be better if it analyses in micro seconds and predictions
should be 1 min future when they released. increase details in formulas and
prediction.. You should remove unnecessary code that isn't in use but present.
Also increase data in app, do 6 times then current data app is getting."* Five
requirements, and the round is only done when all five are true at once.

**1. The analysis is microsecond-resolution now.**
* `backend/core/timebase.py` is the one clock: `now_us()` (epoch microseconds),
  `perf_us()` (monotonic, for timings), `us_to_iso()` (six-digit ISO),
  `format_us()` (µs / ms / s), and `span_us()` / `interval_us()` for the tick
  arrays. The market arrays keep millisecond stamps on purpose - a float64 can
  hold an epoch-ms value to ~0.24 µs, which is finer than any exchange
  timestamp - and the module documents that.
* `backend/core/micro.py` measures the tape itself: interval mean / median / p95
  / min / max in µs, jitter (MAD x 1.4826), tick rate and ticks per second,
  burstiness, last-tick age, micro momentum and trend, micro volatility and
  range, volatility per microsecond, aggression, and how long a quote survives.
* `FormulaEngine.run()` records `timings_us` for every formula and for the whole
  pass, `total_ms` remains for compatibility, and each trace now ends with the
  value's range, its cost in µs, the data window it saw (span + ticks), its
  z-score against the retained history, and its units. `FormulaResult.stats`
  carries mean / sigma / min / max / last / z-score / percentile / trend /
  non-zero rate per formula, over a 360-window history.

**2. Every prediction covers the 60 seconds after its release.**
`prediction.horizon()` publishes `released_at` / `target_at` as six-digit ISO
*and* as epoch microseconds, `seconds_to_target`, `microseconds_to_target`,
`scored_in_seconds`, the wording (`"the next 60 seconds"`), and `base_seconds`
(the specification's 60 s) with `accelerated` saying whether this engine is
running a compressed window. `scored_at` is the instant the outcome is
measured, so "what is this signal for" has exactly one answer: the minute that
starts when it is released.

**3. More detail, in the formulas and in the prediction.**
* `logic.DETAIL` gives all 23 formulas their `units`, `sensitivity`, `range`,
  `misleads` and `corroborates`, on top of the expression / steps / bands / sign
  / why that were already there. The Formula Explorer prints them as chips and a
  "when it misleads" list.
* `prediction.detail()` names the side, the number of formulas evaluated, every
  category score, the top supporters and opponents with their values, the
  confidence parts, the levels with their price distance, the engine block
  (compute µs, publish latency µs, tick interval µs, tape resolution, history
  samples), the agents and the brain read-out. The dashboard renders it inside
  the prediction cell behind a toggle.

**4. Unused code is gone, and stays gone.**
`tools/dead_code.py` walks Python (functions, methods, classes), the browser
bundle (functions, element ids), the stylesheet (classes, comments stripped,
compound selectors split, dynamic compositions such as ``sig-${kind}``
understood) and the Dart client. It reports **0** unused definitions, and
`backend/tests/test_dead_code.py` fails the build if that changes. This round
removed: `api/state.local_agent`, `api/routes_formulas._live_payload`,
`brain.load_matrix` / `save_to`, `health_check.health_loop`,
`matrix_builder.set_edge`, `config.reload_settings`,
`cycle_manager.run_cycle_manager`, `errors.DataUnavailable` /
`BrainUnavailable`, `signal_lock.formula_dict`,
`cross_asset_sync.interpolate_ticks`, `simulator.synthetic_flash_move`,
`_util.safe_div` / `hurst_rs`, `engine.directional_values`,
`synthetic.all_scenarios`, `tai._derivatives`, `news.report_price_event`,
`app.js windowElapsed` and the `#timer` / `#progress` / spinner leftovers, plus
every unused import pyflakes could see.

**5. Six times the data.**
`cfg.DATA_MULTIPLIER = 6`: 3 600 ticks, 120 order-book levels per side, 360
candles, 120 news headlines, 120 scored outcomes, 5 400 spread observations, and
a 360-window formula history. The API surface grew with it: 72 outcomes and 72
history windows per call instead of 20, a 30-headline news block instead of 5,
and the signal lock keeps 1 440 locked windows (a day at the 60-second cadence).

**A bug this round found and fixed.** `#lock-state` and `#utc` in the widget
strip were written *inside* a guard on `#timer`, an element no page has rendered
since the ring replaced the old countdown. The two readouts therefore never
updated after the first render - the dead code was hiding live state. The guard
is gone, both are written on every whole second, and
`tests/test_dead_code.py` keeps the dead panels from returning.

**Verification.** `backend/tests/test_microseconds.py` (18 tests: clock, label
units, the analyser on synthetic tapes, logic detail on all 23 formulas, engine
µs timings, the horizon arithmetic, freshness in µs, the detail block, the x6
multipliers and buffers), the extended `test_prediction.py`,
`test_dashboard_scripts.py`, `test_flutter_contract.py` and `test_dead_code.py`,
plus `tools/dead_code.py` itself as the standing gate.

**Escape hatch.** `DATA_MULTIPLIER` is one constant - set it to 1 for the
draft's original buffer sizes. `BASE_HORIZON_SECONDS` is the forecast window;
`CYCLE_SECONDS` still sets the real window.

## J — The crowd's emotions, from microseconds to minutes

**Asked for, verbatim.** *"shouldn't we should have emotions, it would be great
if we consider market sentiments and peoples emotions every micro seconds to
normal seconds. Since I realised 60 second timeframe market easily get
manipulated by retailers emotions. Emotions could be any fear, Happy,
withdrawal and another 4-5 emotions. Also tell me in ui which emotion it is
more in live condition."*

The premise is right, and it is the premise of the whole module: a 60-second
window is short enough that the tape is often a crowd acting on a feeling
rather than on information. So the engine now reads the crowd continuously and
lets the reading shape the prediction - never the side, always the size.

**1. Eight named emotions, each on five timescales.**
`backend/core/emotions.py` scores `FEAR`, `PANIC`, `CAPITULATION`
(withdrawal), `DENIAL`, `HOPE`, `EUPHORIA` (happy), `FOMO` and `COMPLACENCY`,
each an intensity in [0, 1], each decomposed over `micro` (µs–ms: inter-tick
intervals, jitter, tick surge, quote lifetime), `seconds` (1–15 s: velocity,
aggression, spread), `window` (the 60 s: range position, drawdown, volume
climax), `minutes` (1-minute candles) and `news` (headline sentiment and
flow). Every emotion is a weighted mean of legible ramps, and the three numbers
behind it are printed as *drivers*, so the panel can say *why*.

**2. The scales are the tape's own.** A move counts in multiples of the tape's
typical move over the same horizon (measured over *time*, tick against the
last price at or before the horizon, so bursts and gaps cannot make the scale
and the return disagree). "40 bps in a second" is a panic on a dead tape and a
rounding error on a violent one; measuring the crowd on an absolute scale
would have measured volatility, not surprise.

**3. A manipulation read.** The same features produce a 0–1 *crowding* score
with a named mechanism - retail chase (herding: side skew and same-side runs),
stop hunt (a wick beyond the recent extreme that snaps back), whipsaw
(direction flips), book imbalance - plus a note and the evidence.

**4. Continuity, not flicker.** `EmotionTracker` smooths every emotion with a
fast-attack / slow-release EMA, keeps a five-minute history, and elects the
*dominant* emotion with hysteresis: a challenger must lead by 4 points for two
consecutive samples before the headline switches. The bars stay honest; only
the headline waits for the crowd to mean it. The panel therefore shows *how
long* the emotion has held and how many times it switched this minute.

**5. It shapes the prediction.** `fusion.fuse()` takes the crowd reading of the
*frozen* snapshot (the same immutable inputs the formulas see, so the reading
that dampened a window is reproducible from the snapshot alone). Above the
threshold (`emotion_dampen_threshold = 0.45`) the confidence is multiplied by
`1 − 0.25 × excess` (`emotion_dampen_max`); the side is never touched. The
reasoning gains a `crowd` bullet, the prediction detail a `crowd` block and a
`crowd_dampening` confidence part.

**6. It is live.** The cycle manager samples the live tape every
`EMOTION_INTERVAL_SECONDS` (0.5 s) on its own task and streams a compact
`EMOTION` message; every HELLO / SIGNAL / PULSE carries the full block
(`emotions`, including `locked` - the reading at lock time - and `dampening`).
`GET /api/emotions` and `GET /api/emotions/history` expose the same. The
client still has exactly one timer (the safety net): the backend is the
schedule, the browser only draws.

**7. The UI says which emotion is dominant, live.** The dashboard's **Crowd
Emotion** card: the dominant emotion big and coloured by tone, its intensity,
the timescale it lives on, how long it has held, the runner-up, the one-line
read with a size hint, the eight ranked bars, the signed temperature gauge
(fear ← 0 → euphoria), the manipulation meter with its kind and note, the
five-timescale strip for the dominant emotion, and *what the crowd was feeling
when the locked signal was computed*. An inline `crowd now:` line sits under
the prediction itself. The Flutter client mirrors all of it (`EmotionPanel`,
`_CrowdLine`, models in `signal.dart`).

**A Flutter bug this round found and fixed.** The HELLO handler passed
`event.data['signal']` - the direction *string* - to `_applySignal`, which
ignores non-maps, so the Flutter panel stayed empty until the next minute
boundary. HELLO is a flat signal snapshot and is now applied whole, like a
SIGNAL.

**Verification.** `backend/tests/test_emotions.py` (25 tests: vocabulary, the
right emotion on the right synthetic tape, tape-relative scales, manipulation
bounds, the tracker's smoothing / hold time / hysteresis, the fusion dampener,
the reasoning bullet, the live loop, the `EMOTION` stream cadence, the payload
block, the frozen-snapshot reproducibility, the routes, and the two UIs);
`tools/dashboard_payload_check.js` (+14 live checks); the full suite is 140.

**Escape hatches.** `EMOTION_INTERVAL_SECONDS` (default 0.5),
`EMOTION_HISTORY_SIZE` (720 samples), and the two dampening constants in
`config.Settings`.

## K — Deep reasoning under the emotions, microseconds to seconds

**The ask.** "It is basic - we need ultra advanced deep reasoning formulas for
micro seconds and normal seconds." Round J read the crowd with ramps; this round
puts the quantitative-finance layer underneath them, on the same tape, at the
same cadence, and makes the engine show its work.

1. **`backend/core/deep_micro.py`** - the formulas, each in the tape's own
   units so a dead simulator tape and a violent live one read the same way:
   * *µs band*: Hawkes branching ratio `n = 1 − 1/√F` from the Fano factor of
     250 ms arrival counts (self-excitation: prints causing prints), current
     intensity vs baseline; microprice `P* = (Pa·Qb + Pb·Qa)/(Qa+Qb)` and
     top-5 / top-20 depth pressure; quote-stuffing ratio (surge × share of
     prints that moved nothing).
   * *seconds band*: VPIN over 24 volume buckets (order-flow toxicity);
     Kyle's λ by OLS of 1 s mid changes on signed volume, with R² and a
     normalised "pushability"; Lo-MacKinlay variance ratio with its
     z-statistic; the three-state (calm / trend / stress) Gaussian HMM
     forward filter with sticky transitions; the momentum-ignition detector
     (a ≥2σ 1 s burst that gives ≥60 % back within 5 s); the spoofing proxy
     (large resting imbalance while flow is balanced and price is flat).
   * *seconds → minute*: generalised Hurst exponent from the log-log scaling
     of q-sum variance; Bandt-Pompe permutation entropy (order 3);
     Lillo-Farmer sign-memory (ACF at lags 1-13 and the decay exponent γ);
     Haar wavelet energy per dyadic scale, labelled in µs from the tape's own
     median interval.
   * Three **bands** (`micro` on the tick clock, `seconds` on 250 ms,
     `window` on 1 s) each report H, VR (+z) and entropy.
2. **The Bayesian filter.** Nineteen bounded evidence variables (fast / mid /
   slow moves in surprise units, drawdown, run-up, toxicity, cascade,
   momentum, persistence, disorder, herd memory, regime probabilities, spread
   blow-out, book pressure, news tone) go through a naive-Bayes log-odds
   table (`LIKELIHOOD`, one row per emotion) at temperature 1.4;
   `prior = (1−ρ)·previous + ρ·uniform` with ρ = 0.1 per sample (a ~3 s
   belief half-life, slightly steadier than the panel's EMA) and
   `posterior ∝ prior · exp(Σ w·e)`. The output carries the posterior, its
   entropy / certainty, the KL "surprise" of the sample, and the four
   strongest pieces of evidence for the top two emotions. The state lives on
   the `EmotionTracker` (`deep_state`), so the live loop carries belief
   between samples; the locked reading of a frozen snapshot uses a flat prior
   and is reproducible bit for bit.
3. **The reasoning chain.** Fourteen ordered steps - name, formula (as text),
   inputs, value, unit, one-line reading, timescale, the evidence strength
   and the emotions the step argues for (only when the evidence is material,
   |e| ≥ 0.1). Rendered verbatim on the dashboard and in the Flutter panel.
4. **How it feeds the reading.** Every `EmotionScore` now carries `ramp`
   (round J's intensity) and `belief`; the bar is
   `0.65·ramp + 0.35·min(1, 2.5·belief)`, so the ramps still say how *big*
   the behaviour is and the filter how *consistent* the whole tape is with the
   emotion. `read` ends with "the Bayesian filter agrees (72 % belief)" or
   "leans Panic (41 %)". The five detectors (ignition, toxicity, stuffing,
   spoofing, pushable) join the manipulation read as components with their
   own kinds ("momentum ignition", "toxic flow", "quote stuffing",
   "spoofing"), so the fusion dampener sees them without any new plumbing.
5. **Where it appears.** `deep` in every emotion payload (full on REST /
   PULSE / HELLO, `compact()` on the EMOTION stream - ~10 KB at 0.5 s);
   `GET /api/emotions/deep` (live, locked, and the model's likelihood table);
   `prediction.detail.crowd.deep` (belief, regime, the headline numbers, the
   chain) and a "deep read: …" clause on the crowd bullet of the reasoning.
6. **UI.** Web: a DEEP REASONING block under the Crowd Emotion card - the
   belief line with evidence chips (green agrees / amber leans differently),
   seven verdict tiles, the three-band H / VR / entropy table, five detector
   meters, the wavelet energy strip with the peak scale, and the numbered
   chain (collapsible, open by default); the lock-time chain in the signal
   detail. Flutter: `_DeepBlock` in `emotion_panel.dart` with the same
   sections, `DeepReasoning` / `DeepBand` / `DeepStep` models, and the
   lock-time deep line + chain in `signal_widget_panel.dart`.
7. **Calibration notes.** Regime sigmas are 0.8 / 1.0 / 3.0 × the tape's 1 s
   σ (σ = median|move| / 0.6745), stickiness 0.95; the spread blow-out
   evidence is gated on movement and halved for sub-bps spreads exactly as
   the ramps are; PANIC's momentum weight is small (0.3) so a rhythmic tape
   cannot read as panic; DENIAL needs a drawdown and is penalised by calm;
   sign memory is the *signed* mean ACF so an alternating tape is "no herd".
   Cost: ~5-10 ms per sample on a 1 200-tick tape.

**Verification.** `backend/tests/test_deep_micro.py` (25): each formula on a
series with a known answer (random walk → VR ≈ 1 / H ≈ 0.5; AR(±0.5) →
momentum / reversion; monotone path → zero entropy; Poisson vs bursty arrivals;
one-sided vs alternating flow; a planted λ recovered with R² > 0.95; runs vs
alternation; a planted wavelet scale; stress vs calm regimes; a burst that
fades), the filter (normalised, moved by evidence, sticky but not stubborn),
the layer on synthetic tapes (fourteen steps, tone agreement, short-tape
refusal, compact form), and its consumers (report, tracker, stream, route,
served pages, prediction reasoning). `tools/dashboard_payload_check.js` gains
14 live checks (57 total). Full suite 165.

**Escape hatches.** `DeepState.forgetting` / `temperature`, the `LIKELIHOOD`
table and `BANDS` in `deep_micro.py`; the blend weights in
`emotions.analyze`.

## K.1 — Auto-provisioning fixes ("requirements are not automatically downloading")

Found by running the hooks the way a Codespace does (overlapping, detached):

1. **The lock leak.** `run.sh --bg` started the supervisor from inside the
   locked section, so it inherited fd 9 and held the autostart lock for as
   long as the engine ran; every later hook waited (up to 15 min) and looked
   hung. All long-lived children are now started with `9>&-`.
2. **Overlapping hooks.** postCreate / postStart / postAttach can run at the
   same time; two `pip install`s into one `.venv` produce a half-written
   environment. Provisioning and the engine start are serialised with
   `flock`, and `"waitFor": "postCreateCommand"` keeps the editor away until
   the first pass is done.
3. **Provisioning declared "done" too early.** The import check covered seven
   modules; `pydantic`, `python-dotenv`, `redis`, `ntplib`, `msgpack`,
   `python-multipart` and `pytest` were never verified. Both the autostart
   check and `run.sh`'s `REQUIRED_MODULES` now cover every requirement, and
   provisioning retries up to three passes (clean `.venv` on the second).
4. **Things that could wait for a keyboard.** `setup.sh` used plain `sudo`
   (a password prompt hangs postCreate on any image without NOPASSWD) and
   interactive apt; now `sudo -n`, `DEBIAN_FRONTEND=noninteractive`,
   `--force-confold`. `run_web.sh` asked "[y/N]" *before* checking
   `INSTALL_FLUTTER=1`, so the detached Flutter download could block on
   stdin; the prompt is now skipped when unattended (and times out at 60 s
   when not), the script runs with `</dev/null`, analytics are disabled,
   and `flutter precache --web` is run explicitly with retries.
5. **Small engine fixes.** `POST /api/news/emergency` accepts
   `duration_seconds` (10-900) and honours it; `/api/signal/history` clamps at
   the lock's real buffer (1440) instead of 240.

Guarded by four new tests in `backend/tests/test_autostart.py`.

## L — In sync: the two-stage lock, the reveal at zero, and formulas over emotions

**Symptom (verbatim).** "nothing is in sinc. like predictions coming early
before time, emotions and formulas are not co relating. And why the hell
emotions formulas are not powerful … why their is no prediction lock
predictions changing in middle … why app is not giving formulas more value
than emotions … lag in countdown."

**Causes found.**

1. *Predictions were eight seconds stale, and announced early.* The whole
   pipeline (snapshot → 22 formulas → crowd → agents → fusion) ran once, at
   t−8 s, because the agents' 7 s timeouts need that lead. The formulas take
   ~6 ms, so the prediction was built on a tape eight seconds older than the
   boundary it was supposed to describe, and `computed_at` said so.
2. *The client's clock offset came from one sample.* The first HELLO's
   half-RTT set the offset and every later message could only nudge it by
   120 ms. On a proxied Codespace socket the first round trip is long and
   asymmetric, so the countdown ran ahead or behind the server by up to a few
   seconds: the SIGNAL landed while the digits still showed 2–3 ("coming
   early", "changing in the middle") or after the digits had sat on 0 ("lag").
3. *The countdown stuttered under the emotion stream.* EMOTION was ~10 KB of
   bars plus ~8 KB of deep block, twice a second, and the client rebuilt the
   entire card (including the 14-step chain) on every message.
4. *The prediction cell had a live line in it.* `crowd now: …` changed every
   half second inside the locked prediction, so the lock looked broken even
   though side, confidence and levels never moved.
5. *The emotions never saw the formulas.* The crowd engine read only the tape;
   nothing coupled it to the 22-formula vote, so the two could not correlate
   by construction, and nothing on screen said who had the vote.
6. *The emotion formulas were invisible.* Each emotion was a weighted sum of
   bounded ramps - real formulas - but the payload carried only the result.

**Decisions.**

1. **Two-stage lock** (`cycle_manager._pipelined_loop`). Stage 1 at
   t−`LOCK_DEADLINE_SECONDS` (8 s): the slow half - the agents - on a
   provisional snapshot; `NEXT_WINDOW_READY` says "agents ready". Stage 2 at
   t−`FINAL_LOCK_LEAD_SECONDS` (0.4 s, new): `_refreeze()` takes a fresh
   snapshot and re-runs the formulas, the crowd, the fusion and the level
   geometry with the agents' votes carried over, re-stamping `computed_at`.
   The boundary publishes the stage-2 signal. The window payload gains
   `lock{stages, agents_lead_seconds, final_lock_lead_seconds,
   final_freeze_ms, data_age_at_open_seconds, rule}`; live it reads
   `data_age_at_open_seconds: 0.38`. If stage 2 fails the stage-1 draft is
   published (logged), so a boundary is never missed.
2. **Snap on the boundary SIGNAL, nudge otherwise** (web `applyClock`,
   Flutter `_applyWindow(boundary: true)`). The boundary SIGNAL is sent at
   the exact instant the window opens, so its arrival is the one clock sample
   worth trusting outright. The offset is anchored to the *arrival* (not
   arrival minus half an RTT), which keeps the client behind the server by
   the one-way latency: the countdown can only reach 0 when the next SIGNAL
   is already there. Reveal and zero are the same instant; a prediction can
   never appear early.
3. **A lighter stream, repainted only where it changed.** The emotion loop
   sends the deep block on every fourth sample (2 s) and flags it
   (`deep_included`); the bars still move twice a second. The web client
   keeps the last deep block and repaints it only when a new one arrives
   (`emotionsDeepDirty`); Flutter carries it with `EmotionReading.withDeep`.
   The same every-fourth sample re-runs the 22 formulas (6 ms) so
   `last_live_formulas` is 2 s fresh instead of 15 s.
4. **Nothing live inside the locked cell.** `w-crowd-line` / `_CrowdLine`
   now show the crowd *at the lock* with the verdict against the lock-time
   formulas and the confidence cut - frozen for the window. The live crowd
   card moved below the locked-signal card in the page order.
5. **Formulas into the crowd filter, and the rule on screen.** The 22-formula
   consensus (`prediction.consensus`, EMA-smoothed with the emotions' time
   constant) enters the Bayesian filter as three evidence variables
   (`formula_up`, `formula_down`, `formula_conviction`; step 15 of the chain,
   "22-formula cross-check"), pulling the buying emotions up when the
   formulas lean BUY and the selling ones when they lean SELL - with fewer
   than three voters the formulas do not move the filter. Every reading now
   carries `formula_agreement{verdict: aligned | conflict | crowd flat |
   formulas split | formulas silent, consensus, voters, up/down, crowd_tone,
   alignment, note, weights, rule}`; the note is appended to `read`. The rule
   printed everywhere: *the 22 formulas carry 40 % of the direction vote; the
   crowd never votes - it can only cut confidence by at most 25 % when it
   disagrees* (that is what `fuse(crowd=)` has always done; now it is said).
6. **Emotion formulas, printed.** `EMOTION_FORMULAS` gives each emotion's
   symbolic expression, `TERM_DEFINITIONS` (31 terms) the ramps behind it,
   and every `EmotionScore` carries `formula`, `terms[{weight, term, value,
   contribution}]`, `gate`, `ramp`, `belief`. The web card prints the
   selected emotion's formula with the live numbers substituted (click a bar
   to switch); Flutter has the same block with chips.

**Verification.** `backend/tests/test_countdown.py` (two-stage lock: fresh
`computed_at`, same window, agents carried, the loop order), `test_emotions.py`
(formula + terms on every emotion, the agreement verdicts, the live
cross-check), `test_deep_micro.py` (consensus evidence moves the filter the
right way, silent under three voters, 15-step chain);
`tools/dashboard_payload_check.js` 69/69 against the live engine, including
`data_age_at_open_seconds ≤ 1.5`. A 135 s socket capture: SIGNAL at
`:00.013`, `lock=0.38`, `computed_at :59`, side constant within each window.

**Escape hatches.** `FINAL_LOCK_LEAD_SECONDS` (0.05 … `LOCK_DEADLINE_SECONDS`)
moves the final freeze; setting it equal to the agents' lead restores the
single-stage behaviour.

## M. "Still changing every few seconds", the Flutter download, and the brain ❌ rows

**Complaint 1 — the prediction still changes.** The server payload for a
window was already byte-identical apart from `age_*`/`now_us` (Round L probe),
so the cause had to be client-side. A headless DOM run of the real `app.js`
against the live engine (jsdom, now `tools/lock_watch.js`) sampled every
element inside the PREDICTION cell four times a second for 95 s: the side and
confidence changed only at the boundary, but three things repainted
constantly — `w-fresh` ("updated Ns ago", 1 s), `w-horizon-line` ("N.0s left ·
scored in N.0s", 1 s) and `w-micro-line` (the *live* tape from every PULSE,
~2 s). To a person that is "the prediction is changing". Two real bugs sat in
the detail block too: the category line printed `ANaN BNaN…` (category scores
are `{count,sum,mean,directional}` dicts, rendered as numbers) and it said
"24 formulas evaluated" (`len(values)` counted helper keys such as `_hsi`).

*Fix.* The chip is `🔒 locked HH:MM:SSZ` (age in the tooltip; still flips to
red STALE past `max_age`); the horizon line states the fixed release/target
instants and "frozen until then" (`left`/`scored in` in the tooltip); the tape
line reads `prediction.detail.micro` — the tape at the lock — and
`renderLiveFormulas` no longer touches it; the live tape numbers moved to the
Formula Explorer's timings line. `formula_count` counts `ALL_FORMULAS` names
only (= 22); categories print `mean ×count`. Flutter mirrors all of it:
`AppState.micro` is the lock's tape, `liveMicro` the live one (shown on the
hedge dashboard), the horizon/freshness lines have no running numbers.
Re-run of the watcher after the fix: 3 boundaries, 0 mid-window changes.

**Complaint 2 — "sdk and flutter is not downloading".** The download ran as a
`nohup … &` child of a devcontainer lifecycle hook; when the hook finishes its
process group can be reaped, and a Codespace that sleeps mid-clone leaves a
dead pid file — either way nothing said so, and `/flutter` was a bare 404
JSON. Now `backend/api/flutter_build.py` makes the engine the supervisor: on
start-up (in a Codespace, or `AUTO_FLUTTER=1`) it launches
`frontend/run_web.sh` in its own session if the build is absent and nothing is
running; `GET /api/flutter/status` gives `{built, running, stage, sdk_present,
log_tail, last_error, …}`; `POST /api/flutter/build` restarts it; `/flutter`
is a self-refreshing status page with a Restart button until the bundle
exists. The hook itself uses `setsid`, and `run_web.sh` retries `precache`
five times with a growing back-off and prints the failing lines.
`AUTO_FLUTTER=0` still disables all of it (tests, sandboxes).

**Complaint 3 — ❌ 0-cache / 2-auth / 4-flywire.** Those were never failures:
the cache is empty on the first start by definition, and the two tokens are
optional. Steps now carry `state: pass | skip | fail` (with `ok` true for a
skip so older renderers do not paint ❌); the settings table shows ⏭️, the
Flutter brain screen "–". Fallback matrices are cached too under a
`brain:adjacency:source = fallback` marker: without credentials step 0 passes
and reuses it (status stays the honest `FALLBACK_CSV`); with a token present
the live steps run again so a token pasted on `/settings` (new fields for
both, save triggers `/api/brain/reconnect`) takes effect immediately.

**Verification.** `backend/tests/test_round_m.py` (14 tests: static cell in
both clients, skip semantics, cache pass on second run, token bypass,
settings fields, Flutter supervisor + routes, `setsid`); full suite 189
passed; `tools/dashboard_payload_check.js` 69/69 (now asserts exactly 22
formulas and no running countdown inside the cell); `tools/lock_watch.js 100`
→ `OK: the prediction cell was static inside every window`.

**"We have noticed a change to the dev container configuration" (Round M.1).**
VS Code watches the whole `.devcontainer/` folder, and the autostart wrote its
runtime state there (`.provisioned`, `flutter.pid`, `.autostart.lock`,
`logs/`), so every start looked like a config change and offered a rebuild.
All runtime state now lives in `.run/` (git-ignored); `.devcontainer/` holds
only `devcontainer.json` and `setup.sh`. Dismiss the prompt once (or rebuild
once - both are harmless); it will not come back.

**"Failed to download the Flutter SDK" (Round M.2).** A download killed
half-way (the reaped hook) left `~/flutter` as a non-empty directory without
`.git`; `git clone` then refuses ("destination path already exists") on every
retry, and the script gave up. `frontend/run_web.sh` now wipes such a
directory before cloning, checks for ~3 GB free, tries the shallow clone
three times with back-off, and if github.com is the problem falls back to the
official release archive from `storage.googleapis.com` (the same file
flutter.dev's download button serves); the exact failing lines are printed
into `flutter-setup.log`, which `/flutter` shows. Verified from a sandbox: a
junk `~/flutter` is detected and removed, the clone completes in ~40 s.

**No `git pull` (Round M.3).** "Pull usually removes your connection": a
pull that stops to ask (diverged branch, dirty file) leaves the Codespace on
a detached HEAD or a different branch. So the checkout now updates *itself*,
by fast-forward only: `tools/self_update.sh --check|--apply` fetches its own
branch by explicit refspec, `merge --ff-only`, re-runs setup if
`requirements.txt` changed, restarts the engine; it never checks out, never
switches, never touches `main`, and leaves local edits alone (a refused
fast-forward changes nothing). It runs from the autostart hooks at every
start/attach (`--update` mode too), from the engine (`GET /api/update/status`,
`POST /api/update/apply`, both detached/setsid) and automatically every 10 min
in a Codespace (`backend/api/self_update.py`, `AUTO_UPDATE=0` to disable). The
dashboard top bar shows an "⬆ update available · Apply" chip, checked from the
single safety-net timer every 5 min; after Apply the page reloads in 15 s.
Verified end to end in the sandbox: checkout rolled back one commit, engine's
apply fast-forwarded it and restarted on the new code.

**"Still no prediction lock" (Round M.4).** A wider headless-DOM diff (the
whole widget panel + signal card, no grace period) showed that on this build
side, confidence, levels, crowd and reasoning are static inside a window; the
only mid-window movement left was the PREDICTION ACCURACY row (outcomes scored
just after the boundary, per-side rates on the t+15 pulse). It now paints only
in the first 3 s of a window. Two things were added so the lock is *provable
on the user's own screen*: (1) a lock watchdog in `app.js` snapshots the text
of the prediction and TP/SL cells 3 s into each window and compares on every
safety-net tick; a difference turns the chip red ("LOCK BROKEN - see console")
and prints the diff, otherwise the note under the panel counts the windows it
verified (changes only at boundaries); (2) the Engine line and `/api/health`
/ `/api/system/config` carry `build` = the git short hash, so "is the
Codespace on the new code?" is answered at a glance.

## N — "Predictions are not as accurate as they should be"

What was wrong, honestly: the spec recipe (0.40/0.25/0.20/0.15 + HSI) is a
*fixed* weighting. It cannot notice that, on the asset in front of it, some of
the 22 formulas are right 70 % of the time and others are coin-flips or even
reliably inverted — and it summed correlated formulas as if they were
independent witnesses, so it printed 80–100 % confidence for windows that
realised ~51 %.

What changed (`backend/core/calibration.py`, `EvidenceLedger`):

* Every locked window records the directional vote of each source
  (`f:<formula>`, `brain:CCSv2`, `agent:<name>`, `spec:fusion`, `news:NIV`,
  `crowd:tone`, `micro:ret_1m|ret_5m|order_flow`); when the window is scored
  (60 s later, signed P&L — a SELL into a rising tape is now a loss, which the
  old unsigned buffer hid) every source gets an exponentially decayed hit or
  miss (`CALIBRATION_HALF_LIFE`, default 120 windows).
* Weight = log-odds of the Beta-posterior reliability (prior 4, clipped ±1.5).
  A 50 % source weighs 0; a reliably wrong one votes against itself. Sources
  whose edge is not significant at 2 SE pull at quarter weight and the sum is
  shrunk by √(number of pulling sources) — correlated evidence is not counted
  twice.
* After `CALIBRATION_MIN_SAMPLES` (30) scored windows per asset the sign of the
  weighted vote decides BUY/SELL (`fusion.fuse(..., learned=)`; the spec score
  is kept as `spec_score`). The printed confidence is the *realised* hit rate
  of that confidence bucket once the bucket has ≥15 windows — the engine may
  not claim more than its own record supports.
* Guardrail: the ledger compares its own recent hit rate with the spec
  recipe's (`spec:fusion` is itself a scored source); if the ledger is worse
  by >2 pts it hands control back and says so.
* State: `.run/calibration.json` (survives restarts; `CALIBRATION_PATH`
  override). `GET /api/signal/calibration?asset=` → per-source reliability,
  weight, n, verdict (follow/fade/noise), claimed-vs-realised buckets, Brier,
  `ledger_hit_rate`, `spec_hit_rate`. Prediction detail carries `learned`;
  the reasoning line starts with "learned evidence decides (N windows scored)
  …"; the dashboard shows a "Learned reliability" block.
* Tests: `backend/tests/test_calibration.py` (synthetic good / inverted /
  noise sources: spec follows the inverted source <45 %, ledger >55 %;
  activation threshold; disabled; flat windows not scored; buckets;
  persistence; fusion override + confidence cap).

Measured in the sandbox (built-in simulator, TIME_SCALE=20, 118 windows):
spec recipe 53.8 % right, ledger 69.5 %, Brier 0.225, claimed buckets within
~5 pts of realised. That is the simulator, not the market — the honest
expectation on a real 1-minute tape is a small edge over 50 %, and the
dashboard now prints the realised rate rather than a hopeful one.


## O — Three keys per provider with failover, three local models at once

* `backend/agents/keyring.py` — `KeyRing` per provider (gemini, github,
  cryptopanic, newsapi), three `KeySlot`s. Slot 1 is the primary and is
  always preferred; `candidates()` returns healthy keys in priority order.
  `report_failure(kind)` cools one key: `rate_limited` 10→20→40→80→160→300 s
  (reset on success, honours `Retry-After`), `rejected` 600 s, `error` 60 s.
  When every key is cooling, only a key within 5 s of recovery is offered, so
  a dead provider is not hammered. Keys come from `<NAME>`, `<NAME>_2`,
  `<NAME>_3` (or a comma list in `<NAME>`), all optional.
* Gemini / GitHub agents iterate the ring inside the one 7 s budget: a
  rejected or limited key is cooled and the next one answers the same call;
  `last_failover` and the masked ring status are in the agent health payload.
  Provider status is RATE_LIMITED / INACTIVE only when *all* keys are cooling.
* News engine `_poll_with_ring` does the same for CryptoPanic and NewsAPI
  (`cache.providers` says "ok (n) via key 2" when a stand-in answered).
* Routes: `GET /api/settings/keys` (masked rings), `POST
  /api/settings/keys/{name}` gains `slot`, `PUT /api/settings/keys/{name}`
  takes all three; persistence writes the `_2`/`_3` names.
* `backend/agents/local_pool.py` — `LocalModelPool` with three
  `LocalModelAgent` slots (slot 1 keeps the old store directory). Loaded
  slots answer in parallel; `merge_results` = confidence-weighted majority,
  confidence × (0.5 + 0.5·agreement); reasoning lists each model; a model
  that crashed / timed out is reported in `error`, not hidden.
  `POST /api/agents/local/upload?slot=N`, unload `{slot}`, status carries
  `slots[]`, `models_loaded`, `max_models`.
* Settings page: three boxes per provider with live "primary / backup ·
  in use / cooling Ns (reason)" roles, three model slots. Flutter settings
  has the same three boxes (`saveKey(..., slot:)`).
* Tests: `backend/tests/test_keyring.py` (9).

## P — "Remove weightage of emotions to half"

The crowd-emotion layer never picked the side; its weight on the prediction
was (a) a confidence cut of up to 25 % on a crowded minute and (b) the
`crowd:tone` vote in the learned evidence ledger. Both are halved:
`emotion_dampen_max` 0.25 → 0.125 (threshold unchanged at 0.45), and the
ledger scales `crowd:*` sources by `EMOTION_VOTE_SCALE = 0.5` before they
pull. Payload `formula_agreement.weights` now reports
`crowd_max_confidence_cut: 0.125` and `crowd_vote_scale: 0.5`; the dashboard
line and Flutter default updated. The emotion *formulas* and panel are
unchanged - only their say in the prediction is halved.

## Q — "Add double check in each formula"

Every one of the 22 formulas (and DRG) is now verified three independent
ways on **every pass** (`backend/formulas/double_check.py`, wired in
`FormulaEngine._run_one`):

1. **Replay** – the formula is run a second time on a private `deepcopy` of
   its pre-pass state against the same frozen snapshot; the two answers must
   agree to 1e-9 (catches hidden state, non-determinism, mutation bugs).
2. **Re-derivation** – each formula module now carries `double_check(trace,
   asset)` plus a `DOUBLE_CHECK` rule in words: an independent one-line
   recomputation of the output from the intermediates the formula traced
   while running (traces now carry the exact `raw` number alongside the
   rendered one). Formulas that traced nothing at the final step (AFPR, BAR,
   HRDD, ERC, NIV, SMD, KCAE, CCSv2, MCPE) gained those trace rows.
3. **Range** – finite and inside the declared `RANGE` ([-1, 1]; HSI and
   KCAE [0, 1]).

A formula failing any check is zeroed for that pass with
`errors[name] = "double check failed: …"` (so the §10.1 degradation ladder
sees it), logged, and the verdict is appended to its trace ("✓✓ verified:
replay ✓ · re-derived ✓ · range ✓" + the rule). Payload: `checks{}` and
`double_check{formulas, verified, partial, failed, failed_names, check_us,
rule}` on `/api/formulas/live` and the formula result; the Formula Explorer
shows a "✓✓ double-checked" chip per formula and the summary in the live
note. `FORMULA_DOUBLE_CHECK=0` disables it. Cost ≈ 4–5 ms per pass (the
replay doubles the formula work), kept out of the per-formula timings as
`check_us`.

Verified: 17,664 formula evaluations across all synthetic tapes × both
assets → 17,664 verified, 0 false positives; live engine 23/23 verified.
Tests `backend/tests/test_double_check.py` (14) include a lying formula
(zeroed), a non-deterministic one (replay fails) and an out-of-range one.

## R — "It is always breaking lock"

Found the real cause this time, in the engine rather than the client. The
two-stage lock prepares the **next** window during the current one
(`_prefetch` at t+45, `_refreeze` just before the boundary). Both wrote
their results straight into `self.last_fusion`, `self.emotion_locked`,
`self.conviction_note`, `self.warnings` and `self.degradation` — and those
are exactly what `prediction_payload()` / `prediction_detail()` /
`_reasoning_for()` read to describe the prediction *on screen*. So the side
and levels stayed frozen (they come from the `FrozenSignal`) while the
reasoning bullets, supporters/opponents, confidence parts, crowd-at-lock and
the learned block flipped to the next window's — often the opposite side —
15 s before the countdown ended. Round N made it worse because the learned
sentence is the first bullet.

Fix: prefetch/re-freeze now only **stage** (`_stage_view` →
`_pending_view`); `_apply_pending_view()` swaps everything in at the
boundary inside `_open_window`, next to the existing formula-result
handoff. `_fuse()` takes the fresh crowd reading as an argument instead of
reading the on-screen one. Tests: `test_preparing_the_next_window_changes_
nothing_on_screen` (fails on the previous commit, passes now) and a source
guard that `_prefetch`/`_refreeze` never write those five fields. Live:
two full 60 s windows polled every 2 s → one rendering per window.

## S. Named connectome, token-only neuPrint, in-app docs, self-start visibility

**Symptom (user, Round S).** "Problem in self starting and self downloading
engine. Problem in connecting drosophila. Neurons and synaptids of drosophila
is not coded for app. Also could not open markdown preview."

**Cause.**
1. The fallback 80×80 matrix was a seeded random graph with anonymous
   `KC_cluster_N` nodes - the user is right that no neuron or synapse was
   actually coded.
2. `connect_neuprint` imported `neuprint-python`, which is only a commented
   optional requirement; so a Codespace with a valid token could never reach
   step 2 ("neuprint-python is not installed").
3. `markdown.showPreview` is a VS Code extension command; it was missing in
   the Codespace. Nothing in a repository can install an editor extension.
4. The autostart hooks only printed to a terminal tab; a failed step left no
   trace the app could show.

**Decision.**
* `backend/brain/connectome.py` names all 80 nodes with hemibrain v1.2.1 cell
  types: 20 uniglomerular PNs mapped formula → receptor → glomerulus
  (e.g. TAI → Or67d → DA1_lPN), 50 Kenyon-cell clusters in lobe proportions
  (γ / α'β' / αβ, 1,931 cells), PAM, PPL1, OA-VUMa2, MBON-α3 / γ5β'2a / γ3 /
  γ1pedc>α/β, and three lateral-horn populations. Every edge is a `Synapse`
  with a typical synapse count, a functional sign and a pathway label; the
  GCN weight is `sign × synapses × per-pathway efficacy` (a count is anatomy,
  not drive). The fallback CSV is now generated from it; `NODE_TYPES`, the
  brain trace (`dominant_inputs[].neuron`), the wiring payload and the Flutter
  brain screen use the real names. `GET /api/brain/neurons`; `/matrix` has a
  "Neurons & synapses" table. Counts are labelled *typical* (`live=false`)
  until neuPrint measured them - never presented as measured.
* `HttpNeuprintClient` (httpx; `GET /api/version`, `POST /api/custom/custom`)
  is used whenever `neuprint-python` is absent, so a token alone connects.
  Bad tokens surface as "neuPrint rejected the token" in step 2.
* `/readme` and `/readme/{keys|spec|notes}` render the docs in the app
  (`markdown` added to requirements; the autostart re-provisions on a changed
  requirements.txt by itself).
* The autostart script journals every line to `.run/logs/autostart.journal`;
  `GET /api/system/autostart` and Settings → "Codespace self-start" show the
  verdict, failures, journal and the tails of setup/pip/Flutter/server logs.

**Verification.** `test_round_s.py` (8 tests: named nodes match the GCN
layout, signed synapses, CSV == connectome, offline never "live", mock
neuPrint over HTTP incl. 401, docs page + traversal guard, journal hooks).
Formula self-test (CCSv2 approach on BULL / avoid on BEAR) passes with the
efficacy-scaled connectome; full suite 237; payload check 69.

**Escape hatch.** `python -m backend.brain.generate_fallback_matrix --seed N`
regenerates with a different PN→KC draw; a `NEUPRINT_APPLICATION_CREDENTIALS`
token replaces the typical counts with measured ones at the next start.

## T. The thermodynamic capital layer (13-section "planetary metabolism" prompt)

**Request.** A 13-section specification: BTC as irreversibly consumed work
(Landauer), PAXG as inert rest mass; 60 s rebalancing from thermal occupancy
Θ, solar flux Ω, an E = mc² exchange ratio, AMM/venue/peg microstructure,
VPIN, Ornstein–Uhlenbeck, a REI pendulum, Avellaneda–Stoikov quoting, Kelly
blending and a thermodynamic clamp - plus a multi-node ZK Nash swarm,
lending-protocol liquidation sniping and MEV block sequencing.  The user:
"there could be some contradictions ... apply what is best"; asked, chose
**physics = a new weighted layer** (not a hard clamp), **skip the sections
that need infrastructure the app cannot have**, and **keep news + emotions
as they are**.

**Decision.**
* `backend/physics/` is one more weighted voter in fusion
  (`weight_physics`, default 0.20, env `PHYSICS_WEIGHT`; 0 disables).  The
  22 formulas and the brain still decide; a layer running on modelled
  telemetry alone counts half its weight, each live source (hashrate, gold
  spot, DEX pools, other venues) earns part of the rest.
* Implemented: §1 Landauer valve (k_B·T·ln2, 1024 erased bits per double
  SHA-256, Θ against a 30-day-peak S_max, Θ* 0.85), §2 solar geometry
  (declination, hour angle, 12-hub fleet table, Ω = C_hash/C_grid, α blend
  from trailing-hour variances), §4 energy-per-BTC vs refine+transport+custody
  energy per ounce, the per-cycle exp-update and the phase angle (§12.4),
  §5.1–5.2 AMM price surface from DexScreener pools, §8.1 VPIN (50 volume
  buckets, 0.40), §8.2 fragmentation across Binance/Coinbase/Kraken/DEX,
  §8.3 O-U bridge (AR(1) κ, VWAP-300 s equilibrium, SR₆₀ > 1.5 gate),
  §8.4 pendulum (AR(2) ⇒ damped oscillator, 60 s forecast), §8.5 A-S half-
  spread with κ from the book, §10 PAXG/XAU and wBTC/BTC peg bands, §11.3
  multi-mechanism Kelly (Σ from trailing signed edges, ridge, cap ±0.25),
  §12.1 TSR.
* **Not implemented and said so** (`/api/physics/spec`): §3 (many nodes),
  §6 (geographic nodes), §7 (on-chain positions + execution), §9 (the app
  is not a block builder).  §12.6's "iron floor / cannot lose" is **not
  claimed** anywhere; expected edges are labelled model estimates.
* **Deviation from §11.4.** The physical weight is a portfolio target that
  moves over hours; mapping it directly onto a 60-second side would make the
  layer a constant bias.  So: `w_final = clamp(w_micro, w_composite ± Δw)`
  is computed and shown (Δw 0.15, env `PHYSICS_DELTA_W`), but the window
  vote is `2(w_micro − ½)·(1 − drag)`, where drag (≤ 0.6) applies only to
  votes that lean against the physical target.
* Telemetry (`backend/physics/telemetry.py`): mempool.space → blockchain.info
  hashrate; gold-api.com → CoinGecko PAXG+XAUT; DexScreener; Coinbase +
  Kraken; CoinGecko wBTC.  No keys.  Blocked networks read as "unreachable
  here" with the error, never as the API being unavailable.
* Lock: the report rides inside `FusionResult.physics`, so the Round R
  staging covers it - `/api/physics/current` and the dashboard card cannot
  change mid-window.  `/api/physics/live` is a separate fresh pass.
* UI: dashboard "Thermodynamic capital layer" card (verdict, weight bar,
  Θ/Ω/R/Φ/TSR, mechanism table, every formula with its numbers), a
  `physics` reasoning bullet, Flutter `PhysicsPanel` on the brain screen.

**Verification.** `test_round_t.py`: Landauer 2.87e-21 J at 300 K, equinox
noon irradiance ≈ I₀, 27.87 km terminator sweep, valve direction, energy-
mass sign, VPIN/O-U/pendulum/A-S sanity, drag semantics, PAXG = −BTC vote,
fusion weighs but never vetoes, scope declared.  Full suite + payload check.

## U — Flutter SDK install hardening + faster autostart

* Symptom: `fatal: destination path 'FLUTTER' already exists` — a failed earlier clone left a folder with `.git` but no `bin/flutter`, and the old pre-flight only wiped folders *without* `.git`, so the installer looped forever.
* Fix in `frontend/run_web.sh`: any non-working folder at the SDK path (`~/flutter`, `~/FLUTTER`, `~/Flutter`) is removed or moved aside (fallback home `~/flutter-sdk`), the chosen home is remembered in `.run/flutter_home` (also read by `backend/api/flutter_build.sdk_dir()` and `FLUTTER_HOME` wins over both). Route A is now the official release archive (one resumable `curl -C -` tar.xz, versions from `releases_linux.json` with hard-coded fallbacks), route B a shallow git clone into a *staging* dir followed by `mv` — nothing ever clones into an existing path again.
* Manual path (documented in README and the `/flutter` failure text): download `flutter_linux_<ver>-stable.tar.xz`, extract to `~/flutter`, run `bash frontend/run_web.sh`.
* Autostart speed: `.devcontainer/setup.sh` skips apt entirely when venv/curl/git/unzip/xz already exist (the devcontainer image ships them), pip uses `--prefer-binary`; `devcontainer.json` `waitFor` is `onCreateCommand` so the editor attaches while provisioning continues in the background; the attach-time lock wait dropped 40 s → 20 s.

## V — why the predictions were wrong: five engine bugs, fixed

Found by polling `/api/signal/current` over consecutive windows and probing the brain with unit inputs.

1. **Take-profit / stop-loss never moved.** `realized_volatility_bps` scaled *per-tick* returns by `sqrt(horizon / span)` — ticks are not time, so a 60 bps/min tape read 1.7 bps and every level sat on the 4 bps floor. On top of that only the Binance kline stream filled the candle buffer; CoinGecko and the simulator left it at the warm-up contents forever. Now: every feed rolls its own 1-minute closes from the tape (`MarketDataHub._roll_candle`, Binance klines stay authoritative), and the fallback estimator resamples the tape on a 5 s grid and scales by `sqrt(time)`. Levels now follow the market each window (`test_round_v`).
2. **The brain only heard the bullish half of the formulas.** The PN→KC fan-in is rectified, so NIV = +0.7 read +0.38 at the lateral horn but NIV = −0.7 read −0.02, and LCS = −0.75 changed nothing. Deviation 22-A (zero-vector baseline) could not fix a *gain* asymmetry. **Deviation 22-D**: push-pull read-out — the mirrored ensemble is propagated through the same wiring and `approach = (approach(x)+avoid(−x))/2`, `avoid = (avoid(x)+approach(−x))/2`; the score is now an odd function of the input and the resting offset vanishes by construction.
3. **Nine of twenty inputs were wired with the wrong valence.** Probed alone with +1, DGW, LCS, GCDV, RSV, MCPE, MPS, TWRS, DSKD (and ERC) reached the lateral horn as *avoid*, so a bullish formula set could vote SELL and vice versa — the circuit's output was a scramble, not a consensus. **Deviation 22-E**: `GraphConvolution.calibrate_polarity()` orients every PN at the input so that positive = approach (the ON/OFF channel choice); the wiring is untouched. Together with 22-D, on a flat tape CCSv2 now reads ≈0 (it read +0.42) and `all +0.5 / all −0.5` give ±0.29.
4. **Non-directional formulas pushed the side.** HSI and ERC (`DIRECTIONAL = False`) entered the PNs as positive magnitudes. **Deviation 22-C**: they enter at zero (HSI still reaches the circuit through the octopamine node).
5. **One stale headline held NIV at +0.72 forever.** With the spec's `/ (Σw + eps)` the decay cancels between numerator and denominator. **Deviation 19-A**: `/ (Σw + W0)`, `W0 = 0.5` resting evidence — a lone 25-minute-old wire now reads ≈0.02, five fresh agreeing headlines ≈90 % of their sentiment.
6. **DRG was pinned at +1.** `newest_m × newest_o` used the *signed* P&L, so a −40 bps loss scored (+40): now `|pnl| × outcome`, and the dopamine gate goes to −1 after a loss as designed.

Also in this round: the 22 formulas' weighted consensus is a fusion voter in its own right (`weight_formulas` = 0.30, env `FORMULAS_WEIGHT`; a modelled physics vote at 0.10 could previously out-vote twenty live formulas when the brain was flat); a coherent consensus also feeds the confidence term. The simulator's regime drift is cut from 210 to ~20 bps/min and bursts from three a second to one per 30 s, so the offline tape looks like an exchange instead of a trend generator. `BinanceWebSocket` rotates to `data-stream.binance.vision` (Binance's public market-data host, reachable from the US regions Codespaces run in) when `stream.binance.com` refuses the connection.

What this does **not** change: a 60-second BTC direction call is close to a coin flip for any model; the engine now reports *honest* confidence (single digits to ~30 % when the inputs are weak, instead of 56 % from an 18 % brain read-out) and a hit rate measured in `/api/signal/outcomes`. No yield is claimed.

### V.1 — "lock broken 8×": it was the emergency override firing on politics headlines

The mid-window flip (`SELL 25 %` → `BUY 100 %`, empty `computed` stamp, ⚡ shown as 🔒) was Section 4.3's critical-event override, and it was firing far too often: `critical_keywords_in` did a substring match, so **"war" matched "warning"/"award"/"forward"**, "hack" matched "hackathon", and any Tier-1 wire about elections or trade policy flattened the open position; the same headline re-fired on every 30 s scan; hours-old RSS backlog counted as breaking news. Fixes:

* whole-word keyword match; the headline must also name a market term (bitcoin/crypto/exchange/stablecoin/gold/fed/bank/…); only items published in the last 10 minutes qualify; a headline fires once (dedupe set). Flash-move and NIV-swing triggers are unchanged.
* the override signal now carries `computed_at`, keeps the window's `valid_from/valid_until/window_seconds/risk`, and the chip reads **⚡ emergency re-lock HH:MM:SSZ** (amber) instead of 🔒.
* the dashboard watchdog records an override as "⚡ emergency re-lock" — the one documented mid-window change — and keeps checking the *new* text; anything else is still reported as a broken lock.

## W — "it is not starting automatically"

Two things made a healthy autostart *look* dead. (1) Round U handed the editor over at `onCreateCommand`; the attach hook then blocked silently on the provisioning lock for the minutes pip needs. Reverted to `waitFor: postCreateCommand`, and `with_lock` now reports every 15 s what it is waiting for (last line of the setup log). (2) Codespaces opens port 8000 in the browser as soon as the container exists - before the engine can listen - so the tab showed a connection error. `tools/placeholder_page.py` (stdlib only, runs on the system python before the venv exists) now answers on the port at the very start of every hook with a self-refreshing "setting itself up" page that shows the autostart journal live; `/api/*` answers 503 so the health probes are not fooled; `start_engine` and `run.sh --bg` stop it and take the port over. Rehearsed end to end in the sandbox: placeholder `200`/`503` at 4 s, engine answering, banner printed.

## X — "freeze after the second prediction, only the countdown running"

The countdown is client-side (absolute deadline), so it keeps running when the engine stops talking - which is exactly what happened and why it looked like "calculating on the UI only". Defences now exist on every layer:

* **Engine loop**: a failing window used to retry the *same past deadline* every 0.5 s forever (`index` never advanced), so one persistent exception froze publishing. It now records `last_error`, appends to `warnings`, skips to the next boundary and publishes there (inline compute if nothing was prepared). `/api/health` reports `stalled: true`, `seconds_since_window` and `cycle_manager: "STALLED"` when no window opened for period + 45 s.
* **Supervisor** (`run.sh --supervise`): the engine runs as a child and is health-checked every 10 s (`curl -f /api/health`, 5 s timeout). Six consecutive failures - no answer *or* `stalled: true` - mean TERM, then KILL, then the usual 2-second relaunch. Rehearsed with `SIGSTOP` on the engine: restarted after ~60 s.
* **Browser / Flutter**: a WebSocket that is OPEN but silent for 15 s (the Codespaces forwarder keeps the client side alive after the backend side died) is closed, which reconnects and replays `HELLO`; meanwhile the 5 s safety net polls `/api/signal/current` and the header says "engine silent / unreachable" instead of "live".
* **Most likely trigger**: the Flutter web build saturating a 2-core Codespace ~2 minutes after start (right after the SDK download) and starving the event loop. The build now runs at `nice -n 19` / `ionice -c 3` from both launch paths.
* Simulator warm-up is 600 s (10 one-minute candles), so RSV and the volatility estimate have history from the first window.

## Y — one logic, history that ramps in, asymmetric risk, probability branches

* **Same TP/SL every window** is what a stale Codespace shows (pre-Round-V
  code).  The updater refuses to fast-forward a checkout that has local commits
  or local edits, and the chip used to stay hidden in that case.  Now it shows
  **STALE BUILD · n updates not applied — reason** in red; the footer build hash
  is the thing to compare with the branch head.
* **Decision chain** documented once in `docs/DECISION_CHAIN.md`; every
  contradiction and its resolution listed there.
* **Prediction history → formulas**: the EvidenceLedger's verdict now blends in
  continuously (`λ = scored / CALIBRATION_MIN_SAMPLES`, default 20) instead of
  switching on at 30; a source needs 6 scored votes (was 10) before it pulls.
  Reasoning line: `prediction history weighs 45% of this call (9 of 20 …)`.
* **Risk** is asymmetric by rule: stop = 1.5 σ (clamped), target = `RR_TARGET`
  × stop (1.5). Tests updated (Round F 1:1 contract superseded).
* **Probability branches** (`prediction.branches`): drifted-Brownian fan
  (5/25/50/75/95 %) with drift chosen so P(close on side) = 0.5 + confidence/2 (confidence = edge, one scale everywhere); odds of
  the three endings; P(target before stop) by gambler's ruin. Analytic, no RNG.
  Dashboard draws it with the realised path (white) from `live_price` carried on
  PULSE and EMOTION messages; Flutter prints the odds line.
* **Synesthesia** (opt-in): WebAudio pitch = depth imbalance, loudness = tick
  surge; page glow = dominant-emotion intensity. Off by default.
* Declined honestly: atomic ingestion, neural weave, pre-cognition, guaranteed
  zero slippage, sovereign capital, ghost hedging, femtosecond yield.
* Cache-bust `v=2.17.0`.

## Z — the fresh-Codespace root cause, `&b`, `¶gn`, world news impact, the geometric emotion layer

* **Root cause of "same TP/SL, same signal, stuck after 1–2 predictions"**
  (only reproducible in a fresh Codespace with a real Binance connection):
  `MarketDataHub` started the seeded simulator two seconds after boot
  because Binance had not connected *yet*, and nothing ever stopped it when
  Binance did connect.  Both feeds wrote into one tape (simulator ≈65k
  alternating with the real price).  Realised volatility pinned the levels at
  their ceilings every window, every formula read noise, and - the simulator
  being seeded - every new Codespace failed identically.  Fix: all data enters
  through `hub.ingest(source, …)`; a feed that is not the active source is
  rejected and counted; **any** source change flushes the tape; a real feed
  gets `REAL_FEED_GRACE_SECONDS` (45) before the simulator may start; the
  simulator task is cancelled the moment a real feed is healthy.  The header
  shows `live tape · binance · N ticks` / `SIMULATED TAPE` (red) /
  `connecting to the market…` from `/api/signal/status.tape`, live.
  Tests: `test_round_z.py`.
* **Second real tape**: `backend/data/kraken_ws.py` (Kraken v2 public,
  BTC/USD + PAXG/USD, trade + book 25, reachable from US Codespaces).  Order:
  Binance → Kraken → CoinGecko → simulator.
* **`&b`** (direct internet connection per formula): every `FormulaSpec` has
  `feeds` (tape/book/candles/news/cross/formulas); each pass carries
  `provenance` (source, per-feed liveness and age) and `feed_status`
  per formula; the explorer shows `&b live / simulated / offline` per row.
* **`¶gn`** (sync with the timer): each pass carries `phase`
  (`offset_seconds`, `mark t+15s…`, `window_start`) from the shared UTC
  minute grid; shown per row.
* **News impact** (`backend/news/impact.py`): 15 themes incl. war, terror /
  cyber attack, sanctions / tariffs, hawkish / dovish central banks, dollar,
  recession, gold, hack, depeg, regulation, adoption; signed per asset
  (war: BTC −, PAXG +).  Aggregated with a 20-min half-life and tier weights
  → `fusion` voter `news` (`NEWS_WEIGHT` 0.15, scaled by news mass), per
  asset.  World feeds added to `RSS_FEEDS` (BBC World, Al Jazeera, Fed press,
  CNBC World, Google News query).  News card shows the net impact and the
  theme tag on each headline.
* **Geometric emotion layer** (`backend/core/geometry.py`): TDA of the book
  (0-dim persistence of liquidity cavities, Betti-0 curve, tearing), Takens
  (3,1) embedding + Rosenstein Lyapunov, critical slowing down (Δa₁ × variance
  ratio), BTC↔PAXG entropy production σ = J·X, non-commutative decision
  interference I = P(A) − Σ P(B)P(A|B).  Six new evidence variables in the
  Bayesian emotion filter; "GEOMETRY OF THE CROWD" block in the dashboard.
* Not yet done from the same request (next rounds): premium UI redesign,
  rewrite of countdown / lock weighting / TP-SL as new modules, inverse-RL
  risk-aversion estimation, Flutter rendering of the geometry block.
* Cache-bust `v=2.18.0`.

## AA

**Trigger.** From a fresh Codespace: "&b is offline all over everywhere, hedge
status is always 0.00". `&b offline` (not "simulated") plus every hedge number
at zero means the tape source stayed `none`: neither WebSocket feed connected
and the hedge formulas never saw a PAXG tape. The sandbox cannot reproduce it
(no exchange egress), so this round makes the *reason* visible and adds a
feed that does not need WebSockets at all.

**Feed diagnostics.** `BinanceStatus` / `KrakenStatus` carry `last_error` and
`last_error_at`; `MarketDataHub.feeds_report()` lists every feed (enabled,
connected, failures, messages, server, last_error, age) plus CoinGecko and the
simulator. It is served as `/api/signal/status.tape.feeds` and
`/api/health.feeds`; the header chip prints
`NO MARKET FEED · binance: <error> (×n) · kraken: …` instead of a bare
"offline". CoinGecko starts after **3** consecutive failures of both sockets
(was 10 of Binance alone); the source stays an honest `none` for
grace + 30 s before the simulator is allowed in.

**Kraken REST (`backend/data/kraken_rest.py`).** Real trades
(`/0/public/Trades`, incremental `since`) and a 25-level book
(`/0/public/Depth`) over plain HTTPS GET every 2 s for XBTUSD and PAXGUSD.
It is a `REAL_SOURCES` member (`krakenrest`), started by `_reconcile_source`
as soon as both sockets have failed twice, and takes the tape ahead of
CoinGecko and the simulator (order: Binance → Kraken WS → Kraken REST →
CoinGecko → simulator). Mode `MARKET_DATA_MODE=krakenrest` forces it.

**Risk engine (`backend/core/risk_engine.py`).** TP/SL rebuilt from scratch
on excursions rather than closes: the stop is the q-quantile of the maximum
adverse excursion of the window (`a_q = σ_T · Φ⁻¹(1 − (1−q)/2)`; q = 0.80
BTC → 1.2816 σ_T, 0.78 PAXG), floored by 3× the quoted spread and the
settings clamps; the target is `rr · stop` with
`rr = rr_target + |edge|` clipped to [1.2, 2.5] - a call we barely believe
reaches less far. `risk_levels` keeps its API (plus `edge`, `spread_bps`) and
publishes the engine block (`risk.engine`: sl/tp/rr/sigma_window/mae_z/
spread_floor/method). The note prints the method.

**Lock weights (`backend/core/lock_weights.py`).** One table:
`final(v) = base(v) × reliability(v) × availability(v)`; base from
settings, reliability from the ledger's sources mapped to each fusion voter
(`formulas` ← every `f:*`, `drosophila` ← `brain:CCSv2`, …) as a shrunk hit
rate → multiplier in [0.4, 1.6], availability = physics liveness / news mass
/ agent answered. `fuse()` builds the table (`ledger_sources=` from
`CycleManager._ledger_sources_safe`) and publishes `fusion.lock_weights`;
the "who decides" line shows `(x1.40 earned)` and `50% live` tags.

**Window clock (`backend/core/window_clock.py`).** The countdown's single
source: `grid_offsets`, `grid_marks`, `phase` (progress + label opening /
early / mid / late / closing - the `¶gn` stamp) and `describe(...)` (the
absolute clock block). `CycleManager.tick_grid` / `master_clock` delegate;
`clock.phase` is new in every payload.

**Inverse RL (`backend/core/utility_inversion.py`).** The crowd's utility
recovered from its actions: **λ** loss aversion from
`flow_t = β⁻·min(r_{t−1},0) + β⁺·max(r_{t−1},0)` on 1 s buckets (responses
must be > 1.5 standard errors from zero or they are noise; λ = β⁻/β⁺, or a
t-scaled value when only one side moves the crowd); **γ** risk aversion =
`ln(participation_calm / participation_volatile)` over volatility terciles;
**α** probability weighting from the share of depth beyond 2 σ_window versus
an indifferent book's `(span − 2σ)/span` with `w(p) = p^α`. Published as
`deep.utility` (`lambda`, `gamma`, `alpha`, `intensity`, `read`, `method`);
six new evidence variables (`loss_averse`, `gain_chasing`, `risk_averse`,
`risk_seeking`, `tail_fear`, `complacent`) enter the Bayesian emotion
likelihoods (PANIC/CAPITULATION ← λ, FOMO/EUPHORIA ← 1/λ and −γ, FEAR ←
α<1, COMPLACENCY/DENIAL ← α>1).

**Premium UI.** `styles.css` gained a premium layer (Inter / JetBrains Mono,
aurora backdrop, glass cards with hairline gradient borders, breathing
status dot, SVG countdown ring with the `¶gn` marks drawn on the arc, number
tick animation, pulse breath on every PULSE, value flashes via one
MutationObserver, hover lifts, reduced-motion respected). Haptics: `[5]` at
each pulse, `[6]` on the last three seconds, `[10]` when the next window is
sealed, the reveal unchanged. New dashboard block "WHAT THE CROWD IS
MAXIMISING" (λ / γ / α tiles with scales and the regression detail). Assets
stamped `v=2.19.0`. Still exactly one `setInterval`.

**Flutter.** `DeepReasoning` parses `geometry` (`CrowdGeometry`) and
`utility` (`CrowdUtility`); `emotion_panel.dart` renders GEOMETRY OF THE
CROWD (five meters) and the inverse-RL tiles with animated scale markers.

**Checks.** 278 tests, `tools/dead_code.py` 0, payload check 69/69,
`node --check`, pyflakes clean, `tools/dart_balance.py` OK.

## AB

**Trigger.** "Code the app so it automatically connects to the live internet
as soon as I open it - only Arena is blocked, GitHub Codespaces are not.
Each panel matters."

**Root cause candidates, now handled in code.** (1) Codespaces run in US
regions where Binance answers 451; (2) a socket whose handshake succeeds
but that never delivers data used to count as *healthy*, resetting its
failure counter on every reconnect, so neither CoinGecko nor the simulator
was ever allowed in and the tape stayed `none` ("&b offline", hedge 0.00).

**Connectivity probe (`backend/data/connectivity.py`).** At hub start-up
(any mode but `simulator`) six endpoints are probed in parallel with a 4 s
budget - Binance mirror, Binance, Kraken, CoinGecko, BBC world RSS,
CryptoPanic. The report (`ok`, HTTP status, ms, error - 451 is labelled
*geo-blocked*) is published as `tape.connectivity` on `/api/signal/status`
and inside `/api/health.feeds`, and the header chip appends
`internet: reachable: kraken, coingecko · blocked: binance (HTTP 451 geo-blocked)`.

**Decisions from the probe.** If Binance is unreachable, Kraken REST starts
immediately (first window already has real prices); if no socket host is
reachable the 45 s grace is skipped and the HTTPS tape takes over at once.

**Silent sockets.** `BinanceStatus` / `KrakenStatus` gained `data_messages`
and `healthy()` (= connected ∧ data ∧ not stale). Failures are forgiven only
when real data arrives; a session that closes with zero data raises
"connected but no market data arrived" and counts as a failure. The hub's
reconciliation uses `healthy()`.

**Every panel.** Each card head carries a rail with the same three facts:
`&b <source|simulated|offline>`, `¶gn <opening|early|mid|late|closing>`, and
the age of the panel's last repaint - all repainted from the one frame
loop, so every panel visibly moves together. Panel internals were restyled
(hedge tiles, agent rows, news cards, history rows, physics table, brain
wiring, explorer rows, locked-signal reasoning block). Assets `v=2.20.0`.

**Checks.** 281 tests, dead code 0, payload check 69/69.

## AC — auto-start survives hook teardown; Flutter is pre-built on GitHub

**Auto-start.** The engine supervisor started by `run.sh --bg` is now launched
with `setsid nohup … 9>&-` (own session, lock fd closed), so it survives the
devcontainer lifecycle hook ending; a `.vscode/tasks.json` `folderOpen` task
re-runs `tools/codespace_autostart.sh --attach` whenever the folder opens.

**Flutter root cause.** `frontend/web/` (Flutter's web scaffold: `index.html`,
`manifest.json`, icons) had never been committed, so `flutter build web`
refused on every machine with *"This project is not configured for the web"*
— the SDK download was never the real blocker. The scaffold is now tracked and
both build routes regenerate it (`flutter create . --platforms web`) if lost.

**Pre-built bundle.** `.github/workflows/flutter-web.yml` builds the web client
on GitHub (stable channel) on every push to `arena/**` touching `frontend/**`
and commits `frontend/build/web` back to the same branch (`[skip ci]`, rebase
onto the moved branch, never main; failures are posted as a comment on the
branch's PR because the Actions log is unreadable from the sandbox). `.gitignore`
now ignores `frontend/build/*` except `web/`. `tools/self_update.sh` replaces a
locally built, untracked bundle with the incoming tracked one before the
fast-forward. The engine's `/flutter` route already checks `built()` per
request, so the bundle is served the instant it is on disk.

**Checks.** 281 tests; first GitHub build `586bbee` (42 MB, Flutter 3.47.6).

## AD — nothing left to download: pre-provisioned dev-container image

`.devcontainer/Dockerfile` (built by `.github/workflows/devcontainer-image.yml`
on every change to `requirements.txt`/the Dockerfile, pushed to
`ghcr.io/apidemo495/p-4-dev:latest`) bakes Python 3.11 + all requirements into
`/opt/venv` (`VIRTUAL_ENV`, on `PATH`), plus redis-server and gh.
`devcontainer.json` now uses that image (the github-cli feature is gone - gh
is in the image). `.devcontainer/setup.sh` and `run.sh` adopt `/opt/venv` as
`.venv` (symlink) when `.venv` is missing, and setup.sh skips pip entirely when
every required module already imports. Measured: provision 1.3 s, `run.sh
--setup-only` 0.6 s. Combined with the committed Flutter bundle (§AC) a fresh
Codespace downloads nothing: image pull → engine up → `/flutter` served.

## AE — real wire read 0.00; hub stuck on "none"

**News 0.00.** The spec keyword lists are crypto-desk words; a real world wire
("Israel strikes Gaza", "tariffs on Chinese goods", "drone attack hits power
grid", "stocks fall as yields climb") scored exactly 0.0, so NIV / news impact
read 0.00 in every real Codespace while the offline fixtures looked fine.
`score_headline` now adds macro risk-on / risk-off lists and blends in the
signed BTC impact of the headline's theme (`impact.classify`); the war/attack
themes gained the phrasings wires actually use.

**&b offline / hedge 0.00.** `_reconcile_source` could hold `"none"` while a
real feed (Kraken REST started at boot because Binance is geo-blocked,
CoinGecko) was delivering and having every write counted as *rejected*. The hub
now records `last_delivery[source]` in `ingest` and adopts a delivering real
feed whenever neither socket is healthy — evidence beats state.

**Diagnosis.** `GET /api/feeds/diagnose`: HTTP probes, a real WebSocket
handshake + subscribe on Kraken and Binance, one RSS fetch, the hub's active
source / delivery ages / rejected writes and the news engine's item count and
NIV, with a one-line verdict. Tests: `test_round_ae.py` (284 total).

## AF — Gemini keys, hedge pair on a real tape, graded `&b`

**Gemini.** The default model `gemini-1.5-flash` is retired (404 → "HTTP
error" on Test). `GEMINI_MODEL=auto` now asks ListModels which models *this
key* can call and picks the first of `PREFERRED_MODELS`; a 404 mid-run
re-discovers and retries once. The Test button validates through ListModels
(no prefix check — `AQ.Ab…` and `AIza…` both fine) and names the chosen model.

**Hedge 0.00 on real feeds.** HSI/HRDD/SHRP/GCDV read `synced` — BTC and PAXG
LOCF'd onto a fixed 60 s grid. A real PAXG tape prints a few times a minute
(Kraken) or every 10 s (CoinGecko), so on 60 s it was a flat line: zero
variance, every hedge formula 0.00, while the dense simulator looked fine.
`synchronise()` now widens the grid along 60→120→300→600→900→1800 s until
both legs have ≥ 6 distinct prices (`SyncedSeries.describe()` reports window,
updates, `widened`, `reason`), and an invalid pair says which leg is thin.

**`&b` rewrite.** `provenance()` grades every feed (tape, book, candles, news,
cross) as live / delayed / derived / warming / stale / simulated / offline
with a note, and the pass gets `grade` ∈ {live, partial, simulated, offline}
plus `coverage` (weighted mean). `feed_status()[formula]` carries `grade`,
`coverage`, `detail`. Dashboard chips read `&b live` / `&b partial 62 %` /
`&b simulated` / `&b offline`; tooltips and panel rails list each feed and
why. "offline" now means *no real data at all*.

**Also.** `tools/self_update.sh --check` emitted an all-digit short SHA bare
(`"local":0971472`) → invalid JSON; SHAs are always strings now. Assets
`v=2.21.0`. Tests: `test_round_af.py` (288 total), payload check 69/69.

## AG — one frontend

The user chose to merge the two frontends into one: the Flutter client is gone.
Removed: `frontend/` (Dart sources, web scaffold, the committed bundle), the
`flutter-web` GitHub Action and its commit script, `backend/api/flutter_build.py`,
the `/flutter*` and `/api/flutter/*` routes, `start_flutter` in the autostart,
`AUTO_FLUTTER` / `INSTALL_FLUTTER`, the Dart VS Code extensions and task,
`tools/dart_balance.py`, the Dart scanner in `tools/dead_code.py`, the Flutter
assertions in the tests, and every README / spec reference. Everything the app
offers is the web dashboard on port 8000 - which already carried every feature
(the Flutter client rendered the same payloads). `.gitignore` no longer tracks
a build directory. Tests 270, dead code 0, payload check 69/69.

## AH — Gemini: newest generation wins

`choose_model` ranks the ListModels answer by generation parsed from the id
(3.8 > 3.7 > 3.6 > 3.5 > 3 > 2.5), then flash > flash-lite > pro, then GA >
preview; embedding / image / tts / live / audio ids are excluded. 3.5–3.8 (and
anything newer) are picked automatically; `PREFERRED_MODELS` is only the
offline default. Tests in `test_round_af.py`.

## AI — the Codespace start, rehearsed on GitHub

`.github/workflows/devcontainer-rehearsal.yml` (devcontainers/ci) pulls the
exact image devcontainer.json names, runs postCreate/postStart/postAttach the
way Codespaces does, then checks `/api/health` and `/api/signal/current` and
posts the transcript on PR #1 (Actions logs are unreadable from the sandbox).
First run: provisioned in 2 s (`/opt/venv` adopted), engine answering at once,
Kraken WS connected (Binance 451 from GitHub's US region), first signal BUY
locked 5 s after creation. The container start therefore works unattended;
remaining failure modes are a Codespace created on `main`, recovery mode, or
a hook still running - all documented in the README with the one-click
`codespaces.new?ref=arena/01a0c844-p-4` link. Also: when the probe says
Binance is blocked and Kraken reachable, Kraken is the active tape from boot
(no two seconds of rejected writes).

## AJ — physics rebuilt on the tape, hedge outcomes, &b, strict ledger, news novelty

**User message.** "Fix all the formulas of physics some are 0.00, remove hedge
status instead of that add more powerful hedge outcomes system, try to fix &b
partial. Its predictions are terrible fix them. Change in physics formula add
new ones that needed and remove useless. Fix each formula of maths they are
calculating wrong or their calculations lead to loss. Strict the learning
system. Fix weightage of news since not all time same news affect market
again and again."

**Physics (`backend/physics`).** Retired §1 Landauer, §2 solar, §4 E=mc² and
§5 AMM (`physical.py` deleted, hashrate/DEX telemetry dropped): the global
hashrate is constant inside a minute, so Δln R, Ṙ and the phase angle were
0.00 / "balanced" every window and the AMM needed DexScreener. New
`kinetics.py`: Hawkes branching ratio (variance-to-mean of 1 s arrival
counts), momentum flux z-score with kinetic energy, trade-sign block entropy,
cross-leg lagged diffusion, tape temperature (drag). O-U is cost-gated
(half-life 5-600 s, E[ΔP] > spread, edge ≤ 1 σ₆₀ instead of a 20 bp cap that
dominated the Kelly). Every mechanism carries `active` + reason; the Kelly
runs over active mechanisms only; the vote is halved when gross edge ≤ spread
cost. `/api/physics/spec` marks the retired sections. UI card rewritten.

**Hedge outcomes (`backend/core/hedge_outcomes.py`).** Bivariate-normal joint
matrix (Drezner-Wesolowsky quadrature), β both ways, ρ², four pair actions
with E[bp]/σ/P(profit), scenarios, spread z, regime from HSI/HRDD/GCDV/SHRP.
The lock's conviction maps to P(side) = ½ + 0.45·conviction. Lives in
`signal.hedge.outcomes`; the UI card "Hedge Outcomes" replaces the tiles.

**&b.** `derived` (widened grid, REST without a book) now counts as live;
the grade is over the market feeds only (`MARKET_FEEDS`), news reported beside
it (`news_state`); a quiet tape is allowed three median inter-print gaps
before "delayed"; `holding_back` names the feed when it is partial.

**Maths.** BAR: fraction of resting depth consumed, 5 % = full vote, < 0.5 %
= 0 (it printed −1.000 on a 0.01 BTC bid change). TWRS: returns winsorised at
±4σ, tanh(skew/1.5) gated by the skew's t-statistic (2 → 5 SE) - it was
pinned at −0.9996 by single prints.

**Ledger.** Prior 6, clip 1.2, verdicts at 2.5 SE and ≥ 12 samples, noise
pull 0.10, `CALIBRATION_MIN_SAMPLES` 40, printed probability ≤ realised
bucket rate − 1 SE; `score(move_bps, cost_bps)`: a window inside the spread
is *flat* - no hits, half a miss to every voter, a non-hit in the bucket.
`news:theme:<name>` votes per driving theme.

**News.** `impact.aggregate`: Jaccard ≥ 0.6 duplicates weigh nothing; k-th
headline on a theme weighs 1/k; `habituation()` halves a theme every 2 h of
continuous presence (reset after 90 min away); four-hour cutoff; drivers
carry `theme_key`, `novelty`, `nth_on_theme`, `habituation`.

**Verification.** `test_round_aj.py`, `test_round_t.py` (rewritten),
`test_calibration.py::test_round_aj_*`, `test_round_z.py::test_round_aj_*`;
full suite 274; dead-code 0; assets `v=2.22.0`.

## AK

- neuPrint token test: `/api/databaseInfo` is gone after the neuPrint platform
  migration (404 regardless of token). `neuprint_test_token()` now runs the same
  one-row Cypher the connectome loader uses, `POST /api/custom/custom`
  (`Authorization: Bearer`), and only reports "rejected" on 401/403; a 400 about
  the dataset lists the live dataset names from `/api/dbmeta/datasets`; a missing
  query route falls back to that public metadata route. A 404 of one route is
  never reported as an invalid token.
- Codespace open time: the hooks stamp container age at start
  (`.run/hook-timing.log`, `--status`) to separate GitHub's VM/image time from
  ours; README section 0 documents the Codespaces prebuild setup.

## AL — Formula Genesis Engine v3.0

**Decisions.** New layer beside the 22 formulas (not a replacement); NumPy
only (no GPU / Rust in a Codespace - stated in the UI and `/api/genesis/status`);
Gemini public feed added as a first-class tape; Glassnode / Twelve Data /
LunarCrush as keyed providers that activate only with a key.

**Data.** `backend/data/gemini_ws.py` (trades + L2 + candle REST), hub
priority Binance → Gemini → Kraken WS → Kraken REST → CoinGecko → simulator;
`hub.listeners` hook. `backend/genesis/candles.py`: per-asset minute store
(OHLCV, trades, taker buy/sell, depth, spread, imbalance, VWAP + per-minute
order-book transport columns), bootstrapped from Kraken/Gemini history,
persisted in `.run/genesis/`. `backend/data/keyed_providers.py`: `MacroFeed`
polls configured providers; columns `exch_flow`, `dxy`, `social` (+ `_ret`)
reach the frame as NaN when absent.

**Pool.** `backend/genesis/domains/d01..d10` - 10 × 10 × 21 = 2,100 specs,
`FormulaSpec(fid, kernel, params, layer, definition, interpretation)`; the
spec's named mathematics implemented exactly (`test_round_al.py` checks Lévy
area of a unit loop = 1, Betti-1 of a 4-cycle = 1, Fisher–Rao closed form,
W₁ via CDF = shift of a point mass, max-plus eigenvalue = max cycle mean,
von Neumann entropy 0 / ln 3, v₂(8) = 3, LZ76). Layers: D2, D10 = 4 (every
fifth candle live), D4, D6, D7 = 3, rest 2.

**Fitness** (`fitness.py`): seven metrics on the out-of-sample 30 % and the
in-sample head, unit-mapped and weighted (ic .25, sharpe .20, hit .15, pf /
dd / regime / steady .10), `fitness = 0.6·oos + 0.4·ins − 1.5·overfit −
dead-time`; `select()` greedy at |ρ| ≤ 0.70 over the last 500 signals.

**Lifecycle** (`lifecycle.py`): floors 0.40 (candidate) / 0.45 (active); 2
strikes to decay, 3 to die; a newborn gets three looks; `autopsy()` gives
per-regime IC/P&L, the cause (overfit / everywhere / regime-specific /
decayed) and resurrects regime-specific formulas with `spec.gates` (2 lives).

**Regime** (`regime.py`): Hurst(64), 15-min RV percentile over 500, 3-min
move in σ, taker-flow one-sidedness → trending / mean_reverting / high_vol /
low_vol / cascade; gates 80/70/60/50/40 with preferred domains.

**Breeding** (`symbolic.py`): nested-tuple trees, depth ≤ 4, leaves = frame
columns (incl. keyed macro columns) / survivor signals / constants; 40 %
crossover, 40 % mutation, 20 % random; children scored on the same history.

**Engine** (`engine.py`): one worker thread; first scoring at 240 candles,
re-score every 100, genesis every 4 h (bred population capped at 400);
composite = Σ(fitness − 0.3)·signal / Σ weights, confidence from |vote| ×
agreement × coverage; persisted `genesis_{asset}.json` so a restart votes
immediately with the saved ACTIVE set.

**Fusion.** voter `genesis` (`GENESIS_WEIGHT` 0.25, scaled by firing share,
silent while warming or < 5 firing), `lock_weights.BACKERS["genesis"] =
("genesis:composite",)`, ledger vote `genesis:composite`, `FusionResult.genesis`
travels with the lock → `cycle_manager.genesis_payload()` in
`/api/signal/current`, `routes_genesis.py`, dashboard card `#genesis-card` +
`renderGenesis` (one timer rule intact), Settings → keyed providers.

**Verification.** `test_round_al.py` (20), `tools/dev/genesis_bench.py`
(full pool ≈ 32 s on a 620-minute synthetic tape), dead-code 0, assets `v=2.23.0`.

## AM — flat windows and the live edge guard

**Symptom.** "Win rate 8 % in one hour." A coin-flip scorer cannot print 8 %;
on the simulator the same scorer prints ~50 %. Two real causes, both fixed:

- **Flat windows were losses.** `_evaluate_outcome` scored `exit == entry`
  (a thin PAXG minute with no print) and any move inside the spread as a LOSS
  for whichever side was published. Now `|move| ≤ spread cost` or no move is
  FLAT (outcome 0): reported (`accuracy.flat`, `decided`), never a loss; win
  rate = wins / decided; per-side rates only count decided windows.
- **Anti-correlation was tolerated.** `backend/core/edge_guard.py`: the
  engine's RAW side (before inversion) is scored on every decided window; when
  the Wilson 95 % upper bound of the raw hit rate over the last 30 (≥ 12) is
  below 50 %, fusion publishes the opposite side (`FusionResult.edge_guard`,
  `raw_side`, note in the reasoning) and stays inverted until the raw rate is
  back to 50 % - measured on the raw side, so it cannot oscillate. Shown in
  the prediction panel ("Edge guard" row, "⇄ INVERTED").

`test_round_am.py` (4); full suite 301; assets `v=2.24.0`.
