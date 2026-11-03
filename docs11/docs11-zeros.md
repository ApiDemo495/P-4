# PROMPT — Kill the 0.00 problem, everywhere, for good

Paste everything below this line to the agent working on the multi-asset
prediction app (22 / 44 / 130 / 247-minute windows). It is written as
instructions to that agent. Every rule here comes from a 0.00 we actually
had in a working engine, found its cause, and fixed; the causes repeat
across components, which is why your app "gets it again and again on
different stuff".

---

You are fixing the **0.00 problem**: formulas, physics gauges, hedge status,
emotions, news impact, win rate, confidence and status chips in the app keep
reading `0.00` (or `—`, `NaN`, `waiting`) on real data. Treat this as one
problem with **eleven root causes**, not as a bug per panel. For every 0.00
you find, classify it into one of the causes below, apply that cause's fix,
and add that cause's test. Do not "fix" a 0.00 by adding a constant, a random
jitter or a minimum value — the user will see through it and the predictor
will be fed a lie.

## 0. The rule that governs everything

> **Every number on screen is either a measured value with a reason, or a
> reason instead of a number. Nothing is ever a silent zero.**

Concretely:
- A formula may return 0 only together with a `why` string
  (`FormulaResult.zero_reasons[name] = "needs 40 trades with volume, had 12"`
  or `"computed, genuinely flat: |z| = 0.004 < 0.02"`), and the UI shows an
  amber `why 0.00` chip on that card.
- A value that is small but real must print its first significant digit:
  implement `fx(v, d)` — `d` decimals, but if that would print as `0.00`,
  increase decimals up to 6, then exponent form (`3.1e-7`). A true zero prints
  `0.00`. Use it for **every** human-facing number, server and client (ours:
  `backend/core/numfmt.py`, mirrored in `app.js`).
- A value that is unavailable prints `—` plus the reason in the tooltip and
  in the status line, never `0.00`.
- Add a **flat watch**: for every formula keep the share of the last 50 locks
  it spent inside `±0.02` and the trace line that explains the latest flat
  reading; show it in one line above the formula explorer
  (`flat watch: 3 of 31 formulas flat in >50 % of the last 50 locks · SED 100 %
  (scale used 0.05 ≫ relative effect 1e-4) · …`). This turns "most of them are
  flat" into a list with causes.

## 1. The eleven root causes, with the fix and the test for each

### Cause 1 — Hard thresholds and clipped lines (the biggest one)
**Symptom:** emotions read 0.00 for whole windows; a formula is 0 then jumps
to 0.4; "complacency 0.00, fear 0.00, …".
**Cause:** `score = clip((x − low) / (high − low), 0, 1)` or `if x < T: return 0`.
Everything below `low` is *exactly* 0; five such terms multiplied or summed
are 0 for most of the day.
**Fix:** logistic ramps everywhere a line was clipped:
`ramp(x, low, high) = 1 / (1 + exp(−k · (x − mid) / (high − low)))`, `k = 8`,
`mid = (low + high)/2` → 0.018 at `low`, 0.982 at `high`, ordering and scale
preserved, never exactly 0. Replace `if |x| < T: return 0` by a **soft
materiality gate** `g = x² / (x² + T²)` multiplying the score.
**Test:** on a quiet synthetic tape no emotion and no gated formula is exactly
0.0; on a violent tape ordering is unchanged. **Warning:** any existing test
asserting exact `0.0`/`1.0` at ramp ends will break — change it to
tolerances, do not revert the ramp.

### Cause 2 — Scales and floors sized for another market
**Symptom:** a formula reads 0.00x on every asset forever; its trace shows
`scale used = 0.05` while the measured effect is `1e-4`; or it saturates at
±1 on every tick.
**Cause:** `tanh(x / CONSTANT)` with a constant tuned on one asset/horizon;
or a `SelfScale` floor far above anything the tape produces (we had a floor
400× too high in one formula and 10× in another, for weeks).
**Fix:** normalise by the asset's **own** measured scale: `SelfScale` = EMA of
|x| (decay 0.95–0.99) with `denominator = max(floor, scale)`; set floors from
data (the 5th percentile of |x| over a week), keep them as **code constants**
per asset class, and review them on the flat watch. For z-score-like
quantities compare like with like (a Kalman gap against the running σ of the
gap, not against the window's price move — we divided a 0.04 bp gap by a
110 bp move and got 0.000 everywhere).
**Test:** the same formula on a 1 bp/min tape and a 60 bp/min tape has
|values| in the same order of magnitude; the flat watch for it is < 30 %.

### Cause 3 — State restored from disk pins the scale
**Symptom:** after the fix for cause 2 the value is *still* 0.00 on
machines that ran the old build.
**Cause:** the adaptive scale's floor (or an old EMA) was serialised and
restored, so the persisted state overrode the new constant.
**Fix:** `SelfScale.from_dict(payload, floor=CONSTANT)` — floors and other
constants are never read from state; only the adaptive value and count are.
Bump a `BUILD_VERSION` in the vault payload; a mismatch drops the state and
re-warms.
**Test:** restore a state with `floor = 0.05` into a formula whose constant is
`1e-4`; the restored floor is `1e-4`.

### Cause 4 — Quote prints / empty rows diluting statistics
**Symptom:** volume medians 0, aggressor skew 0, run lengths 0 on thin
assets (gold, small caps) or thin moments; "baseline median vol 0" in a
trace.
**Cause:** the tape stores mid updates between trades as rows with `qty = 0,
side = 0`; on thin assets they are 60–80 % of rows; every median/sign
statistic over "ticks" is dominated by them.
**Fix:** `snapshot.trades(asset)` returns rows with `qty > 0 and side ≠ 0`;
every volume/aggressor/run statistic uses it; price paths may still use all
rows. Show `trade_prints / quote_prints` on the tape chip. On bar data the
same bug appears as bars with `volume = 0` — drop them from volume baselines.
**Test:** a tape with 70 % quote prints gives the same volume/side statistics
as the same tape without them.

### Cause 5 — Not enough history yet, reported as a value
**Symptom:** z-scores 0.00, "history over 0 windows", win rate 0 %, hindsight
multipliers 1.00 forever, learned combiners "watching" with no explanation.
**Cause:** the formula needs N samples, has fewer, returns 0 silently; or the
app started a minute ago and shows statistics as if settled.
**Fix:** (a) **warm start**: at boot pull `30 · W` minutes of 1-minute bars
per asset and run them through the same formula path so z-scores exist from
the first window; (b) persist every learner and restore it; (c) when `n` is
below the minimum print `n = 7 / 40 — not enough to say` instead of a number;
(d) win rate shows `—` until the first settled window, then `x / n` with `n`.
**Test:** on a cold start with a bootstrapped tape, every history-dependent
formula has `history windows ≥ 20` before the first lock.

### Cause 6 — A dead input upstream (the formula is honest, the input is empty)
**Symptom:** hedge/cross-asset formulas 0.00; physics "inactive"; news 0.00;
candle-based formulas frozen.
**Cause:** the second asset's tape is empty or stale; candles never rolled
(only one feed filled them); the news lexicon scored real-world wires at 0;
the feed hub rejected a delivering source as "not active".
**Fix:** make every input report **its own** liveness (`age_s`, rows,
source) and let the consumer print *which* input is missing: "HRDD: PAXG
tape 2 rows in 10 min — bootstrapped from Kraken 1-minute candles, derived".
Roll candles from every feed; bootstrap thin assets from REST candles;
adopt a delivering feed over a state machine that says "none"; macro lexicon
for news.
**Test:** with the secondary asset's live feed disabled, the hedge formulas
still compute from candles and their trace says so.

### Cause 7 — Non-directional values treated as direction (and the reverse)
**Symptom:** "tie probability 0.01 is flat" alarms; stress gauge 0.00 "is a
bug"; or a probability shows up as 0.00 in a signed column.
**Cause:** mixing indicators (probabilities, stress, efficiency) with signed
scores in the same table and the same flatness logic.
**Fix:** tag every output `kind ∈ {signed, probability, gauge}`; the flat
watch ignores probabilities/gauges whose honest reading is near 0; the UI
prints them in their own unit (`P(tie) 1.3 %`, `stress 0.07`), never as
`+0.00`.
**Test:** the flat list never contains a `probability`/`gauge` output.

### Cause 8 — Rounding before the UI
**Symptom:** server sends `round(v, 2)` → `0.0`; UI shows 0.00 for values
like 0.004 that are real and used by the predictor.
**Cause:** rounding in the payload builder; `toFixed(2)` on the client.
**Fix:** send raw floats (or round to 6); format only at render with `fx`.
**Test:** a formula value of 0.0031 renders as `0.0031`, not `0.00`.

### Cause 9 — Exceptions swallowed into zeros
**Symptom:** one formula/physics mechanism is 0.00 on every pass, no trace.
**Cause:** `try: … except: return 0.0`.
**Fix:** `except Exception as exc: ctx.zero_reasons[name] = f"failed: {exc}"`,
log once per minute per name, show the amber chip; count failures in
`/api/system/health`.
**Test:** a formula that raises shows its reason in `zero_reasons` and in the
UI, and the engine's degradation counter counts it.

### Cause 10 — Degradation counted from honest readings
**Symptom:** "degraded evidence — 11+ formulas returned zero" on a calm but
healthy tape; confidence floored.
**Cause:** the degradation rule counted near-zero *readings* as failures.
**Fix:** degradation counts only `zero_reasons` of kind `failed` / `missing
input`; honest near-zero readings are not failures.
**Test:** a calm tape with every formula computing does not trip degradation.

### Cause 11 — Status derived from configuration, not from evidence
**Symptom:** `brain 0.00`, `agent 0.00`, `key: not set`, `&b offline` while
the thing is actually working (or vice versa: "online" while dead).
**Cause:** chips read a boolean from settings or a cached state object.
**Fix:** every chip reads from the component's last real result (`last_ok`,
`age_s`, `last_error`, `model`, `latency_ms`); the client re-fetches state
after any POST; "not set" is only shown when the ring has no slots at all.
**Test:** after saving a key the badge reads the server's tested state; with
a delivering feed the tape chip never reads `offline`.

## 2. The classification procedure (do this for each 0.00)

1. Open the item's trace. No trace? → implement tracing first (inputs seen,
   intermediates, scale used, gate, reading, time, data span, history `n`).
2. Is an input empty/stale? → Cause 6 (fix the input and print its liveness).
3. Is `n` below the formula's minimum? → Cause 5 (warm start, `n` on screen).
4. Is the scale/floor ≫ the measured effect? → Cause 2; still after the fix
   on old machines? → Cause 3.
5. Is the statistic over "ticks" with qty 0 rows? → Cause 4.
6. Is there a `clip`/`if x < T: return 0`? → Cause 1.
7. Is it a probability/gauge? → Cause 7 (it's not a bug; label it).
8. Does the raw payload carry the value but the UI shows 0.00? → Cause 8.
9. Is there a bare `except`? → Cause 9.
10. Does the degradation banner appear on a healthy tape? → Cause 10.
11. Is a status chip contradicted by the component's own log? → Cause 11.

## 3. Where each cause bit us (so you know where to look in your app)

| place | cause(s) | what the user saw |
|---|---|---|
| emotions | 1 | all nine 0.00 on quiet tapes; "complacency" only |
| SED / RSV / DSKD formulas | 2, 3 | 0.00x for weeks; traces showed scale ≫ effect |
| VSD / AFPR / herding | 4 | "baseline median vol 0" on gold and thin BTC moments |
| hedge formulas, hedge status | 6, 5 | `hedge 0.00`, `&b partial` forever (second asset thin) |
| physics mechanisms | 6, 9 | "inactive" with no reason |
| news NIV / impact | 6 | 0.00 on real-world wires |
| TP/SL levels | 6 (candles) | identical levels every window |
| win rate / streak | 5 | 0 % at boot, or three different numbers on one screen |
| brain vote | 6 (input polarity) and a readout bug | ≈ 0 always |
| key badges, feed chips | 11 | "not set", "offline" while working |
| confidence | 10 | floored by a false degradation state |
| every panel | 8 | `0.00` for real small values |

## 4. What to build once, so it stops recurring

1. `trace()` helper used by every computation; the UI renders traces
   generically from `{label, value, unit}` rows.
2. `zero_reasons` on every result object; amber chips; health counters.
3. `fx()/fxs()` formatters server and client; a lint test that greps for
   `toFixed(2)` and `round(..., 2)` in payload builders.
4. `ramp()` and soft gates in one util; a lint test that flags `clip(` in
   scoring code outside the util.
5. `SelfScale` with constructor floors and `from_dict(..., floor=)`.
6. `trades()` accessor on the snapshot; bar-level `volume > 0` filter.
7. Warm start + vault + `BUILD_VERSION`.
8. Liveness (`age_s`, rows, source, last_error) on every input object.
9. Output `kind` tags and the flat watch.
10. Chips bound to last real results.

## 5. Acceptance tests (all must pass before reporting)

1. Quiet synthetic tape: no emotion, no gated formula, no physics mechanism
   reads exactly 0.0; each prints a value with ≥ 1 significant digit or a
   `why`.
2. Thin tape with 70 % quote prints: volume/side formulas equal the trades-
   only result.
3. Cold start with bootstrap: every history-dependent formula has `n ≥ 20`
   before the first lock; the win rate shows `—` then `x / n`.
4. Restored vault with stale floors: constants come from code.
5. A formula that raises: `zero_reasons` and UI chip carry the message; the
   degradation counter increments; honest near-zeros do not.
6. Secondary asset feed off: hedge formulas compute from candles and say so.
7. `fx(0.0031) == "0.0031"`, `fx(3e-7) == "3.0e-07"`, `fx(0) == "0.00"`.
8. Flat watch lists no probability/gauge outputs; its top entry carries a
   reason string.
9. Chips: saved key → tested state within one refresh; delivering feed →
   never `offline`.
10. Lint: no `toFixed(2)` in payload rendering of formula values; no bare
    `except: return 0`.

## 6. Report back

For each panel that showed 0.00: the cause number, the file and line of the
fix, the trace before and after, and the flat-watch share after one hour on
a live tape. Do not report "no more zeros" — report the list of remaining
honest near-zeros with their reasons.
