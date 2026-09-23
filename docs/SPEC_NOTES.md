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
