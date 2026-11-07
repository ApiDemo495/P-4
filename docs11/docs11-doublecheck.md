# PROMPT — The double check: verify the arithmetic, never block the prediction

Paste everything below this line to the agent working on the multi-asset
prediction app (22 / 44 / 130 / 247-minute windows). It is written as direct
instructions to that agent. Context: the app's "double check" has produced
**no prediction for three days**. Every window shows `BULL* · no trade · weak
lean inside noise floor`, `confidence 0.008 · below floor`, `issue floor
0.350`. That is a broken design, not caution. Below is what a double check
is, what it is not, and how to fix the gate so the app predicts every
window again.

---

## 1. What "double check" means (and the only thing it may do)

A double check verifies that **a number was computed correctly**. It runs
per formula, per pass, and has exactly one power: *zero that one formula for
that one pass and record why*. It never touches the prediction, the lock,
the confidence floor or whether a call is issued.

Three independent checks per formula, every pass:

1. **Replay** — run the formula a second time on a private `deepcopy` of its
   pre-pass state against the same frozen snapshot; the two answers must
   agree to `1e-9`. Catches hidden state, non-determinism, mutation bugs.
2. **Re-derivation** — each formula module carries `double_check(trace,
   asset)`: an independent one-line recomputation of its output from the
   intermediates it traced while running (traces carry the exact `raw`
   number next to the rendered one), tolerance `1e-6`, plus a
   `DOUBLE_CHECK = "…rule in words…"` string shown in the UI. If the printed
   intermediates don't reproduce the printed result, the audit trail is
   lying.
3. **Range** — finite and inside the declared `RANGE` (`[-1, 1]`; gauges
   `[0, 1]`).

```python
report = double_check(formula, snapshot, state_before, value, trace)
if not report.ok:
    result.values[name] = 0.0
    result.errors[name] = f"double check failed: {report.verdict}"   # feeds the degradation ladder
    result.zero_reasons[name] = report.verdict
trace.append(f"✓✓ replay {✓|✗ Δ…} · re-derived {✓|✗ Δ…} · range ✓ · {DOUBLE_CHECK}")
```

Payload: `double_check{formulas, verified, partial, failed, failed_names,
check_us, rule}`; UI: a `✓✓ double-checked` chip per formula and one summary
line. Cost ≈ 4–5 ms per pass. Env `FORMULA_DOUBLE_CHECK=0` disables it for
debugging. **That is the whole feature.** Reference numbers from our engine:
17,664 evaluations across every synthetic tape → 17,664 verified, 0 false
positives; a deliberately lying formula, a non-deterministic one and an
out-of-range one are the three tests that must fail.

## 2. What it must never do (this is what broke your app)

Remove every one of these if present:

- ✗ "Do not issue a prediction unless N formulas / agents agree" — a
  consensus gate dressed as a check.
- ✗ "Do not issue unless confidence ≥ 0.35" where confidence is a fused
  score whose natural scale is 0.001–0.05 — a floor on a different unit.
- ✗ "Re-run the whole prediction and only issue if both runs agree in
  sign" — with scores near zero the sign flips on noise; half the windows
  vanish.
- ✗ "Hold the lock until the double check passes" — the lock is a clock
  event, not a quality event.
- ✗ "No trade when inside the noise floor" shown as **the** output — the
  side must still be issued; "weak" is a size hint, not an absence.
- ✗ Counting honest near-zero readings as check failures (see the zeros
  prompt, cause 10) → degradation → confidence floored → nothing issued.

## 3. The rule that replaces all of that

> **Every window gets a side, at the boundary, no exceptions.** The app
> may say *how sure* it is and *how much* to size, never *whether* to speak.

Separate the three things your gate has fused:

| concept | question it answers | allowed values | who uses it |
|---|---|---|---|
| **side** | up or down for this window? | `BUY` / `SELL` — always one of them | the pick, the ledger, the win rate |
| **confidence** | how far from a coin flip? | calibrated probability `P(side) ∈ [0.50, 1.0)` | display, sizing, hindsight |
| **tier / size hint** | how much? | `full / half / quarter / watch-only` | position-size panel only |

"Watch-only" is still a side with a tier; it is *never* "no prediction".
The streak, the win rate and the ledger grade every window.

## 4. Fix the confidence floor properly

The screenshots show `conf 0.008`, `0.012`, `0.023` against `floor 0.350`.
The floor is in the wrong unit. Do this:

1. **Calibrate** the raw fused score `s` into a probability with Platt
   scaling fitted on settled windows per asset and horizon:
   `P(up) = σ(a·s + b)`, `a ∈ [0.05, 5]`, refit hourly, minimum 40 settled
   windows; before 40, `a = 1/median|s|` (so a typical score maps to ≈ 0.73)
   and the UI says `calibration: provisional (n = 12 / 40)`.
2. **Noise floor** for the *tier*, not for issuing:
   `floor = max(0.02, 0.5 · median(|s| over the last 60 windows))`. Below the
   floor → tier `watch-only`, label `weak lean`, side still issued and still
   graded.
3. Tiers from calibrated `P`: `≥ 0.62 full · ≥ 0.56 half · ≥ 0.52 quarter ·
   else watch-only`. Re-fit the cut points monthly from the realised
   accuracy per tier; tiers must be *monotone* (full tier's realised win
   rate ≥ half's ≥ quarter's) — if not, the calibration is wrong, not the
   market.
4. Show on every card: `BUY · P 0.57 · half · s = 0.012 (floor 0.009)` — the
   raw score and the floor in the same unit, next to each other.

## 5. Where a real check *may* change the output (and how)

| situation | allowed reaction |
|---|---|
| formula fails double check | that formula = 0 for the pass, reason shown; fusion renormalises |
| ≥ 1/3 of formulas failed (not "read small") | confidence capped at 0.55, tier ≤ quarter, banner "degraded evidence"; side still issued |
| no tape at all for the asset (0 prints in 3·window) | side from the last valid state, tier watch-only, banner `feed down`; still graded |
| brain/agents absent | weights renormalised across what is present |
| two independent fusions disagree in sign | pick the one with the larger calibrated |P − 0.5|; show both in the trace |

Nothing in the table produces "no prediction".

## 6. Diagnostic for the three silent days

Run this before changing code and put the output in your report:

```
SELECT date, count(*) windows, sum(issued) issued, avg(raw_score), avg(confidence),
       sum(reason='below floor'), sum(reason='double check'), sum(reason='disagree')
FROM window_ledger WHERE ts > now()-3d GROUP BY date, asset, horizon;
```
Then: the distribution of `raw_score` (p10/p50/p90) against the floor; the
`double_check.failed_names` histogram; the number of windows where the
tape was alive but nothing was issued. Every one of those windows is a lost
grade — count them.

## 7. Acceptance tests

1. **Issue rate = 100 %**: over 200 synthetic windows (quiet, trending,
   violent, feed-down), every window has a side at the boundary; the
   ledger has 200 graded rows.
2. Double check: a lying formula, a non-deterministic formula and an
   out-of-range formula are each zeroed with the right verdict; the window
   still issues a side.
3. Honest formulas on a quiet tape: `double_check.failed = 0`; no
   degradation.
4. Floor is in score units: `floor ≤ median|s|` over the last 60 windows;
   the share of `watch-only` windows is ≤ 50 %.
5. Calibration monotone: realised win rate full ≥ half ≥ quarter over the
   last 200 settled windows (or the panel says `calibration not monotone —
   refitting`).
6. Lock timing: the boundary pick is published within 300 ms of the
   boundary regardless of double-check state.
7. UI: no card ever shows "no trade" without a side; `watch-only` carries
   the side and `P`.
8. Lint: no `return None` / `issued = False` path in the prediction builder
   that depends on confidence, agreement or check results.

## 8. Report back

The three-day diagnostic table; the raw-score distribution vs the old and
new floor; the double-check pass/fail counts per formula; the issue rate
after the fix over 24 h (must be 100 %); the calibration table (tier → n →
realised win rate); and the list of removed gates with file and line.
