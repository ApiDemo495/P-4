# The decision chain — one thing, one logic (Round Y)

Every locked call is produced by **one ordered chain**. Nothing outside this
list may change a side, a confidence or a level, and no step is computed twice.

| # | Step | Module | What it may decide |
|---|------|--------|--------------------|
| 1 | Frozen snapshot | `backend/core/frozen_snapshot.py` | the data every later step reads (one tape, one book, one news set, frozen at t−prefetch) |
| 2 | 22 formulas (A–H) | `backend/formulas/` | per-formula values; **Formula consensus** = sign-weighted vote of the *directional* formulas only (indicators such as HSI/MPS never vote) |
| 3 | Brain (CCSv2 + GCN) | `backend/brain/` | one score computed **from the formulas**. It is a *second reading of the same evidence*, so it carries weight 0.40 and the formula consensus 0.30 — together 0.70, never two independent 0.70s |
| 4 | Physics voter | `backend/physics/` | one score, weight 0.20 (halved when model-only) |
| 5 | Agents (Gemini/GitHub/local) | `backend/agents/` | the remaining weight; absent agents forfeit their weight, they are never imputed |
| 6 | News (NIV, critical events) | `backend/news/` | may *override* (emergency) or *dampen*; never votes as a formula |
| 7 | Crowd emotions | `backend/emotions/` | a **modifier** only (`EMOTION_VOTE_SCALE = 0.5`, HSI dampening). Emotions cannot flip a side |
| 8 | Prediction history (EvidenceLedger) | `backend/core/calibration.py` | re-weights every source above by its own decayed hit rate. Its say ramps **linearly** from 0 % (no scored windows) to 100 % (`CALIBRATION_MIN_SAMPLES` = 20 scored windows), and falls to 0 % again if its record is worse than the spec recipe. There is no switch and no second recipe: `score = (1−λ)·spec + λ·learned` |
| 9 | Confidence | same blend | **one scale**: confidence = edge = 2·P(side)−1 (0 = coin flip). The ledger's probabilities are converted to edge before blending; calibrated to the realised hit rate of the confidence bucket once the bucket has ≥15 samples; the engine never prints more certainty than it has earned |
| 10 | Two-stage lock | `backend/core/signal_lock.py` | freezes side + confidence + levels at the boundary; nothing in 1–9 runs again inside the window |
| 11 | Risk | `backend/core/risk.py` | **stop** = clip(1.5 σ, 3…120 bps) sized on realised volatility; **target** = 1.5 × stop (`RR_TARGET`). Loss is bounded before the window opens; the target reaches for the tail |
| 12 | Probability branches | `backend/core/prediction.py::branches` | the dispersion model of the same call: drift fixed so P(close on side) **equals** 0.5 + confidence/2 (the same edge the panel prints), σ = the realised volatility the levels were sized on. Analytic, no sampling, byte-stable per cycle |
| 13 | Outcome scoring | `backend/core/calibration.py::score` | 60 s later the realised close grades the call and every source that voted — this is what feeds step 8 next window |

## Contradictions that were removed

* **1 : 1 vs asymmetric levels** → one rule: target = 1.5 × stop. `rr_target` is a
  setting, printed on every panel, and the tests assert the ratio.
* **"ledger decides" vs "spec decides"** → one blend with a continuous λ, so the
  side can never jump because a sample count crossed 30.
* **Brain and formulas both voting** → declared: they are two readings of the
  same evidence and together hold 0.70; the ledger treats `brain:CCSv2` and
  `f:*` as separate sources and learns which reading is reliable.
* **Confidence vs fan vs levels** → the fan is derived from the confidence and
  the levels, not from a third model.
* **Confidence as "magnitude" vs confidence as "probability"** → one scale,
  the edge; probability is printed next to it on the branch line.

## What was asked for and is *not* built — on purpose

| Request | Why not |
|---------|---------|
| Atomic / Planck-scale ingestion, quantum-state tracking | there is no such data feed; the finest real resolution is the exchange tick (µs timestamps already used) |
| Neural weave / pre-cognitive execution / temporal dilation | no brain interface exists; the engine already shows the prediction before the window opens (two-stage lock), which is the honest version of "before the market moves" |
| Guaranteed zero slippage, sovereign capital, infinite leverage | nobody can guarantee a fill price; claims of infinite capital would be false |
| Ghost hedging in unregulated venues, femtosecond yield routing | unregulated venue arbitrage is not something this app will assist with; any "guaranteed yield" statement would be a false promise |

What *was* built from that list: probability-branch visualisation (step 12),
synesthesia cues (opt-in sound + glow in the dashboard), asymmetric bounded risk
(step 11) and the self-improving loop (steps 8/13).
