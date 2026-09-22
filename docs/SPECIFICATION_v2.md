# DROSOPHILA TRADER v2.0 — Technical Specification

**Status:** normative. The code in `backend/` implements exactly what is written
here; where the implementation intentionally departs from the original v2.0
draft, `docs/SPEC_NOTES.md` records the deviation, the reason and the escape
hatch.

**Scope:** a 60-second scalp-trading engine for BTC and PAXG that produces one
immutable signal per minute, driven by 22 formulas, a Drosophila mushroom-body
connectome model, and up to three external AI agents.

---

## 1. Overview

### 1.1 The 60-second cycle

Every cycle is phase-locked to the UTC minute and follows the same timeline.

| Virtual time | Wall clock at `TIME_SCALE=1` | What happens |
|---|---|---|
| `t = 0.000 s` | `:00.000` | Market snapshot is **frozen**; the lock resets to `COMPUTING` ⏳ |
| `t = 0.000–0.003 s` | first ~3 ms | All 22 formulas run on the frozen snapshot |
| `t = 0.003–8.000 s` | `:00.0 – :08.0` | AI agents are called **concurrently**, 7 s budget each |
| `t = 8.000 s` | `:08` | Fusion → signal **LOCKED** 🔒 and broadcast |
| `t = 8 – 15 s` | `:08 – :15` | (live formula refresh interval) |
| `t = 8–60 s` | `:08 – :60` | Signal panel is frozen; the Formula Explorer may refresh, but the signal cannot change |
| `t = 60 s` | `:60` | 60 s after lock, the outcome of the signal is scored (Appendix B) |
| `t = 60 s` | next `:00` | New cycle, back to `t = 0` |

`TIME_SCALE > 1` compresses *every* duration by the same factor (used for demos
and for the test-suite), so `TIME_SCALE=20` gives a 3-second cycle with a
0.4-second lock deadline and a 0.75-second formula refresh. No ratio changes.

### 1.2 Design invariants

1. **One signal per cycle, computed once, never rewritten.** (Section 2)
2. **Formulas never see live buffers.** They read one immutable snapshot.
3. **A failing subsystem degrades the product, it never crashes it.** (Section 12)
4. **Agent disagreement can only reduce confidence, never manufacture it.**
5. **Every number the UI shows is traceable to a formula, an agent or a clock.**

---

## 2. Signal Lock Protocol

State machine (`backend/core/signal_lock.py`):

```
        new_cycle()                lock()                emergency_override()
   ┌──────────────┐        ┌──────────────┐        ┌────────────────────┐
   │  COMPUTING   │───────▶│    LOCKED    │───────▶│ EMERGENCY_OVERRIDE │
   │      ⏳       │        │      🔒      │        │         ⚡          │
   └──────────────┘        └──────────────┘        └────────────────────┘
           ▲                                                    │
           └──────────────── new_cycle() at t = 60 s ───────────┘
```

* `new_cycle(n)` clears the previous signal completely. `get_current()` raises
  `SignalNotReady` while the cycle is `COMPUTING`, so a client can never render a
  half-computed signal.
* `lock(signal)` may run **at most once per cycle**. A second call (late
  computation, duplicate task, race) is a no-op that returns the already-frozen
  signal: `test_e02` asserts the returned object *is* the locked object.
* The frozen value is a `NamedTuple`, so immutability is enforced by the language
  rather than by convention.
* `emergency_override(event)` re-emits the signal with

  | field | value |
  |---|---|
  | `signal` | always `HOLD` |
  | `confidence` | `1.0` |
  | `lock_state` / icon | `EMERGENCY_OVERRIDE` / ⚡ |
  | `superseded_by` | the direction that was overridden (`BUY`, `SELL` or `COMPUTING`) |
  | `hold_lean` | the overridden direction, so the user still sees what the engine wanted to do |
  | `emergency_headline` | the headline that caused it |

  The override is **HOLD-only by construction**: there is no code path that can
  turn an emergency into a BUY or a SELL.
* Emergency duration is `EMERGENCY_DURATION = 180 s` of virtual time, then the
  lock returns to `LOCKED` and normal cycles resume.
* History is capped at 240 signals (four hours at one per minute).

---

## 3. System architecture

### 3.1 Components

```
        Binance WS ─┐
        CoinGecko ──┼─▶ MarketDataHub ── freeze() ──▶ FrozenMarketSnapshot
        simulator ──┘        │                              │
                             │                    ┌─────────┴──────────┐
   CryptoPanic ─┐            │                    ▼                    ▼
   NewsAPI ─────┼─▶ NewsEngine│              FormulaEngine        AgentOrchestrator
   RSS ─────────┘   (30–60 s) │              (22 formulas)        (Gemini/local/GitHub)
                             │                    │                    │
                             │                    └────────┬───────────┘
                             │                             ▼
                             │                        fusion.fuse()
                             │                             │
                             ▼                             ▼
                        WorldClock ────────────▶  SignalLockController.lock()
                                                           │
                              WebSocket /ws/signals ◀──────┘   (+ REST mirrors)
```

### 3.2 Storage

* **Redis** (optional) keeps formula state, the current signal and history across
  restarts. If Redis is unreachable the engine transparently uses an
  in-process store (`backend/core/redis_bus.py`) — the single-process deployment
  is the common case.
* **Files** `backend/brain/fallback/mb_adjacency_80x80.csv` (the committed
  connectome matrix) and `backend/models/` (uploaded local models).

### 3.3 Repository layout

See Appendix A.

---

## 4. Drosophila brain integration

### 4.1 The mandatory 5-step startup verification

`backend/brain/startup_verification.py` runs these steps in order, each with its
own timeout, and **never raises**. The result is always a usable 80×80 matrix.

| Step | Action | Budget | Failure behaviour |
|---|---|---|---|
| 1 | Load the cached matrix from Redis / disk | 1 s | fall through |
| 2 | neuPrint connectivity probe (`/api/version`) | 5 s | fall through |
| 3 | neuPrint authentication + connectivity query | 5 s | fall through |
| 4 | Query → spectral cluster → build → cache | 30 s | fall through |
| 5 | FlyWire / CAVE cross-check, then commit the CSV | 10 s | fall through |

If every step fails the committed fallback CSV is loaded and the status becomes
`FALLBACK_CSV` — the engine still trades (degradation level 4). Step results
(status, elapsed ms, detail) are exposed at `GET /api/brain/status`.

### 4.2 The circuit (80 nodes)

| Range | Population | Count | Meaning |
|---|---|---|---|
| 0–19 | Projection neurons | 20 | one per input formula, in registry order TAI … SMD |
| 20–69 | Kenyon Cell clusters | 50 | spectrally clustered from ~2000 real KCs |
| 70 | PAM | 1 | dopaminergic, reward (gated by DRG) |
| 71 | PPL1 | 1 | dopaminergic, punishment |
| 72 | OA | 1 | octopaminergic, arousal / hedge (gated by HSI) |
| 73 | MBON-α3 | 1 | **approach** output |
| 74 | MBON-γ5β′2a | 1 | **avoid** output |
| 75 | MBON-β2β′2a | 1 | neutral / gating output |
| 76 | MBON-α′2 | 1 | confidence output |
| 77–79 | Lateral horn | 3 | approach / avoid / neutral read-out |

Kenyon-Cell sparsity is `SPARSITY = 0.10` (a ~10 % active code, as measured in
vivo).

### 4.3 Propagation — 3-layer graph convolution

`backend/brain/graph_convolution.py`:

```
layer 1: drive   = A_norm · a0                 (a0 = PN vector, 20 → 80)
         a1      = ReLU(drive)                 (sparse KC code)
         a1[KC]  = top-10 % of a1[KC] by |value|, remainder zeroed
layer 2: a2      = tanh(A_norm · a1)
layer 3: a3      = tanh(A_norm · a2)
read-out: LH_approach = a3[77], LH_avoid = a3[78], LH_neutral = a3[79],
          MBON outputs = a3[73..76]
```

* `A_norm` is the row-normalised adjacency of `A.T` (the CSV is stored as
  `M[source, target]`).
* A gain factor of **3.2802** is calibrated once at load time
  (`calibrate_gain(target=1.0, probes=16, seed=7)` including the all-ones probe),
  so a saturated input produces a saturated read-out instead of a tenth of it.
* `drive` (the **pre-ReLU** Kenyon-Cell input) is preserved in the trace as
  `kc_drive`. Formula 21 measures it — see Deviation 21-A in `SPEC_NOTES.md`.

### 4.4 Health checks

Every **300 s** (`brain_health_interval_seconds`) the brain is re-checked:

* matrix checksum matches what was loaded,
* a probe vector propagates to a non-zero output (`output_magnitude > 0`),
* neuPrint is optionally re-pinged (never required).

A failed check raises the status to `DEAD` / `CORRUPT` / `ZERO_OUTPUT`, which
degrades the engine to `FALLBACK_BRAIN` or, if even the CSV is unreadable,
`NO_AGENTS` + `MINIMAL`. The matrix itself is refreshed from neuPrint at most
once per 24 h.

---

## 5. Market data layer

### 5.1 Buffers (per asset)

| Buffer | Capacity | Contents |
|---|---|---|
| `TickBuffer` | 600 ticks | `(time_ms, price, quantity, side)` — side is `+1` aggressor-buy, `-1` aggressor-sell |
| `L2Buffer` | 2 snapshots × 20 levels | current and previous order book + a 900-point spread history |
| `CandleBuffer` | 60 candles | 1-minute closes |
| `NewsCache` | 20 items | headlines with tier, sentiment, timestamp |
| `OutcomeBuffer` | 20 outcomes | `(outcome, pnl_bps)` for the DRG reward learner |

### 5.2 Freezing

`MarketDataHub.freeze(news_items, drg_outcomes)` deep-copies every buffer into a
`FrozenMarketSnapshot` and pre-computes the **synchronised BTC↔PAXG grid** used
by the hedge formulas (HRDD, SHRP, GCDV, HSI). Warnings are attached to the
snapshot, not raised:

* `PAXG ticks < 15` → *"Insufficient PAXG data for reliable signal."*
* `PAXG ticks < 30` → *"PAXG tick count low: interpolating for VSD/VSS."*
* `BTC ticks < 15` → *"Insufficient BTC data for reliable signal."* → level 6.
* source ≠ live exchange → *"Live exchange feed unavailable — simulated tape in
  use."*

### 5.3 Sources and fallback chain

`MARKET_DATA_MODE = auto | binance | coingecko | simulator` (default `auto`).

1. **Binance** WebSocket `aggTrade` + `depth20@100ms` for `btcusdt` and
   `paxgusdt` (reconnect with exponential backoff up to 60 s).
2. **CoinGecko** REST polling if Binance is unavailable — reduced tick rate, so
   DGW/LCS/BAR are zeroed (degradation level 5).
3. **Built-in simulator** if `MARKET_ALLOW_SIMULATOR=1` (default) — a
   deterministic random-walk + regime-switching tape (`SIMULATOR_SEED=1337`)
   that keeps the whole product demonstrable offline. It is always flagged in
   `warnings` and in the component status; it is never presented as live data.
   With `MARKET_ALLOW_SIMULATOR=0` the engine instead goes to level 6 (HOLD).

---

## 6. The 22 formulas

### 6.0 Common contract

Every formula module exports:

```python
NAME, TITLE, DESCRIPTION, BRAIN_NODE, LATENCY_MS, DIRECTIONAL: bool
class State: ...                      # per-(formula, asset) rolling state
def compute(snapshot, asset, state, params, ctx) -> float
```

Rules:

* **Isolation.** An exception sets the value to `0.0`, records
  `errors[name]`, and lets the other 21 finish.
* **Range.** Every directional output is in `[-1, +1]`; regime indicators are in
  `[0, 1]` (`DIRECTIONAL = False`).
* **Latency budget** is per formula and is *measured*, not assumed
  (`GET /api/formulas/timings`). Total budget ≈ 3 ms.
* **Per-asset parameters** come from Appendix C through `params[...]`.
* **Zero-count gate.** If 11 or more of the 20 input formulas are exactly zero,
  the cycle is forced to HOLD (Section 10.1).

### 6.1 Category A — Micro-structure

**1. TAI — Tick Acceleration Impulse** *(ORN Or67d, ≤0.2 ms)*
Third derivative (jerk) of price over the last `T` ticks using non-uniform finite
differences, normalised by the tick-time MAD:
`TAI = tanh(jerk / (3·σ_jerk + ε))`.

**2. AFPR — Aggressive Flow Pressure Ratio** *(ORN Or42b, ≤0.1 ms)*
Power-law recency weighting of aggressor volume:
`AFPR = tanh( (Σ wᵢ·vᵢ·sᵢ) / (Σ wᵢ·vᵢ + ε) · power )`, `wᵢ = (1 − i/n)^power`,
`power = 3.0` BTC / `2.0` PAXG.

**3. SED — Spread Elasticity Detector** *(GRN Gr5a, ≤0.1 ms)*
Signed regression of spread changes on aggressor flow:
`SED = tanh(β · sign(flow))` where `β` is the OLS slope of `Δspread` against
signed volume over the last 30 ticks.

**4. VSD — Volume Shock Detector** *(mechanosensory III/IV, ≤0.3 ms)*
Robust volume spike: `z = (median(recent) − median(baseline)) / (1.4826·MAD)`,
`VSD = tanh(z/3) · sign(Δprice)`. Windows: 20 / 100 ticks (BTC), 10 / 50 (PAXG).

### 6.2 Category B — Order book

**5. DGW — Depth Gravity Well** *(chordotonal, ≤0.05 ms)*
Volume-squared centre of mass of each side vs. the mid:
`g = (Σ qᵢ²·dᵢ)/(Σ qᵢ²)`, `DGW = tanh( (g_bid − mid) − (mid − g_ask) )`.

**6. LCS — Liquidity Cliff Score** *(GRN Gr66a, ≤0.05 ms)*
Worst level-to-level drop-off on each side, asymmetry signed bullish when the ask
cliff is deeper: `LCS = tanh( (|C_ask| − |C_bid|) / (|C_ask| + |C_bid| + ε) )`.

**7. BAR — Book Absorption Rate** *(ORN Or22a, ≤0.05 ms)*
Net consumption of resting top-10 liquidity between consecutive snapshots:
`BAR = tanh( (Δask − Δbid) / (|Δask| + |Δbid| + ε) )`.

### 6.3 Category C — Hedge (BTC × PAXG)

All four hedge formulas consume the pre-computed synchronised grid.

**8. HRDD — Hedge Ratio Drift Detector** *(bilateral AL, ≤0.1 ms)*
Instantaneous OLS hedge ratio over 60 synchronised 1-second returns vs. its
60-minute EMA baseline: `d_β = (β_now − β_base)/(|β_base| + ε)`;
`HRDD = −tanh(d_β)` for BTC, `+tanh(d_β)` for PAXG.

**9. SHRP — Safe-Haven Rotation Pulse** *(MB calyx CA, ≤0.15 ms)*
Recency-weighted signed dollar-flow rotation `R` between BTC (risk) and PAXG
(safety); `SHRP = −R` for BTC, `+R` for PAXG.

**10. GCDV — Gold-Crypto Divergence Velocity** *(T4/T5, ≤0.1 ms)*
Speed of separation of the two normalised price paths, averaged over the last 10
first differences; negated for PAXG.

**11. HSI — Hedge Stress Index** *(OA-VUMa2, regime, `[0,1]`)*
`HE = 1 − σ_hedge / ((σ_BTC + σ_PAXG)/2)`, raw `HSI_raw = tanh(2·(1 − HE))`.
The **operative** value is calibrated against the uncorrelated-neutral baseline
(Deviation 11-A): `HSI = clip((σ_H/σ_avg − base)/(1 − base), 0, 1)`. HSI is never
directional; it dampens confidence (Section 8.2) and gates the OA node.

### 6.4 Category D — Volatility & regime

**12. RSV — Regime Switch Velocity** *(clock DN1p, ≤0.3 ms)*
Eight overlapping 20-tick sub-windows give eight rescaled-range Hurst estimates;
`RSV = tanh(slope × 20)`.

**13. VSS — Volatility Surprise Score** *(giant fibre, ≤0.2 ms)*
`realised_1m / baseline_5m` z-score, multiplied by the sign of recent drift.

**14. ERC — Entropy Regime Classifier** *(serotonin CSD, regime, ≤0.5 ms)*
Sample entropy of the return series: `ERC = tanh(SampEn − 1.0)` — negative =
trending, ~0 = transitional, positive = choppy. `O(N²)` matching runs in Cython
when `erc.pyx` is compiled, otherwise in vectorised numpy.

### 6.5 Category E — Temporal pattern

**15. MCPE — Micro-Cycle Phase Estimator** *(clock LNv, ≤0.1 ms)*
OLS-detrend the 60-tick window, count zero crossings `Z`, dominant period
`≈ 2n/Z`, and read the phase: `MCPE = −cos(φ)·min(1, Z/4)`.

**16. MPS — Momentum Persistence Score** *(ORN Or47b, ≤0.15 ms)*
`MPS = tanh(3·(3ρ₁ + 2ρ₂ + ρ₃)/6)` — weighted lag-1/2/3 return autocorrelation.

**17. TWRS — Time-Weighted Return Skewness** *(ORN Or85a, ≤0.1 ms)*
Recency-weighted third standardised moment of tick returns, normalised by the
time-weighted variance.

### 6.6 Category F — Kalman & state estimation

**18. DSKD — Dual-State Kalman Divergence** *(campaniform sensilla, ≤0.01 ms)*
Two random-walk Kalman filters (`q_fast`, `R` from Appendix C) on the same tick
stream; `DSKD = tanh((p_fast − p_slow)/(σ_resid + ε))`.

### 6.7 Category G — News & sentiment (see Section 14)

**19. NIV — News Impact Velocity** *(Johnston's organ, ≤0.01 ms)*
`NIV = Σ sⱼ·cⱼ·e^(−Δtⱼ/300) / (Σ cⱼ·e^(−Δtⱼ/300) + ε)` over the last five items
(`s` = sentiment, `c` = source credibility, 300 s decay).

**20. SMD — Sentiment Momentum Divergence** *(LH multimodal, ≤0.01 ms)*
`SMD = tanh(NIV − TAI)` — what the news says versus what the tape does.

### 6.8 Category H — Drosophila brain output

**21. KCAE — Kenyon Cell Activation Entropy** *(KCs, regime, ≤0.02 ms)*
`pⱼ = |aⱼ|²/Σ|aₖ|²`, `H = −Σ pⱼ ln pⱼ`, `KCAE = 1 − H/ln(50) ∈ [0,1]`.
Measured on the **pre-ReLU** `kc_drive` (Deviation 21-A). Not directional: it is
the brain's confidence term.

**22. CCSv2 — Connectome Consensus Score v2** *(whole MB circuit, ≤0.8 ms)*

```
CCSv2          = tanh(LH_approach − LH_avoid)
brain_confidence = KCAE × |LH_approach − LH_avoid| / (|LH_approach| + |LH_avoid| + ε)
```

`prepare()` runs the graph convolution and `compute()` reads it; `prepare()` is
idempotent within a cycle, and a failed prepare degrades CCSv2 to `0.0` instead
of raising (the `UnboundLocalError` class of bug is structurally impossible).

**DRG — Dopamine Reward Gradient** *(PAM/PPL1, reward meta-parameter, ≤0.05 ms)*
Not one of the 22: it consumes *past outcomes* and modulates the dopamine nodes.

```
V_R      = Σ γ^(R−i)·mᵢ·oᵢ / Σ γ^(R−i),      γ = 0.95
δ_TD     = (m_R·o_R + γ·V_R) − V_{R−1}
DA       = δ^0.8 if δ > 0 else −|δ|^1.2        (loss aversion)
DRG      = tanh(0.1 · DA)
```

---

## 7. AI agent system

Up to three external agents run **concurrently** with a hard 7 s timeout each;
the fusion layer renormalises over whoever answered.

| Agent | Weight | Endpoint | Test on entry |
|---|---|---|---|
| Gemini | 0.25 | `generativelanguage.googleapis.com/…:generateContent` | `POST /api/agents/gemini/test` → key validity |
| Local model | 0.20 | in-process GGUF (llama.cpp) or ONNX | magic-byte validation + test inference |
| GitHub Models | 0.15 | `models.inference.ai.azure.com/chat/completions` | `GET /user` then the models endpoint |

### 7.1 Gemini

Structured output is enforced with a `responseSchema`
(`{decision: enum[BUY,SELL,HOLD], confidence: number, reasoning: string}`), so a
reply is either valid JSON or rejected. HTTP 429 triggers exponential backoff
from 10 s to a 300 s ceiling; the agent reports `RATE_LIMITED` and the cycle
continues without it.

### 7.2 Local model

`POST /api/agents/local/upload` (multipart) → validate → load → test inference →
monitor:

1. **Validation:** `.gguf` must start with the magic bytes `GGUF`, be ≥100 MB,
   and yield a readable metadata block (architecture, tensor count, context
   length, quantisation). `.onnx` must look like a real protobuf graph. Anything
   else is rejected with *"Invalid file format. Please upload a valid .gguf or
   .onnx file."*
2. **Load:** 180 s budget; failure keeps the previous model.
3. **Test inference:** one JSON decision, 60 s budget.
4. **Monitoring:** RSS is sampled every 60 s; above 85 % of system RAM the model
   is unloaded and the agent reports `ERROR`, which drops the engine to level 3
   rather than risking an OOM kill.

`LOCAL_AGENT_STUB=1` loads a deterministic, clearly-labelled `STUB` model that
reads the same prompt numbers with a transparent rule. It exists so the agent
tier, the fusion weights and the UI can be exercised without a 4 GB download —
it never claims to be a real model.

### 7.3 Agent failure isolation

An agent that errors, times out or is rate-limited contributes **nothing** to the
score: it is dropped from `active`, its weight is redistributed proportionally,
and the reason is surfaced in `SIGNAL.agents[name].error` and in the dashboard.
A cycle may therefore complete at degradation level 3 with CCSv2 as the sole
decision maker.

---

## 8. Fusion

### 8.1 Weights

```
drosophila 0.40   gemini 0.25   local 0.20   github 0.15
```

Renormalised over the agents that answered: `wᵢ = Wᵢ / Σ W_active`.

```
score = Σ wᵢ · vᵢ,   v_drosophila = CCSv2,
                     v_agent      = direction(decision) × confidence
```

### 8.2 HSI dampening

If `HSI > 0.80`: `confidence ← confidence × max(0.20, 1 − HSI)`. During extreme
hedge stress even a strong signal is dampened, and the floor keeps it from being
zeroed outright. The reasoning string always states the multiplier.

### 8.3 Confidence

```
magnitude   = min(1, |score| / (2 · signal_threshold))        # 0.25 → full
agreement   = |mean(direction of active contributors)|        # 1.0 = unanimous
brain_term  = 0.6·ccs_confidence + 0.4·mean(agent confidences)
              (or ccs_confidence alone when no agent answered)
confidence  = (0.65·magnitude + 0.35·brain_term)
              × (0.85 + 0.15·agreement)
              × hsi_adjustment
confidence ∈ [0.00, 0.95]
```

Confidence answers *"how much do I believe this reading?"*, magnitude answers
*"how far from neutral is it?"* — the entry gate uses both (Section 10.1).

### 8.4 Hard gates (evaluated before anything else)

| Condition | Result |
|---|---|
| Emergency override active | `HOLD`, confidence 1.0, `forced_reason="Emergency override active"` |
| `zero_count ≥ 11` | `HOLD`, confidence 0.50, `forced_reason="Insufficient formula evidence"` |
| `|CCSv2| ≤ 0.25` or `confidence ≤ 0.55` | `HOLD` (with a `hold_lean` if `|score| > 0.05`) |

---

## 9. World clock

* NTP-corrected (`pool.ntp.org`, 1 h re-sync) with a monotonic-clock fallback;
  a failed sync is never fatal and is reported in `/api/signal/status` as
  `ntp_synced: false`.
* At `TIME_SCALE = 1` cycles are phase-locked to the UTC minute
  (`seconds_until_next_minute()`).
* All UI timestamps are UTC (`YYYY-MM-DDTHH:MM:SSZ`).

---

## 10. Decision logic

### 10.1 Thresholds

| Parameter | Default | Meaning |
|---|---|---|
| `signal_threshold` | 0.25 | minimum `|CCSv2|` for a directional signal |
| `min_fusion_confidence` | 0.55 | minimum fused confidence |
| `max_failed_formulas` | 10 | 11+ zeros ⇒ forced HOLD |
| `hsi_dampen_threshold` | 0.80 | HSI stress threshold |
| `hsi_confidence_floor` | 0.20 | floor of the HSI dampener |
| `emergency_duration` | 180 s | override lifetime |

### 10.2 The HOLD warning box

Whenever the locked signal is `HOLD`, the payload carries this **verbatim** text
plus the lean and its score:

> This prediction is not as powerful as it should be because we are receiving
> HOLD signals. However, if you urgently need to take a trade, you may follow the
> BUY or SELL prediction shown - but proceed with caution.

### 10.3 Timeline guarantees

The lock is published at the **end of the 8-second computation window**
(`_hold_until_lock_deadline`), unless an emergency fires first — so the
"⏳ Computing…" state is a real part of the protocol and not an accident of
latency. Verified by `test_e01`/`test_e02`.

---

## 11. User interface

Two presentation layers share one API:

1. **Web dashboard** (`backend/web/`), served by FastAPI with no build step —
   this is the reference implementation and the live demo.
2. **Flutter app** (`frontend/`), the mobile client; identical information
   architecture, same WebSocket protocol.

### 11.1 Dashboard panels

| Panel | Contents |
|---|---|
| Header | brand, BTC/PAXG toggle (localStorage-persisted), UTC clock, cycle countdown with the lock icon, degradation badge, WebSocket status |
| Signal | BUY/SELL/HOLD badge, confidence bar, reasoning, weights used, freeze timestamp, emergency banner when overridden |
| HOLD box | the verbatim paragraph from Section 10.2 + lean direction and score |
| News | latest headline, source tier, poll age, NIV/SMD, coverage |
| Hedge | HSI/HRDD/SHRP/GCDV with bars, stress label, brain status, CCSv2 + confidence, DRG |
| Agents | one row per agent: status, decision, weight |
| History | last 12 locked signals with lock icon |
| Outcomes | win rate + last 8 scored outcomes in bps |
| Formula Explorer | all 22 formulas grouped in the 8 categories, live values, per-formula description toggle, timings |

### 11.2 Lock iconography

| Icon | State | Meaning |
|---|---|---|
| ⏳ | `COMPUTING` | first 8 s of the cycle, no signal published yet |
| 🔒 | `LOCKED` | **this signal cannot change until the next cycle** |
| ⚡ | `EMERGENCY_OVERRIDE` | forced HOLD, exits any open position |

### 11.3 Formula Explorer

Grouped A–H, each row shows the live value as a signed bar, the value to three
decimals, and (on demand) the human-readable description and the brain node it
maps to. A banner states explicitly that these are live numbers while the signal
stays locked. Latency is shown per formula (`/api/formulas/timings`).

### 11.4 Matrix viewer (`/matrix`)

80×80 heat-map (green = excitatory, red = inhibitory), source/checksum/gain, the
40 strongest edges, the PN ordering, and the last activation trace (KCAE, active
KCs, LH read-outs, CCSv2, both confidence variants).

### 11.5 Settings (`/settings`) and the in-browser key modal

Credentials are managed from the browser; the command line is never required.

* **Dashboard modal (`🔑 API keys` in the header)** — the four agent/news
  credentials plus the neuPrint token. Each field has a **Test** button that
  performs a real authenticated request before the key is accepted
  (`POST /api/agents/{name}/test`), an optional **"remember"** checkbox that
  writes the value to the git-ignored `.env`
  (`POST /api/settings/keys/{name}` with `persist: true`), and immediate effect:
  saving a CryptoPanic/NewsAPI key triggers a news poll, saving a neuPrint token
  offers to rebuild the connectome.
* **First-run banner** — shown on the dashboard while no key is configured,
  listing what each key unlocks; dismissible (remembered in `localStorage`).
* **`/settings`** — the same operations plus local-model upload/unload/stub, the
  brain's five verification steps with reconnect, the news-source status with
  poll/RSS test, and the effective system configuration.
* No credential is ever logged, echoed back to the client, or written anywhere
  except process memory and (opt-in) `.env`.

### 11.6 Flutter client

`frontend/lib` mirrors this: `SignalPanel`, `NewsCard`, `HedgeDashboard`,
`FormulaExplorer`, `AgentList` widgets over a `SignalSocket` service, with a
Flutter-local `CycleTimer` driven from the same `status` payload.

The client is built for the web with `bash frontend/run_web.sh` and served by the
engine itself at **`/flutter`** (`frontend/build/web` is mounted on demand), so
the mobile client and the API share one origin and one forwarded port. When no
`--dart-define=API_BASE` is supplied the client talks to the origin it was served
from (`Uri.base.origin`); native builds default to `http://localhost:8000`.

---

## 12. Graceful degradation

| Level | Trigger | Behaviour |
|---|---|---|
| 1 · Full | everything connected | 22 formulas + 4 agents + live brain + news |
| 2 · No news | every news source fails | NIV = SMD = 0, price-only emergency detection |
| 3 · No agents | Gemini/GitHub unconfigured and no local model | CCSv2 is the sole decision maker |
| 4 · Fallback brain | neuPrint/FlyWire unreachable | CCSv2 on the committed CSV matrix |
| 5 · CoinGecko | Binance unavailable | reduced tick rate; DGW/LCS/BAR = 0 |
| 6 · Minimal | insufficient data (BTC ticks < 15) | forced HOLD, warnings explain why |

Degradation is recomputed every cycle, reported at `/api/system/degradation` and
rendered in the header badge. The engine never raises out of a cycle.

---

## 13. API

### 13.1 WebSocket `/ws/signals`

On connect the server sends one `HELLO` with the full current state, then:

| Message | Payload |
|---|---|
| `CYCLE_START` | `cycle_number`, `asset`, `timestamp`, `lock_state=COMPUTING`, `degradation_level` |
| `SIGNAL` | the frozen signal (below) + `hold_warning`, `fusion`, `pending_asset` |
| `FORMULA_UPDATE` | live values + `note`, every 15 s, **never** a new signal |
| `EMERGENCY_OVERRIDE` | `headline`, `severity`, `reason`, `previous_signal`, `overridden_to=HOLD`, `remaining_seconds`, embedded signal |
| `OUTCOME` | `signal`, `outcome` (±1/0), `pnl_bps`, `win_rate`, entry/exit |
| `ASSET_SWITCH` | `asset`, `pending`, `effective` |

Client → server actions: `{"action":"switch_asset","asset":"PAXG"}` and
`{"action":"ping"}`.

Frozen signal payload: `cycle_number, timestamp, asset, signal, confidence,
reasoning, is_locked, is_emergency_override, lock_state, lock_icon, formulas{}`,
`agents{}`, `hedge{}`, `news{}`, `hold_lean, drg, ccs_value, ccs_confidence,
brain_status, degradation_level, warnings[], price, total_ms,
emergency_headline, superseded_by`.

### 13.2 REST mirror

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | per-component health, degradation, cycle stats |
| `GET /api/system/degradation` · `GET /api/system/config` | degradation + non-secret config |
| `GET /api/signal/current` · `/history` · `/outcomes` · `/status` | signal mirrors and clock |
| `POST /api/assets/switch` | queue an asset switch |
| `GET /api/formulas` · `/current` · `/live` · `/categories` · `/timings` | Formula Explorer |
| `GET /api/agents` · `POST /api/agents/{name}/test` | agent status and key tests |
| `POST /api/settings/keys/{name}` | store a key at runtime (`persist` → `.env`) |
| `POST /api/agents/local/{upload,unload,validate}` · `GET /api/agents/local/status` | local model |
| `GET /api/news` · `/critical` · `POST /api/news/poll` · `/emergency` · `/emergency/clear` | news engine |
| `GET /api/brain/status` · `/matrix` · `/health` · `/trace` · `POST /api/brain/reconnect` | brain |

---

## 14. News sentiment engine

### 14.1 Critical-NEWS-Impact events (the only way to break a lock)

| Trigger | Rule | Source trust |
|---|---|---|
| Extreme keyword | hack, exploit, breach, rug pull, de-peg, bank run, emergency, war, sanction, … | **Tier 1–2 only** |
| Sentiment swing | `|NIV − NIV_prev| ≥ 0.60` within one poll window | any |
| Flash move | `|Δp| ≥ 5 %` inside 30 s | price-based, no source |

An unverified blog screaming "hack" cannot stop trading (test `e06`). Every event
is logged to `/api/news/critical` and can be cleared manually.

### 14.2 Sources

| Source | Interval | Notes |
|---|---|---|
| CryptoPanic | **30 s** | `(pos − neg)/(pos + neg + 1)`, lexicon fallback |
| NewsAPI | 60 s | keyword search on BTC/PAXG/gold |
| RSS | 60 s | CoinDesk, Cointelegraph, The Block (`RSS_FEEDS`) |
| Offline pack | at startup, and whenever the cache is empty or >300 s stale | 8 clearly-synthetic headlines so the panel is never blank |

Coverage is reported as `full` / `limited` / `unavailable` / `disabled`.

### 14.3 Sentiment model

The lexicon (`backend/news/sentiment_lexicon.py`) scores each headline
`tanh((bullish − bearish)/3)` with negation handling and a credibility table
(Reuters/Bloomberg = 1.0 … Tier 4 blog = 0.3). Items older than 300 s decay to
under 37 % weight.

---

## Appendix A — Repository layout

```
backend/
  api/        FastAPI app, WebSocket + REST routers
  core/       config, clock, signal lock, frozen snapshot, cycle manager, store
  data/       ring buffers, Binance WS, CoinGecko, simulator, market hub, sync
  formulas/   22 formulas in 8 category packages + drg.py + engine.py
  brain/      startup verification, connectome query, clustering, GCN, health,
              fallback/mb_adjacency_80x80.csv, generate_fallback_matrix.py
  news/       sentiment lexicon, 3 sources, critical detector, news engine
  agents/     base, gemini, local model, github, fusion, orchestrator
  web/        dashboard (index.html, app.js, styles.css, matrix, settings)
  tests/      smoke.py + test_e2e.py (Appendix E)
frontend/     Flutter client (lib/models, services, screens, widgets)
docs/         SPECIFICATION_v2.md, SPEC_NOTES.md
.devcontainer/ devcontainer.json, setup.sh
```

## Appendix B — Outcome tracking

60 s after a signal locks, the engine prices the same asset again:

```
change_bps = (exit/entry − 1) × 10 000
BUY : outcome = +1 if exit > entry else −1
SELL: outcome = +1 if exit < entry else −1
HOLD: outcome = 0,  pnl = 0
```

The result is pushed to the `OutcomeBuffer` (capacity 20), broadcast as `OUTCOME`,
and becomes the reward history that DRG consumes on the next cycle. HOLD is
neutral, never a loss — otherwise the engine would learn to avoid correct
patience.

## Appendix C — Per-asset parameters

| Parameter | BTC | PAXG |
|---|---|---|
| `tai_ticks` | 30 | 20 |
| `afpr_power` | 3.0 | 2.0 |
| `vsd_baseline_window` | 100 | 50 |
| `vsd_recent_window` | 20 | 10 |
| `erc_tolerance_mult` | 0.2 | 0.3 |
| `dskd_q_fast` | 1.0 | 2.0 |
| `ccsv2_confidence_threshold` | 0.55 | 0.65 |
| `hedge_weight` | 0.5 | 0.5 |
| `forced_hold_ticks` | 15 | 10 |

PAXG has fewer ticks per minute, so its windows are shorter and its confidence
gate stricter.

## Appendix D — Agent prompt template

Every agent receives the identical prompt (`backend/agents/base.py`): asset, price
and timestamp; then the formula groups (micro-structure, order book, hedge,
regime, temporal, Kalman, news, brain, DRG), the latest headline, the brain
status, and the instruction to answer with the Section 7.1 JSON schema only.

## Appendix E — The 13 end-to-end acceptance tests

`backend/tests/test_e2e.py` (`PYTHONPATH=. python -m pytest backend/tests -q`).
Time is compressed 20×; every ratio of the protocol is preserved.

| # | Test | Asserts |
|---|---|---|
| E01 | cold start | first cycle locks 🔒 with all 22 formulas + DRG, no errors, < 50 ms, complete UI payload |
| E02 | immutability | live `FORMULA_UPDATE`s do not move the lock; a duplicate `lock()` returns the same object |
| E03 | formula isolation | an injected `RuntimeError` in TAI zeroes only TAI, records the error, 21 formulas unaffected |
| E04 | evidence gate | 11 zeros ⇒ `insufficient_evidence`, fusion returns HOLD @ 0.50 |
| E05 | emergency | critical event ⇒ ⚡ HOLD @1.0, `superseded_by` preserved, bounded duration |
| E06 | trust gate | Tier 3–4 critical headlines cannot break a lock; Tier 1 can; flash-move threshold works |
| E07 | brain fallback | 80×80 matrix, healthy, signed and connected, non-zero 3-layer read-out |
| E08 | degradation | six ordered levels, driven by real component failures; level 6 forces HOLD |
| E09 | local model | GGUF/ONNX magic-byte and size validation, and a usable labelled STUB |
| E10 | agents | empty keys fail safe, weights are 0.40/0.25/0.20/0.15, a cycle completes with no agents |
| E11 | asset switch | unsupported assets rejected; switch applies at the next boundary; Appendix C per asset |
| E12 | hedge stress | HSI 0.95 ⇒ ×0.20 dampening, forced HOLD, verbatim HOLD box |
| E13 | outcomes | signal scored after the horizon, HOLD neutral, win rate, DRG reward path |

## Appendix F — Environment variables

All optional; see `.env.example` for the annotated list.

`HOST`, `PORT`, `LOG_LEVEL`, `CORS_ORIGINS`,
`NEUPRINT_APPLICATION_CREDENTIALS`, `NEUPRINT_SERVER`, `NEUPRINT_DATASET`,
`CAVE_TOKEN`, `CAVE_SERVER`, `CAVE_DATASET`, `BRAIN_FORCE_FALLBACK`,
`GEMINI_API_KEY`, `GEMINI_MODEL`, `GITHUB_MODELS_TOKEN`, `GITHUB_MODELS_MODEL`,
`CRYPTOPANIC_API_KEY`, `NEWSAPI_API_KEY`, `NEWS_ENABLED`, `NEWS_POLL_SECONDS`,
`RSS_FEEDS`, `MARKET_DATA_MODE`, `MARKET_ALLOW_SIMULATOR`, `SIMULATOR_SEED`,
`REDIS_URL`, `TIME_SCALE`, `LOCK_DEADLINE_SECONDS`, `FORMULA_REFRESH_SECONDS`,
`MODEL_STORE_DIR`, `LOCAL_AGENT_STUB`.

## Appendix G — Deviations from the v2.0 draft

Every deliberate departure from the original wording, with its rationale and
escape hatch, is documented in **`docs/SPEC_NOTES.md`** (11-A, 21-A, 22-A, 22-B,
and the fusion-confidence definition).
