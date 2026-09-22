# DROSOPHILA TRADER v2.0

A 1-minute scalp-trading engine for **BTC** and **PAXG** built around a
Drosophila mushroom-body connectome. One immutable signal per minute, 22
formulas, three AI agents, and a brain that keeps working when neuPrint, the
exchange or your API keys do not.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  t=0s    freeze snapshot ─ formulas (22) ─ agents (concurrent)       │
   │  t=8s    FUSE ─▶ 🔒 LOCK ─▶ broadcast           ⏳ → 🔒 → ⚡         │
   │  t=8-60s the signal cannot change. Period.                           │
   │  t=60s   score the outcome ─▶ reward the dopamine neurons ─▶ repeat  │
   └──────────────────────────────────────────────────────────────────────┘
```

## Quick start

```bash
python -m pip install -r requirements.txt      # or: bash .devcontainer/setup.sh

PYTHONPATH=. python -m backend.api.main        # engine + dashboard
open http://localhost:8000/                    # dashboard
open http://localhost:8000/matrix              # 80×80 connectome viewer
open http://localhost:8000/settings            # keys, models, brain, news
```

No API keys are required to start: the engine detects what is missing and
degrades gracefully (Section 12 of the specification).

* demo speed — `TIME_SCALE=10` gives 6-second cycles
* offline mode — `MARKET_ALLOW_SIMULATOR=1` (default) keeps everything alive
* brain offline — the committed 80×80 matrix is used automatically
* local model — drop a `.gguf`/`.onnx` in Settings, or `LOCAL_AGENT_STUB=1`

## Verify it

```bash
PYTHONPATH=. python -m pytest -q                 # Appendix E - 13 scenarios
PYTHONPATH=. python -m backend.tests.smoke --seconds 90   # cycle-by-cycle dump
```

## Layout

| Path | What it is |
|---|---|
| `backend/core` | config, world clock, **signal lock**, frozen snapshot, cycle manager, store |
| `backend/data` | tick/L2/candle buffers, Binance WS, CoinGecko, simulator, market hub |
| `backend/formulas` | the 22 formulas in 8 categories + `engine.py` + the DRG reward learner |
| `backend/brain` | 5-step startup verification, connectome query, spectral clustering, 3-layer GCN, health checks, fallback matrix |
| `backend/news` | sentiment lexicon, CryptoPanic/NewsAPI/RSS, critical-event detector |
| `backend/agents` | Gemini, local GGUF/ONNX, GitHub Models, fusion, orchestrator |
| `backend/web` | zero-build dashboard (`/`, `/matrix`, `/settings`) |
| `backend/tests` | `smoke.py` + `test_e2e.py` (Appendix E) |
| `frontend` | Flutter client (same protocol, same information architecture) |
| `docs` | [`SPECIFICATION_v2.md`](docs/SPECIFICATION_v2.md), [`SPEC_NOTES.md`](docs/SPEC_NOTES.md) |

## The parts that make it v2.0

* **Signal Lock Protocol.** `COMPUTING ⏳ → LOCKED 🔒 → EMERGENCY_OVERRIDE ⚡`.
  `lock()` is callable once per cycle and the frozen signal is a `NamedTuple`;
  a duplicate or late computation physically cannot rewrite what you are
  looking at. The only exception is a *critical* news event, and it can only
  ever force HOLD — never a BUY or a SELL.
* **22 formulas, 8 categories.** Micro-structure, order book, BTC×PAXG hedge,
  volatility/regime, temporal pattern, Kalman, news, brain output. Each one is
  isolated: an exception zeroes that formula, records the error, and the other
  21 still run. 11 zeros ⇒ forced HOLD.
* **Real connectome, real fallback.** neuPrint/FlyWire are queried with explicit
  timeouts through a 5-step verification that never raises; if the network is
  unavailable the committed 80×80 matrix (285 excitatory / 71 inhibitory edges,
  calibrated gain 3.2802) takes over seamlessly.
* **News engine.** CryptoPanic every 30 s, NewsAPI/RSS every 60 s, an offline
  headline pack so the panel is never blank, and a critical-event detector that
  only trusts Tier ≤2 sources for lock-breaking events.
* **Agents with a spine.** Gemini (structured JSON schema), a local GGUF/ONNX
  model with magic-byte validation + test inference + RAM monitoring, and GitHub
  Models with key-test-on-entry. Weights 0.40/0.25/0.20/0.15 renormalise over
  whoever actually answered; losing an agent can never silently change the
  meaning of a score.

## Documentation

* [`docs/SPECIFICATION_v2.md`](docs/SPECIFICATION_v2.md) — the full specification
  (every formula, every threshold, the API, Appendices A–F).
* [`docs/SPEC_NOTES.md`](docs/SPEC_NOTES.md) — every deliberate deviation from
  the v2.0 draft, with symptom → cause → decision → verification → escape hatch
  (11-A HSI calibration, 21-A pre-ReLU KCAE, 22-A/22-B/22-C matrix construction,
  8-A confidence definition, 5-A/7-A/10-A operational notes).

**Nothing here is financial advice.** The engine is a research instrument: it
emits a locked directional opinion every minute and scores itself afterwards.
