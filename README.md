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

One command. It creates the virtualenv, installs what is missing, starts Redis if
you have it, and prints the exact URL to open:

```bash
bash run.sh
```

```bash
bash run.sh --bg        # background, survives closing the terminal
bash run.sh --check     # diagnose the environment without starting anything
bash run.sh --urls      # print every feature URL (all on ONE port)
bash run.sh --stop      # stop the engine
bash run.sh --clean     # stop the engine + any stray Flutter/Dart dev servers
bash run.sh --port 8020 # if 8000 is taken
```

Then open the URL it prints. **There is exactly one port** — everything the app
offers is served from it, so there is never a second forwarded URL to hunt for:

| Page (same port) | What it is |
|---|---|
| `/` | dashboard — widget panel, brain wiring, hedge, agents, news, history, formulas |
| `/settings` | keys, local model, brain reconnect, news poll |
| `/matrix` | the 80×80 connectome with the last activation trace |
| `/flutter` | the Flutter client, once `bash frontend/run_web.sh` has built it |
| `/docs` | the OpenAPI explorer |
| `/ws/signals` | the WebSocket stream both clients use |

The only command that opens a **second** port is `bash frontend/run_web.sh --dev`
(hot reload), and it is fixed at **8081** — never a random one. Anything else you
see in the PORTS tab belongs to another tool you started; `bash run.sh --clean`
gets rid of them and tells you what is left.

**API keys are optional and are entered in the browser** — click **🔑 API keys**
in the dashboard header. Each key is validated with a live request before it is
accepted, applied to the running engine immediately, and can be written to the
git-ignored `.env` with one checkbox. No terminal, no `nano .env`.

* demo speed — `TIME_SCALE=10` gives 6-second cycles
* offline mode — `MARKET_ALLOW_SIMULATOR=1` (default) keeps everything alive
* brain offline — the committed 80×80 matrix is used automatically
* local model — drop a `.gguf`/`.onnx` in Settings, or `LOCAL_AGENT_STUB=1`

### Running in a GitHub Codespace

1. **Code ▾ → Codespaces → +** on this repository (branch `arena/01a0c844-p-4`).
2. Wait for `postCreateCommand` to finish (`bash .devcontainer/setup.sh`).
3. In the terminal:

```bash
bash run.sh --bg
```

4. Open the **PORTS** tab → port **8000** → globe icon. That is the dashboard.
   The **Run and Debug** panel (`F5`) and the **Terminal → Run Task** menu have
   the same thing wired up as tasks.

Flutter, if you want the mobile client in the browser:

```bash
bash frontend/run_web.sh      # installs the SDK if needed, then builds
```

The build lands in `frontend/build/web` and the engine serves it at `/flutter` —
same origin, same port, no CORS, no second forwarded URL.

### Troubleshooting

| What you see | What it means | Fix |
|---|---|---|
| Browser error page with a sad page icon on port 8000 | **Nothing is listening on 8000** — the server is not running (this is the most common one) | `bash run.sh --bg`, wait for `✔ running in the background`, then reload the tab |
| Several ports listed, each one "not working" | Codespaces keeps a port in the PORTS tab until its **process** dies. The random 6xxx port is a `flutter run` dev server somebody started by hand; 8081 is only used by `run_web.sh --dev`; 6379 was Redis | `bash run.sh --clean` (stops the engine and any stray Flutter/Dart servers, then reports what still listens). Forward **only 8000**. A window reload clears dead entries |
| Everything must be on one port | By design it already is | `bash run.sh` → use `/`, `/settings`, `/matrix`, `/flutter` on that single port. Build the Flutter client with `bash frontend/run_web.sh` (no extra port); only `--dev` adds 8081 |
| `pip install` fails / "downloading requirements" stops | Either `python3-venv` is missing, or PEP 668 blocks the system interpreter, or the network needs a proxy | `bash run.sh` already retries the official index and tries `--user --break-system-packages`; if it still fails: `sudo apt-get update && sudo apt-get install -y python3-venv python3-pip && rm -rf .venv && bash run.sh` |
| Flutter SDK download fails (`~700 MB`) | That download is the only heavy one, and it is optional | Use the dashboard on port 8000 (same engine, same panel). Retry later: `rm -rf ~/flutter && INSTALL_FLUTTER=1 bash frontend/run_web.sh`; check first with `bash frontend/run_web.sh --check` |
| Same, after a Codespace restart | The container stopped and the process is gone with it | `bash run.sh --bg` again |
| `bash: .venv/bin/python: No such file` | The venv was never created (setup did not finish) | `bash run.sh` creates it |
| `ModuleNotFoundError: No module named 'backend'` | Started without the repo root on `PYTHONPATH` | use `bash run.sh`, or prefix `PYTHONPATH=$PWD` |
| `bash run.sh --check` says modules are missing | Dependencies are not installed for the interpreter in use | `bash run.sh` (it installs them) |
| Port 8000 answers only after signing in to GitHub | Forwarded ports are private by default | `bash run.sh --public`, or use it signed in |
| `flutter run` prints nothing / exits immediately | No Chrome/GPU in the container, `flutter` not on `PATH`, and the dev port is not forwarded | `bash frontend/run_web.sh` (release build + `/flutter`) instead |
| `/flutter` returns `{"detail": "The Flutter web build has not been created yet."}` | The build does not exist | `bash frontend/run_web.sh` |
| Dashboard shows `L4 Fallback brain matrix` and a STUB agent chip | neuPrint is unreachable and `LOCAL_AGENT_STUB=1` is set — both are supported modes, clearly labelled | add a neuPrint token in the UI, and load a real `.gguf` in Settings |
| `market_data → simulator` in the log | Binance/CoinGecko unreachable from the network you are on | set `MARKET_DATA_MODE=coingecko`, or keep the simulator (it is always flagged) |

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
| `frontend` | Flutter client (same protocol, same fixed widget layout, **Brain** tab with the full wiring map) |
| `docs` | [`SPECIFICATION_v2.md`](docs/SPECIFICATION_v2.md), [`SPEC_NOTES.md`](docs/SPEC_NOTES.md) |

## The parts that make it v2.0

* **Pipelined countdown.** The 60-second countdown shows the signal that was
  computed *during the previous countdown*, while the engine computes the next
  one in the last ~8 seconds of the current window. The panel is never blank:
  the only thing a cold start shows is a labelled sentinel. `SIGNAL_PIPELINE=0`
  restores the literal "compute at t=0, lock at t=8 s" draft timeline.
* **A dashboard you can read at a glance.** Row 1: prediction · countdown 1–60 ·
  glittering HOLD box. Row 2: take-profit & stop-loss (volatility-derived) ·
  prediction accuracy. An emergency override is a small inline chip under the
  prediction — never a full-screen overlay. The Flutter client adds real haptics
  (`mediumImpact` on a direction change, `heavyImpact` + `vibrate` on an
  override).
* **The fly brain is visible, not decorative.** `/api/brain/wiring` returns the
  formula → neuron map and the five circuit stages; `/api/brain/explain` returns
  what the circuit actually did in the locked window — dominant projection
  neurons, KC sparsity, MBON and lateral-horn read-outs, the dopamine gates, and
  the brain's exact share of the fused score. The **Brain** panel (web) and
  **Brain** tab (Flutter) render both.
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
