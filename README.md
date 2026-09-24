# DROSOPHILA TRADER v2.0

A 1-minute scalp-trading engine for **BTC** and **PAXG** built around a
Drosophila mushroom-body connectome. One immutable signal per minute, 22
formulas, three AI agents, and a brain that keeps working when neuPrint, the
exchange or your API keys do not.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  window N-1   compute signal N   (last ~8 s of the countdown)        │
   │  window N     🔒 LOCK ─▶ broadcast ─▶ compute signal N+1             │
   │  t=60 s       score the outcome ─▶ reward the dopamine neurons       │
   └──────────────────────────────────────────────────────────────────────┘
```

**The whole app runs on ONE port.** Dashboard, API, WebSocket, matrix viewer,
settings page and the Flutter build are all served from port **8000**, so there
is exactly one URL to forward and no second window to keep open.

---

## 1. Install and run (copy-paste, in order)

### Step 1 — get the code

```bash
git clone -b arena/01a0c844-p-4 https://github.com/ApiDemo495/P-4.git
```

```bash
cd P-4
```

Already inside a Codespace for this repository? Skip the clone and just sync the branch:

```bash
git fetch origin && git checkout arena/01a0c844-p-4 && git pull --ff-only
```

### Step 2 — free any stale ports

```bash
bash run.sh --clean
```

### Step 3 — install everything (Python, venv, requirements, redis, .env)

```bash
bash .devcontainer/setup.sh
```

The same work happens automatically inside `bash run.sh`; this explicit step just
makes the download visible. It installs `python3-venv`, `python3-pip` and
`redis-server` through apt, creates `./.venv`, installs `requirements.txt`
(with retries, the official index, and a PEP 668 fallback), and copies
`.env.example` to `.env`.

### Step 4 — confirm the environment is complete

```bash
bash run.sh --check
```

### Step 5 — start the app (the only command you need every day)

```bash
bash run.sh --bg
```

### Step 6 — verify that the port really answers

```bash
bash run.sh --status
```

### Step 7 — make the port public (Codespaces only)

```bash
bash run.sh --public
```

### Step 8 — print every feature URL again at any time

```bash
bash run.sh --urls
```

### Optional — build the Flutter client (≈700 MB, once)

```bash
INSTALL_FLUTTER=1 bash frontend/run_web.sh
```

### Optional — hot reload while editing the Flutter UI

```bash
bash frontend/run_web.sh --dev
```

---

## 2. What each command does

| Command | What it does |
|---|---|
| `bash run.sh` | installs what is missing, starts the engine in the foreground, streams the log, prints the URL |
| `bash run.sh --bg` | same, in the background, under a supervisor that restarts it if it ever dies — safe to run twice |
| `bash run.sh --status` | is the supervisor running, is the port listening, does it answer HTTP, plus the current URL list |
| `bash run.sh --urls` | prints every feature URL (all on the one port) |
| `bash run.sh --check` | full environment diagnosis; starts nothing |
| `bash run.sh --clean` | stops the engine and any stray Flutter/Dart dev servers, then lists what still listens |
| `bash run.sh --stop` | stops the engine and its supervisor |
| `bash run.sh --setup-only` | installs everything, starts nothing |
| `bash run.sh --port 8020` | uses a different port (then forward that one instead) |
| `bash run.sh --public` | flips the Codespaces port to public |
| `bash .devcontainer/setup.sh` | the Codespaces `postCreateCommand`: system packages, venv, requirements, redis, `.env` |
| `bash frontend/run_web.sh` | downloads the Flutter SDK if needed and builds the web client into `frontend/build/web` |
| `bash frontend/run_web.sh --check` | reports whether the SDK is installed and whether a build exists — downloads nothing |
| `bash frontend/run_web.sh --dev` | optional hot-reload dev server, fixed at port 8081 |

---

## 3. The one port

Open the URL `bash run.sh --urls` prints. Everything lives under it:

| Page (same port) | What it is |
|---|---|
| `/` | dashboard — widget panel, brain wiring, hedge, agents, news, history, formulas |
| `/settings` | keys, local model, brain reconnect, news poll |
| `/matrix` | the 80×80 connectome with the last activation trace |
| `/flutter` | the Flutter client, once `bash frontend/run_web.sh` has built it |
| `/docs` | the OpenAPI explorer |
| `/ws/signals` | the WebSocket stream both clients use |

`bash frontend/run_web.sh --dev` is the only command that opens a second port,
and it is pinned to **8081**. Anything else in your PORTS tab belongs to another
tool you started; `bash run.sh --clean` stops what it can and tells you what is
left.

---

## 4. API keys and live data

**Prices need no key.** `MARKET_DATA_MODE=auto` connects to the Binance
WebSocket first (aggTrade + depth20 + 1-minute klines), falls back to CoinGecko,
and only uses the built-in simulator if neither is reachable — and it always
labels which source is active in the UI.

Keys are optional and are entered **in the browser**: click **🔑 API keys** in the
dashboard header, paste, press **Test** (it makes a real authenticated request),
tick *remember* to persist to the git-ignored `.env`.

| Key | What it unlocks |
|---|---|
| `GEMINI_API_KEY` | Gemini agent, 25 % of the fusion |
| `GITHUB_MODELS_TOKEN` | GitHub Models agent, 15 % |
| `CRYPTOPANIC_API_KEY` | 30-second news engine + critical-event detection |
| `NEWSAPI_API_KEY` | NewsAPI backup tier |
| `NEUPRINT_APPLICATION_CREDENTIALS` | live hemibrain connectome (otherwise the committed 80×80 matrix) |

Prefer a file? Copy the template and edit it:

```bash
cp .env.example .env
```

Then set `GEMINI_API_KEY=...`, `CRYPTOPANIC_API_KEY=...`, and leave
`MARKET_DATA_MODE=auto` and `TIME_SCALE=1.0` as they are.

Useful switches: `MARKET_ALLOW_SIMULATOR=1` (offline-safe, default), `BRAIN_FORCE_FALLBACK=1`
(skip every network attempt for the connectome), `LOCAL_AGENT_STUB=1` (exercise
the local-model path without weights), `TIME_SCALE=10` (6-second windows, for demos).

---

## 5. Running in a GitHub Codespace

1. **Code ▾ → Codespaces → +**, and pick the branch `arena/01a0c844-p-4`
   (or open `https://codespaces.new/ApiDemo495/P-4?ref=arena/01a0c844-p-4`).
2. Wait for `postCreateCommand` to finish — it runs `bash .devcontainer/setup.sh`.
3. On every attach the devcontainer runs `bash run.sh --bg`, so the port is
   already answering when you arrive.

If you need to do it by hand:

```bash
bash run.sh --bg
```

```bash
bash run.sh --status
```

4. Open the **PORTS** tab → port **8000** → globe icon (or run `bash run.sh --public`).
   Skip every other port entry; nothing else is needed.

### Why you sometimes see "This page isn't working / HTTP ERROR 502"

The Codespaces forwarder proxies port 8000 to a process **inside** the container.
If nothing is listening there — the server was never started, it crashed, or the
Codespace went to sleep — the forwarder itself answers 502. It is not DNS, CORS
or a firewall problem, and it is never caused by using several features at once.

What the app does about it now:

* the HTTP port opens **before** the engine is warm, so a slow start shows a
  "warming up" banner in the dashboard instead of a browser error page;
* `bash run.sh --bg` verifies the start with a real HTTP request and refuses to
  print a URL that does not answer;
* the background engine runs under a supervisor that restarts it within ~2 s of
  a crash (verified by SIGKILL);
* `bash run.sh --status` and `bash run.sh --check` tell you exactly which of the
  three layers is down: supervisor, listener, or HTTP.

If the whole Codespace is asleep, the port 502s until you wake it: reopen the
Codespace tab and let the attach command run, then reload the URL.

---

## 6. Troubleshooting

| What you see | What it means | Fix |
|---|---|---|
| Sad-page error on port 8000 | nothing is listening | `bash run.sh --bg` then `bash run.sh --status` |
| 502 right after opening the Codespace | the container was asleep; the attach command has not finished | wait a few seconds, reload; `bash run.sh --bg` if needed |
| "Warming up…" banner in the dashboard | the port answers but the brain/market warm-up has not finished | wait ~5 s, the banner clears by itself |
| A notice covering the whole screen | it cannot happen any more: notices are inline chips and a red HOLD box, never a full-screen layer | nothing to fix — if you ever see one, it is a browser cache: reload with `Ctrl+Shift+R` |
| Several ports, each one "not working" | dead processes of other tools (`flutter run` picks a random port) | `bash run.sh --clean`, then forward **only 8000** |
| `pip install` fails or "downloading requirements" stalls | `python3-venv` missing, PEP 668, or a proxy | see the block below |
| Flutter SDK download fails | 700 MB download, optional | `rm -rf ~/flutter` then `INSTALL_FLUTTER=1 bash frontend/run_web.sh` |
| `bash: .venv/bin/python: No such file` | the venv was never created | `bash run.sh` creates it |
| `ModuleNotFoundError: No module named 'backend'` | started without the repo root on `PYTHONPATH` | use `bash run.sh` |
| `/flutter` returns `{"detail": "The Flutter web build has not been created yet."}` | the client was never built | `bash frontend/run_web.sh` |
| `L4 Fallback brain matrix` + a STUB agent chip | neuPrint unreachable, local stub enabled — both are supported modes | add a neuPrint token in the UI, load a real `.gguf` in Settings |
| `market_data → simulator` in the log | Binance and CoinGecko are unreachable from your network | `MARKET_DATA_MODE=coingecko`, or keep the simulator (always labelled) |

### When the requirements download fails

Run this first — it fixes the usual cause (`python3-venv` missing):

```bash
sudo apt-get update && sudo apt-get install -y python3-venv python3-pip
```

```bash
rm -rf .venv && bash run.sh
```

If pip itself is blocked (proxy, VPN, private index):

```bash
.venv/bin/python -m pip install --user --index-url https://pypi.org/simple -r requirements.txt
```

Setup writes the full pip log to `/tmp/pip-install.log`; the launcher prints the
last lines of it when an install fails.

---

## 7. Verify the engine

```bash
PYTHONPATH=. .venv/bin/python -m pytest backend/tests -q
```

The layout guard (`backend/tests/test_ui_layout_guard.py`) is part of that run: it
fails if a class the dashboard applies to a widget ever picks up a full-screen
rule again — the bug that once stretched the HOLD box over the whole page.

```bash
PYTHONPATH=. .venv/bin/python -m backend.tests.smoke --seconds 90
```

```bash
curl -s localhost:8000/api/health | python3 -m json.tool
```

```bash
curl -s localhost:8000/api/brain/explain | python3 -m json.tool
```

```bash
tail -f server.log
```

---

## 8. Layout

| Path | What it is |
|---|---|
| `backend/core` | config, world clock, **signal lock**, frozen snapshot, cycle manager, risk levels, store |
| `backend/data` | tick/L2/candle buffers, Binance WS, CoinGecko, simulator, market hub |
| `backend/formulas` | the 22 formulas in 8 categories + `engine.py` + the DRG reward learner |
| `backend/brain` | 5-step startup verification, connectome query, spectral clustering, 3-layer GCN, `explain.py`, fallback matrix |
| `backend/news` | sentiment lexicon, CryptoPanic/NewsAPI/RSS, critical-event detector |
| `backend/agents` | Gemini, local GGUF/ONNX, GitHub Models, fusion, orchestrator |
| `backend/web` | zero-build dashboard (`/`, `/matrix`, `/settings`) |
| `backend/tests` | `smoke.py` + `test_e2e.py` (Appendix E) |
| `frontend` | Flutter client (same protocol, same fixed widget layout, **Brain** tab) |
| `docs` | [`SPECIFICATION_v2.md`](docs/SPECIFICATION_v2.md), [`SPEC_NOTES.md`](docs/SPEC_NOTES.md) |

---

## 9. The parts that make it v2.0

* **Pipelined countdown.** The 60-second countdown shows the signal computed
  *during the previous countdown*, while the engine computes the next one. The
  panel is never blank; `SIGNAL_PIPELINE=0` restores the literal "compute at
  t=0, lock at t=8 s" timeline.
* **A dashboard you can read at a glance.** Row 1: prediction · countdown 1–60 ·
  glittering HOLD box. Row 2: take-profit & stop-loss (volatility-derived) ·
  prediction accuracy. An emergency override is a small inline chip under the
  prediction — never a full-screen overlay. The Flutter client adds haptics
  (`mediumImpact` on a direction change, `heavyImpact` + `vibrate` on an
  override, `selectionClick` on the first lock).
* **The fly brain is visible, not decorative.** `/api/brain/wiring` returns the
  formula → neuron map and the five circuit stages; `/api/brain/explain` returns
  what the circuit did in the locked window — dominant projection neurons, KC
  sparsity, MBON and lateral-horn read-outs, the dopamine gates, and the brain's
  exact share of the fused score.
* **Signal Lock Protocol.** `COMPUTING ⏳ → LOCKED 🔒 → EMERGENCY_OVERRIDE ⚡`.
  `lock()` is callable once per cycle and the frozen signal is a `NamedTuple`, so
  a late or duplicate computation cannot rewrite what you are looking at. The one
  exception is a *critical* news event, which can only force HOLD.
* **22 formulas, 8 categories.** Each one is isolated: an exception zeroes that
  formula, records the error, and the other 21 still run. 11 zeros ⇒ forced HOLD.
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
  whoever actually answered.

---

## 10. Documentation

* [`docs/SPECIFICATION_v2.md`](docs/SPECIFICATION_v2.md) — the full specification
  (every formula, every threshold, the API, Appendices A–F).
* [`docs/SPEC_NOTES.md`](docs/SPEC_NOTES.md) — every deliberate deviation from the
  v2.0 draft, with symptom → cause → decision → verification → escape hatch.

**Nothing here is financial advice.** The engine is a research instrument: it
emits a locked directional opinion every minute and scores itself afterwards.
