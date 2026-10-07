# DROSOPHILA TRADER v2.0

A scalp-trading engine for **BTC** and **PAXG** built around a Drosophila
mushroom-body connectome. One immutable prediction per **60-second window**,
22 formulas, three AI agents, and a brain that keeps working when neuPrint, the
exchange or your API keys do not.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  window N-1   compute prediction N  (during the previous countdown)  │
   │  window N  t=0   🔒 LOCK ─▶ broadcast the whole dashboard            │
   │            t=15  ⚡ PULSE ─▶ formulas · news · agents · brain · rates │
   │            t=30  ⚡ PULSE ─▶ the same, plus a fresh news poll         │
   │            t=45  ⚡ PULSE ─▶ the same                                 │
   │            t=60  boundary ─▶ next signal; this one is scored         │
   └──────────────────────────────────────────────────────────────────────┘
```

**One clock, one tick.** The countdown is 60 seconds, phase-locked to the UTC
minute, and it is rendered from absolute instants the backend publishes - so it
cannot drift or restart. Every panel (signal, formulas, news, agents, brain,
accuracy) refreshes on the *same* tick, from a single message, instead of each
one polling on a private timer.

Every prediction carries a **side (BUY or SELL - there is no HOLD), a fresh-ness
age, the reasoning behind it, and a 1:1 take-profit / stop-loss pair**.

**The crowd's emotions, live.** A 60-second market is easily pushed around by
retail emotion, so the engine reads the crowd continuously - eight emotions
(fear, panic, capitulation, denial, hope, euphoria, FOMO, complacency), each
measured from microseconds to minutes - and the dashboard says **which emotion
is dominant right now**, how long it has held, how crowded / manipulated the
minute looks, and what the crowd was feeling when the locked signal was
computed. A crowded minute sizes the confidence down; it never flips the side.

**Deep reasoning under the emotions.** The reading is not a rule of thumb: under
every emotion sit fourteen market-microstructure formulas run on the same tape
at the same half-second cadence - Hawkes self-excitation (are prints causing
prints?), microprice lean, VPIN order-flow toxicity, Kyle's lambda (how
pushable the tape is), Lo-MacKinlay variance ratios, the Hurst exponent,
Bandt-Pompe permutation entropy, Lillo-Farmer trade-sign memory, a Haar wavelet
energy spectrum (which timescale carries the action), a three-state regime
filter, and momentum-ignition / quote-stuffing / spoofing detectors - each on
three bands from ticks to the minute. A discrete Bayesian filter with sticky
transitions turns that evidence into a *belief* over the eight emotions, and the
dashboard renders the whole **reasoning chain**: formula, inputs, value, what it
reads as, and which emotions it argues for. The belief is blended into the bars,
the detectors into the manipulation score, and the whole chain is frozen with
the signal at lock time.

**The whole app runs on ONE port.** Dashboard, API, WebSocket, matrix viewer,
settings page are all served from port **8000**, so there
is exactly one URL to forward and no second window to keep open.

---

## 0. Zero commands: open the Codespace and wait

There is nothing to type. Create a Codespace **on this branch** - the one-click
link does exactly that:

**https://codespaces.new/ApiDemo495/P-4?ref=arena/01a0c844-p-4&quickstart=1**

(`main` carries only this README: a Codespace created there has nothing to
start. The **Code → Codespaces** button on the repository page defaults to
`main`; use the link above, or switch the branch picker first.) Then the
container does the rest by itself:

| When | What happens automatically |
| --- | --- |
| container created | the Codespace starts from the **pre-provisioned image** `ghcr.io/apidemo495/p-4-dev` (built by the `devcontainer-image` Action from `.devcontainer/Dockerfile`: Python 3.11 + every line of `requirements.txt` already installed in `/opt/venv`, redis, gh). `--provision` adopts `/opt/venv` as `.venv` in about one second, writes `.env`, starts redis. Nothing is pip-installed and nothing else is downloaded |
| every start / wake | `--start`: self-heals a missing `.venv` or a changed `requirements.txt`, then starts the engine under its supervisor |
| every editor attach | `--attach`: the same, plus it waits until the first prediction is locked and prints the URL |

If a Codespace ever fails to create with *"failed to pull image"*, the GHCR
package is private: open the repository on GitHub → **Packages** → `p-4-dev` →
**Package settings** → *Change visibility* → **Public** (one time). The scripts
also still work on the stock image - they just install instead of adopting.

The three hooks are serialised by a lock (they overlap in a Codespace, and two
pip installs into one `.venv` is how "requirements did not download" used to
happen), the editor is only handed over after `postCreateCommand` finishes,
provisioning makes up to three passes (a clean `.venv` on the second) and
checks that **every** module in `requirements.txt` imports before it calls the
environment ready. Nothing in the automatic path can wait for a keyboard: apt
is non-interactive and `sudo` never asks for a password. Logs:
`.run/logs/setup-*.log`, `/tmp/pip-install.log`.

Port **8000** is forwarded, made **public** and opened in your browser for you,
so the dashboard appears on its own. Everything else is on that one URL:
`/` dashboard · `/settings` API keys · `/matrix` connectome · `/docs` API.

**If it ever looks stuck**, open the Ports tab and click the globe next to
`8000`, or open the URL it prints. The engine restarts itself if it stops.

**Proof it works unattended:** every change to the container files runs the
`devcontainer-rehearsal` GitHub Action, which starts the very same image and
lifecycle hooks on a GitHub runner and posts the transcript on PR #1 — the last
run shows provisioning in 2 s and the first signal locked 5 s after creation.
If your Codespace does not do the same, one of three things is true: it was
created on `main`; it is in **recovery mode** (yellow banner "running in
recovery mode" — the container failed to build, click *Rebuild*); or a hook is
still running — open `/settings` → **Codespace self-start**, or run:

```bash
cat .run/logs/autostart.journal
```

### Slow to open (10–20 minutes)? That is GitHub, not the engine — fix it with a prebuild

The repository's own hooks are measured: provisioning 2 s, engine answering in
under a minute (the rehearsal transcript on PR #1). When the *window* still
takes ten or twenty minutes to appear, that time is spent **before any hook
runs**: GitHub allocating the VM, pulling the container image into a cold
region, and installing the VS Code server. Every hook now logs how long the
container had already been alive when it started, so you can see where the
minutes went:

```bash
bash tools/codespace_autostart.sh --status
```

A large "container age" at the first hook means GitHub was slow, not the
engine. The cure GitHub offers for exactly this is a **Codespaces prebuild**:
GitHub builds the container for this branch ahead of time, in your region, and
a new Codespace then opens in about thirty seconds. It is a repository setting
(one time, two minutes):

1. Repository → **Settings** → **Codespaces** → **Set up prebuild**.
2. Branch: `arena/01a0c844-p-4`. Region: the one nearest you (for India,
   **Southeast Asia**). Trigger: *On configuration change* (default).
3. **Create**. The first prebuild runs as a GitHub Action; from then on the
   Codespace creation page shows a ✓ *Prebuild ready* badge for this branch.

Nothing in the repository changes for this: the prebuild runs the same image
and the same `postCreateCommand`, so the engine is already provisioned inside
the snapshot GitHub keeps for you.

Also pick the smallest machine type (2-core) on the creation page; a bigger VM
takes longer to allocate and the engine does not need it.

### API keys that survive a fresh Codespace

`/settings` saves every key to the running engine **and** to `.env` (default on).
A brand-new Codespace has no `.env`, so put the same names in
**GitHub → Settings → Codespaces → Secrets** (repository access: this repo):
`GEMINI_API_KEY`, `NEWSAPI_API_KEY`, `CRYPTOPANIC_API_KEY`, `GITHUB_MODELS_TOKEN`,
`NEUPRINT_APPLICATION_CREDENTIALS`, `CAVE_TOKEN`. The `/settings` page's
"What each key is doing right now" table (and `/api/settings/effects`) proves
each key is contributing - voting in fusion, headlines on the wire, live
connectome - or says exactly why it is not. Details: `docs/API_KEYS_AND_MODELS.md`.

### Switches (all optional - set them as Codespace secrets or in `.env`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `PORT` | `8000` | the single port everything is served on |
| `AUTO_OPEN` | `1` | try to make the forwarded port public |

### Doing it by hand instead (any machine, no Codespace)

The rest of this section is the manual path - useful outside Codespaces, or if
you prefer to see every step. In a Codespace it is already done for you.

---

## 1. Install and run (copy-paste, in order)

### Step 1 — get the code

*(Skip to [section 0](#0-zero-commands-open-the-codespace-and-wait) if you are in a Codespace.)*

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

### The Codespace updates itself — no `git pull`

A `git pull` that asks questions is how a Codespace ends up detached from its
branch, so nothing here asks you to run one. The checkout fast-forwards **its
own branch** by itself: at every Codespace start/attach, and — while the engine
runs in a Codespace — every 10 minutes (`AUTO_UPDATE=0` turns that off,
`AUTO_UPDATE_MINUTES` changes the cadence). It never checks out, never switches
branches, never touches `main`, and refuses anything that is not a clean
fast-forward. When it applies an update the engine restarts within seconds and
the dashboard reloads on its own. The dashboard's top bar shows an
**⬆ update available · Apply** chip whenever the branch is behind. The same
from a terminal:

```bash
curl -s localhost:8000/api/update/status | python3 -m json.tool
```

```bash
curl -s -X POST localhost:8000/api/update/apply
```

```bash
bash tools/codespace_autostart.sh --update
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
| `bash run.sh --clean` | stops the engine and any stray dev servers, then lists what still listens |
| `bash run.sh --stop` | stops the engine and its supervisor |
| `bash run.sh --setup-only` | installs everything, starts nothing |
| `bash run.sh --port 8020` | uses a different port (then forward that one instead) |
| `CYCLE_SECONDS=60` | length of one prediction window; 60 keeps it locked to the UTC minute |
| `CALIBRATION_ENABLED=1` | learned reliability: after `CALIBRATION_MIN_SAMPLES` (30) scored windows per asset the per-source hit-rate ledger decides the side and the confidence shown is the realised rate; `CALIBRATION_HALF_LIFE=120` windows of memory; inspect with `GET /api/signal/calibration?asset=BTC` |
| `PREDICTION_MAX_AGE_SECONDS=0` | `0` = as long as its own window (60 s + a 5 s grace); set a number to pin a hard cap |
| `RR_TARGET=1.0` | reward:risk target; 1.0 = take-profit and stop-loss are equidistant |
| `OUTCOME_HORIZON_SECONDS=0` | `0` = score each prediction one window later (60 s) |
| `bash run.sh --public` | flips the Codespaces port to public |
| `bash .devcontainer/setup.sh` | the Codespaces `postCreateCommand`: system packages, venv, requirements, redis, `.env` |

---

## 3. The one port

Open the URL `bash run.sh --urls` prints. Everything lives under it:

| Page (same port) | What it is |
|---|---|
| `/` | dashboard — widget panel, brain wiring, hedge, agents, news, history, formulas |
| `/settings` | keys, local model, brain reconnect, news poll |
| `/matrix` | the 80×80 connectome with the last activation trace |
| `/docs` | the OpenAPI explorer |
| `/ws/signals` | the WebSocket stream the dashboard uses (`EMOTION` messages twice a second) |
| `/api/emotions` | the crowd's emotions: live reading, lock-time reading, dampening |
| `/api/emotions/deep` | the deep layer: fourteen microstructure formulas, Bayesian posterior, reasoning chain, the likelihood table |

Nothing in this project opens a second port. Anything else in your PORTS tab
belongs to another tool you started; `bash run.sh --clean` stops what it can
and tells you what is left.

---

## 4. API keys and live data

**Prices need no key.** `MARKET_DATA_MODE=auto` connects to the Binance
WebSocket first (aggTrade + depth20 + 1-minute klines), then the Gemini public
WebSocket (`MARKET_DATA_MODE=gemini` pins it), then Kraken, then CoinGecko,
and only uses the built-in simulator if none is reachable — and it always
labels which source is active in the UI.

**Optional keyed providers (Formula Genesis Engine).** Three more feeds run
*only* while their key is saved in `/settings` → "Keyed providers": Glassnode
(`GLASSNODE_API_KEY`, BTC exchange net-flow), Twelve Data
(`TWELVEDATA_API_KEY`, dollar index minute closes; free plans get the EUR/USD
proxy) and LunarCrush (`LUNARCRUSH_API_KEY`, hourly social sentiment). Each
box has a **Test** button; without a key the matching column is NaN and the
formulas that read it simply do not fire. Nothing else in the app depends on
them.

Keys are optional and are entered **in the browser**: click **🔑 API keys** in the
dashboard header, paste, press **Test** (it makes a real authenticated request),
tick *remember* to persist to the git-ignored `.env`.

| Key | What it unlocks |
|---|---|
| `GEMINI_API_KEY` (+ `_2`, `_3`) | Gemini agent, 25 % of the fusion. Slot 1 is the primary; `_2`/`_3` are optional stand-ins used automatically while slot 1 is rejected / rate limited, primary retried as soon as it recovers |
| `GITHUB_MODELS_TOKEN` (+ `_2`, `_3`) | GitHub Models agent, 15 %, same three-slot failover |
| `CRYPTOPANIC_API_KEY`, `NEWSAPI_API_KEY` (+ `_2`, `_3`) | news sources, same three-slot failover; `GET /api/settings/keys` shows which slot is in use |
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
| Page loads but stays blank / widgets never paint | a dashboard script threw on boot (e.g. a renamed function still being called) | `node --check backend/web/app.js`, then run the guard: `PYTHONPATH=. .venv/bin/python -m pytest backend/tests/test_dashboard_scripts.py -q` |
| 502 right after opening the Codespace | the container was asleep; the attach command has not finished | wait a few seconds, reload; `bash run.sh --bg` if needed |
| "Warming up…" banner in the dashboard | the port answers but the brain/market warm-up has not finished | wait ~5 s, the banner clears by itself |
| A notice covering the whole screen | it cannot happen any more: notices are inline chips and a red HOLD box, never a full-screen layer | nothing to fix — if you ever see one, it is a browser cache: reload with `Ctrl+Shift+R` |
| Several ports, each one "not working" | dead processes of other tools | `bash run.sh --clean`, then forward **only 8000** |
| `pip install` fails or "downloading requirements" stalls | `python3-venv` missing, PEP 668, or a proxy | see the block below |
| `bash: .venv/bin/python: No such file` | the venv was never created | `bash run.sh` creates it |
| `ModuleNotFoundError: No module named 'backend'` | started without the repo root on `PYTHONPATH` | use `bash run.sh` |
| `L4 Fallback brain matrix` + a STUB agent chip | neuPrint unreachable, local stub enabled — both are supported modes | add a neuPrint token in the UI, load a real `.gguf` in Settings |
| VS Code says `command 'markdown.showPreview' not found` | the Markdown extension is not available in that Codespace - it is an editor problem, not an app problem | read the guide in the app instead: open `/readme` (nav → 📖 Guide); every doc is there with copy buttons |
| "Problem in self starting / self downloading" | a hook step failed or the pip download was interrupted | open `/settings` → **Codespace self-start** (or `GET /api/system/autostart`): the journal and the pip/setup logs are shown; it retries on the next start/attach |
| Brain says `Offline connectome` with a neuPrint token pasted | the token was rejected or neuPrint is unreachable from your network | `/settings` → Brain → **Force Reconnect**; the step detail says exactly which (no package install is needed - the client is built in) |
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
last lines of it when an install fails. In a Codespace the automatic passes are
logged to `.run/logs/setup-1.log` … `setup-3.log`; to force a fresh
provisioning pass without recreating the container:

```bash
rm -f .run/.provisioned && bash tools/codespace_autostart.sh --attach
```

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
curl -s localhost:8000/api/emotions | python3 -m json.tool
```

```bash
curl -s localhost:8000/api/emotions/deep | python3 -m json.tool
```

The microsecond contract, checked against the running engine: the forecast
window, its six-digit instants, the per-formula µs timings, the tape's real
resolution, the crowd's eight emotions with the dominant one, and the deep
layer (fourteen reasoning steps, a normalised posterior, three bands, the
detectors) - no browser needed.

```bash
BASE=http://127.0.0.1:8000 node tools/dashboard_payload_check.js
```

Nothing in the tree may be unused. The sweep covers Python, the browser bundle,
the stylesheet, and `backend/tests/test_dead_code.py` fails
the build if it finds anything.

```bash
PYTHONPATH=. .venv/bin/python tools/dead_code.py --list
```

```bash
tail -f server.log
```

---

## 8. Layout

| Path | What it is |
|---|---|
| `backend/core` | config, world clock, **signal lock**, frozen snapshot, cycle manager, risk levels, store, **crowd emotions** (`emotions.py`) |
| `backend/data` | tick/L2/candle buffers, Binance WS, CoinGecko, simulator, market hub |
| `backend/formulas` | the 22 formulas in 8 categories + `engine.py` + the DRG reward learner |
| `backend/brain` | 5-step startup verification, connectome query, spectral clustering, 3-layer GCN, `explain.py`, fallback matrix |
| `backend/news` | sentiment lexicon, CryptoPanic/NewsAPI/RSS, critical-event detector |
| `backend/agents` | Gemini, local GGUF/ONNX, GitHub Models, fusion, orchestrator |
| `backend/web` | zero-build dashboard (`/`, `/matrix`, `/settings`) |
| `backend/tests` | `smoke.py` + `test_e2e.py` (Appendix E) |
| `docs` | [`SPECIFICATION_v2.md`](docs/SPECIFICATION_v2.md), [`SPEC_NOTES.md`](docs/SPEC_NOTES.md) |

---

## 9. The parts that make it v2.0

* **A 60-second countdown you can watch.** The window is 60 seconds, aligned to
  the UTC minute, and the countdown walks 60 → 1 one second at a time. It is
  rendered from the absolute instants the backend publishes
  (`window_started_at_ms`, `window_ends_at_ms`, `server_time_ms`, `cycle_id`),
  and it only re-anchors when the window id changes - a late message can never
  make the number jump, repeat or restart. `tools/clock_check.js` watches a real
  browser-DOM session cross a boundary and asserts exactly that.
* **Nothing inside the prediction cell moves (Round M).** The lock was
  always real server-side; what *looked* like "the prediction changing every
  few seconds" was text inside the cell that repainted every second — an
  "updated 7s ago" chip, a "43.0s left" line and the live tape line. Those are
  gone: the chip prints `🔒 locked HH:MM:SSZ` (it only ever flips to red
  STALE), the horizon line states the fixed release/target instants, the tape
  line shows the tape *as it was at the lock*, and the live tape moved to the
  Formula Explorer. The ring is the only thing that counts. Proof, in a real
  DOM against the live engine (`npm install --no-save jsdom ws` first):

  ```bash
  node tools/lock_watch.js 100
  ```

  It samples every element of the cell four times a second and exits non-zero
  if any of them changes between two boundaries.
* **Brain verification tells skipped from failed (Round M).** A missing
  optional token (`NEUPRINT_APPLICATION_CREDENTIALS`, `CAVE_TOKEN`) and an
  empty cache on the very first start are *skips* (⏭️), not ❌; both tokens
  have fields on `/settings` and saving one re-verifies the brain at once. The
  fallback matrix is cached too (marked as fallback), so `0-cache` passes from
  the second start on — and a token added later still wins, because step 0
  re-runs the live steps whenever credentials are present.
* **In sync, and locked (Round L).** The lock has two stages: the slow half
  (the AI agents, with their 7 s timeouts) is prepared 8 s before the boundary;
  the fast half — a fresh tape snapshot, the 22 formulas, the crowd reading and
  the fusion — is frozen again **0.4 s** before the boundary, so the prediction
  you see was built on data a fraction of a second old (`window.lock.
  data_age_at_open_seconds`). Both clients snap their clock to the arrival of
  the boundary SIGNAL, so the reveal and the countdown's zero are the same
  instant — a prediction can never appear early, and nothing inside the locked
  cell (side, confidence, levels, the crowd at lock) changes until the window
  ends. The crowd is cross-checked against the 22 formulas every 2 s
  (`formula_agreement`: aligned / conflict / crowd flat / formulas split), the
  formula consensus is evidence inside the crowd's Bayesian filter, and the rule
  is printed on every panel: *the 22 formulas carry 40 % of the direction vote;
  the crowd never votes — it can only cut confidence by at most 12.5 % (halved in Round P)*. Every
  emotion is shown as its formula with the live terms substituted.
* **Everything refreshes together.** The backend owns one schedule: a `SIGNAL`
  snapshot at the boundary and one `PULSE` at every grid mark inside the window
  (t+15, t+30, t+45), each carrying the formulas, the news feed, the agents, the
  brain read-out and the accuracy rates. The client has **one** `setInterval`
  (a safety net that does nothing while the socket is healthy) and one animation
  frame loop for the countdown. No panel polls on its own any more.
* **Fresh predictions, provably.** The prediction on screen is *the* signal for
  the window being counted down: it is computed *during the previous countdown*,
  published at the boundary, and every payload carries
  `age_seconds` / `stale` / `expires_at` plus the window it belongs to. The
  dashboard prints "updated 4s ago · max 65s" and turns the chip red if the
  engine ever misses a boundary; `/api/signal/current` calls `ensure_fresh()`
  first, which recomputes out of band rather than answering with an old call.
* **Reasoning, not just a number.** `prediction.reasoning` holds a one-line
  summary plus bullets: the brain read-out and its fusion weight, the formula
  consensus (how many of the directional formulas agree), the strongest
  supporters *and* the dissenters by name and value, the hedge state, the news
  headline, the level geometry and the measured hit rate.
* **1:1 take-profit / stop-loss.** One volatility-derived distance is clamped
  once and applied to both sides, so `tp_bps == sl_bps` and the ratio is exactly
  `1.00:1` (`RR_TARGET` changes it, the default is 1.0). Levels are re-stamped
  at the window boundary with the live price, so they track the market instead
  of the previous minute.
* **Microseconds, not seconds.** The engine times every formula and the whole
  pass in µs (`timings_us`), measures the tape between quotes (interval mean /
  p95, jitter, tick rate, quote lifetime, aggression), and publishes its own
  clock as epoch microseconds. The dashboard prints `812 µs`, `4.31 ms`,
  `1.204 s` — never a rounded "0.0 ms".
* **Every prediction covers the minute that follows its release.** The payload
  carries `horizon` with `released_at` / `target_at` (six-digit ISO *and* epoch
  µs), `microseconds_to_target`, `scored_in_seconds`, and the words "the next
  60 seconds" — so a side is never mistaken for "now". The dashboard shows that
  line under the prediction, plus a `▸ prediction detail` block naming the
  formulas that backed the side and the µs the pass cost.
* **A dashboard you can read at a glance.** Row 1: prediction + reasoning ·
  countdown · side & conviction. Row 2: take-profit & stop-loss (1:1) ·
  prediction accuracy (win rate, per-side hit rates, freshness, scoring
  horizon). An emergency override is a small inline chip under the prediction —
  never a full-screen overlay. The dashboard adds haptics (`navigator.vibrate`
  on a direction change, a longer pattern on an override, a tap on the first
  lock).
* **Six times the data.** `DATA_MULTIPLIER = 6` scales every buffer: 3 600
  ticks, 120 book levels per side, 360 candles, 120 headlines, 120 scored
  outcomes, 5 400 spread observations, a 360-window formula history, and 72
  windows per history/outcome call.
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

## 9b. The physics layer (kinetics of the tape, BTC vs PAXG)

A second, independent view of the pair, built from the physics of the order
flow rather than price patterns, voting beside the brain and the AI agents
(weight `PHYSICS_WEIGHT`, default 0.20; `0` disables). Every mechanism reads
the frozen tape of the window, so it is computable on any real feed; a
mechanism that cannot be estimated is **inactive** (listed with the reason,
excluded from the blend) instead of voting 0.00.

| § | mechanism | what it measures |
|---|---|---|
| 8.3 | Ornstein-Uhlenbeck bridge | reversion of the BTC/PAXG ratio to its 5-min VWAP; tradeable only if the half-life is 5-600 s and E[ΔP] clears the spread; edge capped at one 60 s σ |
| 8.1 | VPIN | flow toxicity; above 0.40 the dominant side is informed |
| 8.6 | Hawkes self-excitation | branching ratio n of trade arrivals; clustered flow carries its sign, n ≥ 0.95 is a cascade about to exhaust |
| 8.7 | Momentum flux (kinetic energy) | p = Σ side·v·\|Δln P\| over 60 s, z-scored against ten minutes |
| 8.8 | Trade-sign entropy | Shannon entropy of sign blocks; low entropy = an order being worked |
| 8.9 | Cross-leg diffusion | lagged BTC↔PAXG cross-correlation on a 5 s grid; the leader's last move predicts the follower |
| 8.4 | REI pendulum | BTC share of pair notional as a damped oscillator, integrated 60 s ahead |
| 8.10 | Tape temperature | realised variance of the last minute vs ten minutes; a hot tape drags every vote |
| 8.5 | Avellaneda-Stoikov | the spread cost the blend must beat |
| 8.2 / 10 | fragmentation · peg | only when 2+ venue quotes / a gold-spot quote are live |
| 11 | Kelly | Σ⁻¹μ over the active mechanisms, temperature drag, spread-cost gate |

The planetary sections of the original document (Landauer valve, solar flux,
E = mc², AMM surface) were retired: inside a 60-second window the global
hashrate is a constant, so they printed 0.00 and "balanced" every window.

Dashboard card "Physics layer" (every mechanism with its numbers and reason),
`GET /api/physics/current` (locked with the window), `/api/physics/live`,
`/api/physics/telemetry`, `/api/physics/spec`. Expected edges are model
estimates; no "cannot lose" claim is made anywhere.

## 9c. Hedge outcomes

The "Hedge Status" tiles were replaced by an outcome system. For the locked
side the engine prints the distribution it is trading on: σ of each leg over
60 s, ρ, the dollar hedge ratio β and the variance a hedge removes; the
**joint outcome matrix** P(BTC↑/↓ × PAXG↑/↓) from a bivariate normal; the
four pair actions consistent with the lock (outright, hedged at β,
pair-neutral, mirror leg) with expected bp, σ and P(profit), the best by μ/σ;
1σ/2σ scenarios; the spread z-score; and the regime read from HSI / HRDD /
GCDV / SHRP ("hedge working", "weakening", "breaking", "ratio drifting",
"paths diverging"). It is in `signal.hedge.outcomes` of `/api/signal/current`.

## 9c-bis. How a window is scored (and the edge guard)

A window is **FLAT** when the price did not move or the move stayed inside
the spread - it is counted and shown, never as a loss. The win rate is over
*decided* windows. The **edge guard** scores the engine's raw side on every
decided window; if that raw side has been significantly worse than a coin
flip (95 % Wilson upper bound < 50 % over the last 12–30 decided windows) the
engine publishes the opposite side, says "⇄ INVERTED" in the accuracy panel,
and releases only when the raw side is right again.

## 9d. The Formula Genesis Engine (v3.0)

A second formula layer that does not replace the 22 named formulas: it
**generates** them. Ten mathematical domains × ten sub-categories × a
21-variant grid (7 look-backs × 3 normalisations, plus each sub-category's own
axes) = **2,100 template formulas**, each with its definition and a reading
guide (`/api/genesis/domains`):

| D | domain | examples of what is computed |
| --- | --- | --- |
| 1 | rough-path signatures | level-2 signature, **Lévy area** ½(∫x dy − ∫y dx), log-signature, signature kernel |
| 2 | persistent homology | Vietoris–Rips **Betti-1 = E − V + C** (Euler formula), barcodes via MST, landscapes |
| 3 | information geometry | **Fisher–Rao distance** √2·arccosh(1 + (Δμ² + 2Δσ²)/4σ₁σ₂), natural gradient, α-divergences |
| 4 | optimal transport | **W₁ = Σ|F_p − F_q|Δx** between bid and ask depth, Sinkhorn, barycenters, trade-flow transport |
| 5 | fractional calculus | Hurst, **Malliavin finite perturbation**, Grünwald–Letnikov D^α, rough vol |
| 6 | tropical geometry | **max-plus polynomial** ⊕ᵢ(aᵢ ⊙ xⁱ), tropical roots, Newton polygon, max-plus eigenvalue |
| 7 | quantum probability | **3-state density matrix** ρ, **von Neumann entropy** −tr ρ ln ρ, interference, quantum walk |
| 8 | p-adic analysis | **p-adic valuation** v_p of tick moves, ultrametric trees, p-adic Haar, Volkenborn |
| 9 | category theory | **natural transformations** between timeframe functors, **sheaf consistency**, limits/colimits |
| 10 | algorithmic information | **Lempel–Ziv 76** complexity, block entropy, K(price|volume), logical depth |

**Life of a formula.** `BIRTH → CANDIDATE → ACTIVE → DECAYING → DEAD →
AUTOPSY`. Every 100 closed candles the whole pool is re-scored on the last
1,500 minutes with seven metrics (hit rate, Spearman IC, Sharpe, profit factor
after 1 bp per unit turnover, drawdown, worst-regime IC, block consistency),
an overfit penalty (in-sample − out-of-sample gap) and a dead-time penalty.
The top 200 by fitness, greedily pruned at |ρ| > 0.7, become ACTIVE. An
autopsy of each dead formula reports the regimes it lost in; one that was
profitable in some regimes is resurrected with a regime gate (two lives max).
Every 4 hours a **genesis** breeds 50 children by crossover / mutation of
expression trees over frame columns and the survivors' signals (no `eval`).

**Every minute.** The regime (trending / mean-reverting / high-vol / low-vol /
cascade, from Hurst, vol percentile and 3-minute σ-moves) gates the active
set to 80 / 70 / 60 / 50 / 40 formulas, preferring the domains the spec names
for that regime. Layer-4 formulas (homology, algorithmic) run every fifth
candle and hold in between. The fitness-weighted composite votes in fusion as
`genesis` (`GENESIS_WEIGHT`, default 0.25; `0` disables), is a ledger source
(`genesis:composite`) and is shown in the 🧬 card (regime, firing/gated/active,
per-domain means, the top contributors, the mathematics drawer).

**Compute, honestly.** Everything is NumPy on one background worker thread:
a Codespace has no GPU and no Rust toolchain, so heavy domains are vectorised
and strided instead of compiled. A full re-score of 2,100 formulas on 1,500
minutes takes about a minute and never blocks the lock. Minute history is
bootstrapped from Kraken (720 min) or Gemini (1,440 min) at start, so the
first scoring happens within seconds of a fresh Codespace instead of after
four hours of tape.

Endpoints: `/api/genesis/current`, `/live`, `/status`, `/formulas?state=&domain=`,
`/formulas/{id}`, `/domains`, `/graveyard`, `POST /rescore`, `POST /breed`.

## 10. Documentation

Every document below is also served **inside the app** at `/readme` (nav →
📖 Guide), rendered with copy buttons - you never need an editor's markdown
preview:

```
https://<your-codespace>-8000.app.github.dev/readme
```

The Drosophila circuit itself - every named neuron, its hemibrain cell type,
compartment, transmitter, cell count and every synaptic connection with its
synapse count and sign - is on `/matrix` ("Neurons & synapses") and at
`GET /api/brain/neurons`.

* [`docs/SPECIFICATION_v2.md`](docs/SPECIFICATION_v2.md) — the full specification
  (every formula, every threshold, the API, Appendices A–F).
* [`docs/API_KEYS_AND_MODELS.md`](docs/API_KEYS_AND_MODELS.md) — how to get every
  key (Gemini, GitHub Models, CryptoPanic, NewsAPI, neuPrint, FlyWire — all
  free), where to paste it, how to test it, and which local AI models to
  download from where and how to load them.
* [`docs/SPEC_NOTES.md`](docs/SPEC_NOTES.md) — every deliberate deviation from the
  v2.0 draft, with symptom → cause → decision → verification → escape hatch.

**Nothing here is financial advice.** The engine is a research instrument: it
emits a locked directional opinion every minute and scores itself afterwards.
