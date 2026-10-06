# Getting every API key and every AI model — step by step

This is the companion to the README. It walks through **each key the app can
use**, where to get it, what it costs (all of them have a free tier or are
entirely free), exactly where to paste it, and how to confirm it works. The
second half covers the **AI side**: which cloud models the two key-based
agents use, and which **local models** you can download, from where, and how
to load them into the app.

> **Nothing here is mandatory.** The app runs with **zero keys**: market data
> comes from Binance/CoinGecko without a key, news comes from RSS without a
> key, the brain falls back to the committed 80×80 matrix, and the AI agents
> simply show as "not configured". Every key you add turns one more panel
> from "fallback" to "live".

---

## 0. Where keys go (read this first)

There are two places, and both are fine. Use whichever you prefer.

### A. In the dashboard (no files, no terminal)

1. Open the dashboard (the port-8000 URL) and click **⚙ Settings** in the
   top bar — or go straight to `/settings`.
2. Each key has its own field. Paste the key, click **Test** next to it —
   the line under the field turns green with what the key can do
   (e.g. `✅ gemini-2.0-flash reachable · 41 ms`) or red with the exact error.
3. Tick **persist to .env** if you want the key to survive an engine restart
   (it is written to the git-ignored `.env` file in the repo root — never
   committed).
4. Click **Save & Return**.

### B. In the `.env` file

The repo root has `.env.example`. The autostart already copied it to `.env`
for you (if not: copy it). Open `.env` in the editor and fill in the value
after the `=` — no quotes, no spaces:

```
GEMINI_API_KEY=AQ.Ab...your...key
```

Then restart the engine so it re-reads the file:

```bash
bash run.sh --stop
```

```bash
bash run.sh --bg
```

### How to know a key is picked up

```bash
curl -s localhost:8000/api/agents/status | python3 -m json.tool
```

```bash
curl -s localhost:8000/api/brain/status | python3 -m json.tool
```

```bash
curl -s localhost:8000/api/news/status | python3 -m json.tool
```

### Security rules

* Never paste a key into a GitHub issue, a commit, a screenshot or a chat.
* `.env` is in `.gitignore`. Keep it that way.
* If a key leaks, **revoke it** on the provider's page (each section below
  says where) and make a new one — do not just delete it from `.env`.
* Codespaces alternative for the security-minded: put keys in
  **GitHub → Settings → Codespaces → Secrets** with the exact variable names
  used below; they are injected as environment variables into every new
  Codespace and never touch the disk.

---

## 0. Three slots per key, three local models (Round O)

Every key box in Settings comes in threes: **primary** (always used) and two
optional **temporary** stand-ins. If the primary is rejected, rate limited or
errors, the engine switches to slot 2, then slot 3, *for that same call*, and
goes back to the primary the moment its cooldown ends (rate limit: 10 s
doubling to 5 min; rejected key: 10 min; other errors: 60 s). In `.env` the
slots are `GEMINI_API_KEY`, `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3` (same
pattern for `GITHUB_MODELS_TOKEN`, `CRYPTOPANIC_API_KEY`, `NEWSAPI_API_KEY`).
`GET /api/settings/keys` shows, masked, which slot is in use and why the
others are resting.

Local models also have three slots: upload into any of them
(`POST /api/agents/local/upload?slot=2`, or the three "Load into slot N"
buttons). Every loaded model answers every cycle in parallel and the answers
are merged into the single local opinion (confidence-weighted majority,
trimmed when they disagree). Slots are independent: unload one, the others
keep answering.

## 1. Gemini API key — `GEMINI_API_KEY` (free)

**What it powers:** the *Gemini agent*, one of the three AI agents whose vote
carries 25 % of the direction decision (formulas 40 %, agents 25 %, brain
20 %, news 15 %).

**Cost:** Google AI Studio gives a free tier with per-minute and per-day
request limits that are far above what this app uses (one call per agent per
60-second window ≈ 1 440 calls/day at most). No credit card required.

**Steps**

1. Go to **https://aistudio.google.com/** and sign in with any Google
   account (a plain Gmail works).
2. Accept the terms on first visit.
3. Click **Get API key** in the left sidebar (or open
   **https://aistudio.google.com/app/apikey** directly).
4. Click **Create API key**.
   * If it asks for a *Google Cloud project*: choose **Create API key in new
     project** — you do not have to open the Cloud console or enable
     billing.
5. A key appears - newer keys begin with `AQ.Ab…`, older ones with `AIza…`;
   both work, the engine never checks the prefix. Click the copy icon. **This
   is the only time it is shown in full**; if you lose it, make a new one.
6. Paste it into **Settings → Gemini API Key** and click **Test**. A green
   line names the model that will be used. (Or `GEMINI_API_KEY=` in `.env`.)

**Which model does it use?** `GEMINI_MODEL=auto` (the default) asks Google
which models *your* key can call and picks the **newest generation** it finds
— 3.8 over 3.7 over 3.5 over 3 over 2.5 — preferring `flash` (the engine needs
an answer inside 7 s) over `flash-lite` over `pro`, and GA over preview. New
generations need no code change: the version is read from the model id. A
retired id (the old default `gemini-1.5-flash` answers 404) can never break
the agent again; if a pinned id returns 404 mid-run the agent re-discovers and
retries once. To pin one anyway:

```
GEMINI_MODEL=gemini-3.8-flash
```

**Common errors**

| Test says | Meaning | Fix |
|---|---|---|
| `API key not valid` | typo / truncated paste | copy again from AI Studio |
| `models/... is not found` | that model name is retired in your region | change `GEMINI_MODEL` (above) |
| `429 RESOURCE_EXHAUSTED` | free-tier rate limit | wait a minute; the app already spaces calls out |
| `User location is not supported` | Gemini API not available in your country | use a VPN region that is, or rely on the GitHub agent |

**Revoke:** https://aistudio.google.com/app/apikey → bin icon next to the key.

---

## 2. GitHub Models token — `GITHUB_MODELS_TOKEN` (free)

**What it powers:** the *GitHub agent* (the second AI vote). It calls the
free **GitHub Models** inference endpoint with the model in
`GITHUB_MODELS_MODEL` (default `gpt-4o-mini`).

**Cost:** free for every GitHub account, with rate limits (per minute / per
day) that comfortably cover one call per window. No card.

**Steps (fine-grained token — recommended)**

1. Sign in at **https://github.com** and go to
   **https://github.com/settings/personal-access-tokens** (Settings → Developer
   settings → Personal access tokens → **Fine-grained tokens**).
2. Click **Generate new token**.
3. **Token name:** anything, e.g. `drosophila-models`.
4. **Expiration:** 90 days is fine (you will get an email before it expires;
   just make a new one and paste it again).
5. **Repository access:** *Public repositories (read-only)* is enough — this
   token is not used for git at all.
6. Scroll to **Permissions → Account permissions** and set
   **Models → Read-only**. This is the only permission the token needs.
7. Click **Generate token**, copy the `github_pat_…` string (shown once).
8. Paste into **Settings → GitHub Models Token → Test**, or
   `GITHUB_MODELS_TOKEN=` in `.env`.

**Steps (classic token — also works)**

1. **https://github.com/settings/tokens** → **Generate new token (classic)**.
2. Give it a name and an expiry; you can leave **all scopes unticked** —
   GitHub Models only needs a valid token that identifies you.
3. Generate, copy the `ghp_…` string, paste as above.

**Which model?** Default `gpt-4o-mini` (fast, small). Other free choices you
can put in `GITHUB_MODELS_MODEL`: `gpt-4o`, `Meta-Llama-3.1-8B-Instruct`,
`Mistral-small`, `Phi-3.5-mini-instruct`. The full list with the exact names
is at **https://github.com/marketplace/models**.

**Common errors**

| Test says | Meaning | Fix |
|---|---|---|
| `401 Unauthorized` | token typo, or expired | regenerate |
| `403 ... models` | fine-grained token without the *Models: read* permission | edit the token → Account permissions → Models → Read |
| `429` | rate limit | normal on the free tier during bursts; the agent times out gracefully and votes next window |

**Revoke:** same page as creation → **Delete**.

---

## 3. CryptoPanic API key — `CRYPTOPANIC_API_KEY` (free)

**What it powers:** the crypto-specific half of the *30-second news engine*
(sentiment, "critical event" detection, the news 15 % weight). Without it the
engine still runs on the free RSS feeds; with it you get faster, tagged,
vote-weighted headlines.

**Cost:** the **Developer / Free** plan is free and covers this app's poll
rate (one request every 30 s). No card.

**Steps**

1. Go to **https://cryptopanic.com/** → **Sign up** (email + password, or
   Google/Twitter).
2. Confirm the email.
3. Open **https://cryptopanic.com/developers/api/** (Developers → API).
4. Your **auth token** is printed on that page in the *Your API auth token*
   box (a 40-character hex string). If you do not see it, click **Generate**.
5. Choose the plan if prompted: **Free / Developer**.
6. Paste into **Settings → CryptoPanic API Key → Test**, or
   `CRYPTOPANIC_API_KEY=` in `.env`.

**Limits to know:** the free plan is *personal, non-commercial*, roughly
1 000–5 000 calls/day depending on their current terms — well above the 2 880
this app makes at the default 30-second poll. If you ever hit `429`, raise
`NEWS_POLL_SECONDS` in `.env` to `60`.

**Revoke / rotate:** the same developers page → **Regenerate**.

---

## 4. NewsAPI key — `NEWSAPI_API_KEY` (free for development)

**What it powers:** the general-market half of the news engine (gold, macro,
"bitcoin OR gold OR crypto" across mainstream outlets).

**Cost:** the **Developer** plan is free: 100 requests/day, headlines with a
24-hour delay, localhost/non-commercial use. Because of the 100/day cap the
app polls NewsAPI far less often than CryptoPanic (it spreads the daily
budget over the day automatically). No card.

**Steps**

1. Go to **https://newsapi.org/register**.
2. Fill in first name, email, password; tick *I am an individual*; agree to
   the terms; **Submit**.
3. The next page shows **Your API key** (32 hex characters). It is also
   emailed to you and always visible at **https://newsapi.org/account** after
   login.
4. Paste into **Settings → NewsAPI Key → Test**, or `NEWSAPI_API_KEY=` in
   `.env`.

**Common errors**

| Test says | Meaning | Fix |
|---|---|---|
| `apiKeyInvalid` | typo | copy from https://newsapi.org/account |
| `rateLimited` | 100/day used | wait until midnight UTC, or leave it — RSS + CryptoPanic keep going |
| `corsNotAllowed` | not relevant here (the *server* calls NewsAPI, not the browser) | — |

---

## 5. RSS feeds — no key (free)

Already on. `RSS_FEEDS` in `.env` is a comma-separated list; the default has
CoinDesk, Cointelegraph and The Block. Add any RSS/Atom URL, e.g. Reuters
business `https://feeds.reuters.com/reuters/businessNews`, or a gold-specific
feed. No signup anywhere.

---

## 6. Market data — Binance and CoinGecko — no key (free)

* **Binance** public REST + WebSocket (`api.binance.com`) needs **no key**
  for prices, order-book and trades, which is all this app reads. It is
  geo-blocked in some countries (US, and intermittently others) — the app
  detects that and switches to CoinGecko by itself.
* **CoinGecko** public API (`api.coingecko.com/api/v3`) needs **no key** at
  its free rate (roughly 10–30 calls/min). If you want higher limits, a free
  **Demo** key is available at **https://www.coingecko.com/en/developers/dashboard**
  (sign up → *Create Demo API key*); the app does not currently require it,
  so you can ignore this unless you see `429` from CoinGecko in
  `server.log`.
* `MARKET_DATA_MODE=auto` picks the best reachable source; set `binance`,
  `gemini`, `kraken` or `coingecko` to force one; `simulator` for offline demos.
* **Gemini exchange** public WebSocket (`api.gemini.com/v1/marketdata`,
  Round AL) needs **no key** - not to be confused with the Gemini *AI* key.

---

## 6b. Keyed providers for the Formula Genesis Engine (optional)

None of these is required; each feed runs **only while its key is saved**
(`/settings` → "Keyed providers", every box has a Test button). Without the
key the matching frame column is NaN and the formulas that read it do not fire.

| Provider | Variable | What it feeds | Where to get it |
| --- | --- | --- | --- |
| Glassnode | `GLASSNODE_API_KEY` | BTC exchange net-flow, hourly (`exch_flow`) | https://studio.glassnode.com/settings/api |
| Twelve Data | `TWELVEDATA_API_KEY` | dollar index 1-minute closes (`dxy`; free plans fall back to 100 / EUR-USD) | https://twelvedata.com/account/api-keys |
| LunarCrush | `LUNARCRUSH_API_KEY` | hourly social sentiment / galaxy score (`social`) | https://lunarcrush.com/developers/api |

Add to `.env` (or tick *persist* in Settings):

```
GLASSNODE_API_KEY=...
```

```
TWELVEDATA_API_KEY=...
```

```
LUNARCRUSH_API_KEY=...
```

`/api/genesis/status` → `providers` shows rows fetched and the last error per
provider; the 🧬 card's footer lists which keyed providers are live.

---

## 7. neuPrint token — `NEUPRINT_APPLICATION_CREDENTIALS` (free)

**What it powers:** the *live* Drosophila **hemibrain** connectome query
(brain verification steps 1–3). Without it the brain uses the committed
80×80 mushroom-body matrix — same maths, static wiring — and the verification
row shows *2-auth: skipped (optional)*, not a failure.

**Cost:** free. neuPrint is a public research service run by HHMI Janelia.

**Steps**

1. Go to **https://neuprint.janelia.org/**.
2. Click **LOGIN** (top right). It uses **Google sign-in** — any Google
   account works; there is no separate registration.
3. On first login accept the terms of use.
4. Click your **avatar / initials** (top right) → **Account**.
5. The page shows **Auth Token**: a long string starting with `eyJ…` (a
   JWT). Click **Copy**. It does not expire quickly (months), but you can
   return to this page any time to copy it again.
6. Paste it into **Settings → neuPrint token** and **Save**. The app
   re-verifies the brain immediately; the table should show
   `2-auth ✅ authenticated` and `3-query ✅ hemibrain mushroom body subgraph`,
   and the brain status becomes `LIVE_CONNECTED`. (Or
   `NEUPRINT_APPLICATION_CREDENTIALS=` in `.env` + restart.)

**Dataset:** `NEUPRINT_DATASET=hemibrain:v1.2.1` (default). You can also use
`manc:v1.0` or the newer `cns` datasets listed on the neuPrint front page;
the mushroom-body query works on hemibrain.

**Common errors**

| Step row says | Meaning | Fix |
|---|---|---|
| `1-connectivity ❌` | neuprint.janelia.org unreachable from the machine | network/firewall; nothing to do with the token |
| `2-auth ❌ 401` | token pasted incompletely (they are ~700 characters) | copy again with the Copy button, not by selecting text |
| `3-query ❌ timed out` | Janelia under load | the app falls back to the CSV for this start and retries on **Force Reconnect** |

---

## 8. FlyWire CAVE token — `CAVE_TOKEN` (free, research use)

**What it powers:** brain verification **step 4** — an alternative live
connectome (FlyWire FAFB) used if neuPrint is unreachable. Purely optional;
most users never need it because neuPrint or the CSV fallback already gives a
brain.

**Cost:** free. Requires accepting FlyWire's data-use terms.

**Steps**

1. Go to **https://flywire.ai/** → **Data Access / Sign up** and register
   with a Google account. Fill in the short form (name, affiliation — a
   personal project is acceptable; state it honestly) and accept the
   **FlyWire principles / terms of use**. Access to the production dataset
   is granted by the team, typically within a day or two; you get an email.
2. Once approved, open **https://global.daf-apis.com/auth/api/v1/create_token**
   while signed in with the same Google account.
3. The page shows a token (a 32-character hex string) as plain text or JSON.
   Copy it. (Creating a new token invalidates the previous one.)
4. Paste into **Settings → FlyWire CAVE token → Save**, or `CAVE_TOKEN=` in
   `.env`. `CAVE_SERVER` and `CAVE_DATASET` defaults are already correct.

If you are not approved yet, leave it empty: the row shows
*4-flywire: skipped (optional)*.

---

## 9. Quick reference — variable names

| Variable | Where from | Cost | Used for |
|---|---|---|---|
| `GEMINI_API_KEY` | aistudio.google.com/app/apikey | free | AI agent 1 |
| `GEMINI_MODEL` | (name, not a key) | — | default `auto` = the best model your key can call (discovered via ListModels) |
| `GITHUB_MODELS_TOKEN` | github.com/settings/personal-access-tokens | free | AI agent 2 |
| `GITHUB_MODELS_MODEL` | github.com/marketplace/models | — | default `gpt-4o-mini` |
| `CRYPTOPANIC_API_KEY` | cryptopanic.com/developers/api | free | crypto news |
| `NEWSAPI_API_KEY` | newsapi.org/account | free (100/day) | general news |
| `RSS_FEEDS` | any RSS URL | free | news, no key |
| `NEUPRINT_APPLICATION_CREDENTIALS` | neuprint.janelia.org → Account | free | live hemibrain |
| `CAVE_TOKEN` | global.daf-apis.com/auth/api/v1/create_token | free (approval) | FlyWire |
| Binance / Gemini / Kraken / CoinGecko | — | free | market data, no key |
| `GLASSNODE_API_KEY` | studio.glassnode.com → Settings → API | optional | genesis engine: exchange net-flow |
| `TWELVEDATA_API_KEY` | twelvedata.com → API keys | optional | genesis engine: dollar index |
| `LUNARCRUSH_API_KEY` | lunarcrush.com/developers | optional | genesis engine: social sentiment |
| `GENESIS_WEIGHT` | — | default `0.25` | fusion weight of the genesis engine (0 disables) |

---

## 10. The AI agents — cloud vs local

The app has **three agent slots**:

| Agent | Needs | Runs where | Latency |
|---|---|---|---|
| Gemini | `GEMINI_API_KEY` | Google's servers | ~0.5–2 s |
| GitHub Models | `GITHUB_MODELS_TOKEN` | GitHub/Azure servers | ~0.5–3 s |
| **Local model** | a `.gguf` (or `.onnx`) file you upload | **your machine / the Codespace CPU** | 2–20 s depending on size |

Each agent gets the same prompt (asset, last prices, formula summary, news
sentiment) and returns a direction and a confidence; the orchestrator gives
each agent 7 s and merges whatever answered. Any slot may be empty. If no
agent answers, the agent share of the vote is simply redistributed — nothing
breaks.

---

## 11. Local AI models — what to download, from where, how

### 11.1 Which file format

The uploader accepts two formats and checks them by **magic bytes**, not by
extension, so a renamed file is rejected with a clear message:

* **`.gguf`** — the llama.cpp format. **This is the one to use.** Every
  popular open model is published in GGUF on Hugging Face, in several
  *quantisations* (sizes). The file must start with the bytes `GGUF` and be
  at least 100 MB (smaller = truncated download).
* **`.onnx`** — accepted and inspected, but only useful if you have an ONNX
  text-generation model *and* `onnxruntime` installed. Skip unless you know
  you want it.

### 11.2 Which model — pick by RAM

The Codespace default machine has **2 cores / 8 GB RAM** (4-core/16 GB
available in the machine-type menu). Rule of thumb: the GGUF file must fit
in RAM with ~1.5 GB to spare, and the smaller the file the faster the answer
(the agent has 7 s). Recommendations, all free and open-weight:

| Model | Parameters | File to download | Size | RAM needed | Speed on 2 CPU cores | Verdict |
|---|---|---|---|---|---|---|
| **Qwen2.5-1.5B-Instruct** | 1.5 B | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | ~1.0 GB | 3 GB | ~2–4 s per answer | **best default for a Codespace** |
| **Llama-3.2-1B-Instruct** | 1 B | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | ~0.8 GB | 2.5 GB | ~2 s | fastest, a bit weaker reasoning |
| **Llama-3.2-3B-Instruct** | 3 B | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | ~2.0 GB | 4.5 GB | ~5–8 s | better answers; needs the 4-core machine to stay under 7 s |
| **Phi-3.5-mini-instruct** | 3.8 B | `Phi-3.5-mini-instruct-Q4_K_M.gguf` | ~2.4 GB | 5 GB | ~6–10 s | strong at structured output; 4-core machine |
| **Mistral-7B-Instruct v0.3** | 7 B | `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` | ~4.4 GB | 7 GB | 15 s+ | too slow for the 7-second budget on CPU; only on a real PC with 16 GB+ |
| TinyLlama-1.1B-Chat | 1.1 B | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | ~0.7 GB | 2 GB | ~1.5 s | fine for testing the pipeline, weak answers |

**Quantisation names** (the suffix): `Q4_K_M` is the sweet spot (4-bit,
small, little quality loss). `Q8_0` is twice the size and slightly better;
`Q2_K` is tiny and noticeably worse. `F16` is the unquantised file — do not
use it on CPU.

### 11.3 Where to download — Hugging Face

All of the above are on **https://huggingface.co**. Downloads of public
models need **no account**; an account (free) is needed only for a few
"gated" repos (Meta's official Llama repos ask you to accept a licence, but
the community GGUF conversions below are not gated).

Reliable GGUF publishers (search these names on Hugging Face):

* **`bartowski`** — GGUF conversions of nearly everything, well labelled.
* **`Qwen`** — official GGUF repos (`Qwen/Qwen2.5-1.5B-Instruct-GGUF`).
* **`lmstudio-community`**, **`QuantFactory`**, **`TheBloke`** (older models).

**Download in the browser (any machine)**

1. Open the repo page, e.g.
   **https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF**.
2. Click the **Files and versions** tab.
3. Find the `…q4_k_m.gguf` file and click the **↓ download** icon at the
   right of its row (not the file name — that opens a viewer).
4. Wait for the full size shown in the row; a shorter file will be rejected
   by the uploader.

**Download inside the Codespace terminal (faster — server-to-server)**

```bash
mkdir -p ~/models
```

```bash
curl -L --progress-bar -o ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Other direct links of the same shape (`/resolve/main/<file>`):

```bash
curl -L --progress-bar -o ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

```bash
curl -L --progress-bar -o ~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

```bash
curl -L --progress-bar -o ~/models/Phi-3.5-mini-instruct-Q4_K_M.gguf https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf
```

Check the file really is a GGUF before uploading (first four bytes must read
`GGUF`):

```bash
head -c 4 ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf; echo
```

Optional: the `huggingface-cli` tool resumes broken downloads:

```bash
pip install -U huggingface_hub
```

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir ~/models
```

### 11.4 Install the runtime once — `llama-cpp-python`

The engine loads GGUF files through **llama.cpp**'s Python bindings. It is
not installed by default because it compiles C++ (2–6 minutes). In the
Codespace terminal:

```bash
.venv/bin/pip install llama-cpp-python
```

If that fails with a compiler error, install the toolchain first, then
repeat:

```bash
sudo apt-get update && sudo apt-get install -y build-essential cmake
```

Pre-built CPU wheels (skip the compile) are available from the project's
index:

```bash
.venv/bin/pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

For **ONNX** models instead:

```bash
.venv/bin/pip install onnxruntime
```

Restart the engine after installing either:

```bash
bash run.sh --stop
```

```bash
bash run.sh --bg
```

### 11.5 Load the model into the app

**From the dashboard (recommended)**

1. **⚙ Settings → Local model**.
2. Click **Choose file**, pick the `.gguf` (from your computer, or from
   `~/models` if you downloaded inside the Codespace — the file picker in the
   browser-based editor can browse the Codespace disk).
3. Click **Validate** first: you get the magic-byte check, the size, the
   header metadata (architecture, context length, estimated parameter count)
   — green, or a red line saying precisely what is wrong.
4. Click **Change Model**. The upload streams to
   `backend/models/` (git-ignored), is validated again and loaded with
   `n_ctx=2048` on half your CPU cores. The status line shows
   `✅ <model name> · <parameters> · loaded in <seconds>`.
5. The **Agents** panel on the dashboard now shows a third chip, **LOCAL**,
   voting every window. The **Local model status** line reports the latency
   of the last answer — if it is consistently above ~6 s, pick a smaller
   quantisation or model (the 7-second budget is enforced; a late answer is
   dropped for that window, not blocked on).

**From the terminal (same endpoint)**

```bash
curl -s -F "file=@$HOME/models/qwen2.5-1.5b-instruct-q4_k_m.gguf" localhost:8000/api/agents/local/upload | python3 -m json.tool
```

```bash
curl -s localhost:8000/api/agents/local/status | python3 -m json.tool
```

Unload (frees the RAM; the other agents continue):

```bash
curl -s -X POST localhost:8000/api/agents/local/unload
```

### 11.6 Local model troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `Invalid file format. Please upload a valid .gguf or .onnx file.` | not a GGUF (renamed zip, HTML error page saved as .gguf) | re-download with the **↓** icon or `curl -L` (the `-L` follows the redirect; without it you save a 1 KB HTML page) |
| `File is only 3.2 MB. A usable GGUF model is at least 100 MB` | truncated download | download again; compare the size with the one on the repo page |
| `Install llama-cpp-python (GGUF) or onnxruntime (ONNX).` | runtime missing | §11.4 |
| upload dies at 100 % / `413` | reverse proxy limit on very large files (Codespaces forwards up to a few GB fine) | download inside the Codespace and upload via the `curl -F` command, which does not go through the browser |
| model loads but answers are `timeout` | too slow for 7 s | smaller model / `Q4_K_M` / 4-core machine |
| Codespace freezes while loading | RAM exhausted | pick a model that leaves ≥1.5 GB free; check with `free -h` |

### 11.7 A note on "downloading AI" for the two cloud agents

Gemini and GitHub Models run on their providers' servers — **there is nothing
to download** for them; the key *is* the setup. Only the local slot needs a
file. If you want a fully offline machine: skip both keys, load a local GGUF,
set `MARKET_DATA_MODE=simulator` and `NEWS_ENABLED=0`, and the whole app runs
without any network at all (with the CSV brain).

---

## 12. Suggested order for a first setup (15 minutes)

1. Gemini key (2 min) — biggest visible change: the Gemini agent chip turns
   green.
2. GitHub Models token (3 min) — second agent.
3. CryptoPanic key (3 min) — news panel fills in with tagged crypto headlines.
4. neuPrint token (2 min) — brain status goes `LIVE_CONNECTED`, the Brain
   Matrix page shows the live hemibrain wiring.
5. NewsAPI key (2 min) — general/gold news.
6. Local model (later, optional) — download Qwen2.5-1.5B `Q4_K_M`, install
   `llama-cpp-python`, upload.
7. CAVE token (only if you applied to FlyWire and were approved).

After each one, look at the **Agents** / **News** / **Brain** panels: every
key you add should move exactly one of them from "fallback" to "live", and
`/settings` should show a green test line for it.
