# Offline (local) model — how it works and what to download

This is the exact behaviour of `backend/agents/local_model_agent.py` and
`backend/agents/local_pool.py` as of Round AS.

## 1. What the local model is asked to do

Every 60 s window, during the prefetch, the local model receives the same
context as Gemini and the GitHub agent (price, volatility, the 31 formulas,
the fly-brain CCSv2 reading, the crowd emotion, the latest headlines) and must
answer with **JSON only**:

```
{"decision":"BUY"|"SELL","confidence":0.0-1.0,"reasoning":"one sentence"}
```

HOLD is not accepted. Its vote enters the fusion with weight **0.20**
(`weight_local`). If the answer is not usable JSON the agent is marked as not
voting for that window; it never blocks the signal.

**Hard limit: 9 seconds per answer** (`local_timeout_seconds = 9.0`). The
model is loaded on CPU with `n_ctx = 2048`, `n_batch = 512`,
`n_threads = cpu_count / 2`, `temperature = 0`. The prompt is ~400–600
tokens and the answer ~40–80 tokens, so in 9 s the model must process ~600
prompt tokens **and** generate ~60 tokens on half your cores. This single
constraint decides which model you can use.

## 2. Accepted files

| format | how it is checked | runtime used |
|---|---|---|
| `.gguf` | magic bytes `GGUF` (0x47 0x47 0x55 0x46), header KV block parsed (architecture, name, file_type → bits/weight → parameter estimate); **minimum 100 MB** (truncated downloads are refused) | `llama-cpp-python` (`Llama.create_chat_completion`) |
| `.onnx` | protobuf header (first byte 0x08), then a load check with onnxruntime | `onnxruntime` CPU provider — only for models that expose a text `generate`; **use GGUF unless you know why you need ONNX** |

Anything else → "Invalid file format. Please upload a valid .gguf or .onnx
file." Upload ceiling 16 GB. Up to **3 model slots** can be loaded at once
(their votes are merged). Files live in `backend/models/` (git-ignored).

**Getting the file into the engine (four ways, best first):**

0. **Do nothing — `models.json`.** Every enabled entry in `models.json` (repo
   root) is (a) baked into the dev-container image by the GitHub Action, so a
   new Codespace starts with the weights already on disk under
   `/opt/drosophila/models` (linked into `backend/models/` by `run.sh`), and
   (b) if it is still missing at runtime, downloaded by the engine on start
   and loaded as soon as the runtime imports. The settings page shows each
   entry's phase. To change the model for all future Codespaces, edit
   `models.json` on the branch (or tell the assistant which model) — a push
   that touches it rebuilds the image. `MODEL_AUTOLOAD=0` disables.

1. **Download INTO the Codespace** (`/settings` → "Download a model INTO the
   Codespace"): paste the Hugging Face link of the `.gguf` (the file page URL,
   the `resolve` download link, or `hf://org/repo/file.gguf`) and press
   *Download & load*. The engine streams it to `backend/models/` with
   progress/ETA, resumes an interrupted download, validates and loads it.
   Use this for anything over ~200 MB. `POST /api/agents/local/fetch`.
2. **Drag the file into `backend/models/` in the VS Code explorer** (that
   channel is not limited by the port proxy), then press "→ slot N" next to
   it in the "on disk" list. `POST /api/agents/local/load-path`.
3. **Browser upload** (the file picker per slot). Works for small files
   only: the Codespaces port-forwarding proxy cuts off large/long uploads and
   the browser shows an HTTP 5xx (you may see 541/502/504) that never
   reached the engine.

Legacy upload path: `/settings` → "Local model" → choose file → slot 1–3 → Upload.
The engine stores → validates → loads → runs a test prompt → reports
`name / parameters / size / load time / inference ms` or a red error.
`POST /api/agents/local/upload?slot=N`, `GET /api/agents/local/status`,
`POST /api/agents/local/unload`.

## 3. RAM arithmetic

```
RAM needed ≈ GGUF file size
           + KV cache for n_ctx 2048 (≈ 0.2 GB for a 1B model, ≈ 0.5 GB for 3B, ≈ 1 GB for 7–8B)
           + the engine itself (≈ 1.5–2.5 GB with the FlyWire connectome and the 2100 Genesis formulas loaded)
           + 1 GB headroom for the OS
```

Quantisation is what sets the file size. For the same model:

| quant | bits/weight | 1B model | 3B model | 7–8B model | quality |
|---|---|---|---|---|---|
| Q8_0 | 8.5 | 1.1 GB | 3.4 GB | 8 GB | near-lossless |
| Q6_K | 6.6 | 0.9 GB | 2.6 GB | 6.2 GB | excellent |
| Q5_K_M | 5.7 | 0.8 GB | 2.3 GB | 5.4 GB | very good |
| **Q4_K_M** | 4.8 | **0.7 GB** | **2.0 GB** | **4.7 GB** | good — the default choice |
| Q3_K_M | 3.9 | 0.6 GB | 1.6 GB | 3.8 GB | noticeably worse at JSON discipline |
| Q2_K | 3.0 | 0.5 GB | 1.3 GB | 2.9 GB | avoid |

## 4. Which machine → which model

### GitHub Codespaces (where the app auto-runs)

The devcontainer asks for the smallest machine (2 cores / 8 GB; you can pick
bigger when creating the Codespace: 4-core/16 GB, 8-core/32 GB,
16-core/64 GB). **No GPU, ever.** CPU speed, not RAM, is the limit: with
`n_threads = cores/2`, prompt processing is roughly

| model size (Q4_K_M) | 2-core (1 thread) | 4-core (2 threads) | 8-core (4 threads) | fits the 9 s budget? |
|---|---|---|---|---|
| 0.5 B | ~2–3 s total | ~1.5 s | <1 s | yes everywhere |
| 1–1.7 B | ~6–9 s | ~3–4 s | ~2 s | 4-core and up |
| 3 B | ~15 s | ~8 s | ~4 s | 8-core only |
| 7–8 B | ~40 s | ~20 s | ~10 s | **no** (times out, vote lost) |

So on Codespaces: **0.5 B on the 2-core machine, 1–1.7 B on the 4-core,
3 B on the 8-core.** A 7B model will load fine and then time out every
window — you get nothing from it.

### Your own PC / laptop (if you run `bash run.sh` locally)

| your RAM | pick | notes |
|---|---|---|
| 8 GB | 1–1.7 B Q4_K_M | leave 4 GB for the engine + OS |
| 16 GB | 3 B Q4_K_M (CPU ≥ 6 cores) | 7B only with a GPU |
| 32 GB + any NVIDIA/Apple GPU | 7–8 B Q4_K_M | needs `llama-cpp-python` built with CUDA/Metal; the agent itself is unchanged |

## 5. Models worth researching (all have ready-made GGUF files on Hugging Face)

**Round BE — the five that ship (3–4B class).** `models.json` lists five
models for the five slots, all enabled, all voting at the same time:
Qwen3-4B-Instruct-2507 (the non-thinking Qwen3 release), Phi-4-mini-instruct
3.8B, Gemma-3-4B-it, Llama-3.2-3B-Instruct and Qwen2.5-3B-Instruct — Q4_K_M,
~12 GB in total, so **a 4-core / 16 GB Codespace** (on 8 GB set `"enabled":
false` on two entries). Each model is loaded with `(cores − 1) / 5` threads
(at least 1; `LOCAL_MODEL_THREADS` overrides) so five generations share the
CPU instead of fighting for it; a model whose *measured* vote does not fit the
9 s budget votes every fifth window — on 4 cores expect two or three of the
five to answer every window and the rest every fifth, on 8+ cores all five
every window. The measured tok/s, the token budget and who is skipping are
printed per slot in `/settings`. Gemma 3 and Qwen3 need llama-cpp-python
≥ 0.3.8 (the image has it); if a download link changes, edit the `url` in
`models.json` — the Settings page shows each slot's state.

Look for the official or **bartowski / Qwen / unsloth** GGUF repos and download
the `*-Q4_K_M.gguf` file. Instruction-tuned ("Instruct") variants only — base
models do not follow the JSON instruction.

| model | params | Q4_K_M size | why |
|---|---|---|---|
| **Qwen2.5-0.5B-Instruct** | 0.5 B | ~0.4 GB | the only one comfortably inside 9 s on a 2-core Codespace; surprisingly disciplined JSON |
| **Qwen2.5-1.5B-Instruct** | 1.5 B | ~1.0 GB | best quality/speed on a 4-core Codespace; strong at structured output |
| Llama-3.2-1B-Instruct | 1.2 B | ~0.8 GB | good JSON, slightly weaker numeric reasoning than Qwen |
| SmolLM2-1.7B-Instruct | 1.7 B | ~1.1 GB | small, fast, decent |
| Gemma-3-1B-it | 1.0 B | ~0.8 GB | fine; verbose unless told otherwise (our system prompt does) |
| Qwen2.5-3B-Instruct | 3 B | ~2.0 GB | 8-core Codespace or a laptop; clearly better reasoning |
| Llama-3.2-3B-Instruct | 3.2 B | ~2.0 GB | same tier |
| Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct | 7–8 B | ~4.7–4.9 GB | PC with GPU only |

Skip for this job: reasoning/"thinking" models (DeepSeek-R1 distills, QwQ,
Qwen3 in thinking mode) — they spend the whole 9 s thinking and never reach
the JSON; coder models; anything > 8 B on CPU; "abliterated"/role-play
finetunes (worse instruction following).

## 6. What a local model realistically adds

It is one of four voters at 20 % weight, behind the fly brain, the formulas
and Gemini. A 1 B model will not out-predict the formula layer; its value is
(a) a vote that keeps working when Gemini/GitHub are rate-limited or offline,
and (b) a second, uncorrelated reading of the headlines. Do not expect the
competition win rate to move much from the model alone — the boundary pick
and the formulas carry that.

## 7. The runtime (honest status)

`llama-cpp-python` and `onnxruntime` are **optional** in `requirements.txt`
(commented out). PyPI ships `llama-cpp-python` as source only, so installing
it compiles llama.cpp with CMake — about 5–10 minutes on a 2-core Codespace
and it needs `cmake` + `g++`. Until it is installed, an uploaded GGUF is
validated and stored but the load step reports
"Install llama-cpp-python (GGUF) or onnxruntime (ONNX)". The dev-container image
now carries a prebuilt wheel of it (Round AV), so in a Codespace it is simply
there. On other machines, since Round AT `run.sh`
installs it **automatically in the background** (prebuilt CPU wheel first,
source build only if no wheel matches) the first time the engine
starts on a machine that has `cmake` and `g++` (Codespaces do); progress is in
`logs/llama_runtime.log`, and the Load button tells you "being built in the
background" until it is ready. `LOCAL_RUNTIME_AUTOINSTALL=0` opts out. On a
machine without the compilers run, once:

```
.venv/bin/pip install llama-cpp-python
```


## 8. Making a new Codespace start in seconds (Prebuilds)

The image already holds the venv, the GGUF runtime, the connectome and the
`models.json` weights, so a new Codespace only pulls one image (~2.5 GB, about
a minute on GitHub's network). To remove even that: GitHub → repository
**Settings → Codespaces → Prebuild configurations → Set up prebuild**, choose
branch `arena/01a0c844-p-4`, your region, and "on configuration change". From
then on GitHub keeps a ready-made Codespace snapshot and "Create codespace"
hands you a running engine almost immediately. (Prebuild minutes are billed
like Codespaces storage; the free tier covers one small configuration.)

Alternatively keep **one** Codespace: the engine self-updates fast-forward
from the branch on every start (`/api/update/status`), so nothing is lost by
not recreating it.
