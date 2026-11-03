# PROMPT — Import the Drosophila connectome and let it vote "according to its will"

Paste everything below this line to the agent working on the multi-asset
prediction app (22 / 44 / 130 / 247-minute windows). It is written as direct
instructions to that agent. Everything here is what we built and run in a
working engine: the real FlyWire whole-brain connectome (138,639 neurons,
15.09 M connections) plus a mushroom-body sub-circuit, downloaded, cached,
driven by the market as if it were an odour, and read out through the fly's
own valence pathways — not a hand-written matrix pretending to be a brain.

---

You are building the **brain component**. It must (1) download and cache the
real connectome, (2) verify in five steps and fall back honestly, (3) convert
market state into glomerular input, (4) propagate through the real wiring,
(5) read the fly's own approach/avoid balance as the vote, (6) learn only
through the fly's own learning rule (dopamine reward-prediction error), and
(7) show its wiring in the UI. **"According to its will"** means: never
override the circuit's valence with your own sign, never replace it with a
random or hand-tuned matrix presented as the brain, and never fake its
output when it is not loaded.

## 1. Data: what to import and from where

| artefact | source | size | contents |
|---|---|---|---|
| connectivity | `https://github.com/philshiu/Drosophila_brain_model` → `Connectivity_783.parquet` (raw URL first, `api.github.com/repos/…/contents/…` with `Accept: application/vnd.github.raw` as fallback) | ≈ 95 MB, `EXPECTED_MIN_BYTES = 90 MB` | FlyWire v783 edges: `pre_root_id, post_root_id, syn_count, nt_type` (already thresholded at ≥ 5 synapses) |
| annotations | `https://github.com/flyconnectome/flywire_annotations` → `supplemental_files/Supplemental_file1_neuron_annotations.tsv` | ≈ 30 MB, min 25 MB | `root_id, cell_type, cell_class, super_class, side, hemibrain_type, top_nt` |

Rules:
- Download with resume, 60 s connect/600 s total timeouts, size check
  against `EXPECTED_MIN_BYTES`; a short file is a failed download, not a
  dataset.
- Build once, cache as `.npz` (`flywire_783_whole.npz` and
  `flywire_783_mushroom_body.npz`) with a `BUILD_VERSION` inside; bump it
  whenever the build changes; a mismatch rebuilds.
- Cache directory outside the repo (`~/.cache/<app>/connectome/`); never
  commit the parquet or the npz.
- The user may also supply a **neuPrint** token for `hemibrain:v1.2.1`.
  Test it with `GET /api/dbmeta/datasets` (the old `/api/databaseInfo` is
  gone); a 401 is "invalid key", a timeout is "unreachable" — different
  messages. **Never store the user's neuPrint key in the repo, `.env` or any
  file**; keep it in process memory only.

## 2. Build (do not OOM a 2-core / 8 GB Codespace)

Our first build loaded the whole parquet into pandas and peaked at 1.7 GB →
OOM-killed with no error on screen. The fix:

1. Read the parquet **one column at a time** with `pyarrow` (`columns=[…]`),
   converting each to a compact NumPy dtype immediately (`int64` ids,
   `uint16` synapse counts, `int8` sign) — peak ≈ 0.6 GB.
2. Map `root_id → index` with `np.unique(…, return_inverse=True)` on the
   concatenated pre/post ids.
3. Sign from `nt_type`: `ACh → +1`, `GABA → −1`, `Glu → −1`,
   `DA / OA / 5-HT → +1` (neuromodulators are treated as excitatory at the
   fast-pathway level; their gain modulation is applied separately, §5).
4. Normalise per post-synaptic neuron:
   `W[post, pre] = sign · syn_count / total_in[post]` (row-stochastic in
   magnitude) → store as CSR (`scipy.sparse.csr_matrix`,  float32).
5. Build the mushroom-body sub-circuit as the subgraph of neurons with
   `cell_class ∈ {ALPN, Kenyon_Cell, DAN, MBON, MBIN, LHLN, LHCENT}`
   (≈ 8,331 neurons, ≈ 836 k connections) — this is the fast path.
6. Run the build in a **worker process** (`multiprocessing`, `spawn`) so a
   crash never takes the API down; the API polls progress
   (`downloading 42 %`, `building column 3/4`, `normalising`, `cached`).

## 3. Startup verification — five steps, every outcome logged and shown

```
step 0  cache      npz present, BUILD_VERSION matches            → "cached (v3)"
step 1  reach      neuPrint host answers /api/dbmeta/datasets    → "neuPrint reachable" | "unreachable (timeout 8 s)"
step 2  auth       token accepted                                → "neuPrint auth ok" | "invalid key" | "no key"
step 3  query      one Cypher query returns a known MBON         → "neuPrint query ok"
step 4  flywire    parquet + annotations downloaded or cached    → "FlyWire v783 ready (138,639 neurons)"
step 5  fallback   80×80 GCN matrix loaded                       → "FALLBACK 80×80 — not the connectome"
```

- Steps 1–3 only run when a neuPrint key exists; they are **not** required
  for the brain to work (FlyWire is the primary source).
- Step 5 is a **declared** fallback: the UI brain chip reads
  `fallback 80×80` in amber, the vote's weight is halved, and the trace says
  so. Never present the fallback as the real connectome; never remove it
  either — it keeps the app alive offline.
- Expose `GET /api/brain/status` with each step's outcome, duration, error
  message and the neuron/edge counts; `POST /api/brain/reconnect` re-runs
  the steps without restarting the app.
- Never block the first prediction on the build: the engine starts on the
  fallback and hot-swaps to the connectome when the worker finishes.

## 4. Market → odour: the input encoding

The antennal lobe projection neurons (ALPNs, ≈ 685 in v783) are the fly's
smell input. We present each **formula/feature** as an odour channel with
**ON and OFF glomerular groups** (push–pull), so the brain sees sign.

- Sort ALPNs by `cell_type` (deterministic), split into `2 · N_INPUTS`
  contiguous groups. Our N_INPUTS = 28 (19 formula categories + 9 window
  inputs: realised drift, volatility ratio, imbalance, trade intensity,
  time-in-window, …). For your app: **one input vector per asset per
  horizon**; use the horizon's own formula outputs (signed, in [−1, 1]) as
  channels.
- For input `x_i ∈ [−1, 1]`: ON group rate `max(0, x_i)`, OFF group rate
  `max(0, −x_i)`. Flat market → all rates 0 → brain output exactly 0 by
  construction, which is the honest reading (tag it `gauge`-style; see the
  zeros prompt).
- Apply the push–pull mirror in the readout: `vote(x) = ½ (f(x) − f(−x))`
  so `f` is antisymmetric and a sign-flipped market gives a sign-flipped
  vote. Test this.

## 5. Propagation — the fly's own dynamics

```
h₀ = injected ALPN rates (§4)
for hop in 1..VALENCE_HOPS (= 3):
    h = ReLU(W @ h)                     # sparse CSR matvec, float32
    KC sparsity: keep the top 5 % Kenyon cells, zero the rest (APL feedback)
    gains: h[PAM]  *= 1 + 0.5·max(0, +drg)
           h[PPL1] *= 1 + 0.5·max(0, −drg)     # dopamine, §6
           h[OA]   *= 1 + 0.3·hsi              # octopamine ← HSI / stress index
```

- Three hops = ALPN → KC → MBON → downstream; more hops just blur.
- The whole-brain pass takes 0.1–0.4 s on 2 cores: run it on a **worker
  thread every 2 s** and publish the latest; the MB sub-circuit pass takes
  milliseconds and runs on every evaluation (it drives the two brain
  formulas, §7).
- Never run the pass inside the request handler or the lock path.

## 6. Readout — valence "according to its will"

MBONs are typed by neurotransmitter (Aso et al. 2014): cholinergic and
GABAergic MBONs drive **approach**, glutamatergic MBONs drive **avoidance**.

```
approach = Σ h[MBON, nt ∈ {ACh, GABA}]
avoid    = Σ h[MBON, nt = Glu]
balance  = tanh( 2 · (approach − avoid) / (|approach| + |avoid| + ε) )
vote     = ½ (balance(x) − balance(−x))          # push–pull mirror
```

- `vote > 0` = the circuit approaches the "up" odour = buy; `< 0` = sell.
- **Polarity calibration (do this once per build, never by hand):** probe
  each input channel alone with `x_i = +1`; if the circuit's `balance` is
  negative for a channel whose positive value means "up", record a flip
  `s_i = −1` for that channel. The flips are a property of the wiring (some
  glomeruli are innately aversive) and are shown in the UI as the "polarity
  map". Without this we had a vote that was ≈ 0 for weeks because channels
  cancelled.
- Add a descending-neuron share: fraction of activity reaching
  `super_class = descending_neuron` → confidence of the vote (`0.2–1.0`).
- Fuse: ours is 0.40 brain / 0.30 primary AI agent / 0.15 / 0.15 other
  agents, then the hourly hindsight multiplier `clip(1 + 0.25·z, 0.25, 1.5)`
  on the brain's own realised accuracy per horizon. Start yours at 0.25 and
  let hindsight move it; **never hand-edit the sign of the vote.**

## 7. Learning — dopamine RPE only

The fly learns through dopamine neurons (PAM = reward, PPL1 = punishment)
modulating KC→MBON synapses. Implement exactly that and nothing else:

```
expectation E = EMA of recent brain confidence-weighted outcomes
outcome    R  = +1 if the window closed in the vote's direction else −1
RPE        δ  = R − E
drg (dopamine release gain) = tanh(λ · δ), λ = 2.25
learning rate α ∈ [0.10, 0.50] scaled by |δ|
tonic level 0.85, phasic decay 0.6 per window
update: KC→MBON weights of the *active* KCs: w += α · drg · pre · post (sign per MBON class)
appetite (risk appetite) only *reduces* effective size, never flips direction
```

- Updates apply only at window boundaries, persist to the vault, and are
  rolled back if the hourly hindsight review finds the brain worse than
  before the change.
- Never retrain `W` globally, never use gradient descent on the vote — that
  would make it a neural net wearing a fly costume.

## 8. The two brain formulas (fast MB sub-circuit path)

- **CCSv2** (circuit consensus): `tanh(LH_approach − LH_avoid)` over the
  lateral-horn + MBON readout of the MB pass on the current input vector.
- **KCAE** (Kenyon-cell assembly entropy): entropy of the active KC set
  across the last 10 evaluations; low entropy = stable percept = higher
  confidence; feeds confidence, not direction.

Both go through the same tracing / `zero_reasons` / flat-watch machinery as
every other formula.

## 9. UI

- Brain chip: `FlyWire v783 · 138,639 n · pass 0.21 s · 12 s ago` (green),
  `fallback 80×80` (amber), `building 63 %` (blue), `failed: <reason>` (red).
- Brain page: block matrix (`cell_class × cell_class` summed |W|, log
  colour), live activation trace per hop (ALPN → KC → MBON → DN bars),
  approach vs avoid bars, polarity map, dopamine panel (E, δ, drg, α, tonic),
  five-step verification table, reconnect button.
- Every prediction card shows `brain +0.31 (conf 0.62)` with a tooltip of
  the top three contributing input channels.

## 10. Errors we hit — do not repeat

| id | error | fix |
|---|---|---|
| B-1 | pandas full-parquet load → 1.7 GB → silent OOM | pyarrow per-column, compact dtypes, worker process, progress |
| B-2 | brain "connected" with vote ≈ 0 for weeks | polarity calibration + push-pull mirror + flat watch on the vote |
| B-3 | 80×80 fallback shown as the brain | declared fallback chip, halved weight, trace text |
| B-4 | `/api/databaseInfo` removed by neuPrint → key test always failed | `/api/dbmeta/datasets`; distinguish 401 / timeout |
| B-5 | neuPrint key written to `.env` | memory only; lint test greps the repo for token patterns |
| B-6 | whole-brain pass on the request thread → UI stalls | worker thread every 2 s; MB path for per-tick |
| B-7 | raw.githubusercontent blocked in some networks | `api.github.com` contents fallback with raw Accept header |
| B-8 | cached npz from an old build loaded after a logic change | `BUILD_VERSION` in the npz |
| B-9 | KC layer dense → every input looked the same | top-5 % sparsity (APL) |
| B-10 | vote flipped by a hand-set sign "to improve accuracy" | forbidden; only RPE updates KC→MBON, only hindsight re-weights |
| B-11 | first prediction blocked by the download | start on fallback, hot-swap |

## 11. Acceptance tests

1. Build from a 200-row synthetic parquet: `W` is CSR float32, rows
   normalised, signs follow `nt_type`; `BUILD_VERSION` is in the npz.
2. Verification reports all five steps with durations; with no network it
   ends on step 5 and the chip reads `fallback 80×80`.
3. Input groups: `2 · N_INPUTS` contiguous ALPN groups, deterministic across
   runs.
4. Flat input → vote exactly 0 with reason `flat input`; `vote(−x) == −vote(x)`
   to 1e-6.
5. Polarity probe: each channel alone at +1 yields a non-zero balance; flips
   recorded; after flips every channel's `+1` probe is positive.
6. KC sparsity: active KC share after hop 1 is ≤ 5 % + 1 neuron.
7. RPE: a run of wins raises `E`, a loss produces negative `drg`, weights
   change only at boundaries, rollback restores them.
8. Pass timing: MB path < 20 ms, whole brain < 1 s on 2 cores; neither runs
   on the request thread (assert thread name in trace).
9. No neuPrint token string appears in any repo file or log line.
10. Brain's vote in the fusion carries its horizon-specific hindsight
    multiplier and the fallback halving.

## 12. Report back

Neuron/edge counts per build, build peak RSS, five-step table with
durations, polarity map (which channels flipped), 24 h of brain-only
accuracy per horizon with `n`, and the list of any place where the brain's
sign is modified outside the RPE rule (there must be none).
