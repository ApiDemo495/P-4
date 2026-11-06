# PROMPT — Charts that move, and TP / SL that are computed, drawn and settled correctly

Paste everything below this line to the agent working on the multi-asset
prediction app (22 / 44 / 130 / 247-minute windows). It is written as direct
instructions to that agent. Two problems are being fixed at once because they
share causes: **(A) every chart is flat** and **(B) take-profit / stop-loss are
not executed properly on the chart**. Everything here is taken from a working
engine (`risk.py`, `risk_engine.py`, `renderBranches()` in `app.js`).

---

## PART A — Why charts are flat, in the order you should check

A flat line on a chart is never a style problem. It is one of these, checked
top to bottom:

### A-1 The y-axis range is the whole price, not the window's range
**Symptom:** price 4,120.37 drawn on an axis 0 … 5,000 → a 0.3 % move is one
pixel. Everything looks flat even though the data moves.
**Fix:** compute `lo`/`hi` from **the data on screen only** (path points,
fan quantiles, TP, SL, entry), then pad:
```js
let lo = Math.min(...values), hi = Math.max(...values);
const pad = (hi - lo) * 0.08 || entry * 0.0005;   // 8 % of range; 5 bps if range is 0
lo -= pad; hi += pad;
const Y = (p) => H - 18 - ((p - lo) / (hi - lo)) * (H - 30);
```
Never start the axis at 0; never share one axis across assets; never use a
fixed ±1 % band on a 22-minute chart of a 2 bps/minute asset.

### A-2 The path array is never appended (the usual real cause)
**Symptom:** the canvas shows one dot or a horizontal line at the entry.
**Causes we had:** (a) the WebSocket price message was consumed by the
ticker widget but never pushed to `state.path`; (b) the path was reset on
every render instead of every **window**; (c) `t` (seconds since window
start) was computed from `Date.now()` while the window start came from the
server clock → negative/NaN `t`, filtered out; (d) points were stored
with the same timestamp so `X(t)` collapsed to one column.
**Fix:**
```js
function pushPath(price, cycleNumber) {
  if (cycleNumber !== state.pathCycle) { state.pathCycle = cycleNumber; state.path = []; }
  const t = (serverNowMs() - state.clock.startedAtMs) / 1000;  // server clock on both sides
  state.path.push({ t, p: Number(price), at: serverNowMs() });
  if (state.path.length > 4000) state.path.shift();
}
```
Call `pushPath` from the **price** message handler, not from the signal
handler (signals arrive once per window; prices arrive every tick). Keep
`serverNowMs()` = `Date.now() + (serverTime − clientTime at last sync)`.

### A-3 The server sends one price per window
**Symptom:** the path has 1–2 points per window; the line is a straight
segment.
**Fix:** stream prices at ≥ 1 Hz per asset over the same WebSocket as
signals (`{type:"PRICE", asset, price, ts}`); for 130/247-minute windows
also send the 1-minute candles so the chart has a body before the live path
starts. Downsample for drawing (`≤ 2 points per pixel`) — never for storage.

### A-4 The source itself is flat
**Symptom:** the data really is one value: the feed is a REST poll of a
cached quote, the candle endpoint returns the same last bar, or the asset is
closed (gold / equities outside session) and the app draws the last price
forever.
**Fix:** show the **tape chip** (`rows in last 60 s`, `age_s`, `source`) on
the chart itself; if `age_s > 3·poll interval` draw the line in grey and
write `stale — <source> last print 04:12 ago`; if the market is closed,
say `closed — opens 09:15 IST` instead of drawing a flat line. Feed
liveness beats chart styling.

### A-5 Rounding / integer prices
**Symptom:** prices arrive as `Math.round(price)` or `toFixed(0)` and the
asset moves by less than 1 unit per minute.
**Fix:** send full floats; format only in labels. (Same rule as the
zeros prompt, cause 8.)

### A-6 Canvas is drawn once
**Symptom:** correct data, chart never updates.
**Fix:** render from a single `requestAnimationFrame` loop (or the one
`setInterval` the app already has for the countdown) that redraws every
chart from `state`; set `canvas.width/height` from `clientWidth ×
devicePixelRatio` on resize; `clearRect` before every draw. Never create a
new canvas or chart object per message.

### A-7 Everything above is right but the window is long
On a 247-minute window a 1 Hz path is 14,820 points; a 2 bps/minute asset
moves ~30 bps across the window — a healthy chart **is** nearly flat
relative to the fan. Fix the perception, not the data: draw the fan
quantiles, TP/SL rules and the entry line, so "flat" has a scale next to
it; show `range so far 31 bps · σ-expected 58 bps` in the chart note.

**Test for Part A:** feed a synthetic tape that moves +40 bps then −40 bps
inside one window; assert `state.path.length > 100`, `max(p) − min(p)` in
pixels ≥ 40 % of canvas height, and the chart note range reads `80 bps`.

---

## PART B — TP / SL: compute, draw, settle

### B-1 Compute them from measured moves, not from a percentage
Every window gets a `risk` block built once, at the lock, from the
**measured excursions of this asset at this horizon**:

```
inputs: entry (first print at/after the boundary), direction,
        σ_bps (realised vol scaled to the horizon),
        moves = quantiles of |window move| and of max adverse excursion (MAE)
                over the last ≥ 5 (prefer 60) completed windows of the same length
                (bootstrapped from 1-minute candles at start, so it exists at window 1)
sanity:  if σ > 2 · q80(|move|)/1.2816  →  σ = q80/1.2816   (one bad print cannot inflate σ)
stop:    sl_bps = q80(MAE)                      # 80 % of windows never go further against you
target:  tp_bps = clip(rr · sl_bps · f(edge), q50(|move|), q95(|move|))
         f(edge) = 0.8 + 0.4 · edge_in_[0,1]    # stronger edge reaches further
         rr target 1.5; never below 1.0
spread:  tp_bps += spread_bps/2 ; sl_bps += spread_bps/2
levels:  TP = entry · (1 + s·tp_bps/1e4),  SL = entry · (1 − s·sl_bps/1e4),  s = +1 buy / −1 sell
note:    "138 bps target / 92 bps stop = 1.50:1 — q80 MAE of 61 windows — full size (high conviction)"
```
- Per asset **and** per horizon: 22-minute gold and 247-minute NIFTY have
  different `moves` tables. Keep the tables in the vault; show `n` in the
  note; below `n = 5` print `levels provisional (n = 3)`.
- **Never** let TP/SL be the same number on every window: if two
  consecutive windows produce identical `tp_bps`, the `moves` table is not
  updating — fix the candle roll (we had this: candles only rolled from
  one feed).
- The block is **frozen at the lock** with the rest of the signal. Nothing
  recomputes TP/SL mid-window.

### B-2 Draw them in price space on the same axis as the path
The common bug: TP/SL drawn with a different scale (or in bps) than the
price path, or drawn before `entry` exists (so `entry = 0` → the rule is
off-canvas and nothing shows).
```js
const risk = state.signal?.risk || {};
const rule = (price, colour, label) => {
  if (!(price > 0)) return;                 // no entry yet → draw nothing, show "levels pending"
  const y = Y(price);                       // SAME Y as the path and the fan
  if (y < 0 || y > H) return;               // outside range → include it in lo/hi instead (see A-1)
  ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(44, y); ctx.lineTo(W - 12, y);
  ctx.strokeStyle = colour; ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = colour; ctx.font = "10px monospace"; ctx.fillText(label, 46, y - 3);
};
rule(risk.take_profit, "rgba(61,220,132,.8)", `TP ${fmt(risk.take_profit)} (+${risk.tp_bps} bps)`);
rule(risk.stop_loss,   "rgba(255,92,92,.8)",  `SL ${fmt(risk.stop_loss)} (−${risk.sl_bps} bps)`);
rule(risk.entry,       "rgba(255,255,255,.35)", `entry ${fmt(risk.entry)}`);
```
- Include `take_profit`, `stop_loss` and `entry` in the `lo/hi` computation
  of A-1 so they are always on screen.
- Draw the **expected path fan** (q5/q25/q50/q75/q95 of
  `entry·exp(drift·t ± z·σ·√t)`) under the price path; TP/SL rules on top
  of it; the live path on top of everything.
- Tint TP green / SL red for BUY and swap nothing — TP is always above
  entry for BUY, below for SELL; the formula in B-1 already does that.

### B-3 "Execute" them: touch detection, first-touch wins, same price source
The app does not trade, but it must **settle the levels honestly** on the
chart and in the ledger:

```
on every price print inside the window (server side, not the browser):
  if BUY:  hit_tp = price ≥ TP ; hit_sl = price ≤ SL
  if SELL: hit_tp = price ≤ TP ; hit_sl = price ≥ SL
  first touch wins; record {level, price, ts, seconds_into_window}
  if both touched on the same print (gap through) → record SL (conservative)
mark on chart: a dot + label "TP hit 14:02 (+138 bps)" at the touch point;
               after a hit, draw the rest of the path dimmed
window end:  outcome = direction win/loss by the settlement rule of the window
             AND level outcome ∈ {TP first, SL first, neither}  — two separate columns
```
- Touch detection uses the **same price stream** as the chart and the
  settlement (last trade, not mid, not a different exchange). We lost
  consistency once by settling on exchange A and drawing exchange B.
- Settle on the server, broadcast the event (`{type:"LEVEL_HIT", …}`); the
  browser only draws it. Reload must show the hit (persist it in the
  window's ledger row).
- Report `P(TP before SL)` at the lock (from the fan: first-passage
  probability of a drifted Brownian motion between two barriers) and track
  realised `TP-first %` against it per asset/horizon in the hourly review;
  if realised is far below predicted, the `moves` table or σ is wrong.

### B-4 Levels for long windows
For 130/247-minute windows draw TP/SL on the candle chart as well as the
window chart, with the window boundaries as vertical rules, so the user sees
the levels in the context of the day.

---

## Errors we hit — do not repeat

| id | error | fix |
|---|---|---|
| C-1 | y-axis from 0 → every chart flat | range from on-screen data + 8 % pad |
| C-2 | path reset on each render | reset only on window change |
| C-3 | browser clock vs server clock → points filtered out | `serverNowMs()` offset on both |
| C-4 | prices pushed from the signal message (1 per window) | push from the price stream |
| C-5 | `entry = 0` at lock → TP/SL = 0 → rules off-screen, "TP 0.00" | levels only when `entry > 0`; "levels pending" otherwise |
| C-6 | same TP/SL every window | candles rolled from one feed only; `moves` never updated |
| C-7 | σ inflated by one bad print → 500-point target on a 10-point minute | σ capped at `q80/1.2816`; TP ≤ q95 of measured moves |
| C-8 | TP/SL recomputed mid-window as price moved | frozen in the lock |
| C-9 | TP/SL drawn with their own scale | one `Y()` for path, fan, rules |
| C-10 | touch detected in the browser on a different feed | server-side on the settlement stream |
| C-11 | gap through both levels counted as TP | SL wins on the same print |
| C-12 | flat chart during market close looked like a bug | `closed — opens …` state |

## Acceptance tests

1. Synthetic ±40 bps tape: path > 100 points, drawn range ≥ 40 % of canvas,
   TP/SL/entry rules visible and inside the axis.
2. Entry absent at lock: no rules drawn, note reads `levels pending`; on the
   first print the block fills and the rules appear.
3. Two consecutive windows on a live tape produce different `tp_bps`
   (moves table updating).
4. σ test: inject one 300 bps print into a 5 bps/min tape; `tp_bps` ≤ q95 of
   the clean moves.
5. Touch: BUY with TP at +50 bps, tape crosses +50 at t = 310 s → `LEVEL_HIT
   TP 310 s`; a tape that gaps through both levels → SL.
6. Same stream: the price used for the hit equals the price drawn at that
   timestamp and the price used for settlement.
7. Reload mid-window: path (from server buffer), rules and any hit marker
   are restored.
8. Market closed asset: chart shows the closed state, not a flat line.
9. One `Y()`; lint: no second scale function in the chart module.
10. The hourly review prints predicted vs realised `TP-first %` per
    asset/horizon with `n`.

## Report back

Screenshots of one window per horizon with path, fan, entry/TP/SL rules and
a hit marker; the `moves` table (`n`, q50/q80/q95, MAE q80) per asset and
horizon; predicted vs realised TP-first % after 24 h; and the list of
places that still compute a price range from 0 or from a fixed percentage
(there must be none).
