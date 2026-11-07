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
let lo = Math.min(...candleLowsOnScreen, entry), hi = Math.max(...candleHighsOnScreen, entry);
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
  if (y < 0 || y > H) { pinTag(price, label, colour, y < 0 ? "top" : "bottom"); return; }
  ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(44, y); ctx.lineTo(W - 12, y);
  ctx.strokeStyle = colour; ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = colour; ctx.font = "10px monospace"; ctx.fillText(label, 46, y - 3);
};
rule(risk.take_profit, "rgba(61,220,132,.8)", `TP ${fmt(risk.take_profit)} (+${risk.tp_bps} bps)`);
rule(risk.stop_loss,   "rgba(255,92,92,.8)",  `SL ${fmt(risk.stop_loss)} (−${risk.sl_bps} bps)`);
rule(risk.entry,       "rgba(255,255,255,.35)", `entry ${fmt(risk.entry)}`);
```
- **Do not** stretch the y-axis to fit TP/SL. The axis is scaled to the
  candles/path on screen (plus `entry`); a level outside that range is shown
  as a **pinned tag at the top/bottom edge** (`▲ TP 77,050 · +132 bps`), not
  by squashing the price into a flat line. (We did it the other way once —
  the price became a 3-pixel squiggle between two far lines. See Part C.)
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

## PART C — Make it a real trading chart (what the screenshots show is wrong)

What the current chart does, read from the screenshots: the y-axis runs
75,400 → 77,200 because it was stretched to the TP (77,050) and SL (75,544);
the price (76,046) is a 3-pixel squiggle in the middle; a huge translucent
fan dominates the canvas; the path starts at 13:15 with nothing before it;
volume is two giant blocks; the four mini cards are red noise lines. Nobody
can read a trade off that. A trading chart is **candles, auto-scaled to the
candles, with levels as thin horizontal lines carrying price tags on the
right axis**. Build exactly that.

### C-1 Use a real charting library — do not hand-draw candles on a canvas
Use **TradingView Lightweight Charts** (`lightweight-charts` on npm, Apache
2.0, ~45 kB, no account, no iframe). Vendor the single file
`lightweight-charts.standalone.production.js` into `static/vendor/` at
setup time (download from `https://registry.npmjs.org/lightweight-charts/-/lightweight-charts-<ver>.tgz`
and extract `dist/`), so the app works offline and never depends on a CDN.
It gives candlesticks, volume histogram, price lines with axis tags,
markers, crosshair, time axis, zoom/pan, autoscale — everything the
hand-drawn canvas lacks.

### C-2 One main chart per asset, this exact layout
```js
const chart = LightweightCharts.createChart(el, {
  layout: { background: { color: "#0b0f14" }, textColor: "#9aa4b2" },
  grid: { vertLines: { color: "#161c24" }, horzLines: { color: "#161c24" } },
  rightPriceScale: { borderColor: "#222", scaleMargins: { top: 0.08, bottom: 0.25 } },
  timeScale: { borderColor: "#222", timeVisible: true, secondsVisible: false, rightOffset: 12 },
  crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  // NOTE: the library has no timezone option — see D-3 for IST labels
});
const candles = chart.addCandlestickSeries({
  upColor: "#26a69a", downColor: "#ef5350", borderVisible: false,
  wickUpColor: "#26a69a", wickDownColor: "#ef5350",
  priceFormat: { type: "price", precision: 2, minMove: 0.01 },   // per asset: gold 2 dp, BTC 1 dp, NIFTY 2 dp
});
const volume = chart.addHistogramSeries({
  priceFormat: { type: "volume" }, priceScaleId: "",            // own overlay scale at the bottom
});
volume.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
```
- **Candle resolution per timeframe:** TF 22 m → 1-minute candles,
  44 m → 1-minute, 130 m → 5-minute, 247 m → 5-minute (switchable 1/5/15).
  Show **the last 3 windows of history plus the current window** so the
  current window has context (22-minute TF → the last ~90 minutes).
- **Candles come from the server**, built from the same trade stream the
  settlement uses, bootstrapped from the exchange's REST klines at start
  (Binance `/api/v3/klines`, Kraken OHLC, your equity feed's bars) so the
  chart is full from the first second, never "starts at 13:15".
- Update the live candle with `candles.update({time, open, high, low, close})`
  on every price tick (≥ 1 Hz); push a new bar when the minute rolls over.
  Never `setData` the whole array per tick.

### C-3 Levels: entry / TP / SL as price lines with axis tags
```js
const line = (price, color, title, style) => candles.createPriceLine({
  price, color, lineWidth: 1, lineStyle: style, axisLabelVisible: true, title,
});
state.lines = {
  entry: line(risk.entry, "#c9d1d9", "ENTRY", LightweightCharts.LineStyle.Dotted),
  tp:    line(risk.take_profit, "#26a69a", `TP +${risk.tp_bps} bps`, LightweightCharts.LineStyle.Dashed),
  sl:    line(risk.stop_loss,   "#ef5350", `SL −${risk.sl_bps} bps`, LightweightCharts.LineStyle.Dashed),
};
// at the next lock: candles.removePriceLine(old) for each, create new ones.
```
- Levels do **not** change the autoscale (`autoscaleInfoProvider` on the
  candle series returns the candles' own min/max). If TP is off-screen the
  library still shows its tag on the price axis edge — that is the correct
  behaviour, and it tells the user "the target is 132 bps away, above the
  screen", which is itself information.
- Shade the TP zone (entry→TP) faint green and the SL zone (entry→SL) faint
  red **only between the lock time and the window end**, using two
  area/baseline series or a canvas overlay aligned with
  `chart.timeScale().timeToCoordinate` / `candles.priceToCoordinate`. This
  is the standard "position tool" look.
- Markers: `candles.setMarkers([{ time: lockTime, position: "belowBar",
  shape: "arrowUp", color: "#26a69a", text: "LOCK BUY" }, { time: hitTime,
  position: "aboveBar", shape: "circle", color, text: "TP hit +138 bps" }])`.
  Vertical window boundaries: thin dotted vertical lines drawn on an overlay
  at every boundary time; the current window's end labelled with the
  countdown.

### C-4 The fan and the first-touch replay are secondary, not the chart
- The quantile fan belongs **inside the current window only**, drawn as a
  thin `q50` dashed line plus a faint q25–q75 band — never a q5–q95 cone
  painted over the whole canvas. Make it a toggle (`expected path: on/off`),
  default off on the main chart, on in the probability panel.
- The "NEITHER 96 % / STOP" first-touch bar and the replay belong in the
  probability panel, not over the chart.

### C-5 The four timeframe mini-cards
Each card gets a small Lightweight Chart (height 90 px) with the same candle
series and the three price lines, **no volume, no fan, no red noise line**;
`handleScroll/handleScale: false`; candle resolution per card as in C-2;
the current window shaded. The card's number line stays below.

### C-6 Volume
Volume is the histogram of **trade quantity per candle** from the same
stream (not two blocks): colour by candle direction; for assets where the
feed has no volume (some FX/CFD feeds), hide the pane and say `no volume on
this feed` instead of drawing fake bars.

### C-7 Axis, formatting, time
- Right price axis with the asset's tick size (`minMove`), left axis off.
- Time axis in the user's zone (IST), `HH:mm`, with the window boundaries
  labelled; `rightOffset` ≈ 10 bars so the current window's remainder is
  visible as empty space up to the window end.
- Last-price tag on the axis (built in), crosshair with OHLC tooltip
  (`subscribeCrosshairMove` → a legend line `O 76,040 H 76,112 L 76,020
  C 76,046 · vol 12.4`).
- Resize with `ResizeObserver` → `chart.applyOptions({ width, height })`.

### C-8 Data contract (server → browser)
```
GET  /api/chart/{asset}?tf=1m&limit=400       → [{time, open, high, low, close, volume}]
WS   {type:"CANDLE", asset, tf, bar}           every tick for the live bar
WS   {type:"LEVELS", asset, horizon, entry, take_profit, stop_loss, tp_bps, sl_bps, lock_time, window_end}
WS   {type:"LEVEL_HIT", asset, horizon, level, price, time}
GET  /api/chart/{asset}/markers?since=…        → locks, hits, boundaries (for reload)
```
The browser never computes candles from ticks on its own; it only applies
updates. A reload rebuilds the chart from the two GETs in < 1 s.

### C-9 Acceptance tests for Part C
1. Main chart shows ≥ 90 one-minute candles for the 22 m TF at first paint,
   oldest from REST bootstrap, newest live.
2. Autoscale range equals the on-screen candles' low/high (±margins) and
   does **not** change when TP/SL move; an off-screen TP shows its axis tag.
3. Entry/TP/SL price lines exist with tags; after a new lock the old lines
   are removed (count stays 3).
4. Live candle's `close` equals the last price in the tape within one tick.
5. Markers: lock marker at the boundary; hit marker at the hit time, same
   price as the server's `LEVEL_HIT`.
6. Volume histogram sum over a candle equals the trade quantity sum for
   that minute from the tape.
7. No fan on the main chart by default; toggle draws q50 + q25–q75 band
   only within the current window.
8. Mini cards: candles + 3 price lines, no red noise line, same scale rule.
9. Works with the vendored library file; no external CDN request is made.
10. Reload mid-window restores candles, lines and markers from the API.

---

## PART D — Candles below the live price (chart desync): diagnosis and fix

Observed after the Part C rebuild: the candle series runs 72,000 → 74,000 and
ends at 09:50, while the live price is 76,081 (tag at the top of the axis),
ENTRY 76,078 and SL 75,576 are drawn correctly, the ticker says BTC +12.15 %,
PAXG +53 %, ETH −36 %, and the system clock says 15:21 IST. Read together,
this is **not a drawing bug** — the candles are not the same data as the
price. Three facts prove it:

1. The candles are a smooth, evenly-sized ramp. Real 1-minute BTC candles
   are jagged. A smooth ramp is **synthesised** history (a bootstrap that
   generated bars when the REST kline fetch failed, or a simulator/replay
   feed used as the chart source).
2. The percentage changes (+12 %, +53 %, −36 %) are the live price compared
   with the **first synthetic bar**, which is why they are absurd.
3. The last candle ends at "09:50" = 15:20 IST shown in UTC — the bars were
   stamped backwards from *now*, so they are current in time but fake in
   price.

### D-1 The rule
> **The chart may only show bars that come from the exchange (REST klines)
> or that the app itself built from real trades. If neither exists, the
> chart is empty with a reason. No bar is ever synthesised, interpolated,
> replayed from a simulator, or extrapolated backwards.**

### D-2 Diagnose it in two minutes
```
GET /api/chart/BTC?tf=1m&limit=3        → look at the last bar's close and time
compare with GET /api/price/BTC          → last trade
```
- close ≠ price by > 0.2 %  → different sources (D-4).
- last bar time ≠ now ± 60 s → bootstrap stale, live bar not updated (D-5).
- bars perfectly regular (|close−open| nearly constant, volume uniform)
  → synthetic generator; find and delete it (D-4).
- percentage on the ticker uses `bars[0].open` → wrong reference (D-6).

### D-3 Fix the sources (server)
```
bootstrap(asset, tf):
    klines = exchange_rest_klines(symbol, tf, limit=400)      # Binance /api/v3/klines, Kraken OHLC, Gemini candles, equity provider bars
    if klines fails (network / 451 / symbol):  retry 3× with back-off, then
        bars = []  and  chart_status = "no history — <provider> <error>"   # NOT a generator
    store bars; verify: last_close within 0.5 % of the live price, else
        chart_status = "history source disagrees with live feed (<provider> vs <feed>)"; discard the bars
live:
    on every trade (price, qty, ts) from THE SAME stream that feeds the ticker and the settlement:
        bar = bars.last if same minute else new bar(open=price)
        bar.high = max, bar.low = min, bar.close = price, bar.volume += qty
        broadcast CANDLE
```
- One source of truth per asset: the trade stream that settles windows.
  REST klines are only for the **past**; from the first live trade onward,
  bars are built from trades. If the REST source is a different venue from
  the live stream (e.g. Kraken bars, Binance trades), say so on the chart
  and accept the small gap — never scale bars to "match".
- Delete every code path that fabricates bars: random walks,
  `np.linspace` ramps, "replay" tapes used as history, simulator output fed
  to the chart. The simulator may exist for self-tests but **must not be a
  chart or ticker source**; if a sim source is active, print `SIM` in red on
  the chart and in the ticker (the user asked for this honesty elsewhere).
- Time: store bar times as **UTC epoch seconds**. Lightweight Charts has no
  timezone option; to label in IST feed it `time + 19800` and keep the real
  UTC time in the bar object for logic. Do this in one adapter function so
  logic never sees shifted times.

### D-4 Reconcile continuously (browser + server)
```
every 5 s:  desync = |lastBar.close − lastPrice| / lastPrice
   if desync > 0.002 →  badge "CHART DESYNC 2.8 % — rebuilding"; re-fetch /api/chart; log it
   if lastBar.time < now − 2·tf →  badge "live bar stalled"; resubscribe
```
The ticker, the last-price axis tag, the live candle's close and the
settlement price must be the **same number**. Assert it in a test.

### D-5 The live bar in the browser
- `candles.update(bar)` on every `CANDLE` message; new minute → new bar
  (the library appends when `time` is greater than the last).
- On `LEVELS` after a lock, the entry line must sit **on** the current
  candle's close (entry is the first trade after the boundary). If the
  entry line is 2,000 points above the candles, the bars are wrong, not the
  line.

### D-6 Ticker percentage
`change % = (last − session_open) / session_open` where `session_open` is
the exchange's day-open (Binance 24 h ticker `/api/v3/ticker/24hr` →
`priceChangePercent`, or your equity provider's previous close), never the
first bar of whatever history the chart happens to hold.

### D-7 Tests for Part D
1. `last_bar.close == last_price` within one tick, for every asset, 1,000
   consecutive checks on a live tape.
2. With the REST kline endpoint disabled, the chart is empty with
   `no history — <provider> <error>`, the live bar still forms from trades,
   and no bar exists with time earlier than the first live trade.
3. Lint: no `linspace`, `random`, `replay` or simulator import in the chart
   / candle module.
4. Bars' `time` are UTC epoch seconds; the axis shows IST (`15:21`, not
   `09:51`) with the adapter offset applied only at render.
5. Ticker change % equals the exchange's 24 h figure within 0.05 %.
6. Injecting a 3 % wrong bar triggers the DESYNC badge and a re-fetch
   within 5 s.

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
| C-13 | y-axis stretched to TP/SL → price a 3-px squiggle | autoscale to candles only; off-screen levels as axis tags |
| C-14 | q5–q95 fan painted over the whole chart | q50 + q25–q75 inside the window, toggle, default off |
| C-15 | hand-drawn canvas path instead of candles | Lightweight Charts candlesticks from server candles |
| C-16 | chart begins at app start (no history) | REST kline bootstrap, last 3 windows + current |
| D-1 | candles synthesised when the kline fetch failed → bars 2.8 % below the live price, ticker +12 % / +53 % | no generator; empty chart with reason; reconcile badge |
| D-2 | time axis in UTC for an IST user | UTC storage, +19800 s adapter at render |

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
