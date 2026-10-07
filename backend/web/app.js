/* ==========================================================================
   Drosophila Trader v2.0 - dashboard client.

   The client NEVER derives a signal of its own.  It renders exactly what the
   backend locked, and it structurally cannot show a different signal mid-cycle
   because the only SIGNAL message it can receive is the locked one.
   ========================================================================== */

const $ = (id) => document.getElementById(id);

const state = {
  asset: "BTC",
  pendingAsset: null,
  lockState: "COMPUTING",
  cycleNumber: 0,
  cyclePeriod: 60,
  secondsIntoMinute: 0,
  lastCycleAt: null,
  signal: null,
  formulas: {},
  liveFormulas: {},
  liveReadings: {},
  liveTraces: {},
  liveStats: {},        // per-formula history statistics (Round I)
  liveMicro: {},        // the tape measured in microseconds (Round I)
  liveTimingsUs: {},    // per-formula cost in µs from the last live pass
  liveChecks: {},       // per-formula double check (replay · re-derived · range)
  liveHistoryWindow: 0,
  emotions: null,        // the crowd's live emotion reading (Round J, EMOTION stream)
  emotionsLocked: null,  // the reading taken on the frozen snapshot at lock time
  emotionDampening: null,
  lastEmotionDominant: null,
  emotionsDeep: null,        // the last deep block received (streamed every 4th sample)
  emotionsDeepDirty: false,  // repaint the deep block only when a new one arrived
  emotionFormulaFor: null,   // which emotion's formula is printed (default: dominant)
  formulaTraces: {},
  formulaReadings: {},
  formulaMeta: [],
  selfTest: null,
  openLogic: {},
  convictionNote: null,
  prediction: null,
  predictionAge: null,
  emergency: null,
  emergencyUntil: 0,
  window: null,
  /* ---------------------------------------------------------------- clock --
     ONE clock for the whole page.  The backend publishes the window as two
     absolute instants (start and end, epoch ms) plus its own time; we measure
     the offset between its clock and ours once, then count down to that fixed
     instant.  Within a window the deadline NEVER moves: an unrelated message
     arriving late cannot make the number jump, repeat or restart - which is
     exactly what "too glitchy" was.  Only a new cycle id re-anchors. */
  clock: null,            // {period, windowId, startedAtMs, endsAtMs, offsetMs}
  serverOffsetMs: null,
  clockSamples: 0,
  lastSecondShown: null,
  tickParts: null,
  outcomes: [],
  wiring: null,
  explain: null,
  wiringOpen: false,
  lastPrediction: null,
  config: null,
  ready: false,
  warmingSince: null,
  startError: null,
  ws: null,
  connected: false,
  explain: false,
  collapsed: {},
};

/* ------------------------------------------------------------------ utils */
const fmtPct = (v) => (v === null || v === undefined || Number.isNaN(Number(v))) ? "—" : `${(Number(v) * 100).toFixed(0)}%`;
const fmtSigned = (v, d = 3) => (v >= 0 ? "+" : "") + Number(v).toFixed(d);

function pctClass(v) {
  if (v > 0.15) return "pos";
  if (v < -0.15) return "neg";
  return "neutral";
}

/* Haptics: navigator.vibrate where the browser supports it (Android Chrome). */
function haptic(pattern = 18) {
  try {
    if (navigator.vibrate) navigator.vibrate(pattern);
  } catch (e) { /* unsupported - never break the render for a buzz */ }
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

const fmtMoney = (v) => (v || v === 0)
  ? `$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 })}`
  : "—";

async function getJSON(url, options) {
  try {
    const res = await fetch(url, { cache: "no-store", ...(options || {}) });
    if (!res.ok) {
      // Surface the backend's error message instead of swallowing it: the key
      // modal needs to tell "wrong key" apart from "server unreachable".
      try {
        const payload = await res.json();
        return { error: payload.detail || `HTTP ${res.status}`, valid: false };
      } catch (e) {
        return { error: `HTTP ${res.status}`, valid: false };
      }
    }
    return await res.json();
  } catch (e) {
    return { error: String(e && e.message ? e.message : e), valid: false, offline: true };
  }
}

/* --------------------------------------------------------------- websocket */
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/signals`);
  state.ws = ws;

  ws.onopen = () => {
    state.connected = true;
    $("ws-dot").className = "dot on";
    $("ws-label").textContent = "live";
  };
  ws.onclose = () => {
    state.connected = false;
    $("ws-dot").className = "dot off";
    $("ws-label").textContent = "reconnecting…";
    setTimeout(connect, 2000);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (event) => {
    state.lastMessageAt = Date.now();
    let msg;
    try { msg = JSON.parse(event.data); } catch (e) { return; }
    handle(msg);
  };
}

function send(payload) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(payload));
  }
}

function handle(msg) {
  switch (msg.type) {
    case "HELLO":
    case "SIGNAL": {
      // Both carry the complete snapshot.  SIGNAL means "a new window just
      // opened": the countdown re-anchors to the new boundary and every panel
      // redraws in the same pass.
      const d = msg.data;
      const boundary = msg.type === "SIGNAL";
      applySnapshot(d, { rttMs: state.lastRttMs, force: boundary, boundary });
      state.cycleNumber = d.cycle_number ?? d.status?.cycle?.cycle_number ?? state.cycleNumber;
      // The snapshot normally carries history and outcomes itself; this is the
      // belt-and-braces fetch when the payload was the cold-start sentinel.
      if (boundary && !d.history) refreshHistory();
      break;
    }
    case "PULSE": {
      // One message, every panel: this is the "in parallel" contract.
      recordPath(msg.data.live_price, msg.data.cycle_number);
      applySnapshot(msg.data, { rttMs: state.lastRttMs });
      pulseBreath();
      break;
    }
    case "CYCLE_START": {
      // Kept for the non-pipelined mode (SIGNAL_PIPELINE=0), where the panel
      // really does go blank for the first seconds of a window.
      state.cycleNumber = msg.data.cycle_number;
      state.asset = msg.data.asset || state.asset;
      state.lockState = "COMPUTING";
      state.lastCycleAt = Date.now();
      renderSignal();
      $("degradation").textContent = degradationLabel(msg.data.degradation_level);
      break;
    }
    case "NEXT_WINDOW_READY": {
      // The next window's signal is computed and held.  This changes a pipeline
      // flag, not any value the user reads, so it is picked up by the frame
      // loop on its next second rather than repainting panels mid-window.
      state.nextReady = msg.data;
      if (state.window) state.window.prefetch_ready = true;
      break;
    }
    case "FORMULA_UPDATE": {
      // Legacy shape (older backends / SIGNAL_PIPELINE=0).  Same one-pass path.
      applySnapshot({ live_formulas: msg.data }, { rttMs: state.lastRttMs });
      break;
    }
    case "EMERGENCY_OVERRIDE": {
      // Safety-critical, so it is not deferred to the next tick: the flip is
      // applied the instant it arrives, in the same single render pass.
      state.emergency = msg.data;
      state.emergencyUntil = Date.now() + (msg.data.remaining_seconds || 180) * 1000;
      if (msg.data.signal) {
        state.signal = msg.data.signal;
        state.lockState = "EMERGENCY_OVERRIDE";
        state.convictionNote = msg.data.signal.conviction_note || null;
      }
      if (msg.data.clock) applyClock(msg.data, { rttMs: state.lastRttMs });
      renderEmergency();
      renderSignal();
      break;
    }
    case "OUTCOME":
      refreshOutcomes();
      break;
    case "EMOTION": {
      // The crowd's mood, streamed twice a second on the backend's schedule.
      // Only the emotion panel repaints: nothing else on the page is touched,
      // so the countdown and the locked panels stay exactly where they are.
      recordPath(msg.data.live_price, msg.data.cycle_number);
      adoptEmotions(msg.data);
      renderEmotions();
      renderBranches();
      synesthesia.update(msg.data.emotions);
      break;
    }
    case "ASSET_SWITCH": {
      state.pendingAsset = msg.data.pending;
      renderAssetToggle();
      break;
    }
  }
}

/* ------------------------------------------------------------------ clock */
/* ONE clock, ONE tick.

   The backend owns the schedule: a 60-second window aligned to the UTC minute,
   with refresh marks at fixed offsets inside it (t+15, t+30, t+45).  It sends
   the window as absolute instants and the client counts down to a fixed point
   in time - so the digits are smooth, they never jump when an unrelated message
   arrives, and every panel refreshes on the same tick instead of each one
   polling on a private timer ("not in parallel with other features"). */

function serverNowMs() {
  return Date.now() + (state.serverOffsetMs || 0);
}

/* Adopt the server clock from any payload that carries it.
   Within a window the deadline is frozen; only a new cycle id re-anchors.  The
   offset estimate is nudged (never snapped) so a slow network cannot shift the
   deadline under the running countdown. */
function applyClock(payload, opts = {}) {
  const clock = payload?.clock || payload?.window?.clock || null;
  if (!clock || typeof clock.window_ends_at_ms !== "number") {
    if (payload?.window) state.window = payload.window;
    return false;
  }

  // One-way delay estimate: half the round trip of the request that carried it.
  const rttMs = typeof opts.rttMs === "number" ? opts.rttMs : 0;
  const sample = (clock.server_time_ms + rttMs / 2) - Date.now();
  const arrival = clock.server_time_ms - Date.now();
  if (state.serverOffsetMs === null) {
    state.serverOffsetMs = sample;
    state.clockSamples = 1;
  } else if (opts.boundary) {
    // A boundary SIGNAL is sent at the exact instant the window opens, so its
    // arrival is the one clock sample worth trusting outright.  Snapping here
    // (and only here) is what keeps the reveal and the zero of the countdown
    // on the same instant: had the first sample been wrong (a proxied
    // Codespace socket has a long, asymmetric first round trip), the prediction
    // used to land while the digits still showed 2 or 3 - "coming early".
    // Anchoring to the arrival (not arrival minus half an RTT) means the client
    // runs *behind* the server by the one-way latency, so the countdown can only
    // ever reach zero when the next SIGNAL is already here - never before.
    state.serverOffsetMs = arrival;
    state.clockSamples += 1;
    state.lastBoundarySnapMs = Date.now();
  } else {
    // Trust the new sample slowly: at most 120 ms per message, so the numbers
    // on screen never lurch, and never enough to move this window's deadline.
    const step = clamp(sample - state.serverOffsetMs, -120, 120);
    state.serverOffsetMs += step;
    state.clockSamples += 1;
  }

  const sameWindow = state.clock && state.clock.windowId === clock.cycle_id;
  const period = Number(clock.window_seconds || clock.period_seconds || state.cyclePeriod);
  state.cyclePeriod = period > 0 ? period : state.cyclePeriod;
  if (payload?.window) state.window = payload.window;

  if (!sameWindow || opts.force) {
    // New window: re-anchor to the instants the server published.
    state.clock = {
      windowId: clock.cycle_id,
      period: state.cyclePeriod,
      startedAtMs: clock.window_started_at_ms,
      endsAtMs: clock.window_ends_at_ms,
      minuteAligned: !!clock.minute_aligned,
      ticks: clock.ticks || [],
      freshnessMaxAge: clock.freshness_max_age_seconds,
      windowEndsAt: clock.next_boundary_at,
    };
    state.lastSecondShown = null;   // force one redraw of the digits
    state.newWindow = true;
  } else {
    // Same window: keep the deadline, refresh only the descriptive fields.
    state.clock.ticks = clock.ticks || state.clock.ticks;
    state.clock.minuteAligned = !!clock.minute_aligned;
  }
  return !sameWindow;
}

/* Round AB - every panel wears the same two factors: &b (where its data
   comes from right now) and ¶gn (where in the window we are), plus the age
   of its last repaint.  Renderers call markPanel(name); the frame loop
   repaints the rails once a second, so all of them move together. */
const PANEL_SEEN = {};
function markPanel(name) { PANEL_SEEN[name] = Date.now(); }
function panelFeedClass() {
  // Round AF: the engine's own grade of the last pass wins (live / partial /
  // simulated / offline); the tape source is the fallback before any pass.
  const grade = state.liveProvenance?.grade;
  if (grade) return grade;
  const src = state.tape?.source;
  if (!src || src === "none") return "offline";
  return src === "simulator" ? "simulated" : "live";
}
function renderPanelRails() {
  const feed = panelFeedClass();
  const src = state.tape?.source || "none";
  const phase = state.clock ? (() => {
    const p = 1 - windowRemaining() / (state.cyclePeriod || 60);
    return p < 0.1 ? "opening" : p < 0.4 ? "early" : p < 0.7 ? "mid" : p < 0.9 ? "late" : "closing";
  })() : "—";
  document.querySelectorAll("[data-panel]").forEach((panel) => {
    const name = panel.dataset.panel;
    const head = panel.classList.contains("card-head") ? panel : panel.querySelector(".card-head");
    if (!head) return;
    let rail = head.querySelector(".panel-rail");
    if (!rail) {
      rail = document.createElement("div");
      rail.className = "panel-rail";
      rail.innerHTML = `<span class="rail-feed" title="&b - the internet source behind this panel right now"></span>` +
        `<span class="rail-phase" title="¶gn - where in the 60 s window we are"></span>` +
        `<span class="rail-age" title="seconds since this panel last repainted"></span>`;
      head.appendChild(rail);
    }
    const f = rail.querySelector(".rail-feed");
    f.className = `rail-feed ${feed}`;
    const cov = state.liveProvenance?.coverage;
    const holding = state.liveProvenance?.holding_back || [];
    f.textContent = `&b ${feed === "live" ? src : feed === "partial" ? `${src} ${Math.round((cov ?? 0) * 100)}%${holding.length ? " · " + holding.join("+") : ""}` : feed}`;
    const rows = state.liveProvenance?.feeds || {};
    f.title = "&b - " + (Object.keys(rows).length
      ? Object.entries(rows).map(([k, r]) => `${k}: ${r.state}${r.note && r.state !== "live" ? ` (${r.note})` : ""}`).join(" · ")
      : "the internet source behind this panel right now");
    rail.querySelector(".rail-phase").textContent = `¶gn ${phase}`;
    const seen = PANEL_SEEN[name];
    const age = seen ? Math.max(0, Math.round((Date.now() - seen) / 1000)) : null;
    const a = rail.querySelector(".rail-age");
    a.textContent = age === null ? "waiting" : age === 0 ? "just now" : `${age}s ago`;
    a.className = "rail-age";
    if (age !== null && age > 90) a.classList.add("old");
  });
}

/* Round AA - the ¶gn mark: one soft breath over the whole surface and a
   whisper of haptic, so the user feels the panels refresh together. */
function pulseBreath() {
  document.body.classList.remove("pulse");
  void document.body.offsetWidth;   // restart the animation
  document.body.classList.add("pulse");
  haptic([5]);
}

/* The ¶gn marks drawn on the countdown ring itself (SVG, pathLength 100). */
function renderRingMarks() {
  const g = $("w-ring-marks");
  if (!g) return;
  const period = state.cyclePeriod || 60;
  const ticks = state.clock?.ticks || [];
  const key = ticks.map((t) => `${t.offset_seconds}:${t.done ? 1 : 0}`).join("|") + `@${period}`;
  if (g.dataset.key === key) return;
  g.dataset.key = key;
  g.innerHTML = ticks.map((t) => {
    // ring is rotated -90deg in CSS, so angle 0 is the top; marks sit on the arc radius
    const a = (Number(t.offset_seconds) / period) * 2 * Math.PI;
    const x1 = 60 + 46 * Math.cos(a), y1 = 60 + 46 * Math.sin(a);
    const x2 = 60 + 58 * Math.cos(a), y2 = 60 + 58 * Math.sin(a);
    return `<line x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}" class="${t.done ? "done" : ""}"><title>${(t.parts || []).join(" + ")} at +${t.offset_seconds}s</title></line>`;
  }).join("");
}

function windowRemaining() {
  if (!state.clock) return state.cyclePeriod || 60;
  return Math.max(0, (state.clock.endsAtMs - serverNowMs()) / 1000);
}

function anchorWindow(window) {
  // Window blocks carry the clock too; this keeps the old call sites working.
  if (!window) return false;
  return applyClock({ window, clock: window.clock });
}

/* The single frame loop.  The ring animates every frame; the digits and the
   clock-driven panels (freshness age, pipeline bar, emergency countdown) are
   written only when the whole second changes, so nothing flickers. */
function frame() {
  const remaining = windowRemaining();
  const period = state.cyclePeriod || 60;
  const secs = clamp(Math.ceil(remaining - 1e-6), 0, Math.ceil(period));
  const ring = $("w-ring");
  if (ring) {
    ring.style.setProperty("--frac", clamp(remaining / period, 0, 1).toFixed(4));
    const ready = !!state.window?.prefetch_ready;
    if (ready && !ring.classList.contains("ready")) haptic([10]);   // the next window is sealed
    ring.classList.toggle("ready", ready);
    ring.classList.toggle("urgent", secs <= 5);
  }
  if (secs !== state.lastSecondShown) {
    const previous = state.lastSecondShown;
    state.lastSecondShown = secs;
    const el = $("w-countdown");
    if (el) {
      el.textContent = String(secs);
      el.classList.remove("tick");
      void el.offsetWidth;
      el.classList.add("tick");
    }
    // minor haptics: the last three seconds tap, the reveal is felt by the
    // SIGNAL handler itself (never twice).
    if (previous !== null && secs > 0 && secs <= 3) haptic([6]);
    if (state.clock?.ticks) {
      const elapsed = period - remaining;
      state.clock.ticks.forEach((t) => { t.done = Number(t.offset_seconds) <= elapsed; });
    }
    renderRingMarks();
    // state + utc used to be written inside a guard on the long-gone #timer
    // panel, so they only moved when that element existed - which it never did.
    if ($("lock-state")) $("lock-state").textContent = state.lockState;
    if ($("utc")) $("utc").textContent = new Date(serverNowMs()).toISOString().substr(11, 8) + "Z";
    if ($("w-countdown-sub")) $("w-countdown-sub").innerHTML = countdownSubLine();
    renderFreshness(state.prediction);
    renderHorizon(state.prediction);
    renderPipelineProgress();
    renderWindowStrip();
    renderPanelRails();
    if (state.emergencyUntil > Date.now()) renderConvictionBox();
  }
  state.raf = requestAnimationFrame(frame);
}

/* What the countdown is counting down to: the window boundary, on the same grid
   every other panel refreshes on. */
function countdownSubLine() {
  const t = state.clock?.ticks || [];
  const next = t.find((mark) => !mark.done && mark.seconds_until > 0);
  const parts = next ? (next.parts || []).join(" + ") : "";
  const every = state.clock?.minuteAligned ? "minute-aligned" : `every ${Math.round(state.cyclePeriod)}s`;
  if (next) {
    return `next refresh t+${Math.round(next.offset_seconds)}s <b>${Math.round(next.seconds_until)}s</b>` +
      (parts ? ` · ${parts}` : "") + ` · ${every}`;
  }
  return `window closes at the boundary · every panel refreshes together · ${every}`;
}

/* REST is the safety net, not the schedule: it only runs while the socket is
   down, so a healthy dashboard makes no polls of its own. */
/* ---- self-update chip -------------------------------------------------------
   Zero commands, never `git pull`: the engine fast-forwards its own branch.
   Checked every 5 minutes from the ONE safety-net timer (no extra interval);
   in a Codespace the engine applies it by itself, the chip is the manual
   route and the notice. */
let updateTick = 0;
async function checkForUpdate() {
  const chip = $("update-chip");
  if (!chip) return;
  const res = await getJSON("/api/update/status");
  if (!res || res.error || !res.ok) return;
  const behind = Number(res.behind || 0);
  chip.classList.remove("blocked");
  if (behind > 0 && res.can_fast_forward) {
    $("update-text").textContent = `${behind} update${behind === 1 ? "" : "s"} available` +
      (res.auto ? " · auto-applies" : "");
    chip.title = `origin/${res.branch} is at ${res.remote} (you are at ${res.local}): ${res.latest || ""}`;
    chip.classList.remove("hidden");
  } else if (behind > 0) {
    // Round Y: a Codespace that is behind but CANNOT fast-forward used to
    // show nothing - the user then judged the latest build by stale code.
    // Say so, in red, with the reason the updater refuses.
    const why = Number(res.ahead || 0) > 0 ? `this checkout has ${res.ahead} local commit${res.ahead === 1 ? "" : "s"} the branch does not`
      : (res.dirty_files || []).length ? `local edits in ${(res.dirty_files || []).slice(0, 3).join(", ")}`
      : res.fetch_error ? `fetch failed: ${res.fetch_error}` : "unknown reason";
    $("update-text").textContent = `STALE BUILD · ${behind} update${behind === 1 ? "" : "s"} not applied — ${why}`;
    chip.title = `you are on ${res.local}; origin/${res.branch} is at ${res.remote}. The self-updater only fast-forwards clean checkouts.`;
    chip.classList.remove("hidden");
    chip.classList.add("blocked");
  } else {
    chip.classList.add("hidden");
  }
}
async function applyUpdate() {
  const chip = $("update-chip");
  chip.classList.add("busy");
  $("update-text").textContent = "updating… the engine restarts in a few seconds";
  const res = await getJSON("/api/update/apply", { method: "POST" });
  if (!res || res.error || !res.started) {
    $("update-text").textContent = `could not start: ${res?.reason || res?.error || "unknown"}`;
    chip.classList.remove("busy");
    return;
  }
  // The engine goes away and comes back with new code; a full reload picks
  // up the new app.js (cache-busters change with every release).
  setTimeout(() => location.reload(), 15000);
}

async function safetyNet() {
  updateTick += 1;
  lockWatchdog();
  if (updateTick % 60 === 1) checkForUpdate();   // 5 s x 60 = every 5 min, first at boot
  // A socket that is OPEN but silent is the proxy keeping our side alive
  // after the backend side died or stalled (Codespaces does exactly this):
  // the countdown kept running while nothing else moved.  The engine streams
  // the crowd twice a second, so 15 s of silence is never normal - close the
  // socket (which reconnects) and fall through to the poll below.
  const silentMs = state.lastMessageAt ? Date.now() - state.lastMessageAt : 0;
  if (state.ws && state.ws.readyState === WebSocket.OPEN && silentMs > 15000) {
    console.warn(`engine silent for ${Math.round(silentMs / 1000)}s - reconnecting`);
    $("ws-label").textContent = `engine silent ${Math.round(silentMs / 1000)}s — reconnecting…`;
    $("ws-dot").className = "dot off";
    state.lastMessageAt = Date.now();
    try { state.ws.close(); } catch (e) { /* onclose reconnects */ }
  }
  // Nothing to do while the socket is healthy - the backend is the schedule.
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
    const t0 = performance.now();
    const current = await getJSON("/api/signal/current");
    if (current && !current.error) {
      const rttMs = performance.now() - t0;
      applyClock({ clock: current.clock, window: current.window }, { rttMs });
      applySnapshot(current, { source: "poll" });
    }
    const status = await getJSON("/api/signal/status");
    if (status && !status.error) renderStatus(status);
    if (current && current.error) {
      $("ws-label").textContent = current.offline ? "engine unreachable — the supervisor restarts it" : `engine error: ${current.error}`;
    }
  }
  if (!state.ready || state.startError) await pollReadiness();
}

async function syncClock() {
  const t0 = performance.now();
  const status = await getJSON("/api/signal/status");
  if (!status) return;
  const rttMs = performance.now() - t0;
  applyClock({ clock: status.window?.clock || status.clock, window: status.window }, { rttMs });
  state.secondsIntoMinute = status.clock?.seconds_into_minute || 0;
  if (status.lock) {
    state.lockState = status.lock.state;
    if (status.lock.emergency_active) {
      state.emergencyUntil = Date.now() + status.lock.emergency_remaining * 1000;
    }
  }
  if (status.asset) state.asset = status.asset;
  state.pendingAsset = status.pending_asset;
  renderStatus(status);
  if (!state.signal) {
    const current = await getJSON("/api/signal/current");
    if (current && !current.error) applySnapshot(current, { source: "poll" });
  }
  renderAssetToggle();
  renderAll();
}

/* Microseconds, printed the way the engine measures them.  A pass costs tens
   of microseconds and a tick arrives every few hundred - showing "0.0 ms"
   everywhere would hide both. */
function fmtUs(us) {
  const n = Number(us);
  if (!Number.isFinite(n) || n <= 0) return "—";
  if (n < 1000) return `${Math.round(n)} µs`;
  if (n < 1e6) return `${(n / 1000).toFixed(n < 1e4 ? 2 : 1)} ms`;
  return `${(n / 1e6).toFixed(3)} s`;
}

function fmtClockUs(iso) {
  // 2026-09-27T09:14:02.123456Z -> 09:14:02.123456
  if (!iso) return "—";
  const match = String(iso).match(/T(\d\d:\d\d:\d\d(?:\.\d+)?)Z?/);
  return match ? match[1] : String(iso);
}

/* What the prediction is *for*: the 60 seconds that start when it is released.
   This is the Round-I contract made visible - release instant to the
   microsecond, the instant it targets, and how much of the window is left. */
function renderHorizon(prediction) {
  const el = $("w-horizon-line");
  if (!el) return;
  const h = prediction?.horizon;
  if (!h || !h.released_at_us) {
    el.textContent = "forecast: the next 60 seconds — waiting for the first release";
    el.className = "horizon-line muted";
    return;
  }
  // Counted down locally against the shared server clock, not frozen at the
  // value the payload carried: the window closes on a fixed instant, and the
  // line has to walk toward it exactly like the ring does.
  const nowUs = serverNowMs() * 1000;
  const left = h.target_at_us
    ? Math.max(0, (h.target_at_us - nowUs) / 1e6)
    : (h.seconds_to_target ?? Math.max(0, (h.microseconds_to_target || 0) / 1e6));
  const scored = h.scored_at
    ? Math.max(0, (h.scored_at_us ? (h.scored_at_us - nowUs) / 1e6 : h.scored_in_seconds))
    : h.scored_in_seconds;
  el.className = "horizon-line";
  // Round L.1: no per-second numbers inside the prediction cell.  The ring is
  // the only countdown; this line states the fixed instants of the forecast
  // and how the window is scored.  (``left``/``scored`` stay in the tooltip.)
  el.innerHTML =
    `forecast <b>${escapeHtml(h.forecast_for || h.label || "the next 60 seconds")}</b>` +
    ` · released <b>${fmtClockUs(h.released_at_precise || h.released_at)}</b>` +
    ` · targets <b>${fmtClockUs(h.target_at_precise || h.target_at)}</b>` +
    ` · <span class="muted">side, confidence and levels are frozen until then</span>`;
  el.title = `${Number(left).toFixed(1)}s left in this window` +
    (scored !== undefined ? ` · scored in ${Number(scored).toFixed(1)}s` : "");
}

/* The tape itself: how fast quotes arrive and how finely the engine can see.
   Shown from the live formula pass (or the prediction detail when the socket
   is still warming). */
function renderMicro(prediction) {
  const el = $("w-micro-line");
  if (!el) return;
  // Round L.1: this line sits inside the locked prediction cell, so it shows
  // the tape *as it was at the lock* (from the prediction detail) and never
  // the live pass - nothing in the cell may move during a window.  The live
  // tape numbers stay in the Formula Explorer.
  const micro = prediction?.detail?.micro || null;
  if (!micro || !micro.resolution_us) {
    el.textContent = "tape at lock — waiting for the first lock";
    el.className = "micro-line muted";
    return;
  }
  el.className = "micro-line";
  el.innerHTML =
    `tape at lock <b>${escapeHtml(micro.resolution_label || fmtUs(micro.resolution_us))}</b> per tick` +
    ` · <b>${Number(micro.tick_rate_hz || 0).toFixed(1)}</b> Hz` +
    ` · jitter <b>${fmtUs(micro.jitter_us)}</b>` +
    ` · quote life <b>${fmtUs(micro.quote_lifetime_us)}</b>` +
    ` · aggression <b>${fmtSigned(micro.aggression || 0, 2)}</b>`;
}

function detailList(label, rows, cls) {
  const items = (rows || []).map((row) => {
    const name = escapeHtml(row.name || row.key || "?");
    const value = typeof row.value === "number" ? fmtSigned(row.value, 3) : escapeHtml(row.value ?? "");
    return `<li><span class="trace-label">${name}</span><b class="${cls}">${value}</b></li>`;
  }).join("");
  return items
    ? `<div><div class="logic-h">${escapeHtml(label)}</div><ul class="logic-trace">${items}</ul></div>`
    : "";
}

/* The prediction detail block: which formulas argued for the side, which
   against, what each category scored, how the confidence was assembled, and
   what the engine cost in microseconds. */
function renderPredictionDetail(prediction) {
  const box = $("w-detail");
  if (!box) return;
  const detail = prediction?.detail;
  if (!detail) {
    box.innerHTML = `<div class="muted">no detail yet — it arrives with the first live prediction</div>`;
    return;
  }
  const cats = Object.entries(detail.category_scores || {})
    .map(([key, raw]) => {
      // Each category is {count, sum, mean, directional} (Round I); older
      // payloads carried a bare number.
      const value = typeof raw === "number" ? raw : Number(raw?.mean ?? 0);
      const count = typeof raw === "object" && raw ? raw.count : null;
      return `<span class="kv"><span>${escapeHtml(key)}${count ? `×${count}` : ""}</span>` +
        `<b class="${value >= 0 ? "pos" : "neg"}">${fmtSigned(value, 3)}</b></span>`;
    })
    .join(" ");
  const parts = Object.entries(detail.confidence_parts || {})
    .map(([key, value]) => `<li><span class="trace-label">${escapeHtml(key.replace(/_/g, " "))}</span><b>${fmtSigned(value, 3)}</b></li>`)
    .join("");
  const engine = detail.agreement?.engine || {};
  const levels = detail.levels || {};
  const micro = detail.micro || {};

  box.innerHTML =
    `<div class="logic-h">${escapeHtml(String(detail.side || prediction.signal || ""))} · ` +
    `${detail.formulas_evaluated || 0} formulas evaluated · ` +
    `agreement ${fmtSigned(detail.agreement?.score ?? 0, 3)}` +
    `${detail.agreement?.weighted ? ` (weighted ${fmtSigned(detail.agreement.weighted, 3)})` : ""}</div>` +
    (cats ? `<div class="detail-cats">categories ${cats}</div>` : "") +
    `<div class="logic-cols">` +
    detailList("Supporting the side", detail.supporters, "pos") +
    detailList("Arguing against", detail.opponents, "neg") +
    `</div>` +
    (parts ? `<div><div class="logic-h">Confidence parts</div><ul class="logic-trace">${parts}</ul></div>` : "") +
    `<div><div class="logic-h">Risk levels</div><div class="detail-line">` +
    `tp ${fmtSigned(levels.tp_bps ?? 0, 1)} bps · sl ${fmtSigned(levels.sl_bps ?? 0, 1)} bps · ` +
    `rr ${Number(levels.rr ?? 1).toFixed(2)} : 1` +
    (levels.distance_price !== undefined ? ` · distance ${fmtMoney(levels.distance_price)}` : "") +
    `</div></div>` +
    `<div><div class="logic-h">Engine microsecond budget</div><div class="detail-line">` +
    `pass <b>${fmtUs(engine.compute_us)}</b> · publish latency <b>${fmtUs(engine.publish_latency_us)}</b> · ` +
    `tick interval <b>${fmtUs(engine.tick_interval_us)}</b> · tape resolution <b>${fmtUs(engine.resolution_us)}</b> · ` +
    `history <b>${engine.history_samples ?? 0}</b> samples` +
    (micro.tick_rate_hz ? ` · <b>${Number(micro.tick_rate_hz).toFixed(1)}</b> Hz tape` : "") +
    `</div></div>` +
    (Object.keys(detail.agents || {}).length
      ? `<div class="detail-line">agents: ${Object.entries(detail.agents).map(([name, payload]) =>
          `${escapeHtml(name)} ${escapeHtml(String(payload.decision ?? payload.status ?? "—"))}`
            + (payload.confidence !== undefined && payload.confidence !== null ? ` (${fmtPct(payload.confidence)})` : "")).join(" · ")}</div>`
      : "") +
    (detail.brain
      ? `<div class="detail-line">brain: ${escapeHtml(String(detail.brain.status ?? "—"))} · gain ${fmtSigned(detail.brain.gain ?? 0, 3)} · ${escapeHtml(String(detail.brain.message || ""))}</div>`
      : "") +
    crowdDetailHtml(detail.crowd) +
    learnedDetailHtml(detail.learned) +
    (detail.formula_stats
      ? `<div class="muted" style="font-size:11px">per-formula history: ${Object.keys(detail.formula_stats).length} formulas tracked</div>`
      : "");
}

/* Learned reliability (Round N): which sources have actually been right on
   this asset, and whether that record — not the spec recipe — decided the
   side.  Shown honestly: watch-only until enough windows are scored. */
function learnedDetailHtml(learned) {
  if (!learned) return "";
  const scored = learned.scored ?? 0;
  const need = learned.min_samples ?? 30;
  if (!learned.active) {
    const why = learned.handed_back
      ? `the spec recipe is scoring better right now (ledger ${fmtPct(learned.ledger_hit_rate)} vs spec ${fmtPct(learned.spec_hit_rate)}) — ledger is watch-only`
      : scored < need
        ? `watch-only until ${need} windows are scored on this asset (${scored} so far)`
        : "watch-only";
    return `<div><div class="logic-h">Learned reliability</div><div class="detail-line muted">${escapeHtml(why)}</div></div>`;
  }
  const row = (v, cls) =>
    `<li><span class="trace-label">${escapeHtml(v.source)} ${(v.vote === "up" || v.vote > 0) ? "▲" : "▼"}</span>` +
    `<b class="${cls}">${fmtPct(v.reliability)} right · n ${Number(v.n ?? 0).toFixed(0)}</b></li>`;
  const rows = (learned.for || []).slice(0, 5).map((v) => row(v, "pos")).join("") +
    (learned.against || []).slice(0, 3).map((v) => row(v, "neg")).join("");
  const realised = learned.realised_at_this_confidence;
  return `<div><div class="logic-h">Learned reliability decided this side</div>` +
    `<div class="detail-line">${scored} windows scored · ledger ${fmtPct(learned.ledger_hit_rate)} right vs spec recipe ${fmtPct(learned.spec_hit_rate)} · ` +
    `earned confidence <b>${fmtPct(learned.p_side)}</b>` +
    (realised !== null && realised !== undefined ? ` (this bucket realised ${fmtPct(realised)})` : "") +
    `</div><ul class="logic-trace">${rows}</ul></div>`;
}

/* What the crowd was feeling when this side was locked, and whether that was
   strong enough to dampen the confidence (Round J). */
function crowdDetailHtml(crowd) {
  if (!crowd || !crowd.available) return "";
  const bars = (crowd.emotions || []).slice(0, 8)
    .map((e) => `${escapeHtml(e.label || e.name)} ${Number(e.percent || 0).toFixed(0)}%`)
    .join(" · ");
  const parts = state.prediction?.detail?.confidence_parts || {};
  const damp = Number(parts.crowd_dampening ?? 1);
  return `<div><div class="logic-h">Crowd at lock time</div><div class="detail-line">` +
    `dominant <b>${escapeHtml(String(crowd.dominant || "—"))}</b> ${Number(crowd.percent || 0).toFixed(0)}% ` +
    `on the <b>${escapeHtml(String(crowd.timescale || "—"))}</b> timescale · ` +
    `temperature ${fmtSigned(crowd.tone_bias ?? 0, 2)} · ` +
    `manipulation <b>${fmtPct(crowd.manipulation || 0)}</b> (${escapeHtml(String(crowd.manipulation_kind || "none"))})` +
    (damp < 0.999 ? ` · confidence <b>x${damp.toFixed(2)}</b> for the crowd` : " · no crowd dampening") +
    `</div>` +
    (bars ? `<div class="detail-line" style="font-size:10.5px">${bars}</div>` : "") +
    (crowd.read ? `<div class="muted" style="font-size:11px">${escapeHtml(String(crowd.read))}</div>` : "") +
    agreementLockedHtml(crowd.formula_agreement) +
    deepLockedHtml(crowd.deep) +
    `</div>`;
}

/* Round L: the locked crowd against the lock-time formulas, and who decided. */
function agreementLockedHtml(agree) {
  if (!agree || !agree.verdict) return "";
  return `<div class="logic-h" style="margin-top:8px">Crowd vs the 22 formulas at lock time</div>` +
    `<div class="detail-line">verdict <b>${escapeHtml(String(agree.verdict))}</b> · formulas ${fmtSigned(Number(agree.consensus || 0), 2)} ` +
    `(${agree.voters || 0} voted, ${agree.up || 0} up / ${agree.down || 0} down) · crowd ${fmtSigned(Number(agree.crowd_tone || 0), 2)} · ` +
    `alignment ${fmtSigned(Number(agree.alignment || 0), 2)}</div>` +
    `<div class="muted" style="font-size:11px">${escapeHtml(String(agree.note || ""))} — ${escapeHtml(String(agree.rule || ""))}</div>`;
}

/* Round K: the deep-reasoning layer as it was when the side was locked -
   the filter's belief, the microstructure verdicts and the chain. */
function deepLockedHtml(deep) {
  if (!deep || !deep.available) return "";
  const num = (v, d = 2) => (typeof v === "number" ? v.toFixed(d) : "—");
  const chain = (deep.chain || []).map((step) =>
    `<li><b>${escapeHtml(String(step.name))}</b> <span class="mono">${escapeHtml(String(step.value))} ${escapeHtml(String(step.unit || ""))}</span>` +
    ` — ${escapeHtml(String(step.reads || ""))}</li>`).join("");
  return `<div class="logic-h" style="margin-top:8px">Deep reasoning at lock time</div>` +
    `<div class="detail-line">Bayesian filter <b>${escapeHtml(String(deep.belief || "—"))}</b> ${fmtPct(Number(deep.belief_probability || 0))} belief · ` +
    `certainty ${num(deep.certainty)} · regime <b>${escapeHtml(String(deep.regime || "—"))}</b> · ` +
    `Hawkes n ${num(deep.branching_ratio)} · VPIN ${num(deep.vpin)} · Kyle λ ${num(deep.kyle_lambda_bps, 4)} (R² ${num(deep.kyle_r2)}) · ` +
    `VR ${num(deep.variance_ratio)} · H ${num(deep.hurst)} · entropy ${num(deep.entropy)} · sign memory ${num(deep.sign_memory)} · ` +
    `computed in ${deep.compute_us ? fmtUs(deep.compute_us) : "—"}</div>` +
    (chain ? `<ol class="deep-chain compact">${chain}</ol>` : "");
}

/* ------------------------------------------------------------ emotions --
   Round J.  The card answers, live: which emotion is dominant, on which
   timescale, for how long, how crowded the minute is, and what the crowd was
   feeling when the locked signal was computed.  The data arrives in the
   EMOTION stream (twice a second) and in every PULSE / snapshot; the client
   only draws it - it never measures anything itself. */
function adoptEmotions(data) {
  const block = data?.emotions;
  if (!block || typeof block !== "object") return;
  state.emotions = block;
  // The deep block is heavy, so the stream carries it on every fourth sample
  // only; the bars and the read-out still move twice a second.  The last deep
  // block is kept and repainted only when a new one arrives.
  if (block.deep && block.deep.available !== undefined) {
    state.emotionsDeep = block.deep;
    state.emotionsDeepDirty = true;
  }
  if (block.locked !== undefined) state.emotionsLocked = block.locked;
  if (block.dampening) state.emotionDampening = block.dampening;
  if (data.dampening) state.emotionDampening = data.dampening;
}

function toneClass(tone) {
  return tone === "negative" || tone === "positive" ? tone : "neutral";
}

function emotionTimescaleLabel(key) {
  return { micro: "µs", seconds: "sec", window: "60 s", minutes: "min", news: "news" }[key] || key;
}

function renderEmotions() {
  markPanel("emotions");
  const e = state.emotions;
  const dominantEl = $("emotion-dominant");
  if (!dominantEl) return;
  const line = $("w-crowd-text");
  const dot = $("w-crowd-dot");
  if (!e || !e.available) {
    dominantEl.textContent = "—";
    dominantEl.className = "emotion-dominant neutral";
    $("emotion-dominant-sub").textContent = e?.reason || "the crowd has not been measured yet";
    $("emotion-meta").textContent = "waiting for the first sample…";
    if (line) line.textContent = "crowd — measuring the tape…";
    if (dot) dot.className = "crowd-dot neutral";
    return;
  }
  const top = e.dominant || {};
  const tone = toneClass(top.tone);
  const changed = state.lastEmotionDominant !== null && state.lastEmotionDominant !== top.name;
  dominantEl.textContent = top.label || top.name || "—";
  dominantEl.className = `emotion-dominant ${tone}`;
  if (changed) {
    dominantEl.classList.remove("pop");
    void dominantEl.offsetWidth; // restart the animation
    dominantEl.classList.add("pop");
    haptic([8]);
  }
  state.lastEmotionDominant = top.name || null;

  $("emotion-dominant-sub").textContent =
    `${Number(top.percent || 0).toFixed(0)}% intensity · ${top.family || ""} family · ` +
    `strongest on the ${emotionTimescaleLabel(top.dominant_timescale)} timescale`;
  const held = Number(e.held_seconds ?? top.held_seconds ?? 0);
  const runner = e.runner_up;
  $("emotion-held").innerHTML =
    `held for <b>${held >= 60 ? `${Math.floor(held / 60)}m ${Math.round(held % 60)}s` : `${held.toFixed(1)} s`}</b>` +
    (runner ? ` · runner-up <b>${escapeHtml(runner.label)}</b> ${Number(runner.percent || 0).toFixed(0)}%` : "") +
    (e.churn_per_minute ? ` · ${e.churn_per_minute} switch${e.churn_per_minute === 1 ? "" : "es"} this minute` : "");
  $("emotion-read").textContent = e.read || "";
  $("emotion-drivers").innerHTML = (top.drivers || []).map((d) => `<li>${escapeHtml(d)}</li>`).join("");
  $("emotion-meta").textContent =
    `${e.asset || state.asset} · sampled every ${Number(e.interval_seconds || 0.5).toFixed(1)} s · ` +
    `tape resolution ${e.resolution_label || fmtUs(e.resolution_us)} · ${e.ticks || 0} ticks · ` +
    `${e.samples || 0} samples`;

  /* the eight bars, ranked */
  const bars = $("emotion-bars");
  const rows = (e.emotions || []).map((item) => {
    const cls = toneClass(item.tone);
    const isTop = item.name === top.name;
    const selected = (state.emotionFormulaFor || top.name) === item.name;
    return `<div class="emotion-row${selected ? " selected" : ""}" data-emotion="${escapeHtml(item.name)}" title="click to print this emotion's formula">` +
      `<span class="name${isTop ? " dominant" : ""}">${escapeHtml(item.label || item.name)}<span class="fam">${escapeHtml(item.family || "")}</span></span>` +
      `<div class="bar"><div class="bar-fill ${cls}" style="width:${clamp(Number(item.percent || 0), 0, 100)}%"></div></div>` +
      `<span class="pct">${Number(item.percent || 0).toFixed(0)}%</span>` +
      `<span class="band">${emotionTimescaleLabel(item.dominant_timescale)}</span>` +
      `</div>`;
  });
  bars.innerHTML = rows.join("");
  renderEmotionFormula(e, top);
  renderFormulaAgreement(e.formula_agreement);

  /* temperature */
  const toneBias = clamp(Number(e.tone_bias || 0), -1, 1);
  $("emotion-tone-marker").style.left = `${50 + toneBias * 50}%`;
  $("emotion-tone").innerHTML =
    `<b>${fmtSigned(toneBias, 2)}</b> ` +
    (toneBias < -0.25 ? "— the crowd is afraid" : toneBias > 0.25 ? "— the crowd is chasing" : "— the crowd is neither afraid nor chasing");

  /* manipulation */
  const manip = e.manipulation || {};
  const score = clamp(Number(manip.score || 0), 0, 1);
  const bar = $("emotion-manip-bar");
  bar.style.width = `${(score * 100).toFixed(0)}%`;
  bar.className = `bar-fill ${score >= 0.45 ? "neg" : score >= 0.25 ? "" : "neutral"}`;
  if (score >= 0.25 && score < 0.45) bar.style.background = "var(--amber)"; else bar.style.background = "";
  const damp = Number(state.emotionDampening?.applied ?? 1);
  $("emotion-manip").innerHTML =
    `<b>${fmtPct(score)}</b> ${escapeHtml(String(manip.kind || "none"))}` +
    (damp < 0.999 ? ` · confidence <b>x${damp.toFixed(2)}</b>` : "");
  $("emotion-manip-note").textContent = manip.note || "";

  /* timescales of the dominant emotion */
  const topRow = (e.emotions || []).find((item) => item.name === top.name) || {};
  const bands = topRow.by_timescale || {};
  $("emotion-timescales").innerHTML = ["micro", "seconds", "window", "minutes", "news"].map((key) => {
    const value = Number(bands[key] || 0);
    const peak = key === top.dominant_timescale;
    return `<div class="timescale-cell${peak ? ` peak ${tone}` : ""}">${emotionTimescaleLabel(key)}<b>${(value * 100).toFixed(0)}%</b></div>`;
  }).join("");

  /* at lock time */
  const locked = state.emotionsLocked;
  const lockedTop = locked?.dominant;
  $("emotion-locked").innerHTML = lockedTop
    ? `<b>${escapeHtml(lockedTop.label || lockedTop.name)}</b> ${Number(lockedTop.percent || 0).toFixed(0)}% · ` +
      `manipulation ${fmtPct(Number(locked.manipulation?.score || 0))} (${escapeHtml(String(locked.manipulation?.kind || "none"))})` +
      `<span class="muted">${escapeHtml(String(state.emotionDampening?.note || "no crowd dampening on this signal"))}</span>`
    : `<span class="muted">the first locked window will record the crowd's mood</span>`;

  /* inline line in the prediction cell: the crowd *at the lock*, which does
     not move during the window - the live reading is the card above. */
  renderLockedCrowdLine(line, dot);
  if (state.emotionsDeepDirty) {
    state.emotionsDeepDirty = false;
    renderDeep(state.emotionsDeep, top);
  }
}

function renderLockedCrowdLine(line, dot) {
  if (!line) return;
  const locked = state.emotionsLocked;
  const lockedTop = locked?.dominant;
  if (!lockedTop) {
    line.textContent = "crowd at lock — waiting for the first lock…";
    if (dot) dot.className = "crowd-dot neutral";
    return;
  }
  const agree = locked.formula_agreement || {};
  const damp = Number(state.emotionDampening?.applied ?? 1);
  const verdict = agree.verdict ? ` · vs formulas: <b>${escapeHtml(agree.verdict)}</b>` : "";
  line.innerHTML = `crowd at lock: <b>${escapeHtml(lockedTop.label || lockedTop.name)}</b> ` +
    `${Number(lockedTop.percent || 0).toFixed(0)}%${verdict}` +
    (damp < 0.999 ? ` · confidence x${damp.toFixed(2)}` : " · no confidence cut") +
    ` · <span class="muted">frozen for this window</span>`;
  if (dot) dot.className = `crowd-dot ${toneClass(lockedTop.tone)}`;
}

/* Round L: the emotion's formula, printed with the live numbers.  The eight
   emotions are weighted sums of bounded ramps over the tape's own surprise
   units, times a gate; every term is shown with its weight, its live value and
   its contribution so the reading can be checked by hand. */
function renderEmotionFormula(e, top) {
  const el = $("emotion-formula");
  if (!el) return;
  const name = state.emotionFormulaFor || top.name;
  const item = (e.emotions || []).find((row) => row.name === name) || (e.emotions || [])[0];
  if (!item || !item.formula) { el.innerHTML = ""; return; }
  const terms = (item.terms || []).map((t) =>
    `<tr><td class="w">${Number(t.weight).toFixed(2)}</td><td class="t">${escapeHtml(t.term)}</td>` +
    `<td class="v">${Number(t.value).toFixed(3)}</td><td class="c">${Number(t.contribution).toFixed(3)}</td></tr>`).join("");
  const raw = (item.terms || []).reduce((acc, t) => acc + Number(t.contribution || 0), 0);
  const gate = Number(item.gate ?? 1);
  el.innerHTML =
    `<div class="formula-head"><b>${escapeHtml(item.label || item.name)}</b> <span class="muted">formula · live terms</span></div>` +
    `<code class="formula-text">${escapeHtml(item.formula)}</code>` +
    `<table class="term-table"><thead><tr><th>w</th><th>term</th><th>value</th><th>w·value</th></tr></thead><tbody>${terms}</tbody></table>` +
    `<div class="formula-foot">Σ = <b>${raw.toFixed(3)}</b> × gate <b>${gate.toFixed(3)}</b>` +
    ` → ramp <b>${Number(item.ramp ?? 0).toFixed(3)}</b> · Bayesian belief <b>${(Number(item.belief || 0) * 100).toFixed(0)}%</b>` +
    ` · shown <b>${Number(item.percent || 0).toFixed(0)}%</b> <span class="muted">(0.65·ramp + 0.35·min(1, 2.5·belief), band-lifted, EMA-smoothed)</span></div>`;
}

/* Round L: the crowd against the 22 formulas.  The formulas carry the vote
   (40% of the fusion, the largest weight); the crowd is a bounded confidence
   modifier.  This block says whether the two agree and repeats the rule. */
function renderFormulaAgreement(agree) {
  const el = $("emotion-agreement");
  if (!el) return;
  if (!agree || !agree.verdict) { el.innerHTML = `<span class="muted">waiting for the first formula pass…</span>`; return; }
  const consensus = clamp(Number(agree.consensus || 0), -1, 1);
  const tone = clamp(Number(agree.crowd_tone || 0), -1, 1);
  const cls = agree.verdict === "aligned" ? "positive" : agree.verdict === "conflict" ? "negative" : "neutral";
  const names = [...(agree.up_names || []).map((n) => `${n}↑`), ...(agree.down_names || []).map((n) => `${n}↓`)].slice(0, 6).join(" ");
  el.innerHTML =
    `<div class="agree-verdict ${cls}">${escapeHtml(agree.verdict)}</div>` +
    `<div class="agree-row"><span>formulas</span><div class="agree-gauge"><div class="tone-zero"></div><div class="tone-marker" style="left:${50 + consensus * 50}%"></div></div><b>${fmtSigned(consensus, 2)}</b></div>` +
    `<div class="agree-row"><span>crowd</span><div class="agree-gauge"><div class="tone-zero"></div><div class="tone-marker" style="left:${50 + tone * 50}%"></div></div><b>${fmtSigned(tone, 2)}</b></div>` +
    `<div class="agree-note">${escapeHtml(agree.note || "")}</div>` +
    (names ? `<div class="agree-names muted">${escapeHtml(names)}</div>` : "") +
    `<div class="agree-rule muted">${escapeHtml(agree.rule || "")}</div>`;
}

/* ------------------------------------------------------------ deep --------
   Round K.  The microstructure formulas behind the reading (Hawkes, VPIN,
   Kyle's lambda, variance ratio, Hurst, permutation entropy, sign memory,
   wavelet spectrum, regime filter, ignition / stuffing / spoofing) and the
   Bayesian filter's belief, plus the ordered reasoning chain.  Everything is
   computed on the server at the emotion cadence; the client only draws it. */
function renderGeometry(geo) {
  const el = $("deep-geometry");
  if (!el) return;
  if (!geo || !geo.available) { el.innerHTML = ""; return; }
  const g = (v, d = 2) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(d));
  const bar = (v) => `<span class="geo-bar"><span style="width:${Math.round(Math.max(0, Math.min(1, Number(v) || 0)) * 100)}%"></span></span>`;
  const tda = geo.tda || {}, tak = geo.takens || {}, csd = geo.csd || {}, th = geo.thermo || {}, qi = geo.quantum || {};
  el.innerHTML =
    `<div class="geo-head">GEOMETRY OF THE CROWD <span class="muted">${geo.live_measurements}/5 live · stress ${g(geo.stress)}</span></div>` +
    `<div class="geo-grid">` +
    `<div title="0-dim persistent homology of the liquidity surface: gaps ≥ 3× the median gap are cavities; tearing = depth behind them"><span>TDA · book manifold</span>${bar(tda.tearing)}<b>${g(tda.tearing)}</b><i>${tda.cavities ?? 0} cavities · β₀ ${(tda.betti0 || []).join("/")}</i></div>` +
    `<div title="Takens (3,1) delay embedding of 1 s returns; Rosenstein local Lyapunov exponent"><span>Takens · attractor</span>${bar(tak.chaotic)}<b>λ ${g(tak.lyapunov)}</b><i>${Number(tak.lyapunov) > 0.2 ? "spiralling" : "stable"}</i></div>` +
    `<div title="lag-1 autocorrelation and variance both rising = precursor of a phase transition"><span>Critical slowing</span>${bar(csd.csd)}<b>${g(csd.csd)}</b><i>a₁ ${g(csd.a1_now)} · var×${g(csd.variance_ratio)}</i></div>` +
    `<div title="σ = J·X: capital flux between BTC (hot) and PAXG (cold) times the return differential"><span>Entropy production</span>${bar(th.entropy_production)}<b>${g(th.entropy_production)}</b><i>${escapeHtml(th.heat_direction || "—")}</i></div>` +
    `<div title="order effect: P(up) − [P(buyer)P(up|buyer) + P(seller)P(up|seller)] on interleaved halves"><span>Quantum interference</span>${bar(qi.polarisation)}<b>I ${g(qi.interference, 3)}</b><i>${Number(qi.polarisation) > 0.3 ? "non-classical" : "classical"}</i></div>` +
    `</div>` +
    `<div class="geo-read muted">${escapeHtml(geo.read || "")}</div>`;
}

/* Round AA - the inverse-RL utility read (λ, γ, α) under the geometry. */
function renderUtility(util) {
  let el = $("deep-utility");
  if (!el) {
    const host = $("deep-geometry");
    if (!host || !host.parentNode) return;
    el = document.createElement("div");
    el.className = "deep-utility";
    el.id = "deep-utility";
    host.parentNode.insertBefore(el, host.nextSibling);
  }
  if (!util || !util.available) { el.innerHTML = ""; return; }
  const g = (v, d = 2) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(d));
  const lam = util.lambda || {}, gam = util.gamma || {}, alp = util.alpha || {};
  const pos = (v, lo, hi) => Math.max(0, Math.min(1, (Number(v) - lo) / (hi - lo))).toFixed(3);
  const tile = (title, big, sub, p, tip) =>
    `<div title="${escapeHtml(tip)}"><span>${title}</span><b>${big}</b><i>${escapeHtml(sub)}</i><span class="util-scale" style="--pos:${p}"></span></div>`;
  el.innerHTML =
    `<div class="geo-head">WHAT THE CROWD IS MAXIMISING <span class="muted">inverse RL · ${util.live_measurements}/3 live</span></div>` +
    `<div class="util-grid">` +
    (lam.available ? tile("loss aversion λ", g(lam.lambda), lam.lambda > 2.5 ? "sells losses far harder than it buys gains" : lam.lambda < 0.5 ? "chases gains, ignores losses" : "symmetric (KT population ≈ 2.25)", pos(Math.log(lam.lambda || 1), -1.5, 1.5),
      `flow_t = β⁻·min(r,0) + β⁺·max(r,0) over ${lam.buckets} one-second buckets; β⁻ ${g(lam.beta_loss, 3)} (t ${g(lam.t_loss, 1)}), β⁺ ${g(lam.beta_gain, 3)} (t ${g(lam.t_gain, 1)}) — ${lam.significance || ""}`) : `<div><span>loss aversion λ</span><b>—</b><i>needs 24 s of tape</i></div>`) +
    (gam.available ? tile("risk aversion γ", g(gam.gamma), gam.gamma > 0.3 ? "steps back when the tape gets wild" : gam.gamma < -0.3 ? "chases volatility" : "indifferent to volatility", pos(gam.gamma, -2, 2),
      `γ = ln(participation in calm seconds / participation in volatile seconds) = ln(${g(gam.participation_calm, 0)} / ${g(gam.participation_volatile, 0)})`) : `<div><span>risk aversion γ</span><b>—</b><i>needs 24 s of tape</i></div>`) +
    (alp.available ? tile("probability weighting α", g(alp.alpha), alp.alpha < 0.85 ? "book braced for a jump (tails overweighted)" : alp.alpha > 1.25 ? "book complacent (tails neglected)" : "tails priced about right", pos(alp.alpha, 0.2, 2.0),
      `w(p) = p^α; α = ln(tail share ${g(alp.tail_share, 3)}) / ln(uniform tail ${g(alp.uniform_tail, 3)}), tail = beyond ${g(alp.tail_bps, 0)} bps`) : `<div><span>probability weighting α</span><b>—</b><i>${escapeHtml(alp.reason || "book too shallow")}</i></div>`) +
    `</div>` +
    `<div class="util-read muted">${escapeHtml(util.read || "")}</div>`;
}

function renderDeep(deep, top) {
  const belief = $("deep-belief");
  if (!belief) return;
  if (!deep || !deep.available) {
    belief.textContent = deep?.reason || "the deep layer needs a little more tape";
    $("deep-verdicts").innerHTML = ""; $("deep-bands").innerHTML = "";
    $("deep-detectors").innerHTML = ""; $("deep-spectrum").innerHTML = ""; $("deep-chain").innerHTML = "";
    return;
  }
  const num = (v, d = 2) => (typeof v === "number" ? v.toFixed(d) : "—");
  const post = deep.posterior || {};
  const agrees = top && post.argmax === top.name;
  const reg = deep.regime || {};
  const flow = deep.flow || {};
  const hawkes = deep.hawkes || {};
  const book = deep.book || {};
  renderGeometry(deep.geometry);
  renderUtility(deep.utility);
  $("deep-meta").textContent =
    `${deep.ticks || 0} ticks · ${(deep.chain || []).length} formulas · computed in ${deep.compute_us ? fmtUs(deep.compute_us) : "—"}`;
  const evidenceChips = ((post.evidence_for || {})[post.argmax] || []).slice(0, 4).map((ev) =>
    `<span class="${ev.log_odds >= 0 ? "chip pos" : "chip neg"}">${escapeHtml(String(ev.label))} ${fmtSigned(Number(ev.log_odds), 2)}</span>`).join("");
  belief.innerHTML =
    `Bayesian filter: <b class="${agrees ? "ok" : "warn"}">${escapeHtml(String(post.argmax || "—")).toLowerCase()}</b> ` +
    `${fmtPct(Number(post.argmax_probability || 0))} belief ` +
    `<span class="muted">(${agrees ? "agrees with the reading" : "leans differently from the reading"} · ` +
    `certainty ${num(post.certainty)} · surprise ${num(post.surprise_kl)} nats · runner-up ${escapeHtml(String(post.runner_up || "—")).toLowerCase()})</span>` +
    `<div class="deep-evidence">${evidenceChips}</div>`;

  const verdict = (label, value, reads, cls = "") =>
    `<div class="deep-verdict ${cls}"><span class="k">${label}</span><b>${value}</b><span class="r">${escapeHtml(String(reads))}</span></div>`;
  const n = Number(hawkes.branching_ratio || 0);
  const vp = Number(flow.vpin || 0);
  $("deep-verdicts").innerHTML = [
    verdict("Hawkes n", num(n), n >= 0.6 ? "cascade" : n >= 0.3 ? "clustered" : "Poisson", n >= 0.6 ? "neg" : ""),
    verdict("intensity", `${num(hawkes.intensity_hz, 1)}/s`, `baseline ${num(hawkes.baseline_hz, 1)}/s`),
    verdict("VPIN", num(vp), vp >= 0.6 ? "one-sided" : vp >= 0.4 ? "leaning" : "balanced", vp >= 0.6 ? "neg" : ""),
    verdict("Kyle λ", num(flow.kyle_lambda_bps, 4), `R² ${num(flow.kyle_r2)} · impact ${num(flow.impact_norm)}x`),
    verdict("sign memory", num(flow.sign_memory), `γ ${num(flow.sign_gamma)}`),
    verdict("microprice", `${fmtSigned(Number(book.microprice_bps || 0), 3)} bps`, `top-5 ${fmtSigned(Number(book.pressure_top5 || 0), 2)}`),
    verdict("regime", escapeHtml(String(reg.label || "—")),
      `calm ${fmtPct(Number(reg.calm || 0))} · trend ${fmtPct(Number(reg.trend || 0))} · stress ${fmtPct(Number(reg.stress || 0))}`,
      reg.label === "stress" ? "neg" : reg.label === "calm" ? "ok" : ""),
  ].join("");

  const bands = deep.bands || {};
  $("deep-bands").innerHTML =
    `<div class="deep-band head"><span>band</span><span>clock</span><span>H</span><span>VR</span><span>entropy</span></div>` +
    ["micro", "seconds", "window"].map((key) => {
      const b = bands[key] || {};
      const vr = Number(b.variance_ratio || 1);
      const h = Number(b.hurst || 0.5);
      return `<div class="deep-band"><span class="k">${key}</span><span class="mono">${escapeHtml(String(b.clock || "—"))}</span>` +
        `<span class="${h > 0.58 ? "pos" : h < 0.42 ? "neg" : ""}">${num(h)}</span>` +
        `<span class="${vr > 1.15 ? "pos" : vr < 0.85 ? "neg" : ""}">${num(vr)}</span>` +
        `<span>${num(b.entropy)}</span></div>`;
    }).join("");

  const man = deep.manipulation || {};
  $("deep-detectors").innerHTML = [
    ["ignition", "momentum ignition"], ["toxicity", "toxic flow"], ["stuffing", "quote stuffing"],
    ["spoofing", "spoofing"], ["pushable", "pushable tape"],
  ].map(([key, label]) => {
    const v = clamp(Number(man[key] || 0), 0, 1);
    return `<div class="deep-detector"><span class="k">${label}</span>` +
      `<div class="bar"><div class="bar-fill ${v >= 0.5 ? "neg" : v >= 0.25 ? "" : "neutral"}" style="width:${(v * 100).toFixed(0)}%"></div></div>` +
      `<span class="pct">${fmtPct(v)}</span></div>`;
  }).join("");

  const spec = deep.spectrum || [];
  const peak = spec.reduce((a, b) => (Number(b.share) > Number(a?.share || 0) ? b : a), null);
  const specCells = spec.map((sc) =>
    `<span class="${sc === peak ? "spec-cell peak" : "spec-cell"}"><b>${escapeHtml(String(sc.scale_label))}</b>${fmtPct(Number(sc.share || 0))}</span>`).join("");
  $("deep-spectrum").innerHTML = spec.length ? `<span class="k">where the energy is</span>${specCells}` : "";

  $("deep-chain-meta").textContent = `${(deep.chain || []).length} steps, microseconds → minute`;
  $("deep-chain").innerHTML = (deep.chain || []).map((step) => {
    const chips = (step.feeds || []).map((f) => `<span class="chip">${escapeHtml(String(f))}</span>`).join("");
    const feeds = chips ? ` <span class="feeds">→ ${chips}</span>` : "";
    return `<li><div class="step-h"><b>${escapeHtml(String(step.name))}</b>` +
      `<span class="mono val">${escapeHtml(String(step.value))} <span class="muted">${escapeHtml(String(step.unit || ""))}</span></span>` +
      `<span class="band">${escapeHtml(String(step.timescale || ""))}</span></div>` +
      `<div class="mono formula">${escapeHtml(String(step.formula || ""))}</div>` +
      `<div class="reads">${escapeHtml(String(step.reads || ""))}${feeds}</div></li>`;
  }).join("");
}

/* The freshness contract: the prediction may never be older than the
   maximum age the backend publishes.  The chip is the visible proof, and the
   age is advanced locally between server messages so it ticks like a clock. */
function renderFreshness(prediction) {
  const chip = $("w-fresh");
  if (!chip) return;
  const p = prediction || state.prediction;
  if (!p || !p.computed_at) {
    chip.textContent = "—";
    chip.className = "fresh-chip";
    return;
  }
  const maxAge = p.max_age_seconds || state.clock?.freshnessMaxAge || 65;
  let age = typeof p.age_seconds === "number" ? p.age_seconds : null;
  if (age !== null && typeof state.predictionAnchoredServerMs === "number") {
    // Advanced on the *server* clock, the same one the countdown runs on.
    age += Math.max(0, (serverNowMs() - state.predictionAnchoredServerMs) / 1000);
  }
  if (age === null) {
    chip.textContent = "—";
    chip.className = "fresh-chip";
    return;
  }
  const stale = age > maxAge;
  // Round L.1: the chip is static for the whole window - a ticking "updated
  // Ns ago" inside the prediction cell read as "the prediction is changing".
  // It names the lock instant and only ever flips to STALE (red) if the
  // engine misses a boundary; the age is in the tooltip.
  const lockedAt = fmtClockUs(p.horizon?.released_at_precise || p.horizon?.released_at || p.computed_at).slice(0, 8);
  const override = state.signal?.is_emergency_override || state.lockState === "EMERGENCY_OVERRIDE";
  chip.textContent = override
    ? `⚡ emergency re-lock ${lockedAt}Z`
    : stale
      ? `STALE ${Math.round(age)}s > ${Math.round(maxAge)}s`
      : `🔒 locked ${lockedAt}Z`;
  chip.className = "fresh-chip " + (override ? "override" : stale ? "stale" : "live");
  chip.title = `computed ${p.computed_at} · age ${Math.round(age)}s · max ${Math.round(maxAge)}s · expires ${p.expires_at || "—"}`;
}

/* The reasoning bullets: what supports the side (▸) and what argues against
   it (▾).  Rendered verbatim from the API - the client never composes a case
   of its own. */
function renderReasoning(prediction, targetId) {
  const list = $(targetId);
  if (!list) return;
  const reasoning = prediction?.reasoning;
  const bullets = reasoning?.bullets || [];
  list.innerHTML = "";
  if (!bullets.length) {
    const li = document.createElement("li");
    li.textContent = "reasoning unavailable for this window";
    list.appendChild(li);
    return;
  }
  // Most important first: supporters, then the counterpoints, capped so the
  // cell keeps its height.
  const ordered = [...bullets].sort((a, b) => Number(b.supports) - Number(a.supports));
  ordered.slice(0, 6).forEach((b) => {
    const li = document.createElement("li");
    li.className = b.supports ? "" : "against";
    li.textContent = b.text;
    list.appendChild(li);
  });
}

function renderPipelineProgress() {
  if (!$("w-next")) return;
  const w = state.window || {};
  const fill = $("w-pipeline-fill");
  const ready = !!w.prefetch_ready;
  const lead = state.config?.lock_deadline_seconds || 8;
  const period = state.cyclePeriod || 60;
  // Progress of the *next* window's computation: idle until the prefetch lead
  // window opens (the engine deliberately computes late so its snapshot is
  // fresh), then 0 -> 100 %.
  const remaining = windowRemaining();
  const progress = ready ? 1 : clamp((lead - remaining) / lead, 0, 1);
  fill.style.width = `${Math.round(progress * 100)}%`;
  fill.classList.toggle("ready", ready);
  const nextCycle = (state.cycleNumber || 0) + 1;
  const lock = w.lock || {};
  const age = lock.data_age_at_open_seconds;
  const proof = typeof age === "number"
    ? ` · this window was frozen <b>${age.toFixed(1)}s</b> before it opened (agents prepared ${Math.round(lock.agents_lead_seconds || lead)}s early, formulas + crowd re-frozen at the boundary)`
    : "";
  $("w-next").innerHTML = ready
    ? `agents for <b>#${nextCycle}</b> ready — tape, formulas and crowd freeze at the boundary, revealed at 0${proof}`
    : `preparing signal <b>#${nextCycle}</b> … ${Math.round(progress * 100)}%${proof}`;
}

/* ---------------------------------------------------------------- renders */
function degradationLabel(level) {
  return {
    1: "Level 1 · Full",
    2: "Level 2 · No news APIs",
    3: "Level 3 · No AI agents",
    4: "Level 4 · Fallback brain",
    5: "Level 5 · CoinGecko feed",
    6: "Level 6 · Minimal mode",
  }[level] || "—";
}

function renderStatus(status) {
  if (!status) return;
  state.cycleNumber = status.cycle?.cycle_number ?? state.cycleNumber;
  if ($("cycle-number")) $("cycle-number").textContent = state.cycleNumber;
  const lock = status.lock || {};
  state.lockState = lock.state || state.lockState;
  $("degradation").textContent = degradationLabel(status.degradation_level);
  $("foot-status").textContent =
    `cycle ${state.cycleNumber} · ${status.asset} · redis=${status.infrastructure?.redis ?? "?"}` +
    ` · win rate ${fmtPct(status.infrastructure?.win_rate ?? 0)} · ${status.clock?.utc ?? ""}`;
  if (status.clock?.ntp_synced === false) {
    $("foot-status").textContent += " · NTP: system clock";
  }
  anchorWindow(status.window);
  if (status.tape) state.tape = status.tape;
  renderWidgetPanel();
  renderWindowStrip();
  renderWarming(status);
  renderTape();
}

/* Round Z: the tape's provenance, live.  "simulated" is said in red; a feed
   that is still connecting is said as such instead of showing a fake price. */
function renderTape() {
  const chip = $("tape-chip");
  if (!chip || !state.tape) return;
  const t = state.tape;
  chip.classList.remove("sim", "none", "live");
  if (t.source === "simulator") {
    chip.textContent = `SIMULATED TAPE · not live market data`;
    chip.classList.add("sim");
  } else if (!t.source || t.source === "none") {
    // Round AA: say WHY - the last error of every real feed, so "offline"
    // never stands alone.
    const f = (t.feeds && t.feeds.feeds) || t.feeds || {};
    const why = ["binance", "gemini", "kraken", "krakenrest"].map((n) => {
      const r = f[n] || {};
      if (!r.enabled) return null;
      if (r.connected) return `${n}: connected, waiting for data`;
      return `${n}: ${r.last_error ? r.last_error : (r.polling ? "polling" : "connecting")}${r.failures ? ` (×${r.failures})` : ""}`;
    }).filter(Boolean).join(" · ");
    const net = (t.feeds && t.feeds.connectivity) || t.connectivity || null;
    const netLine = net && net.summary && net.summary !== "not probed" ? ` · internet: ${net.summary}` : "";
    chip.textContent = `NO MARKET FEED · ${why || "connecting…"}${netLine}`;
    chip.classList.add("none");
  } else {
    chip.textContent = `live tape · ${t.source} · ${t.btc_ticks} ticks`;
    chip.classList.add("live");
  }
  const net = (t.feeds && t.feeds.connectivity) || null;
  chip.title = `source for ${t.source_age_seconds}s; writes rejected from other feeds: ${JSON.stringify(t.rejected_writes || {})}`
    + (net && net.summary ? `\ninternet probe at start-up: ${net.summary}` : "");
}

/* The port answers before the engine is ready; say so instead of looking dead. */
function renderWarming(status) {
  const banner = $("warming-banner");
  if (!banner || !status) return;
  const ready = status.ready !== false;
  const error = status.start_error || null;
  state.ready = ready;
  state.startError = error;
  if (ready && !error) {
    banner.classList.add("hidden");
    return;
  }
  banner.classList.remove("hidden");
  if (error) {
    $("warming-title").textContent = "⚠️ Warm-up hit a problem — the app is still serving";
    $("warming-text").innerHTML =
      `The engine reported: <code>${escapeHtml(error)}</code>. The dashboard, the API and ` +
      `the WebSocket stay up; market/news panels run in their degraded modes. ` +
      `Check <a href="/api/health">/api/health</a>, then restart with ` +
      `<code>bash run.sh --stop &amp;&amp; bash run.sh --bg</code>.`;
    $("warming-progress").style.width = "100%";
    return;
  }
  const uptime = status.uptime_seconds || 0;
  $("warming-elapsed").textContent = `up ${Math.round(uptime)}s`;
  $("warming-progress").style.width = `${Math.min(96, uptime * 6)}%`;
}

function renderWindowStrip() {
  const w = state.window;
  if (!w || !$("w-window-span")) return;
  $("w-window-span").textContent =
    `${(w.valid_from || "").substr(11, 8)}–${(w.valid_until || "").substr(11, 8)}Z`;
  const bootstrap = !state.signal || state.signal.preview;
  const period = state.clock?.period || state.cyclePeriod || 60;
  const cadence = state.clock?.minuteAligned
    ? `${Math.round(period)}s window, minute-aligned`
    : `${Math.round(period)}s window`;
  const ticks = (state.clock?.ticks || []).map((m) => `t+${Math.round(m.offset_seconds)}s`).join(" · ");
  $("w-clock-note").textContent = !w.pipeline
    ? "SIGNAL_PIPELINE=0 · literal 8-second computing window"
    : `${cadence} · every panel refreshes together${ticks ? ` at ${ticks}` : ""}` +
      (bootstrap ? " · bootstrapping" : ` · window ${state.cycleNumber}`);
  $("lock-state").textContent = state.lockState;
}

function renderAssetToggle() {
  document.querySelectorAll("#asset-toggle .asset").forEach((btn) => {
    const isActive = btn.dataset.asset === state.asset;
    btn.classList.toggle("active", isActive);
    btn.querySelector(".radio").textContent = isActive ? "●" : "○";
  });
  const pending = $("pending-switch");
  if (state.pendingAsset && state.pendingAsset !== state.asset) {
    pending.textContent = `Switching to ${state.pendingAsset} at next cycle…`;
    pending.classList.remove("hidden");
  } else {
    pending.classList.add("hidden");
  }
}

function renderSignal() {
  markPanel("signal");
  const s = state.signal;
  const pending = !s || s.signal === null || s.signal === undefined;

  $("signal-body")?.classList.toggle("hidden", false);
  if (pending) {
    // Sentinel: the layout stays complete, the cells explain the wait.
    if ($("signal-note")) $("signal-note").textContent = "computing the first window…";
    if ($("signal-cycle-badge")) $("signal-cycle-badge").textContent = "cycle —";
    if ($("signal-frozen")) $("signal-frozen").textContent = "";
    renderWidgetPanel();
    renderConvictionBox();
    return;
  }

  if (state.signal.prediction) {
    state.prediction = state.signal.prediction;
    state.predictionAnchoredAt = Date.now();
  }
  renderFreshness(state.prediction);
  renderReasoning(state.prediction, "reasoning-list");

  const badge = $("signal-badge");
  if (badge) {
    badge.textContent = s.signal;
    badge.className = "signal-badge " + String(s.signal).toLowerCase();
  }
  if ($("signal-icon")) $("signal-icon").textContent = s.lock_icon || (s.is_emergency_override ? "⚡" : "🔒");
  if ($("signal-note")) {
    $("signal-note").textContent = s.is_emergency_override
      ? "emergency override — exit side locked in"
      : "locked for this window";
  }
  if ($("signal-cycle-badge")) $("signal-cycle-badge").textContent = `cycle ${s.cycle_number}`;
  if ($("signal-frozen")) {
    const computed = (s.computed_at || "").substr(11, 8);
    $("signal-frozen").textContent = computed
      ? `computed ${computed}Z · valid ${(s.valid_from || "").substr(11, 8)}–${(s.valid_until || "").substr(11, 8)}Z`
      : "";
  }
  if ($("confidence-value")) $("confidence-value").textContent = fmtPct(s.confidence);
  const bar = $("confidence-bar");
  if (bar) {
    bar.style.width = `${Math.round((s.confidence || 0) * 100)}%`;
    bar.className = "bar-fill " + (s.signal === "BUY" ? "" : "neg");
  }
  if ($("reasoning")) $("reasoning").textContent = s.reasoning || "";

  if ($("weights")) {
    const w = s.fusion?.weights_used || {};
    const lw = s.fusion?.lock_weights || {};
    // Round AA: the share each voter actually got, with the ledger's
    // reliability multiplier when it differs from 1 (x1.40 = a 70% source).
    const parts = Object.keys(w).map((k) => {
      const row = lw[k] || {};
      const rel = Number(row.reliability);
      const tag = Number.isFinite(rel) && Math.abs(rel - 1) >= 0.05 ? ` (x${rel.toFixed(2)} earned)` : "";
      const avail = Number(row.availability);
      const live = Number.isFinite(avail) && avail < 0.999 ? ` ${Math.round(avail * 100)}% live` : "";
      return `${k} ${fmtPct(w[k])}${tag}${live}`;
    });
    const crowdCut = Number(s.fusion?.crowd_adjustment ?? 1);
    const crowd = ` · crowd: confidence x${crowdCut.toFixed(2)} (a modifier of at most 12.5%, never a vote)`;
    $("weights").textContent = parts.length
      ? `who decides: ${parts.join(" · ")}${crowd} · window #${s.cycle_number} · frozen at the boundary`
      : `window #${s.cycle_number}`;
  }

  // The conviction note replaces the old HOLD box: same slot, but it always
  // names a side and says how much size to give it.
  const hw = $("conviction-note");
  if (hw) {
    const note = state.convictionNote || s.conviction_note;
    if (note && note.visible !== false) {
      hw.classList.remove("hidden");
      $("hw-text").textContent = `“${note.text}”`;
      const bits = [];
      if (note.direction) bits.push(`side ${note.direction}`);
      if (note.conviction) bits.push(`conviction ${note.conviction}`);
      if (typeof note.edge === "number") bits.push(`edge ${fmtSigned(note.edge, 2)}`);
      bits.push(`confidence ${fmtPct(note.confidence || s.confidence || 0)}`);
      if (note.source) bits.push(note.source);
      $("hw-lean").textContent = bits.join(" · ");
    } else {
      hw.classList.add("hidden");
    }
  }

  renderHedge(s);
  renderNews(s);
  if (!state.inRenderAll) renderWidgetPanel();
  renderConvictionBox();
  if ($("price")) $("price").textContent = fmtMoney(s.price);
}

/* ------------------------------------------------- Round Y: branches */
/* The realised path of the current window: one point per PULSE/EMOTION
   message (the backend's schedule, no client timer).  Reset when the cycle
   number changes, so the overlay always belongs to the fan it is drawn on. */
function recordPath(price, cycleNumber) {
  if (!price || !Number.isFinite(Number(price))) return;
  if (cycleNumber !== undefined && cycleNumber !== state.pathCycle) {
    state.pathCycle = cycleNumber;
    state.path = [];
  }
  state.path = state.path || [];
  const started = state.clock?.startedAtMs;
  const t = started ? (serverNowMs() - started) / 1000 : null;
  state.path.push({ t, p: Number(price), at: serverNowMs() });
  if (state.path.length > 400) state.path.shift();
}

function renderBranches() {
  const canvas = $("w-branches");
  if (!canvas) return;
  const br = state.prediction?.branches;
  const note = $("w-branch-note");
  const odds = $("w-branch-odds");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!br || !br.available || !br.fan?.length) {
    if (note) note.textContent = "waiting for the first lock…";
    if (odds) odds.textContent = "";
    return;
  }
  const fan = br.fan;
  const T = br.horizon_seconds || 60;
  const path = state.path || [];
  const t0 = path.length ? path[0].at : null;
  const pts = path.map((r) => ({ t: r.t ?? (t0 ? (r.at - t0) / 1000 : 0), p: r.p })).filter((r) => r.t >= 0 && r.t <= T);
  let lo = Math.min(...fan.map((r) => r.q5)), hi = Math.max(...fan.map((r) => r.q95));
  for (const r of pts) { lo = Math.min(lo, r.p); hi = Math.max(hi, r.p); }
  const pad = (hi - lo) * 0.08 || 1;
  lo -= pad; hi += pad;
  const X = (t) => 44 + (t / T) * (W - 56);
  const Y = (p) => H - 18 - ((p - lo) / (hi - lo)) * (H - 30);
  const band = (a, b, colour) => {
    ctx.beginPath();
    fan.forEach((r, i) => (i ? ctx.lineTo(X(r.t), Y(r[a])) : ctx.moveTo(X(r.t), Y(r[a]))));
    [...fan].reverse().forEach((r) => ctx.lineTo(X(r.t), Y(r[b])));
    ctx.closePath();
    ctx.fillStyle = colour;
    ctx.fill();
  };
  const buy = (state.prediction?.side || state.signal?.signal) === "BUY";
  const tint = buy ? "61,220,132" : "255,92,92";
  band("q5", "q95", `rgba(${tint},0.10)`);
  band("q25", "q75", `rgba(${tint},0.18)`);
  const line = (key, colour, width, dash) => {
    ctx.beginPath();
    ctx.setLineDash(dash || []);
    fan.forEach((r, i) => (i ? ctx.lineTo(X(r.t), Y(r[key])) : ctx.moveTo(X(r.t), Y(r[key]))));
    ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.stroke();
    ctx.setLineDash([]);
  };
  line("q50", `rgba(${tint},0.9)`, 1.5, [4, 3]);
  const risk = state.signal?.risk || {};
  const rule = (price, colour, label) => {
    if (!price) return;
    const y = Y(price);
    if (y < 0 || y > H) return;
    ctx.beginPath(); ctx.moveTo(44, y); ctx.lineTo(W - 12, y);
    ctx.strokeStyle = colour; ctx.lineWidth = 1; ctx.setLineDash([2, 3]); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = colour; ctx.font = "10px monospace"; ctx.fillText(label, 46, y - 3);
  };
  rule(risk.take_profit, "rgba(61,220,132,0.8)", `TP ${fmtMoney(risk.take_profit)}`);
  rule(risk.stop_loss, "rgba(255,92,92,0.8)", `SL ${fmtMoney(risk.stop_loss)}`);
  rule(br.entry, "rgba(255,255,255,0.35)", `entry ${fmtMoney(br.entry)}`);
  if (pts.length > 1) {
    ctx.beginPath();
    pts.forEach((r, i) => (i ? ctx.lineTo(X(r.t), Y(r.p)) : ctx.moveTo(X(r.t), Y(r.p))));
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2; ctx.stroke();
    const last = pts[pts.length - 1];
    ctx.beginPath(); ctx.arc(X(last.t), Y(last.p), 3, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill();
  }
  ctx.fillStyle = "rgba(255,255,255,0.45)"; ctx.font = "10px monospace";
  ctx.fillText(fmtMoney(hi), 2, 10); ctx.fillText(fmtMoney(lo), 2, H - 4);
  [0, 15, 30, 45, 60].filter((t) => t <= T).forEach((t) => ctx.fillText(`${t}s`, X(t) - 6, H - 4));
  if (note) {
    note.textContent = `drift ${Number(br.drift_bps).toFixed(1)} bps · σ ${Number(br.sigma_bps).toFixed(1)} bps · ` +
      `P(close ${buy ? "up" : "down"}) ${fmtPct(br.p_close_for)} · P(target before stop) ${fmtPct(br.p_tp_first)}` +
      ` · 5/25/50/75/95 % bands, realised path in white`;
  }
  if (odds) {
    odds.innerHTML = (br.branches || []).map((b) => {
      const cls = b.name === "target reached" ? "pos" : b.name === "stop hit" ? "neg" : "";
      return `<span class="${cls}">${b.name} <b>${fmtPct(b.p)}</b> <span class="muted">(${b.move_bps > 0 ? "+" : ""}${Number(b.move_bps).toFixed(1)} bps)</span></span>`;
    }).join("");
  }
}

/* --------------------------------------------- Round Y: synesthesia */
/* Hear the book.  Off by default; a click starts a WebAudio oscillator whose
   pitch follows the depth imbalance (bids heavier = higher) and whose loudness
   follows tape activity (tick surge).  The page glow follows the dominant
   emotion's intensity.  Purely a rendering of numbers already on screen. */
const synesthesia = {
  ctx: null, osc: null, gain: null, on: false,
  toggle() {
    this.on = !this.on;
    const btn = $("synesthesia-toggle");
    if (this.on) {
      try {
        this.ctx = this.ctx || new (window.AudioContext || window.webkitAudioContext)();
        this.osc = this.ctx.createOscillator();
        this.gain = this.ctx.createGain();
        this.osc.type = "sine";
        this.osc.frequency.value = 220;
        this.gain.gain.value = 0.0;
        this.osc.connect(this.gain).connect(this.ctx.destination);
        this.osc.start();
        if (this.ctx.state === "suspended") this.ctx.resume();
      } catch (e) { this.on = false; }
    } else if (this.osc) {
      try { this.osc.stop(); } catch (e) { /* already stopped */ }
      this.osc = null;
    }
    if (btn) { btn.textContent = this.on ? "🔊 sound on" : "🔈 sound off"; btn.classList.toggle("on", this.on); }
  },
  update(emotions) {
    const f = emotions?.features || {};
    const dom = emotions?.dominant || {};
    const glow = Math.max(0, Math.min(1, Number(dom.intensity || 0)));
    const tone = dom.tone === "negative" ? "255,92,92" : dom.tone === "positive" ? "61,220,132" : "120,140,255";
    document.body.classList.add("glow");
    document.body.style.boxShadow = `inset 0 0 ${Math.round(40 + 120 * glow)}px rgba(${tone},${(0.04 + 0.16 * glow).toFixed(3)})`;
    if (!this.on || !this.osc) return;
    const imb = Math.max(-1, Math.min(1, Number(f.depth_imbalance || 0)));
    const surge = Math.max(0, Math.min(3, Number(f.tick_surge || 0)));
    const now = this.ctx.currentTime;
    this.osc.frequency.linearRampToValueAtTime(220 * Math.pow(2, imb), now + 0.4);
    this.gain.gain.linearRampToValueAtTime(0.02 + 0.06 * (surge / 3), now + 0.4);
  },
};

/* ============================ WIDGET PANEL ==============================
   Row 1: prediction | countdown (1-60) | signal + conviction box
   Row 2: take profit & stop loss | prediction accuracy
   Row 3: probability branches (Round Y)
   ======================================================================== */
function renderWidgetPanel() {
  markPanel("window");
  const s = state.signal;
  const has = s && s.signal;
  const prediction = has ? s.signal : "·  ·  ·";
  const el = $("w-prediction");
  if (!el) return;

  const changed = state.lastPrediction !== null && state.lastPrediction !== prediction;
  el.textContent = prediction;
  el.className = "prediction-value " +
    (has ? String(s.signal).toLowerCase() : "pending");
  if (changed && has) {
    el.classList.remove("pop");
    void el.offsetWidth;              // restart the animation
    el.classList.add("pop");
  }
  state.lastPrediction = has ? prediction : null;

  if (s && s.prediction) {
    state.prediction = s.prediction;
    state.predictionAnchoredAt = Date.now();
  }
  renderFreshness(state.prediction);
  renderHorizon(state.prediction);
  renderMicro(state.prediction);
  renderPredictionDetail(state.prediction);
  renderReasoning(state.prediction, "w-reasoning");

  const w = state.window || {};
  $("w-window-label").textContent = has ? `window #${s.cycle_number}` : "starting up";
  $("w-prediction-sub").textContent = !has
    ? "the first window is being computed"
    : (s.preview
        ? "bootstrap window — computed at the boundary"
        : `computed ${(s.computed_at || "").substr(11, 8)}Z · confidence ${fmtPct(s.confidence)}`);

  // ---- row 2: risk ------------------------------------------------------
  const risk = s?.risk || {};
  $("w-entry").textContent = fmtMoney(risk.entry || s?.price);
  const tradeable = !!risk.tradeable;
  $("w-tp").textContent = tradeable ? fmtMoney(risk.take_profit) : "—";
  $("w-sl").textContent = tradeable ? fmtMoney(risk.stop_loss) : "—";
  $("w-sl").className = tradeable ? "neg" : "muted";
  $("w-tp").className = tradeable ? "pos" : "muted";
  $("w-rr").textContent = tradeable && risk.rr ? `${Number(risk.rr).toFixed(2)} : 1` : "—";
  $("w-vol").textContent = risk.volatility_bps ? `${Number(risk.volatility_bps).toFixed(1)} bps` : "—";
  $("w-levels").textContent = tradeable
    ? `${Math.round(risk.tp_bps)} / ${Math.round(risk.sl_bps)} bps`
    : "no position";
  $("w-levels").className = tradeable ? "" : "muted";
  $("w-risk-note").textContent = risk.note || "—";

  // ---- row 2: accuracy --------------------------------------------------
  // Frozen with the rest of the panel (Round M.4): scored outcomes arrive a
  // moment after the boundary and the per-side rates with the t+15 pulse; a
  // cell that re-paints mid-window inside the prediction panel reads as "the
  // prediction changed".  The row is painted in the first 3 s of a window
  // (the boundary settle) and then left alone until the next one.
  const windowId = state.clock?.windowId;
  const sinceOpen = state.clock?.startedAtMs ? (serverNowMs() - state.clock.startedAtMs) / 1000 : 0;
  if (state.accuracyWindowId === windowId && sinceOpen > 3) return;
  state.accuracyWindowId = windowId;
  const rows = state.outcomes || [];
  // Round AM: the rate is over DECIDED windows; flat windows (no print or a
  // move inside the spread) are shown beside it, never counted as losses.
  const decidedRows = rows.filter((r) => r.outcome !== 0);
  const flats = rows.length - decidedRows.length;
  const wins = decidedRows.filter((r) => r.outcome > 0).length;
  const winRate = decidedRows.length ? wins / decidedRows.length : null;
  let streak = 0;
  for (let i = rows.length - 1; i >= 0; i -= 1) {
    if (rows[i].outcome === 0) break;
    if (streak === 0) { streak = rows[i].outcome > 0 ? 1 : -1; continue; }
    if ((rows[i].outcome > 0 ? 1 : -1) === streak) streak += streak > 0 ? 1 : -1;
    else break;
  }
  $("w-winrate").textContent = winRate === null ? "—" : fmtPct(winRate);
  $("w-samples").textContent = rows.length ? `${decidedRows.length} · ${flats} flat` : "0";
  const guard = (state.prediction && state.prediction.accuracy && state.prediction.accuracy.edge_guard) || {};
  if ($("w-guard")) {
    $("w-guard").textContent = guard.inverted
      ? `⇄ INVERTED — ${guard.note || "the raw side has been losing; publishing the opposite"}`
      : (guard.note || "arming");
    $("w-guard").className = guard.inverted ? "neg" : "muted";
  }
  $("w-streak").textContent = streak === 0 ? "—" : (streak > 0 ? `${streak} win${streak > 1 ? "s" : ""}` : `${-streak} loss${streak < -1 ? "es" : ""}`);
  $("w-streak").className = streak > 0 ? "pos" : streak < 0 ? "neg" : "muted";
  const last = rows[rows.length - 1];
  $("w-last-outcome").textContent = last
    ? `${last.outcome > 0 ? "WIN" : last.outcome < 0 ? "LOSS" : "FLAT"} ${fmtSigned(last.pnl_bps, 1)} bps`
    : "waiting for the first window to close";
  $("w-last-outcome").className = last ? (last.outcome > 0 ? "pos" : last.outcome < 0 ? "neg" : "muted") : "muted";

  // Per-side hit rates: which direction the engine has actually been getting
  // right, and how long a prediction is given before it is scored.
  const livePrediction = state.prediction || {};
  const accuracy = livePrediction.accuracy || {};
  const perSide = accuracy.per_side || {};
  const sideRate = (side) => {
    const row = perSide[side];
    if (!row || !row.evaluated) return "—";
    return `${fmtPct(row.win_rate)} of ${row.evaluated}`;
  };
  $("w-buy-rate").textContent = sideRate("BUY");
  $("w-sell-rate").textContent = sideRate("SELL");
  $("w-horizon").textContent = livePrediction.horizon_seconds
    ? `${Math.round(livePrediction.horizon_seconds)}s`
    : (state.window?.window_seconds ? `${Math.round(state.window.window_seconds)}s` : "—");

  const src = state.tape?.source || state.config?.market_source || "live feed";
  const engine = (state.tape ? state.tape.simulated : state.config?.simulated)
    ? `simulated market data · ${src}`
    : `${src} · ${state.config?.cycle_period_seconds || 60}s windows`;
  $("w-engine").textContent =
    `${engine}${state.window?.pipeline ? " · pipelined" : ""}` +
    ` · ${Math.round(state.window?.window_seconds || state.cyclePeriod || 60)}s predictions` +
    ` · build ${state.config?.build || "?"}`;
  renderBranches();
}

/* ---- lock watchdog ----------------------------------------------------------
   The page checks its own promise.  Three seconds into a window it snapshots
   the text of the prediction cell and the TP/SL cell; every safety-net tick
   until the boundary it compares.  A difference is a broken lock: the chip
   turns red, the diff goes to the console, and the note under the panel says
   so.  Otherwise the note counts the windows it has verified - a number that
   moves only at boundaries. */
const lockWatch = { windowId: null, snapshot: null, verified: 0, broken: 0, lastDiff: "" };
function lockCellText() {
  const cell = document.querySelector(".cell-prediction");
  const tpsl = $("w-entry")?.closest(".widget-cell");
  return `${cell ? cell.textContent : ""}\u0001${tpsl ? tpsl.textContent : ""}`.replace(/\s+/g, " ");
}
function lockWatchdog() {
  const windowId = state.clock?.windowId;
  if (windowId === undefined || windowId === null || !state.clock?.startedAtMs) return;
  const sinceOpen = (serverNowMs() - state.clock.startedAtMs) / 1000;
  if (lockWatch.windowId !== windowId) {
    // New window: the previous one closed without a diff -> verified.
    if (lockWatch.snapshot !== null) lockWatch.verified += 1;
    lockWatch.windowId = windowId;
    lockWatch.snapshot = null;
    renderLockProof();
    return;
  }
  if (sinceOpen < 3) return;                       // boundary settle
  const now = lockCellText();
  if (lockWatch.snapshot === null) { lockWatch.snapshot = now; return; }
  const override = state.signal?.is_emergency_override || state.lockState === "EMERGENCY_OVERRIDE";
  if (override && lockWatch.overrideWindow !== windowId) {
    // The one documented way a lock changes mid-window (Section 4.3): a
    // critical event flattens the open side.  That is a re-lock, not a
    // broken lock - record it as such and watch the new text from here on.
    lockWatch.overrideWindow = windowId;
    lockWatch.overrides = (lockWatch.overrides || 0) + 1;
    lockWatch.snapshot = now;
    renderLockProof();
    return;
  }
  if (now !== lockWatch.snapshot) {
    lockWatch.broken += 1;
    let i = 0;
    while (i < now.length && now[i] === lockWatch.snapshot[i]) i += 1;
    lockWatch.lastDiff = `at ${sinceOpen.toFixed(1)}s: "${lockWatch.snapshot.slice(Math.max(0, i - 30), i + 40)}" -> "${now.slice(Math.max(0, i - 30), i + 40)}"`;
    console.error("LOCK BROKEN inside window", windowId, lockWatch.lastDiff);
    lockWatch.snapshot = now;
    const chip = $("w-fresh");
    if (chip) { chip.textContent = "LOCK BROKEN — see console"; chip.className = "fresh-chip stale"; chip.title = lockWatch.lastDiff; }
    renderLockProof();
  }
}
function renderLockProof() {
  const el = $("w-lock-proof");
  if (!el) return;
  if (lockWatch.broken) {
    el.textContent = `⚠ lock broken ${lockWatch.broken}× — ${lockWatch.lastDiff}`;
    el.className = "lock-proof broken";
  } else {
    const overrides = lockWatch.overrides ? ` · ⚡ ${lockWatch.overrides} emergency re-lock${lockWatch.overrides === 1 ? "" : "s"} (critical news flattened the open side - the only permitted mid-window change)` : "";
    el.textContent = (lockWatch.verified
      ? `🔒 lock verified: the prediction cell did not change inside the last ${lockWatch.verified} window${lockWatch.verified === 1 ? "" : "s"}`
      : "🔒 lock watchdog armed — checks the cell every 5 s, reports at the boundary") + overrides;
    el.className = "lock-proof";
  }
}

/* The conviction box: the signal, its conviction and the size it implies.
   Every window is BUY or SELL - there is no third state to wait on - so this
   box explains *how much* to trust the side rather than whether to trade.
   It is small and inline: never an overlay.  The state class is `override`
   (scoped as `.hold-box.override`).  It must not be called `emergency`: that
   bare name used to match a leftover full-screen rule in styles.css and
   stretched this box over the whole viewport. */
function renderConvictionBox() {
  const box = $("w-hold");
  if (!box) return;
  const s = state.signal;
  const emergency = (state.emergencyUntil > Date.now()) || (s && s.is_emergency_override);
  const conviction = s?.conviction || "—";
  const note = state.convictionNote || s?.conviction_note;

  box.classList.remove("override", "neutral");
  if (emergency) {
    box.classList.add("override");
    const exit = s?.signal ? `${s.signal}` : "—";
    $("w-hold-title").textContent = `⚡ EXIT → ${exit}`;
    const headline = state.emergency?.headline || s?.emergency_headline || s?.reasoning || "";
    $("w-hold-text").textContent = s?.closed_signal
      ? `${s.signal} closes the open ${s.closed_signal}: emergency override, flat is the only safe state.`
      : "Emergency override. Exit any open position now.";
    $("w-hold-meta").textContent = headline ? `“${headline}”` : "";
  } else if (s) {
    box.classList.add("neutral");
    $("w-hold-title").textContent = `${s.signal} · ${conviction}`;
    $("w-hold-text").textContent = note?.text
      || s.direction_reason
      || `${s.signal} is live (${String(conviction).toLowerCase()} conviction) — levels are on the left.`;
    $("w-hold-meta").textContent = [
      s.direction_source ? `why: ${s.direction_source}` : null,
      s.edge !== undefined ? `edge ${fmtSigned(s.edge || 0, 2)}` : null,
      `confidence ${fmtPct(s.confidence || 0)}`,
    ].filter(Boolean).join(" · ");
  } else {
    box.classList.add("neutral");
    $("w-hold-title").textContent = "…";
    $("w-hold-text").textContent = "waiting for the first window";
    $("w-hold-meta").textContent = "";
  }

  // Inline emergency chip under the prediction (item 2).
  const chip = $("w-emergency");
  chip.classList.toggle("hidden", !emergency);
  if (emergency) {
    const headline = state.emergency?.headline || s?.emergency_headline || "";
    const exit = s?.signal ? `exit ${s.signal}` : "exit the position";
    $("w-emergency-text").textContent = headline
      ? `emergency override — ${exit}: “${headline}”`
      : `emergency override — ${exit}`;
  }
}

function renderHedge(s) {
  markPanel("hedge");
  const h = s.hedge || {};
  const o = h.outcomes || {};
  const regime = o.regime || {};
  const stress = h.stress === "high" ? "high stress" : "";
  $("hedge-regime").textContent = regime.label ? `${regime.label}${stress ? " · " + stress : ""}` : (h.stress ? `${h.stress} stress` : "—");
  $("hedge-regime").title = regime.detail || "";
  if (!o.valid) {
    $("hedge-best").textContent = o.reason ? `outcomes pending — ${o.reason}` : "waiting for the first locked window…";
    ["hj-uu", "hj-ud", "hj-du", "hj-dd"].forEach((id) => { $(id).textContent = "—"; $(id + "-bar").style.width = "0%"; });
    ["hk-beta", "hk-sigma", "hk-spread", "hk-rotation"].forEach((id) => { $(id).textContent = "—"; });
    $("hedge-actions").innerHTML = "";
    $("hedge-scenarios").textContent = "";
    $("hedge-logic").innerHTML = "";
  } else {
    const best = o.best || {};
    $("hedge-best").innerHTML =
      `<b>${escapeHtml(best.action || "")}</b> — ${escapeHtml(best.note || "")} · ` +
      `E <b class="${best.expected_bps >= 0 ? "sig-BUY" : "sig-SELL"}">${fmtSigned(best.expected_bps, 2)} bp</b> ± ${Number(best.sigma_bps).toFixed(2)} · ` +
      `P(profit) <b>${fmtPct(best.p_profit)}</b> · hedging removes ${fmtPct(o.variance_reduction)} of variance`;
    const j = o.joint || {};
    [["hj-uu", j.btc_up_paxg_up], ["hj-ud", j.btc_up_paxg_down], ["hj-du", j.btc_down_paxg_up], ["hj-dd", j.btc_down_paxg_down]].forEach(([id, v]) => {
      $(id).textContent = fmtPct(v);
      $(id + "-bar").style.width = `${Math.round((v || 0) * 100)}%`;
    });
    $("hk-beta").textContent = `ρ ${fmtSigned(o.rho, 3)} · β ${fmtSigned(o.beta_paxg_on_btc, 3)}`;
    $("hk-sigma").textContent = `${Number(o.sigma_btc_bps).toFixed(1)} · ${Number(o.sigma_paxg_bps).toFixed(1)} bp`;
    $("hk-spread").textContent = `${fmtSigned((o.spread || {}).z, 2)} — ${(o.spread || {}).read || ""}`;
    $("hk-rotation").textContent = regime.rotation || "—";
    $("hedge-actions").innerHTML = (o.actions || []).map((a) =>
      `<tr class="${a.action === best.action ? "best" : ""}"><td>${escapeHtml(a.action)}<div class="muted">${escapeHtml(a.note || "")}</div></td>` +
      `<td class="mono ${a.expected_bps >= 0 ? "sig-BUY" : "sig-SELL"}">${fmtSigned(a.expected_bps, 2)}</td>` +
      `<td class="mono">${Number(a.sigma_bps).toFixed(2)}</td><td class="mono">${fmtPct(a.p_profit)}</td></tr>`).join("");
    $("hedge-scenarios").innerHTML = (o.scenarios || []).map((sc) =>
      `<span class="scenario">if ${escapeHtml(sc.if)} → ${escapeHtml(sc.then)} · best action ${fmtSigned(sc.best_action_pnl_bps, 1)} bp</span>`).join("");
    $("hedge-logic").innerHTML = (o.logic || []).map((l) => `<div>${escapeHtml(l)}</div>`).join("");
  }
  $("brain-status").textContent = s.brain_status || "—";
  $("brain-detail").textContent = `CCSv2 ${fmtSigned(s.ccs_value)} @ ${fmtPct(s.ccs_confidence)}`;
  $("drg-value").textContent = fmtSigned(s.drg || 0);
}

function renderNews(s) {
  markPanel("news");
  const n = s.news || {};
  $("news-headline").textContent = n.latest_headline || "No headlines available";
  $("news-source").textContent = n.source ? `${n.source} · Tier ${n.tier}` : "";
  const sent = $("news-sentiment");
  sent.textContent = `sentiment ${fmtSigned(n.niv || 0, 2)}`;
  sent.className = "pill " + pctClass(n.niv || 0);
  $("news-niv").textContent = `NIV ${fmtSigned(n.niv || 0)}`;
  $("news-smd").textContent = `SMD ${fmtSigned(n.smd || 0)}`;
  if (n.last_poll_seconds_ago !== null && n.last_poll_seconds_ago !== undefined) {
    $("news-age").textContent = `last updated ${n.last_poll_seconds_ago}s ago`;
  }
}

function renderNewsList(data) {
  if (!data || data.error) return;
  $("news-coverage").textContent = `coverage ${data.status?.coverage ?? "—"}`;
  const list = $("news-list");
  list.innerHTML = "";
  // Round Z: the wire's net impact on each asset - the number the fusion
  // votes with - and the headlines that produced it.
  const imp = data.impact;
  const impactEl = $("news-impact");
  if (impactEl && imp) {
    const cell = (asset) => {
      const v = Number(imp[asset] || 0);
      const cls = v > 0.05 ? "pos" : v < -0.05 ? "neg" : "";
      const top = (imp.drivers?.[asset] || [])[0];
      return `<span class="impact-cell"><span class="muted">${asset}</span> <b class="${cls}">${fmtSigned(v, 2)}</b>` +
        (top ? `<span class="muted"> ← ${escapeHtml(top.theme)}</span>` : "") + `</span>`;
    };
    const themes = Object.entries(imp.themes || {}).sort((a, b) => b[1].headlines - a[1].headlines).slice(0, 3)
      .map(([t, v]) => `${t}×${v.headlines}${v.habituation < 0.9 ? ` (${Math.round(v.habituation * 100)}% fresh)` : ""}`).join(", ");
    impactEl.innerHTML = `${cell("BTC")} ${cell("PAXG")} <span class="muted">· ${imp.classified || 0} themed, ${imp.world_items || 0} world` +
      `${imp.duplicates ? `, ${imp.duplicates} duplicates ignored` : ""}${themes ? ` · ${escapeHtml(themes)}` : ""}</span>`;
    impactEl.title = "weight = tier × age decay × novelty (1/k-th headline on the theme × habituation: halves every 2 h a theme stays on the wire)";
  }
  (data.items || []).forEach((item) => {
    const row = document.createElement("div");
    row.className = "news-item";
    const sentClass = item.sentiment > 0.1 ? "pos" : item.sentiment < -0.1 ? "neg" : "";
    const ii = item.impact;
    const tag = ii && ii.theme !== "neutral"
      ? `<span class="impact-tag ${ii.scope}" title="${escapeHtml(`matched “${ii.matched}” → BTC ${ii.btc >= 0 ? "+" : ""}${ii.btc}, PAXG ${ii.paxg >= 0 ? "+" : ""}${ii.paxg}`)}">` +
        `${escapeHtml(ii.label)} · BTC ${ii.btc > 0 ? "▲" : ii.btc < 0 ? "▼" : "·"} PAXG ${ii.paxg > 0 ? "▲" : ii.paxg < 0 ? "▼" : "·"}</span>`
      : "";
    row.innerHTML =
      `<span class="tier">T${item.tier}</span>` +
      `<span style="flex:1">${escapeHtml(item.headline)}<br><span class="muted">${escapeHtml(item.source)} · ${Math.round(item.age_seconds)}s ago</span> ${tag}</span>` +
      `<span class="sent ${sentClass}">${fmtSigned(item.sentiment, 2)}</span>`;
    list.appendChild(row);
  });
}

async function refreshNewsList() {
  renderNewsList(await getJSON("/api/news?limit=5"));
}

function renderAgents(data) {
  markPanel("agents");
  if (!data || data.error) return;
  const weights = data.weights || {};
  const rows = [
    { key: "drosophila", label: "🧠 Drosophila CCSv2", weight: weights.drosophila },
    { key: "gemini", label: "🤖 Gemini AI", weight: weights.gemini },
    { key: "local", label: "💻 Local model", weight: weights.local },
    { key: "github", label: "🐙 GitHub Models", weight: weights.github },
  ];
  const s = state.signal;
  const list = $("agent-list");
  list.innerHTML = "";
  rows.forEach((row) => {
    // The Drosophila brain is not an external agent: it is always live and its
    // status comes from the brain module, not from /api/agents.
    const info = row.key === "drosophila"
      ? { status: "LIVE", detail: data.brain_status || "mushroom-body circuit" }
      : (data[row.key] || {});
    let cls = "st-disabled";
    if (["ACTIVE", "LIVE"].includes(info.status)) cls = "st-active";
    else if (info.status === "STUB") cls = "st-stub";
    else if (info.status === "ERROR") cls = "st-error";
    else if (["RATE_LIMITED", "TIMEOUT", "INACTIVE"].includes(info.status)) cls = "st-warn";

    let decision = "";
    if (s && row.key !== "drosophila") {
      const a = (s.agents || {})[row.key];
      if (a && a.decision) decision = `${a.decision} (${fmtPct(a.confidence ?? 0)})`;
      else if (a && a.error) decision = a.error.slice(0, 40);
    } else if (s && row.key === "drosophila") {
      decision = `${s.signal} (${fmtPct(s.ccs_confidence)}) · ${s.conviction || "—"}`;
    }
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="agent-name">${row.label}<br><span class="muted">${escapeHtml(info.detail || info.model || "")}</span></span>` +
      `<span class="agent-state ${cls}">${info.status || "—"}</span>` +
      `<span class="muted" style="min-width:120px;text-align:right">${escapeHtml(decision)}</span>` +
      `<span class="muted" style="min-width:44px;text-align:right">${fmtPct(row.weight || 0)}</span>`;
    list.appendChild(li);
  });
  if (data.brain?.status) {
    $("brain-status").textContent = data.brain.status;
  }
}

async function refreshAgents() {
  renderAgents(await getJSON("/api/agents"));
}

/* ---------------- brain wiring card (item 6): where the fly is used ---- */
async function loadWiring() {
  const data = await getJSON("/api/brain/wiring");
  if (!data || data.error) return;
  state.wiring = data;
  renderBrainPipeline();
  renderWiringTable();
}

function renderBrainPipeline() {
  const host = $("brain-pipeline");
  if (!host || !state.wiring) return;
  const stages = state.wiring.stages || [];
  const e = state.explain && state.explain.available ? state.explain : null;
  const live = [
    e ? `${Object.keys(e.all_inputs || {}).length}/20 formulas written onto PNs` : "20 formulas → PNs 0-19",
    e ? `${e.kenyon_cells?.active ?? 0}/${e.kenyon_cells?.of ?? 50} KC clusters active (top 10%)` : "50 KC clusters, ReLU + top-10 % sparsity",
    e ? `DRG ${fmtSigned(e.dopamine?.drg ?? 0)} → PAM · HSI ${fmtSigned(e.dopamine?.hsi ?? 0)} → OA` : "DRG → PAM/PPL1 · HSI → OA",
    e ? `approach ${fmtSigned(e.mbons?.approach ?? 0)} · avoid ${fmtSigned(e.mbons?.avoid ?? 0)} · conf ${fmtPct(e.mbons?.confidence ?? 0)}` : "4 MBONs, 3 conv layers, gain 3.2802",
    e ? `CCSv2 ${fmtSigned(e.ccs_value)} @ ${fmtPct(e.ccs_confidence)} → 40% of fusion` : "CCSv2 + KCAE → fusion (0.40 weight)",
  ];
  host.innerHTML = "";
  stages.forEach((stage, i) => {
    const div = document.createElement("div");
    div.className = "brain-stage";
    div.innerHTML =
      `<div class="idx">${i + 1}</div>` +
      `<div><h4>${escapeHtml(stage.name)}</h4>` +
      `<p>${escapeHtml(stage.role)}</p>` +
      `<div class="detail">${escapeHtml(stage.detail)}</div>` +
      `<div class="detail live">▸ ${escapeHtml(live[i] || "")}</div></div>`;
    host.appendChild(div);
  });
}

function renderWiringTable() {
  const host = $("brain-wiring");
  if (!host || !state.wiring) return;
  const rows = (state.wiring.projection_neurons || []).map((p) =>
    `<div class="wiring-row"><span><span class="pn">PN ${p.pn}</span> ${escapeHtml(p.formula)}</span>` +
    `<span class="muted">${escapeHtml(p.brain_node || "")}</span></div>`).join("");
  host.innerHTML =
    `<div class="card-head"><h2 style="font-size:13px">Formula → neuron map</h2>` +
    `<span class="muted">${(state.wiring.projection_neurons || []).length} projection neurons · ` +
    `fusion weight ${state.wiring.fusion_weight}</span></div>` +
    `<div class="wiring-table">${rows}</div>` +
    `<p class="muted" style="margin-bottom:0">Same circuit, once per window: formulas → PNs → KC sparse code ` +
    `→ MBONs → lateral horn → CCSv2/KCAE → ${Math.round((state.wiring.fusion_weight || 0.4) * 100)}% of the fused decision.</p>`;
}

function renderBrainExplain(data) {
  markPanel("brain");
  if (!data || data.error) return;
  state.explain = data;
  if (!data.available) {
    $("brain-verdict").textContent = data.detail || "no completed window yet";
    return;
  }
  $("brain-verdict").textContent = data.verdict;
  $("b-dominant").textContent = (data.dominant_inputs || [])
    .slice(0, 4).map((d) => `${d.formula} ${fmtSigned(d.value, 3)}`).join(" · ") || "—";
  $("b-kc").textContent = `${data.kenyon_cells?.active ?? "—"} / ${data.kenyon_cells?.of ?? 50} (KCAE ${fmtSigned(data.kenyon_cells?.kcae ?? 0, 3)})`;
  $("b-mbon").textContent = `${fmtSigned(data.mbons?.approach ?? 0)} / ${fmtSigned(data.mbons?.avoid ?? 0)}`;
  $("b-lh").textContent = `appr ${fmtSigned(data.lateral_horn?.approach ?? 0)} · avoid ${fmtSigned(data.lateral_horn?.avoid ?? 0)}`;
  const dan = data.dopamine || {};
  $("b-dan").textContent =
    `DRG ${fmtSigned(dan.drg ?? 0)} → PAM ${fmtSigned(dan.pam ?? 0)} · PPL1 ${fmtSigned(dan.ppl1 ?? 0)}` +
    ` · OA ${fmtSigned(dan.octopamine ?? 0)}`;
  $("b-ccs").textContent = `${fmtSigned(data.ccs_value)} → ${data.weights?.in_fusion ?? 0.4} × weight` +
    (data.weights?.share_of_score !== null && data.weights?.share_of_score !== undefined
      ? ` (${fmtSigned(data.weights.share_of_score)} of the score)`
      : "");
  $("b-source").textContent = `${data.source?.status || "—"} · ${data.source?.message || ""}` +
    ` · checksum ${data.source?.checksum || "—"} · gain ${data.source?.gain ?? "—"}`;
  renderBrainPipeline();
}

/* The physics layer (Round T, rebuilt in Round AJ): the report locked with the
   window on screen.  It travels inside the fusion, so it cannot change
   mid-window either. */
function renderPhysics(payload) {
  markPanel("physics");
  const r = payload && payload.report;
  const card = $("physics-card");
  if (!card) return;
  if (!r) {
    $("ph-verdict").textContent = payload && payload.weight === 0
      ? "disabled (PHYSICS_WEIGHT=0)" : "waiting for the first locked window…";
    return;
  }
  const w = r.weights || {}, comp = r.composite || {};
  const side = r.vote >= 0 ? "BUY" : "SELL";
  const byKey = {};
  (r.mechanisms || []).forEach((m) => { byKey[m.key] = m; });
  $("ph-head").textContent =
    `${payload.locked ? "🔒 locked with this window" : "live"} · weight ${fmtPct(payload.weight)} of fusion · ${r.elapsed_us} µs`;
  $("ph-verdict").innerHTML =
    `<span class="sig-${side}">${side} ${r.asset}</span> vote <b>${Number(r.vote).toFixed(3)}</b> at ${fmtPct(r.confidence)} · ` +
    `Kelly BTC weight <b>${fmtPct(w.w_micro)}</b> from ${comp.voting ?? 0} voting of ${comp.active ?? 0} active mechanisms` +
    `${w.drag > 0 ? ` · temperature drag ${fmtPct(w.drag)}` : ""}${w.below_cost ? " · <span class=\"sig-SELL\">edge below spread cost — vote halved</span>" : ""}`;
  $("ph-bar-fill").style.width = `${Math.round(Number(w.w_micro) * 100)}%`;
  $("ph-bar-micro").style.left = `${Math.round(Number(w.w_micro) * 100)}%`;
  $("ph-bar-text").textContent = `Kelly ${fmtPct(w.w_micro)} · agreement ${fmtPct(comp.agreement)}`;
  const T = byKey.temperature || {}, H = byKey.hawkes || {}, E = byKey.entropy || {}, Kn = byKey.kinetic || {};
  const act = (m, txt) => (m && m.active ? txt : `inactive — ${String((m && m.logic) || "").split(" - ")[0].replace(/^inactive: /, "")}`);
  $("ph-temp").textContent = act(T, `${Number(T.value).toFixed(2)} · σ₆₀ ${Number(T.sigma_60s_bps).toFixed(1)} bp · drag ${fmtPct(T.drag)}`);
  $("ph-hawkes").textContent = act(H, `${Number(H.value).toFixed(3)} · ${Number(H.arrivals_per_s).toFixed(2)} prints/s${H.critical ? " · CRITICAL" : ""}`);
  $("ph-entropy").textContent = act(E, `${Number(E.value).toFixed(3)} (${Number(E.entropy_bits).toFixed(2)} bits) · ${E.informed ? "informed flow" : "noise"}`);
  $("ph-kinetic").textContent = act(Kn, `${Number(Kn.value).toFixed(2)} σ · KE ${Number(Kn.ke_ratio).toFixed(2)}× baseline`);
  $("ph-edge").textContent = `${Number(comp.gross_edge_bps).toFixed(2)} − ${Number(comp.cost_bps).toFixed(2)} = ${Number(comp.net_edge_bps).toFixed(2)} bp`;
  const tele = r.telemetry || {};
  $("ph-live").textContent = `${(r.active || []).length} active · ${Object.keys(r.inactive || {}).length} inactive · gold spot ${tele.xau_usd?.source || "—"} · venues ${tele.venues?.source || "—"}`;
  const body = $("ph-mechanisms");
  body.innerHTML = "";
  (r.mechanisms || []).forEach((m) => {
    const tr = document.createElement("tr");
    if (!m.active) tr.className = "muted";
    const dir = !m.active ? '<span class="muted">inactive</span>'
      : m.direction > 0 ? '<span class="sig-BUY">BTC +</span>' : m.direction < 0 ? '<span class="sig-SELL">PAXG +</span>' : '<span class="muted">regime</span>';
    const unit = m.unit || (m.key === "vpin" ? "" : m.key === "pendulum" ? " REI" : m.key === "ou" ? " dev" : " bp");
    const reading = m.active ? `${Number(m.value).toFixed(3)}${escapeHtml(unit)}` : "—";
    tr.innerHTML = `<td class="muted">${escapeHtml(m.section)}</td><td>${escapeHtml(m.name)}</td>` +
      `<td class="mono">${reading}</td><td>${dir}</td>` +
      `<td class="mono">${m.active ? Number(m.edge_bps || 0).toFixed(2) + " bp" : ""}</td><td class="muted">${escapeHtml(m.active ? String(m.source || "") : String(m.logic || "").replace(/^inactive: /, "").split(";")[0])}</td>`;
    body.appendChild(tr);
  });
  const logic = [
    ...(r.mechanisms || []).map((m) => [`§${m.section} ${m.name}`, m.logic]),
    ["§11 Kelly", (r.kelly || {}).logic], ["§11 vote", w.logic],
  ];
  $("ph-logic").innerHTML = logic.map(([k, v]) =>
    `<div class="ph-logic-row"><b>${escapeHtml(k)}</b><div>${escapeHtml(String(v || ""))}</div></div>`).join("");
  $("ph-note").textContent = comp.note || "";
}

/* Round AL - the Formula Genesis Engine card. */
let gnDomainsLoaded = false;
async function loadGenesisDomains() {
  if (gnDomainsLoaded) return;
  const res = await getJSON("/api/genesis/domains");
  if (!res || res.error || !$("gn-logic")) return;
  gnDomainsLoaded = true;
  $("gn-logic").innerHTML = (res.domains || []).map((d) =>
    `<div class="ph-logic-row"><b>D${d.domain} · ${escapeHtml(d.name)}</b><div class="muted">${escapeHtml(d.description)}</div>` +
    (d.subcategories || []).map((s) =>
      `<div class="gn-sub"><span class="gn-sub-name">${escapeHtml(s.name)}</span> <span class="muted">layer ${s.layer}</span>` +
      `<div>${escapeHtml(s.definition)}</div><div class="muted">${escapeHtml(s.interpretation)}</div></div>`).join("") +
    `</div>`).join("");
}

function renderGenesis(payload) {
  markPanel("genesis");
  const card = $("genesis-card");
  if (!card) return;
  const r = payload && payload.report;
  const st = (payload && payload.status) || {};
  const states = st.states || {};
  const cs = st.candles || {};
  loadGenesisDomains();
  $("gn-pool").textContent = `${st.pool ?? 0} in pool · ${states.ACTIVE || 0} active · ${states.CANDIDATE || 0} candidates · ` +
    `${states.DECAYING || 0} decaying · ${st.graveyard || 0} autopsied · ${st.bred || 0} bred`;
  const nextG = st.next_genesis_in_s;
  $("gn-gen").textContent = `generation ${st.generation ?? 0} · ` +
    (nextG == null ? "genesis after the first scoring" : nextG <= 0 ? "genesis due now" : `next genesis in ${Math.round(nextG / 60)} min`);
  const nextR = st.next_rescore_in_candles;
  $("gn-candles").textContent = `${cs.minutes ?? 0} min (${cs.live_minutes ?? 0} live${cs.bootstrapped_from ? `, history ${cs.bootstrapped_from}` : ""}) · ` +
    (nextR == null ? "first scoring at 240" : `re-score in ${nextR} candles`) +
    (st.rescore_seconds ? ` · last took ${st.rescore_seconds}s` : "");
  if (!r || r.status !== "live") {
    $("gn-head").textContent = payload && payload.weight === 0 ? "disabled (GENESIS_WEIGHT=0)" : `weight ${fmtPct((payload || {}).weight || 0)} of fusion · not voting yet`;
    $("gn-verdict").textContent = (r && r.note) || "warming up — the pool needs 240 closed minutes before its first scoring…";
    $("gn-note").textContent = "NumPy on one worker thread (no GPU / Rust in a Codespace): a full re-score of the pool takes ~1 minute and runs in the background.";
    return;
  }
  const side = r.vote > 0 ? "BUY" : r.vote < 0 ? "SELL" : "NEUTRAL";
  const reg = r.regime || {};
  $("gn-head").textContent = `${payload.locked ? "🔒 locked with this window" : "live"} · weight ${fmtPct(payload.weight)} of fusion · ¶gn minute ${r.candle_ts_ms ? new Date(r.candle_ts_ms).toISOString().slice(11, 16) : "—"}`;
  $("gn-verdict").innerHTML =
    `<span class="sig-${side}">${side} ${escapeHtml(r.asset)}</span> composite <b>${Number(r.vote).toFixed(3)}</b> at ${fmtPct(r.confidence)} · ` +
    `${r.votes_up} formulas up / ${r.votes_down} down · regime <b>${escapeHtml(reg.regime || "")}</b>`;
  $("gn-bar-marker").style.left = `${Math.round((Number(r.vote) + 1) * 50)}%`;
  $("gn-bar-text").textContent = `vote ${Number(r.vote).toFixed(3)} · agreement ${fmtPct(r.agreement)}`;
  $("gn-regime").textContent = `${reg.regime || "—"} · ${reg.note || ""}`;
  $("gn-firing").textContent = `${r.firing} / ${r.gated} / ${r.active}`;
  $("gn-agree").textContent = `${fmtPct(r.agreement)} · confidence ${fmtPct(r.confidence)}`;
  $("gn-domains").innerHTML = (r.domains || []).map((d) => {
    const cls = d.mean > 0.05 ? "sig-BUY" : d.mean < -0.05 ? "sig-SELL" : "muted";
    return `<span class="pill gn-dom" title="${escapeHtml(d.name)}: ${d.firing}/${d.count} firing, mean signal ${Number(d.mean).toFixed(3)}">` +
      `D${d.domain} ${escapeHtml(d.name.split(" ")[0])} <b class="${cls}">${fmtSigned(d.mean, 2)}</b> <span class="muted">×${d.count}</span></span>`;
  }).join("");
  const body = $("gn-top");
  body.innerHTML = "";
  (r.top || []).forEach((f) => {
    const tr = document.createElement("tr");
    const cls = f.signal > 0 ? "sig-BUY" : f.signal < 0 ? "sig-SELL" : "muted";
    tr.innerHTML = `<td class="mono muted">${escapeHtml(f.id)}</td><td>${escapeHtml(f.name)}</td>` +
      `<td class="muted">${f.domain ? "D" + f.domain : "bred"}</td><td class="mono ${cls}">${fmtSigned(f.signal, 3)}</td>` +
      `<td class="mono">${Number(f.raw).toPrecision(3)}</td><td class="mono">${Number(f.fitness).toFixed(3)}</td><td class="muted">${escapeHtml(f.state)}</td>`;
    body.appendChild(tr);
  });
  const prov = payload.providers || {};
  const keyed = Object.entries(prov).filter(([, p]) => p.configured).map(([k, p]) => `${k} ${p.error ? "✗" : `${p.rows} rows`}`);
  $("gn-note").textContent = `${r.note || ""} · &b ${keyed.length ? `keyed providers: ${keyed.join(", ")}` : "public tape only (Glassnode / Twelve Data / LunarCrush activate with keys in Settings)"}`;
}

async function refreshBrainExplain() {
  renderBrainExplain(await getJSON("/api/brain/explain"));
}

function renderBrainStatus(data) {
  if (!data || data.error) return;
  $("brain-status").textContent = data.status;
  $("brain-detail").textContent =
    `${data.health?.message || data.message} · checksum ${data.matrix?.checksum || "—"} · gain ${data.gain}`;
}

async function refreshBrain() {
  renderBrainStatus(await getJSON("/api/brain/status"));
}

function renderHistory(data) {
  markPanel("history");
  if (!data || data.error) return;
  const list = $("history");
  list.innerHTML = "";
  (data.history || []).forEach((s) => {
    const kind = s.is_emergency_override ? "EMERGENCY" : s.signal;
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="sig-${kind}">${(s.is_emergency_override ? "⚡" : "🔒")} ${kind}</span>` +
      `<span style="flex:1">${s.asset} ${fmtPct(s.confidence)}</span>` +
      `<span class="muted">#${s.cycle_number} ${(s.timestamp || "").substr(11, 8)}</span>`;
    list.appendChild(li);
  });
}

async function refreshHistory() {
  renderHistory(await getJSON("/api/signal/history?limit=12"));
}

function renderOutcomes(data) {
  if (!data || data.error) return;
  state.outcomes = data.rows || [];
  if ($("win-rate")) $("win-rate").textContent = `win rate ${fmtPct(data.win_rate)} (${data.count} outcomes)`;
  renderWidgetPanel();
  const list = $("outcomes");
  list.innerHTML = "";
  (data.rows || []).slice().reverse().slice(0, 8).forEach((row) => {
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="${row.outcome > 0 ? "sig-BUY" : row.outcome < 0 ? "sig-SELL" : "sig-neutral"}">` +
      `${row.outcome > 0 ? "WIN" : row.outcome < 0 ? "LOSS" : "FLAT"}</span>` +
      `<span style="flex:1">${fmtSigned(row.pnl_bps, 1)} bps</span>`;
    list.appendChild(li);
  });
}

function wirePredictionDetailToggle() {
  const button = $("w-detail-toggle");
  const box = $("w-detail");
  if (!button || !box) return;
  button.onclick = () => {
    const open = box.classList.toggle("hidden") === false;
    button.textContent = `${open ? "▾" : "▸"} prediction detail`;
  };
}

async function refreshOutcomes() {
  renderOutcomes(await getJSON("/api/signal/outcomes"));
}

function renderTimings(data) {
  if (!data || (!data.timings_ms && !data.timings_us)) return;
  // Microseconds are the real unit: a formula takes tens of µs, and rounding to
  // milliseconds turns most of the 23 into the same number.
  const source = data.timings_us && Object.keys(data.timings_us).length
    ? Object.entries(data.timings_us).map(([k, v]) => `${k} ${fmtUs(v)}`)
    : Object.entries(data.timings_ms).map(([k, v]) => `${k} ${Number(v).toFixed(3)}ms`);
  const top = source.slice(0, 5).join(" · ");
  const total = data.total_us ? fmtUs(data.total_us)
    : (data.total_ms !== undefined ? `${Number(data.total_ms).toFixed(2)}ms` : "?");
  const resolution = data.resolution_us ? ` · tape resolution ${fmtUs(data.resolution_us)}` : "";
  // The live tape (rate, jitter, quote life, aggression) belongs here, with
  // the live formulas - not inside the locked prediction cell.
  const m = state.liveMicro || {};
  const tape = m.resolution_us
    ? ` · live tape ${Number(m.tick_rate_hz || 0).toFixed(1)} Hz · jitter ${fmtUs(m.jitter_us)}` +
      ` · quote life ${fmtUs(m.quote_lifetime_us)} · aggression ${fmtSigned(m.aggression || 0, 2)}`
    : "";
  $("timings").textContent =
    `total formula pass ${total} (budget ${data.budget_ms}ms)${resolution}${tape} · slowest: ${top}`;
}

async function refreshTimings() {
  renderTimings(await getJSON("/api/formulas/timings"));
}

/* ------------------------------------------------------------ formular UI */
async function loadFormulaMeta() {
  const data = await getJSON("/api/formulas");
  if (!data) return;
  state.formulaMeta = data.categories || [];
  state.rewardMeta = data.reward;
  renderFormulas();
}

/* The self-test answers "do the 22 formulas do what they claim?".  It is the
   audit trail behind the numbers, so it is loaded once and shown per formula. */
async function loadSelfTest() {
  const data = await getJSON("/api/formulas/self-test");
  if (!data) return;
  state.selfTest = data;
  renderFormulas();
}

function selfTestVerdict(name) {
  const list = state.selfTest?.formulas;
  if (!Array.isArray(list)) return null;
  return list.find((f) => f.name === name) || null;
}

function renderLiveFormulas(data) {
  if (!data || data.error) return;
  state.liveFormulas = data.formulas || {};
  state.liveReadings = data.readings || {};
  state.liveTraces = data.traces || {};
  state.liveStats = data.stats || {};
  state.liveTimingsUs = data.timings_us || {};
  state.liveChecks = data.checks || {};
  state.liveMicro = data.micro || {};
  state.liveProvenance = data.provenance || state.liveProvenance || {};
  state.livePhase = data.phase || state.livePhase || {};
  state.liveFeedStatus = data.feed_status || state.liveFeedStatus || {};
  if (data.double_check && data.double_check.formulas) {
    const note = $("live-note");
    const dc = data.double_check;
    if (note) {
      note.textContent = `double check: ${dc.verified}/${dc.formulas} verified` +
        (dc.failed ? ` · ${dc.failed} failed (${(dc.failed_names || []).join(", ")}) → zeroed` : "") +
        ` · ${fmtUs(dc.check_us)} · values refresh every 15s · signal stays locked`;
      note.title = dc.rule || "";
    }
  }
  state.liveHistoryWindow = data.history_window || 0;
  if (data.note) {
    const note = $("live-note");
    if (note) note.textContent = "live values " + data.note;
  }
  if (data.timings_ms || data.timings_us) renderTimings(data);
  renderFormulas();
}

async function refreshLiveFormulas() {
  renderLiveFormulas(await getJSON("/api/formulas/live"));
}

function renderFormulas() {
  markPanel("formulas");
  const container = $("formula-explorer");
  const live = Object.keys(state.liveFormulas).length > 0;
  const values = live ? state.liveFormulas : state.formulas;
  const readings = live ? state.liveReadings : state.formulaReadings;
  const traces = live ? state.liveTraces : state.formulaTraces;
  if (!state.formulaMeta.length) return;
  state.readings = readings;
  state.traces = traces;

  container.innerHTML = "";
  state.formulaMeta.forEach((cat) => {
    const collapsed = state.collapsed[cat.key];
    const wrap = document.createElement("div");
    wrap.className = "category";
    const head = document.createElement("div");
    head.className = "category-head";
    head.innerHTML = `<span>${collapsed ? "▶" : "▼"} Category ${cat.key}: ${cat.name}</span>` +
      `<span class="count">${cat.formulas.length} formula${cat.formulas.length > 1 ? "s" : ""}</span>`;
    head.onclick = () => { state.collapsed[cat.key] = !collapsed; renderFormulas(); };
    wrap.appendChild(head);

    if (!collapsed) {
      const body = document.createElement("div");
      body.className = "category-body";
      cat.formulas.forEach((f) => {
        const v = values[f.name];
        body.appendChild(formulaRow(f.name, v, f.description, f, readings, traces));
      });
      container.appendChild(wrap);
      wrap.appendChild(body);
      return;
    }
    container.appendChild(wrap);
  });

  if (state.rewardMeta) {
    const wrap = document.createElement("div");
    wrap.className = "category";
    wrap.innerHTML = `<div class="category-head"><span>Reward learning</span>` +
      `<span class="count">modulates the brain, not a market signal</span></div>`;
    const body = document.createElement("div");
    body.className = "category-body";
    body.appendChild(
      formulaRow("DRG", values.DRG, state.rewardMeta.description, state.rewardMeta,
        readings, traces)
    );
    wrap.appendChild(body);
    container.appendChild(wrap);
  }
}

function formulaRow(name, value, description, meta, readings = {}, traces = {}) {
  const row = document.createElement("div");
  const known = typeof value === "number";
  const v = known ? value : 0;
  let posClass = "neutral", fillClass = "fill-pos";
  if (v > 0.15) { posClass = "pos"; }
  else if (v < -0.15) { posClass = "neg"; fillClass = "fill-neg"; }
  const width = Math.min(50, Math.abs(v) * 50);

  const logic = meta?.logic || null;
  const verdict = selfTestVerdict(name);
  const open = !!state.openLogic[name];
  const reading = readings[name] || logic?.reading || "";
  const stats = state.liveStats[name] || null;
  const timingUs = state.liveTimingsUs?.[name];

  const verdictChip = verdict
    ? `<span class="verdict ${verdict.passed ? "pass" : "fail"}" ` +
      `title="${escapeHtml(verdict.claim + " — " + verdict.detail)}">` +
      `${verdict.passed ? "✔ self-test" : "✘ self-test"}</span>`
    : "";
  // The double check of this very pass: replay on a private state copy,
  // re-derivation from the traced intermediates, range.
  const check = state.liveChecks?.[name];
  const checkChip = check
    ? `<span class="verdict ${check.ok ? "pass" : "fail"}" ` +
      `title="${escapeHtml((check.rule || "") + (check.notes && check.notes.length ? " — " + check.notes.join("; ") : ""))}">` +
      `${check.verdict === "verified" ? "✓✓ double-checked" : check.ok ? "✓ checked" : "✗ check failed → 0"}</span>`
    : "";

  // Round Z: &b - the live feeds behind this value, green only when every one
  // of them is a real connected feed; ¶gn - the window phase of the pass.
  // Round AF: &b is a graded coverage - live / partial n% / simulated /
  // offline - and the tooltip names every feed and WHY it is not live.
  const feed = state.liveFeedStatus?.[name];
  const phase = state.livePhase || {};
  const src = state.liveProvenance?.source || "";
  const feedChip = feed ? (() => {
    const grade = feed.grade || (feed.live ? "live" : (src === "simulator" ? "simulated" : "offline"));
    const cls = grade === "live" ? "pass" : grade === "partial" ? "warn" : "fail";
    const pct = Math.round((feed.coverage ?? (feed.live ? 1 : 0)) * 100);
    const label = grade === "partial" ? `partial ${pct}%` : grade;
    const tip = `&b · source ${src || "none"} · ` + (feed.detail || `feeds: ${feed.feeds.join(", ")}`);
    return `<span class="verdict ${cls}" title="${escapeHtml(tip)}">&amp;b ${escapeHtml(label)}</span>`;
  })() : "";
  const phaseChip = phase.label
    ? `<span class="verdict phase" title="¶gn — this pass sits at ${escapeHtml(phase.label)}; every panel shares the same minute grid">¶gn ${escapeHtml(phase.mark || "")}</span>`
    : "";

  const html =
    `<div class="formula-row">` +
    `  <div class="formula-name" title="${escapeHtml(meta?.brain_node || "")}">${name}</div>` +
    `  <div class="formula-bar"><div class="zero"></div>` +
    `    <div class="fill ${fillClass}" style="${v >= 0 ? "left:50%" : `right:50%`};width:${width}%"></div>` +
    `  </div>` +
    `  <div class="formula-value ${posClass}">${known ? fmtSigned(v, 3) : "—"}</div>` +
    `  <div class="formula-head">` +
    `    <span class="formula-reading">${escapeHtml(reading)}</span>` +
    (stats
      ? `<span class="formula-stats" title="this value against its own history">` +
        `${fmtSigned(stats.zscore || 0, 2)}σ · p${Math.round(stats.percentile || 0)} · ` +
        `${stats.samples || 0} windows` +
        (timingUs ? ` · ${fmtUs(timingUs)}` : "") +
        `</span>`
      : (timingUs ? `<span class="formula-stats">${fmtUs(timingUs)}</span>` : "")) +
    `    ${verdictChip} ${checkChip} ${feedChip} ${phaseChip}` +
    `    <button class="logic-toggle" type="button">${open ? "▾ hide logic" : "▸ logic"}</button>` +
    `  </div>` +
    (state.explain && description ? `<div class="formula-desc">${escapeHtml(description)}</div>` : "") +
    (open ? logicBlock(name, logic, verdict, traces[name]) : "") +
    `</div>`;
  row.innerHTML = html;
  const toggle = row.querySelector(".logic-toggle");
  if (toggle) {
    toggle.onclick = () => {
      state.openLogic[name] = !state.openLogic[name];
      renderFormulas();
    };
  }
  return row;
}

/* The logic panel: the equation, the steps the code actually performs, how to
   read the value, and - the point of the whole thing - the intermediate
   numbers the formula recorded while computing tonight's number. */
function logicBlock(name, logic, verdict, trace) {
  if (!logic) {
    return `<div class="logic-block"><div class="muted">No logic entry for ${escapeHtml(name)}.</div></div>`;
  }
  const steps = (logic.steps || [])
    .map((step, index) => `<li><span class="step-no">${index + 1}</span>${escapeHtml(step)}</li>`)
    .join("");
  const bands = (logic.bands || [])
    .map((band) => `<li><code>${band.lo}</code> → <code>${band.hi}</code> : ${escapeHtml(band.text)}</li>`)
    .join("");
  const traceRows = (trace || [])
    .map((item) => {
      const value = typeof item.value === "number"
        ? Math.abs(item.value) < 1e-4 && item.value !== 0
          ? item.value.toExponential(3)
          : item.value.toFixed(4)
        : escapeHtml(String(item.value));
      return `<li><span class="trace-label">${escapeHtml(item.label)}</span>` +
        `<b>${value}</b><span class="trace-unit">${escapeHtml(item.unit || "")}</span></li>`;
    })
    .join("");
  const reads = (logic.reads || []).map((r) => `<code>${escapeHtml(r)}</code>`).join(" ");

  const factChips = [
    logic.range ? `<span class="pill">range ${escapeHtml(logic.range)}</span>` : null,
    logic.units ? `<span class="pill">units ${escapeHtml(logic.units)}</span>` : null,
    logic.sensitivity ? `<span class="pill">sensitivity ${escapeHtml(logic.sensitivity)}</span>` : null,
  ].filter(Boolean).join(" ");
  const misleads = (logic.misleads || []).map((line) => `<li>${escapeHtml(line)}</li>`).join("");
  const corroborates = (logic.corroborates || []).map((c) => `<code>${escapeHtml(c)}</code>`).join(" ");

  return (
    `<div class="logic-block">` +
    `  <div class="logic-expression"><code>${escapeHtml(logic.expression)}</code></div>` +
    (factChips ? `  <div class="logic-facts">${factChips}</div>` : "") +
    `  <div class="logic-reads muted">reads: ${reads}</div>` +
    `  <div class="logic-cols">` +
    `    <div><div class="logic-h">How it is computed</div><ol class="logic-steps">${steps}</ol></div>` +
    `    <div><div class="logic-h">How to read the value</div><ul class="logic-bands">${bands}</ul>` +
    `      <div class="logic-sign">${escapeHtml(logic.sign || "")}</div>` +
    (logic.why ? `<div class="logic-why muted">${escapeHtml(logic.why)}</div>` : "") +
    `    </div>` +
    `  </div>` +
    (traceRows
      ? `<div class="logic-h">Numbers behind this window</div><ul class="logic-trace">${traceRows}</ul>`
      : `<div class="logic-h">Numbers behind this window</div>` +
        `<div class="muted">not computed yet — they appear after the first live pass</div>`) +
    (misleads || corroborates
      ? `<div class="logic-facts-detail">` +
        (misleads ? `<div><div class="logic-h">When it misleads</div><ul class="logic-bands">${misleads}</ul></div>` : "") +
        (corroborates ? `<div class="logic-h">Corroborated by ${corroborates}</div>` : "") +
        `</div>`
      : "") +
    (verdict
      ? `<div class="logic-verdict ${verdict.passed ? "pass" : "fail"}">` +
        `<b>Self-test:</b> ${verdict.passed ? "PASS" : "FAIL"} — ${escapeHtml(verdict.claim)}. ` +
        `BULL ${fmtSigned(verdict.observed.BULL, 3)} · BEAR ${fmtSigned(verdict.observed.BEAR, 3)} · ` +
        `FLAT ${fmtSigned(verdict.observed.FLAT, 3)} · STRESS ${fmtSigned(verdict.observed.STRESS, 3)}.` +
        `<br><span class="muted">${escapeHtml(verdict.detail)}</span></div>`
      : "") +
    `</div>`
  );
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ------------------------------------------------------------- emergency */
/* The emergency path no longer takes over the screen: it lights the inline
   conviction box and a small chip under the prediction, both inside the page. */
function renderEmergency() {
  renderConvictionBox();
}

/* ------------------------------------------------------- API keys modal */
/* Adding keys without a terminal: the same endpoints the Settings page uses,
   reachable from the dashboard header.  Every key is tested before it is
   accepted, applied to the running engine immediately, and can optionally be
   written to the git-ignored .env. */

const KEY_SLOTS = ["gemini", "cryptopanic", "newsapi", "github", "neuprint"];

function openKeys() {
  $("keys-modal").classList.remove("hidden");
  renderKeyStates();
}
function closeKeys() {
  $("keys-modal").classList.add("hidden");
}

function setResult(slot, text, cls = "") {
  const el = $(`r-${slot}`);
  if (!el) return;
  el.className = "key-result " + cls;
  el.textContent = text;
}

async function testKey(slot, inputId, button) {
  const key = $(inputId).value.trim();
  if (!key) return setResult(slot, "enter a key first", "warn");
  button.disabled = true;
  setResult(slot, "testing…", "warn");
  const res = await getJSON(`/api/agents/${slot}/test`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key }),
  }) || { valid: false, error: "no response (is the backend still running?)" };
  button.disabled = false;
  if (res.valid) {
    setResult(slot, `✅ ${res.detail || "valid"}`, "ok");
  } else {
    setResult(slot, `❌ ${res.error || "invalid"}${res.hint ? " — " + res.hint : ""}`, "err");
  }
}

async function saveKeys() {
  const persist = $("k-persist").checked;
  const saved = [];
  const failed = [];
  for (const slot of KEY_SLOTS) {
    const input = $(`k-${slot}`);
    if (!input || !input.value.trim()) continue;
    const res = await getJSON(`/api/settings/keys/${slot}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: input.value.trim(), persist }),
    });
    if (res && !res.error) saved.push(slot);
    else failed.push(slot);
  }
  if (!saved.length) {
    setResult("save", failed.length ? `❌ could not save: ${failed.join(", ")}` : "nothing entered yet",
      failed.length ? "err" : "warn");
    return;
  }
  setResult("save", `✅ saved ${saved.join(", ")}${persist ? " (written to .env)" : ""}`, "ok");
  for (const slot of saved) $(`k-${slot}`).value = "";

  // apply immediately: agents re-read their keys, news re-polls with the new token
  await refreshAgents();
  await refreshNewsList();
  if (saved.includes("cryptopanic") || saved.includes("newsapi")) {
    await fetch("/api/news/poll", { method: "POST" });
    await refreshNewsList();
  }
  if (saved.includes("neuprint")) {
    setResult("save", "✅ token saved — rebuilding the connectome (up to 60 s)…", "warn");
    const brain = await getJSON("/api/brain/reconnect", { method: "POST" });
    setResult("save", brain && brain.status
      ? `✅ connectome: ${brain.status} — ${brain.detail}`
      : "✅ token saved; the fallback matrix stays in use until the next reconnect",
      brain && brain.status === "LIVE" ? "ok" : "warn");
    await refreshBrain();
  }
  renderKeyStates();
}

function renderKeyStates() {
  const configured = (state.config && state.config.configured) || {};
  KEY_SLOTS.forEach((slot) => {
    const el = $(`state-${slot}`);
    if (!el) return;
    const isSet = !!configured[slot];
    el.className = "key-state" + (isSet ? " set" : "");
    el.textContent = isSet ? "configured ✓" : "not set";
  });
}

function maybeShowSetupBanner() {
  const configured = (state.config && state.config.configured) || {};
  const dismissed = localStorage.getItem("drosophila.setup.dismissed") === "1";
  const nothingSet = !configured.gemini && !configured.cryptopanic && !configured.newsapi && !configured.github;
  $("setup-banner").classList.toggle("hidden", dismissed || !nothingSet);
}

document.querySelectorAll("[data-reveal]").forEach((btn) => {
  btn.onclick = () => {
    const input = $(btn.dataset.reveal);
    input.type = input.type === "password" ? "text" : "password";
  };
});

document.querySelectorAll("[data-test]").forEach((btn) => {
  btn.onclick = () => testKey(btn.dataset.test, btn.dataset.input, btn);
});

$("open-keys").addEventListener("click", (event) => { event.preventDefault(); openKeys(); });
$("setup-open").addEventListener("click", openKeys);
$("synesthesia-toggle")?.addEventListener("click", () => synesthesia.toggle());
$("keys-close").onclick = closeKeys;
$("keys-cancel").onclick = closeKeys;
$("keys-save").onclick = saveKeys;
$("setup-dismiss").onclick = () => {
  localStorage.setItem("drosophila.setup.dismissed", "1");
  maybeShowSetupBanner();
};
$("keys-modal").addEventListener("click", (event) => {
  if (event.target.id === "keys-modal") closeKeys();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeKeys();
});

/* ---------------------------------------------------------------- events */
$("asset-toggle").addEventListener("click", (event) => {
  const btn = event.target.closest(".asset");
  if (!btn) return;
  const asset = btn.dataset.asset;
  if (asset === state.asset) return;
  send({ action: "switch_asset", asset });
  // optimistic label only - the backend decides when the switch happens
  state.pendingAsset = asset;
  renderAssetToggle();
});

$("w-emergency-clear").addEventListener("click", async () => {
  await fetch("/api/news/emergency/clear", { method: "POST" });
  state.emergencyUntil = 0;
  state.emergency = null;
  renderEmergency();
});

$("emotion-bars").addEventListener("click", (event) => {
  const row = event.target.closest(".emotion-row");
  if (!row || !row.dataset.emotion) return;
  state.emotionFormulaFor = state.emotionFormulaFor === row.dataset.emotion ? null : row.dataset.emotion;
  renderEmotions();
});
$("brain-toggle").addEventListener("click", () => {
  state.wiringOpen = !state.wiringOpen;
  $("brain-wiring").classList.toggle("hidden", !state.wiringOpen);
  $("brain-toggle").textContent = state.wiringOpen ? "Hide the wiring diagram" : "Show the wiring diagram";
});

$("toggle-explain").addEventListener("click", () => {
  state.explain = !state.explain;
  $("toggle-explain").textContent = state.explain ? "Hide descriptions" : "Show descriptions";
  renderFormulas();
});

$("collapse-all").addEventListener("click", () => {
  const anyOpen = Object.values(state.collapsed).some((v) => !v);
  state.formulaMeta.forEach((c) => { state.collapsed[c.key] = anyOpen; });
  renderFormulas();
});

/* Is this object the signal payload the panel renders?  It has the frozen
   levels and the lock state; its `signal` field is the direction ("BUY"). */
function isSignalPayload(value) {
  return !!value && typeof value === "object" &&
    "lock_state" in value && "risk" in value && "cycle_number" in value;
}

/* ------------------------------------------------------- one snapshot, one pass
   Every panel on the page is rendered from a single payload.  The boundary
   message (SIGNAL) and the mid-window heartbeat (PULSE) carry the same feature
   set, so the signal, the formulas, the news list, the agents, the brain and
   the accuracy panel all change on the same frame - in parallel, never one
   panel at a time on a timer of its own. */
function applySnapshot(data, opts = {}) {
  if (!data || data.error) return;
  const anchored = applyClock(data, { rttMs: opts.rttMs, force: opts.force, boundary: !!opts.boundary });
  if (data.asset) state.asset = data.asset;
  if ("pending_asset" in data) state.pendingAsset = data.pending_asset;
  if (data.lock_state) state.lockState = data.lock_state;
  if (data.conviction_note !== undefined) state.convictionNote = data.conviction_note;
  if (data.degradation_level !== undefined) {
    const el = $("degradation");
    if (el) el.textContent = degradationLabel(data.degradation_level);
  }
  if (data.status) renderStatus(data.status);
  // A signal *payload* is the object the panel renders: it carries the frozen
  // levels and the lock state, and its own `signal` field is the direction.
  // (A pulse carries `locked_side` instead - the two must never be confused.)
  if (isSignalPayload(data)) {
    state.signal = data;
    state.formulas = data.formulas || state.formulas;
    state.formulaReadings = data.readings || state.formulaReadings;
    state.formulaTraces = data.traces || state.formulaTraces;
  } else if (isSignalPayload(data.signal)) {
    state.signal = data.signal;
  }
  if (data.prediction) {
    state.prediction = data.prediction;
    state.predictionAnchoredAt = Date.now();
    state.predictionAnchoredServerMs = serverNowMs();
  }
  if (data.live_formulas) renderLiveFormulas(data.live_formulas);
  if (data.news_feed) renderNewsList(data.news_feed);
  if (data.emotions) adoptEmotions(data);
  if (data.agents_status) renderAgents(data.agents_status);
  if (data.brain_explain) renderBrainExplain(data.brain_explain);
  if (data.brain_status) renderBrainStatus(data.brain_status);
  if (data.physics) renderPhysics(data.physics);
  if (data.genesis) renderGenesis(data.genesis);
  if (data.history) renderHistory(data.history);
  if (data.outcomes) renderOutcomes(data.outcomes);
  if (data.accuracy && state.prediction) state.prediction.accuracy = data.accuracy;
  if (data.window) state.window = data.window;
  state.lastSnapshotAt = Date.now();
  state.snapshotCount = (state.snapshotCount || 0) + 1;
  renderAll();
  if (anchored) {
    // A new window: the prediction may have flipped - feel it and say it.
    const previous = state.lastPrediction;
    if (state.signal?.signal && state.signal.signal !== previous) {
      haptic(state.signal.is_emergency_override ? [24, 60, 24] : [18]);
    }
  }
}

/* One pass, in a fixed order: the panels that read state.signal render after it
   has been replaced, and nothing repaints twice inside the pass. */
function renderAll() {
  state.inRenderAll = true;
  try {
    renderAssetToggle();
    renderSignal();
    renderFormulas();
    renderWidgetPanel();
    renderConvictionBox();
    renderEmotions();
  } finally {
    state.inRenderAll = false;
  }
}

async function pollReadiness() {
  const health = await getJSON("/api/health");
  if (!health || health.error) return;
  renderWarming({
    ready: health.ready !== false,
    warming_up: health.warming_up,
    start_error: health.start_error,
    uptime_seconds: health.uptime_seconds,
  });
}

window.addEventListener("load", pollReadiness, { once: true });

async function boot() {
  state.config = await getJSON("/api/system/config");
  if (state.config && !state.config.error) {
    state.cyclePeriod = state.config.cycle_period_seconds || 60;
    state.asset = (state.config.assets || ["BTC"])[0];
  } else {
    state.config = {};
  }
  renderKeyStates();
  maybeShowSetupBanner();
  wirePredictionDetailToggle();

  /* Static material: fetched once, it does not change while the page is open. */
  await loadFormulaMeta();
  await loadSelfTest();
  await loadWiring();

  /* Everything else arrives in the HELLO snapshot, and then in one message per
     tick.  Before the socket is up (or if it is down) one REST pass paints the
     page so it is never blank. */
  const t0 = performance.now();
  state.lastRttMs = 0;
  await syncClock();
  state.lastRttMs = performance.now() - t0;

  /* ONE frame loop drives the countdown and every clock-derived readout. */
  state.raf = requestAnimationFrame(frame);

  /* Round AA: numbers that change glow for a moment (no timers - a single
     MutationObserver on the text of the numeric cells). */
  if (window.MutationObserver) {
    const flash = new MutationObserver((records) => {
      records.forEach((r) => {
        const b = (r.target.nodeType === 3 ? r.target.parentElement : r.target)?.closest?.(".kv-item b, .hedge-item b, .formula-value");
        if (!b) return;
        b.classList.remove("changed");
        void b.offsetWidth;
        b.classList.add("changed");
      });
    });
    flash.observe(document.body, { subtree: true, characterData: true, childList: true });
  }

  /* ONE safety net, and it does nothing while the socket is healthy. */
  const applyBtn = $("update-apply");
  if (applyBtn) applyBtn.onclick = applyUpdate;
  setInterval(safetyNet, 5000);

  connect();
}

boot();
