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
const fmtPct = (v) => `${(v * 100).toFixed(0)}%`;
const fmtSigned = (v, d = 3) => (v >= 0 ? "+" : "") + Number(v).toFixed(d);

function pctClass(v) {
  if (v > 0.15) return "pos";
  if (v < -0.15) return "neg";
  return "neutral";
}

/* Haptics: navigator.vibrate where the browser supports it (Android Chrome),
   the Flutter client uses real HapticFeedback for the same events. */
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
      applySnapshot(msg.data, { rttMs: state.lastRttMs });
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
  if (state.serverOffsetMs === null) {
    state.serverOffsetMs = sample;
    state.clockSamples = 1;
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

function windowRemaining() {
  if (!state.clock) return state.cyclePeriod || 60;
  return Math.max(0, (state.clock.endsAtMs - serverNowMs()) / 1000);
}

function windowElapsed() {
  return Math.max(0, (state.cyclePeriod || 60) - windowRemaining());
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
    ring.classList.toggle("ready", !!state.window?.prefetch_ready);
    ring.classList.toggle("urgent", secs <= 5);
  }
  if ($("progress")) {
    $("progress").style.width = `${clamp(100 - (remaining / period) * 100, 0, 100)}%`;
  }
  if (secs !== state.lastSecondShown) {
    state.lastSecondShown = secs;
    const el = $("w-countdown");
    if (el) el.textContent = String(secs);
    if ($("timer")) {
      const mm = String(Math.floor(remaining / 60)).padStart(2, "0");
      const ss = String(Math.floor(remaining % 60)).padStart(2, "0");
      const icon = state.lockState === "COMPUTING" ? "\u23f3"
        : state.lockState === "EMERGENCY_OVERRIDE" ? "\u26a1" : "\ud83d\udd12";
      $("timer").innerHTML = `${mm}:${ss} <span id="lock-icon">${icon}</span>`;
      $("timer").className = "timer" + (state.lockState === "EMERGENCY_OVERRIDE" ? " override" : "");
      $("lock-state").textContent = state.lockState;
      $("progress").className = "progress-fill" + (state.lockState === "EMERGENCY_OVERRIDE" ? " override" : "");
      $("utc").textContent = new Date(serverNowMs()).toISOString().substr(11, 8) + "Z";
    }
    if ($("w-countdown-sub")) $("w-countdown-sub").innerHTML = countdownSubLine();
    renderFreshness(state.prediction);
    renderPipelineProgress();
    renderWindowStrip();
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
async function safetyNet() {
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
  chip.textContent = stale
    ? `STALE ${Math.round(age)}s > ${Math.round(maxAge)}s`
    : `updated ${Math.round(age)}s ago · max ${Math.round(maxAge)}s`;
  chip.className = "fresh-chip " + (stale ? "stale" : "live");
  chip.title = `computed ${p.computed_at} · expires ${p.expires_at || "—"}`;
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
  const computedAgo = state.window?.computed_seconds_ago;
  const proof = computedAgo
    ? ` · this window's signal was computed <b>${Math.round(computedAgo)}s</b> before it opened`
    : "";
  $("w-next").innerHTML = ready
    ? `next signal <b>#${nextCycle}</b> computed and held — revealed at the boundary${proof}`
    : `computing signal <b>#${nextCycle}</b> … ${Math.round(progress * 100)}%${proof}`;
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
  renderWidgetPanel();
  renderWindowStrip();
  renderWarming(status);
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
    const parts = Object.keys(w).map((k) => `${k} ${fmtPct(w[k])}`);
    $("weights").textContent = parts.length
      ? `fusion weights: ${parts.join(" · ")} · window #${s.cycle_number}`
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

/* ============================ WIDGET PANEL ==============================
   Row 1: prediction | countdown (1-60) | signal + conviction box
   Row 2: take profit & stop loss | prediction accuracy
   ======================================================================== */
function renderWidgetPanel() {
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
  const rows = state.outcomes || [];
  const wins = rows.filter((r) => r.outcome > 0).length;
  const winRate = rows.length ? wins / rows.length : null;
  let streak = 0;
  for (let i = rows.length - 1; i >= 0; i -= 1) {
    if (rows[i].outcome === 0) break;
    if (streak === 0) { streak = rows[i].outcome > 0 ? 1 : -1; continue; }
    if ((rows[i].outcome > 0 ? 1 : -1) === streak) streak += streak > 0 ? 1 : -1;
    else break;
  }
  $("w-winrate").textContent = winRate === null ? "—" : fmtPct(winRate);
  $("w-samples").textContent = rows.length ? String(rows.length) : "0";
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

  const engine = state.config?.simulated
    ? `simulated market data · ${state.config?.market_source || "simulator"}`
    : `${state.config?.market_source || "live feed"} · ${state.config?.cycle_period_seconds || 60}s windows`;
  $("w-engine").textContent =
    `${engine}${state.window?.pipeline ? " · pipelined" : ""}` +
    ` · ${Math.round(state.window?.window_seconds || state.cyclePeriod || 60)}s predictions`;
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
  const h = s.hedge || {};
  const set = (id, value) => { $(id).textContent = value === undefined ? "—" : fmtSigned(value, 3); };
  set("h-hsi", h.hsi); set("h-hrdd", h.hrdd); set("h-shrp", h.shrp); set("h-gcdv", h.gcdv);
  $("h-hsi-bar").style.width = `${Math.min(100, Math.abs(h.hsi || 0) * 100)}%`;
  $("h-hsi-bar").className = "bar-fill " + (h.hsi > 0.8 ? "neg" : "neutral");
  [["h-hrdd-bar", h.hrdd], ["h-shrp-bar", h.shrp], ["h-gcdv-bar", h.gcdv]].forEach(([id, v]) => {
    const el = $(id);
    el.style.width = `${Math.min(100, Math.abs(v || 0) * 100)}%`;
    el.className = "bar-fill " + pctClass(v || 0);
  });
  $("hedge-stress").textContent = (h.stress || "—") + " stress";
  $("brain-status").textContent = s.brain_status || "—";
  $("brain-detail").textContent = `CCSv2 ${fmtSigned(s.ccs_value)} @ ${fmtPct(s.ccs_confidence)}`;
  $("drg-value").textContent = fmtSigned(s.drg || 0);
}

function renderNews(s) {
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
  (data.items || []).forEach((item) => {
    const row = document.createElement("div");
    row.className = "news-item";
    const sentClass = item.sentiment > 0.1 ? "pos" : item.sentiment < -0.1 ? "neg" : "";
    row.innerHTML =
      `<span class="tier">T${item.tier}</span>` +
      `<span style="flex:1">${escapeHtml(item.headline)}<br><span class="muted">${escapeHtml(item.source)} · ${Math.round(item.age_seconds)}s ago</span></span>` +
      `<span class="sent ${sentClass}">${fmtSigned(item.sentiment, 2)}</span>`;
    list.appendChild(row);
  });
}

async function refreshNewsList() {
  renderNewsList(await getJSON("/api/news?limit=5"));
}

function renderAgents(data) {
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

async function refreshOutcomes() {
  renderOutcomes(await getJSON("/api/signal/outcomes"));
}

function renderTimings(data) {
  if (!data || !data.timings_ms) return;
  const top = Object.entries(data.timings_ms).slice(0, 5)
    .map(([k, v]) => `${k} ${v.toFixed(3)}ms`).join(" · ");
  $("timings").textContent =
    `total formula pass ${data.total_ms?.toFixed(2) ?? "?"}ms (budget ${data.budget_ms}ms) · slowest: ${top}`;
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
  if (data.note) {
    const note = $("live-note");
    if (note) note.textContent = "live values " + data.note;
  }
  if (data.timings_ms) renderTimings(data);
  renderFormulas();
}

async function refreshLiveFormulas() {
  renderLiveFormulas(await getJSON("/api/formulas/live"));
}

function renderFormulas() {
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

  const verdictChip = verdict
    ? `<span class="verdict ${verdict.passed ? "pass" : "fail"}" ` +
      `title="${escapeHtml(verdict.claim + " — " + verdict.detail)}">` +
      `${verdict.passed ? "✔ self-test" : "✘ self-test"}</span>`
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
    `    ${verdictChip}` +
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

  return (
    `<div class="logic-block">` +
    `  <div class="logic-expression"><code>${escapeHtml(logic.expression)}</code></div>` +
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
  $("flutter-link").classList.toggle("hidden", !(state.config && state.config.flutter_web));
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
  const anchored = applyClock(data, { rttMs: opts.rttMs, force: opts.force });
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
  if (data.agents_status) renderAgents(data.agents_status);
  if (data.brain_explain) renderBrainExplain(data.brain_explain);
  if (data.brain_status) renderBrainStatus(data.brain_status);
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

  /* ONE safety net, and it does nothing while the socket is healthy. */
  setInterval(safetyNet, 5000);

  connect();
}

boot();
