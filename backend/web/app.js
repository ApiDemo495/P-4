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
  formulaMeta: [],
  holdWarning: null,
  emergency: null,
  emergencyUntil: 0,
  config: null,
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

async function getJSON(url) {
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    return null;
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
    case "HELLO": {
      const d = msg.data;
      state.asset = d.asset;
      state.pendingAsset = d.pending_asset;
      state.cycleNumber = d.status?.cycle?.cycle_number ?? 0;
      state.degradation = d.status?.degradation_level;
      renderAssetToggle();
      renderStatus(d.status);
      if (d.signal) {
        state.signal = d.signal;
        state.lockState = d.signal.lock_state || "LOCKED";
        state.holdWarning = d.hold_warning;
      }
      state.formulas = d.formulas || {};
      syncClock();
      renderAll();
      break;
    }
    case "CYCLE_START": {
      state.cycleNumber = msg.data.cycle_number;
      state.asset = msg.data.asset || state.asset;
      state.lockState = "COMPUTING";
      state.signal = null;
      state.holdWarning = null;
      state.lastCycleAt = Date.now();
      renderSignal();
      $("cycle-number").textContent = state.cycleNumber;
      $("degradation").textContent = degradationLabel(msg.data.degradation_level);
      break;
    }
    case "SIGNAL": {
      state.signal = msg.data;
      state.lockState = msg.data.lock_state || "LOCKED";
      state.holdWarning = msg.data.hold_warning || null;
      state.formulas = msg.data.formulas || {};
      renderSignal();
      refreshHistory();
      break;
    }
    case "FORMULA_UPDATE": {
      state.liveFormulas = msg.data.formulas || {};
      $("live-note").textContent = "live values " + (msg.data.note || "");
      renderFormulas();
      refreshTimings();
      break;
    }
    case "EMERGENCY_OVERRIDE": {
      state.emergency = msg.data;
      state.emergencyUntil = Date.now() + (msg.data.remaining_seconds || 180) * 1000;
      if (msg.data.signal) {
        state.signal = msg.data.signal;
        state.lockState = "EMERGENCY_OVERRIDE";
        state.holdWarning = msg.data.signal.hold_warning || null;
      }
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
  renderAll();
}

/* ------------------------------------------------------------------ clock */
async function syncClock() {
  const status = await getJSON("/api/signal/status");
  if (!status) return;
  renderStatus(status);
  state.cyclePeriod = status.clock?.cycle_period_seconds || 60;
  state.secondsIntoMinute = status.clock?.seconds_into_minute || 0;
  if (status.lock) {
    state.lockState = status.lock.state;
    if (status.lock.emergency_active) {
      state.emergencyUntil = Date.now() + status.lock.emergency_remaining * 1000;
    }
  }
  if (status.asset) state.asset = status.asset;
  state.pendingAsset = status.pending_asset;
  if (!state.signal) {
    const current = await getJSON("/api/signal/current");
    if (current?.signal) {
      state.signal = current.signal;
      state.holdWarning = current.hold_warning;
      state.formulas = current.signal.formulas || {};
    }
  }
  renderAssetToggle();
  renderAll();
}

function tickClock() {
  const period = state.cyclePeriod;
  let elapsed;
  if (state.lastCycleAt) {
    elapsed = (Date.now() - state.lastCycleAt) / 1000;
    if (elapsed > period + 2) {           // missed a CYCLE_START; resync
      state.lastCycleAt = null;
      syncClock();
    }
  } else {
    elapsed = state.secondsIntoMinute;
    state.secondsIntoMinute = (elapsed + 0.1) % period;
  }
  const remaining = Math.max(0, period - elapsed);
  const mm = String(Math.floor(remaining / 60)).padStart(2, "0");
  const ss = String(Math.floor(remaining % 60)).padStart(2, "0");
  const icon = state.lockState === "COMPUTING" ? "⏳"
    : state.lockState === "EMERGENCY_OVERRIDE" ? "⚡" : "🔒";
  $("timer").innerHTML = `${mm}:${ss} <span id="lock-icon">${icon}</span>`;
  $("timer").className = "timer" + (state.lockState === "EMERGENCY_OVERRIDE" ? " emergency" : "");
  $("lock-state").textContent = state.lockState;

  const pct = state.lockState === "COMPUTING" ? Math.min(100, (period - remaining) / 8 * 100) : 100;
  $("progress").style.width = `${Math.max(0, Math.min(100, 100 - (remaining / period) * 100))}%`;
  $("progress").className = "progress-fill" + (state.lockState === "EMERGENCY_OVERRIDE" ? " emergency" : "");
  $("utc").textContent = new Date().toISOString().substr(11, 8) + "Z";

  if (state.emergencyUntil > Date.now()) {
    const left = Math.round((state.emergencyUntil - Date.now()) / 1000);
    $("em-remaining").textContent = `${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`;
  }
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
  $("cycle-number").textContent = state.cycleNumber;
  const lock = status.lock || {};
  state.lockState = lock.state || state.lockState;
  $("degradation").textContent = degradationLabel(status.degradation_level);
  $("foot-status").textContent =
    `cycle ${state.cycleNumber} · ${status.asset} · redis=${status.infrastructure?.redis ?? "?"}` +
    ` · win rate ${fmtPct(status.infrastructure?.win_rate ?? 0)} · ${status.clock?.utc ?? ""}`;
  if (status.clock?.ntp_synced === false) {
    $("foot-status").textContent += " · NTP: system clock";
  }
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
  const computing = state.lockState === "COMPUTING" || !state.signal;
  $("computing").classList.toggle("hidden", !computing);
  $("signal-body").classList.toggle("hidden", computing);
  $("hold-warning").classList.toggle("hidden", true);
  if (computing) return;

  const s = state.signal;
  const badge = $("signal-badge");
  badge.textContent = s.signal;
  badge.className = "signal-badge " + s.signal.toLowerCase();
  $("signal-icon").textContent = s.lock_icon || "🔒";
  $("signal-note").textContent = s.is_emergency_override
    ? "emergency override — forced HOLD"
    : "this signal is locked for the remainder of the cycle";
  $("confidence-value").textContent = fmtPct(s.confidence);
  const bar = $("confidence-bar");
  bar.style.width = `${Math.round(s.confidence * 100)}%`;
  bar.className = "bar-fill " + (s.signal === "BUY" ? "" : s.signal === "SELL" ? "neg" : "neutral");
  $("reasoning").textContent = s.reasoning || "";

  const w = s.fusion?.weights_used || {};
  const parts = Object.keys(w).map((k) => `${k} ${fmtPct(w[k])}`);
  $("weights").textContent = parts.length
    ? `fusion weights: ${parts.join(" · ")} · cycle ${s.cycle_number} · frozen at ${s.timestamp}`
    : `cycle ${s.cycle_number} · frozen at ${s.timestamp}`;

  if (s.signal === "HOLD" && state.holdWarning) {
    $("hold-warning").classList.remove("hidden");
    $("hw-text").textContent = `“${state.holdWarning.text}”`;
    const lean = state.holdWarning.lean;
    $("hw-lean").textContent = lean
      ? `Lean direction: ${lean} (score ${fmtSigned(state.holdWarning.lean_score)}, ` +
        `confidence ${fmtPct(state.holdWarning.confidence)})`
      : "No directional lean this cycle.";
  }

  renderHedge(s);
  renderNews(s);
  $("price").textContent = s.price ? `$${Number(s.price).toLocaleString(undefined, { maximumFractionDigits: 2 })}` : "—";
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

async function refreshNewsList() {
  const data = await getJSON("/api/news?limit=5");
  if (!data) return;
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

async function refreshAgents() {
  const data = await getJSON("/api/agents");
  if (!data) return;
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
      decision = s.signal === "HOLD" ? "HOLD" : `${s.signal} (${fmtPct(s.ccs_confidence)})`;
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

async function refreshBrain() {
  const data = await getJSON("/api/brain/status");
  if (!data) return;
  $("brain-status").textContent = data.status;
  $("brain-detail").textContent =
    `${data.health?.message || data.message} · checksum ${data.matrix?.checksum || "—"} · gain ${data.gain}`;
}

async function refreshHistory() {
  const data = await getJSON("/api/signal/history?limit=12");
  if (!data) return;
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

async function refreshOutcomes() {
  const data = await getJSON("/api/signal/outcomes");
  if (!data) return;
  $("win-rate").textContent = `win rate ${fmtPct(data.win_rate)} (${data.count} outcomes)`;
  const list = $("outcomes");
  list.innerHTML = "";
  (data.rows || []).slice().reverse().slice(0, 8).forEach((row) => {
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="${row.outcome > 0 ? "sig-BUY" : row.outcome < 0 ? "sig-SELL" : "sig-HOLD"}">` +
      `${row.outcome > 0 ? "WIN" : row.outcome < 0 ? "LOSS" : "FLAT"}</span>` +
      `<span style="flex:1">${fmtSigned(row.pnl_bps, 1)} bps</span>`;
    list.appendChild(li);
  });
}

async function refreshTimings() {
  const data = await getJSON("/api/formulas/timings");
  if (!data || !data.timings_ms) return;
  const top = Object.entries(data.timings_ms).slice(0, 5)
    .map(([k, v]) => `${k} ${v.toFixed(3)}ms`).join(" · ");
  $("timings").textContent =
    `total formula pass ${data.total_ms?.toFixed(2) ?? "?"}ms (budget ${data.budget_ms}ms) · slowest: ${top}`;
}

/* ------------------------------------------------------------ formular UI */
async function loadFormulaMeta() {
  const data = await getJSON("/api/formulas");
  if (!data) return;
  state.formulaMeta = data.categories || [];
  state.rewardMeta = data.reward;
  renderFormulas();
}

function renderFormulas() {
  const container = $("formula-explorer");
  const values = Object.keys(state.liveFormulas).length ? state.liveFormulas : state.formulas;
  if (!state.formulaMeta.length) return;

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
        body.appendChild(formulaRow(f.name, v, f.description, f));
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
    body.appendChild(formulaRow("DRG", values.DRG, state.rewardMeta.description, state.rewardMeta));
    wrap.appendChild(body);
    container.appendChild(wrap);
  }
}

function formulaRow(name, value, description, meta) {
  const row = document.createElement("div");
  const known = typeof value === "number";
  const v = known ? value : 0;
  let posClass = "neutral", fillClass = "fill-pos";
  if (v > 0.15) { posClass = "pos"; }
  else if (v < -0.15) { posClass = "neg"; fillClass = "fill-neg"; }
  const width = Math.min(50, Math.abs(v) * 50);

  const html =
    `<div class="formula-row">` +
    `  <div class="formula-name" title="${escapeHtml(meta?.brain_node || "")}">${name}</div>` +
    `  <div class="formula-bar"><div class="zero"></div>` +
    `    <div class="fill ${fillClass}" style="${v >= 0 ? "left:50%" : `right:50%`};width:${width}%"></div>` +
    `  </div>` +
    `  <div class="formula-value ${posClass}">${known ? fmtSigned(v, 3) : "—"}</div>` +
    (state.explain && description ? `<div class="formula-desc">${escapeHtml(description)}</div>` : "") +
    `</div>`;
  row.innerHTML = html;
  return row;
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ------------------------------------------------------------- emergency */
function renderEmergency() {
  const active = state.emergencyUntil > Date.now();
  $("emergency").classList.toggle("hidden", !active);
  $("shade").classList.toggle("hidden", !active);
  if (active && state.emergency) {
    $("em-headline").textContent = `“${state.emergency.headline}”`;
  }
}

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

$("em-ack").addEventListener("click", () => {
  state.acknowledged = true;
  renderEmergency();
  $("emergency").classList.add("hidden");
  $("shade").classList.add("hidden");
});

$("em-dismiss").addEventListener("click", async () => {
  await fetch("/api/news/emergency/clear", { method: "POST" });
  state.emergencyUntil = 0;
  state.emergency = null;
  if (state.lockState === "EMERGENCY_OVERRIDE") state.lockState = "LOCKED";
  renderEmergency();
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

/* ------------------------------------------------------------------ boot */
function renderAll() {
  renderAssetToggle();
  renderSignal();
  renderFormulas();
  renderEmergency();
}

async function boot() {
  state.config = await getJSON("/api/system/config");
  if (state.config) {
    state.cyclePeriod = state.config.cycle_period_seconds || 60;
    state.asset = (state.config.assets || ["BTC"])[0];
  }
  await loadFormulaMeta();
  await syncClock();
  await refreshHistory();
  await refreshOutcomes();
  await refreshNewsList();
  await refreshAgents();
  await refreshBrain();
  await refreshTimings();
  connect();
  setInterval(tickClock, 100);
  setInterval(syncClock, 20000);
  setInterval(refreshAgents, 5000);
  setInterval(refreshNewsList, 15000);
  setInterval(refreshBrain, 20000);
  setInterval(refreshHistory, 30000);
  setInterval(refreshOutcomes, 20000);
  setInterval(renderEmergency, 1000);
}

boot();
