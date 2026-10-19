/* Brain Matrix page (Round AZ): the real FlyWire v783 connectome first -
   mushroom-body and whole-brain block matrices, populations, the latest
   whole-brain read-out - and the 80x80 stand-in only as a labelled fallback. */

const $ = (id) => document.getElementById(id);

async function getJSON(url) {
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch (e) { return null; }
}

function buildGrid(edges, nodeTypes) {
  const grid = $("matrix-grid");
  grid.style.gridTemplateColumns = `repeat(80, 11px)`;
  const matrix = Array.from({ length: 80 }, () => new Array(80).fill(0));
  let maxAbs = 0;
  edges.forEach((e) => {
    matrix[e.source][e.target] = e.weight;
    maxAbs = Math.max(maxAbs, Math.abs(e.weight));
  });

  const frag = document.createDocumentFragment();
  const peak = maxAbs || 1;
  for (let r = 0; r < 80; r++) {
    for (let c = 0; c < 80; c++) {
      const w = matrix[r][c];
      const cell = document.createElement("div");
      cell.className = "matrix-cell";
      if (w !== 0) {
        const alpha = 0.25 + 0.75 * Math.min(1, Math.abs(w) / peak);
        cell.style.opacity = alpha.toFixed(2);
        cell.className += w > 0 ? " pos" : " neg";
      }
      cell.title = `${nodeTypes[r] || r} → ${nodeTypes[c] || c}\nweight ${w.toFixed(4)}`;
      frag.appendChild(cell);
    }
  }
  grid.innerHTML = "";
  grid.appendChild(frag);
}

function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function fillTrace() {
  const trace = await getJSON("/api/brain/trace");
  const body = $("trace-body");
  if (!trace || !trace.available) {
    body.innerHTML = `<tr><td class="muted">No trace yet — wait for the next signal cycle.</td></tr>`;
    return;
  }
  const rows = [
    ["cycle", trace.cycle_number],
    ["asset", trace.asset],
    ["KCAE (confidence)", trace.kcae],
    ["circuit", trace.kenyon_cells?.connectome ? `${trace.kenyon_cells.connectome} (real)` : "80×80 stand-in"],
    [`active Kenyon cells (of ${trace.kenyon_cells?.of ?? 50})`, trace.active_kcs],
    ["LH approach", trace.lh_approach],
    ["LH avoid", trace.lh_avoid],
    ["LH neutral", trace.lh_neutral],
    ["CCSv2", trace.ccs],
    ["confidence (operative)", trace.confidence],
    ["confidence (literal spec term)", trace.diagnostics?.confidence_spec],
    ["read-out margin", trace.diagnostics?.decisiveness],
    ["neutral share", trace.diagnostics?.neutral_share &&
      Number(trace.diagnostics.neutral_share).toFixed(4)],
  ];
  body.innerHTML = rows
    .map(([k, v]) => `<tr><td>${k}</td><td>${typeof v === "number" ? v.toFixed(4) : v ?? "—"}</td></tr>`)
    .join("");
}

function fmt(n) { return Number(n || 0).toLocaleString(); }

/* Population x population heatmap with row labels and rotated column labels. */
function buildBlockGrid(hostId, block) {
  const host = $(hostId);
  if (!block || !block.labels || !block.labels.length) { host.innerHTML = `<span class="muted">not loaded yet</span>`; return; }
  const k = block.labels.length;
  host.style.gridTemplateColumns = `150px repeat(${k}, 14px)`;
  const peak = Math.max(1e-9, ...block.matrix.flat().map((v) => Math.abs(v)));
  const frag = document.createDocumentFragment();
  const corner = document.createElement("div"); corner.className = "lbl"; corner.textContent = "target ↓ · source →";
  frag.appendChild(corner);
  block.labels.forEach((label) => {
    const c = document.createElement("div"); c.className = "lbl col"; c.textContent = label; c.title = label;
    frag.appendChild(c);
  });
  for (let r = 0; r < k; r++) {
    const l = document.createElement("div"); l.className = "lbl";
    l.textContent = `${block.labels[r]} [${fmt(block.sizes[r])}]`; l.title = l.textContent;
    frag.appendChild(l);
    for (let c = 0; c < k; c++) {
      const w = block.matrix[r][c];
      const cell = document.createElement("div");
      cell.className = "bc" + (w > 0 ? " pos" : w < 0 ? " neg" : "");
      if (w !== 0) cell.style.opacity = (0.2 + 0.8 * Math.min(1, Math.sqrt(Math.abs(w) / peak))).toFixed(2);
      cell.title = `${block.labels[c]} → ${block.labels[r]}\ndrive ${w.toFixed(4)}`;
      frag.appendChild(cell);
    }
  }
  host.innerHTML = "";
  host.appendChild(frag);
}

function renderFlyWire(fw) {
  const live = !!(fw && fw.loaded);
  $("src-dot").className = "dot " + (live ? "on" : "");
  $("src-label").textContent = live ? "FlyWire v783 connectome LIVE" : `connectome ${fw?.data?.phase || "not loaded"} · 80×80 stand-in in use`;
  $("fw-state").textContent = fw ? fw.status : "backend unreachable";
  if (!live) {
    const d = (fw && fw.data) || {};
    $("fw-summary").textContent = `${fw?.message || ""} · connectome ${d.phase || "idle"}${d.percent ? ` ${d.percent}%` : ""}${d.detail ? ` — ${d.detail}` : ""}${fw?.error ? ` · ${fw.error}` : ""}` +
      " · the engine is on the 80×80 stand-in until this finishes (it downloads ~130 MB from GitHub and builds the arrays once; afterwards it loads in under a second).";
    $("fallback-note").textContent = "IN USE RIGHT NOW: the real connectome is not loaded yet - see the state above.";
    $("fallback-details").open = true;
    return;
  }
  const mb = fw.mushroom_body || {}; const wb = fw.whole_brain || {};
  $("fw-summary").textContent = fw.message;
  const pops = [
    ["neurons", wb.neurons], ["connections", wb.connections], ["synapses", wb.synapses],
    ["mushroom body neurons", mb.neurons], ["MB connections", mb.connections],
    ["MB passes (every formula pass)", mb.passes], ["last MB pass", `${fmt(mb.last_pass_us)} µs`],
    ["whole-brain passes", wb.passes],
  ];
  $("fw-populations").innerHTML = pops.map(([k, v]) => `<span><span class="muted">${esc(k)}</span> <b>${typeof v === "number" ? fmt(v) : esc(v)}</b></span>`).join("");
  $("mb-stats").textContent = `${mb.labels?.length || 0} populations · ${fmt(mb.excitatory_blocks)} excitatory / ${fmt(mb.inhibitory_blocks)} inhibitory blocks · ${fmt(mb.connections)} real connections`;
  $("wb-stats").textContent = `${wb.labels?.length || 0} super-classes · ${fmt(wb.connections)} connections`;
  buildBlockGrid("mb-grid", mb);
  buildBlockGrid("wb-grid", wb);
  const edges = $("edge-body"); edges.innerHTML = "";
  (mb.edges || []).slice(0, 60).forEach((e) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${esc(mb.labels[e.source])}</td><td>${esc(mb.labels[e.target])}</td><td>${Math.abs(e.weight).toFixed(3)}</td>` +
      `<td style="color:${e.weight > 0 ? "#22c55e" : "#ef4444"}">${e.weight > 0 ? "+ excitatory" : "− inhibitory"}</td>`;
    edges.appendChild(tr);
  });
  renderReadout(wb.readout);
  $("fallback-note").textContent = "Not in use: the real connectome is loaded. Kept only for the seconds before the connectome is on disk on a brand-new machine.";
}

function renderReadout(r) {
  const body = $("wb-readout");
  if (!r) { body.innerHTML = `<tr><td class="muted">first whole-brain pass pending</td></tr>`; $("wb-when").textContent = "—"; return; }
  $("wb-when").textContent = `${Math.max(0, (Date.now() / 1000 - r.computed_at)).toFixed(0)} s ago · ${fmt(r.elapsed_us)} µs · ${r.hops} hops`;
  const rows = [
    ["balance (approach − avoid)", r.balance], ["confidence", r.confidence], ["MBON approach", r.approach], ["MBON avoid", r.avoid],
    ["active Kenyon cells", r.active_kcs], ["descending drive", r.descending],
    ["active per hop", (r.hop_active || []).join(" → ")],
    ["top MBONs", (r.top_mbons || []).slice(0, 4).map((m) => Array.isArray(m) ? m.join(" ") : (m.type || m.name || JSON.stringify(m))).join(", ")],
  ];
  body.innerHTML = rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${typeof v === "number" ? v.toFixed(4) : esc(v ?? "—")}</td></tr>`).join("");
}

async function refreshFlyWire() {
  const fw = await getJSON("/api/brain/matrix/flywire");
  renderFlyWire(fw);
  return fw;
}

(async function boot() {
  let fw = await refreshFlyWire();
  const data = await getJSON("/api/brain/matrix");
  if (data) {
    $("matrix-source").textContent = `${data.source} · checksum ${data.stats?.checksum} · gain ${data.gain}`;
    $("matrix-stats").textContent =
      `${data.stats?.edges} edges · excitatory ${data.stats?.positive_edges} · ` +
      `inhibitory ${data.stats?.inhibitory_edges} · mean |w| ${data.stats?.mean_abs_weight}`;
    $("pn-order").textContent = (data.node_types || []).slice(0, 20).join(" · ");
    buildGrid(data.edges || [], data.node_types || []);
  } else if (!fw) {
    $("src-label").textContent = "backend unreachable";
  }
  await fillTrace();
  setInterval(async () => {
    await fillTrace();
    const now = await getJSON("/api/brain/matrix/flywire");
    if (!now) return;
    if (!now.loaded || !(fw && fw.loaded)) { renderFlyWire(now); fw = now; }   // still loading, or just loaded: redraw
    else renderReadout(now.whole_brain && now.whole_brain.readout);
  }, 10000);
})();
