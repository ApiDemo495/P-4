/* 80x80 brain matrix viewer: heatmap, stats, strongest edges, last trace. */

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

/* The named circuit (Round S): neuron table + synapse-count edge list. */
async function fillNeurons() {
  const data = await getJSON("/api/brain/neurons?limit=60");
  if (!data) return;
  const t = data.totals || {};
  $("neuron-totals").textContent =
    `${t.nodes} nodes standing for ${t.cells_represented} cells · ${t.edges} connections · ` +
    `${(t.synapses || 0).toLocaleString()} synapses · ${t.excitatory_edges} excitatory / ${t.inhibitory_edges} inhibitory`;
  $("neuron-source").textContent =
    (data.live ? "✅ synapse counts measured live from neuPrint" : "ℹ " + (data.counts_note || "")) +
    ` · ${data.source || ""}`;
  const body = $("neuron-body");
  body.innerHTML = "";
  (data.neurons || []).forEach((n) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${n.index}</td><td><b>${esc(n.name)}</b>${n.receptor ? `<br><span class="muted">${esc(n.receptor)}</span>` : ""}</td>` +
      `<td>${esc(n.cell_type)}</td><td>${esc(n.population)}</td><td>${esc(n.compartment)}</td>` +
      `<td>${esc(n.transmitter)}</td><td>${n.cells}</td><td>${n.synapses_in}</td><td>${n.synapses_out}</td>` +
      `<td class="muted">${esc(n.role)}</td>`;
    body.appendChild(tr);
  });
  const edges = $("edge-body");
  edges.innerHTML = "";
  const names = (data.neurons || []).map((n) => n.name);
  (data.synapses || []).forEach((e) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${esc(names[e.source] || e.source)}</td><td>${esc(names[e.target] || e.target)}</td>` +
      `<td>${e.synapses}</td><td class="muted">${esc(e.pathway)}</td>` +
      `<td style="color:${e.sign > 0 ? "#22c55e" : "#ef4444"}">${e.sign > 0 ? "+" : "−"}</td>`;
    edges.appendChild(tr);
  });
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
    ["active Kenyon Cells (of 50)", trace.active_kcs],
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

(async function boot() {
  const data = await getJSON("/api/brain/matrix");
  if (!data) {
    $("src-label").textContent = "backend unreachable";
    return;
  }
  $("src-dot").className = "dot " + (data.source === "FALLBACK_CSV" ? "" : "on");
  $("src-label").textContent = data.source;
  $("matrix-source").textContent = `${data.source} · checksum ${data.stats?.checksum} · gain ${data.gain}`;
  $("matrix-stats").textContent =
    `${data.stats?.edges} edges · excitatory ${data.stats?.positive_edges} · ` +
    `inhibitory ${data.stats?.inhibitory_edges} · mean |w| ${data.stats?.mean_abs_weight}`;
  $("pn-order").textContent = (data.node_types || []).slice(0, 20).join(" · ");
  buildGrid(data.edges || [], data.node_types || []);
  await fillNeurons();
  await fillTrace();
  setInterval(fillTrace, 10000);
})();
