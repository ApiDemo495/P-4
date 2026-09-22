/* Settings screen: key testing, model upload, brain control, news sources. */

const $ = (id) => document.getElementById(id);

async function getJSON(url, options) {
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try { detail = (await res.json()).detail || detail; } catch (e) {}
      return { error: detail };
    }
    return await res.json();
  } catch (e) {
    return { error: String(e) };
  }
}

function line(id, text, cls = "") {
  const el = $(id);
  el.className = "status-line " + cls;
  el.textContent = text;
}

/* ---------------------------------------------------------------- keys */
document.querySelectorAll("[data-reveal]").forEach((btn) => {
  btn.onclick = () => {
    const input = $(btn.dataset.reveal);
    input.type = input.type === "password" ? "text" : "password";
  };
});

document.querySelectorAll("[data-test]").forEach((btn) => {
  btn.onclick = async () => {
    const slot = btn.dataset.test;
    const key = $(btn.dataset.input).value.trim();
    line(`status-${slot}`, "testing…", "warn");
    btn.disabled = true;
    const res = await getJSON(`/api/agents/${slot}/test`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    });
    btn.disabled = false;
    if (res.error) return line(`status-${slot}`, `❌ ${res.error}`, "err");
    if (res.valid) return line(`status-${slot}`, `✅ ${res.detail || "valid"}`, "ok");
    line(`status-${slot}`, `❌ ${res.error || "invalid"}`, "err");
  };
});

$("save-keys").onclick = async () => {
  const persist = $("persist").checked;
  const slots = ["gemini", "github", "cryptopanic", "newsapi"];
  const saved = [];
  for (const slot of slots) {
    const input = $(`key-${slot}`);
    if (!input || !input.value.trim()) continue;
    const res = await getJSON(`/api/settings/keys/${slot}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: input.value.trim(), persist }),
    });
    if (!res.error) saved.push(slot);
  }
  line("keys-note", saved.length ? `saved: ${saved.join(", ")}` : "nothing to save", "ok");
  setTimeout(() => { location.href = "/"; }, 700);
};

/* --------------------------------------------------------- local model */
async function refreshLocal() {
  const res = await getJSON("/api/agents/local/status");
  if (res.error) return line("local-status", res.error, "err");
  const mem = res.memory || {};
  const ready = ["ACTIVE", "STUB"].includes(res.status);
  const text =
    `${ready ? "✅" : "❌"} ${res.model || "No model loaded"}` +
    ` · status ${res.status}` +
    (res.parameter_count ? ` · ${res.parameter_count}` : "") +
    (mem.rss_human ? ` · RAM ${mem.rss_human}` : "") +
    (res.detail ? ` · ${res.detail}` : "");
  line("local-status", text, ready ? "ok" : "");
}

$("upload-model").onclick = async () => {
  const file = $("model-file").files[0];
  if (!file) return line("upload-status", "Pick a .gguf or .onnx file first", "warn");
  line("upload-status", `uploading ${file.name} (${(file.size / 1048576).toFixed(1)} MB)…`, "warn");
  const form = new FormData();
  form.append("file", file);
  const res = await getJSON("/api/agents/local/upload", { method: "POST", body: form });
  if (res.error) return line("upload-status", `❌ ${res.error}`, "err");
  if (res.success) {
    line("upload-status",
      `✅ ${res.model_name} (${res.parameter_count}) ready · test inference ${res.test_inference_ms}ms`,
      "ok");
  } else {
    line("upload-status", `❌ ${res.error}`, "err");
  }
  refreshLocal();
};

$("unload-model").onclick = async () => {
  const res = await getJSON("/api/agents/local/unload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ delete_file: false }),
  });
  line("upload-status", res.error ? `❌ ${res.error}` : "model unloaded", res.error ? "err" : "ok");
  refreshLocal();
};

$("stub-model").onclick = async () => {
  const res = await getJSON("/api/agents/local/stub?enabled=true", { method: "POST" });
  line("upload-status",
    res.error ? `❌ ${res.error}`
      : "dev stub loaded — clearly labelled STUB in the dashboard, no real weights",
    res.error ? "err" : "warn");
  refreshLocal();
};

/* ---------------------------------------------------------------- brain */
async function refreshBrain() {
  const res = await getJSON("/api/brain/status");
  if (res.error) return line("brain-status", res.error, "err");
  const h = res.health || {};
  line("brain-status",
    `${res.is_live ? "🟢" : "🟡"} ${res.status} · ${res.message}` +
    (res.matrix ? ` · matrix ${res.matrix.shape.join("×")} · checksum ${res.matrix.checksum}` : "") +
    (res.gain ? ` · gain ${res.gain}` : ""),
    res.is_live ? "ok" : "warn");

  const steps = $("brain-steps");
  steps.innerHTML = "";
  (res.steps || []).forEach((step) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${step.ok ? "✅" : "❌"}</td><td>${step.step}</td>` +
      `<td>${escapeHtml(step.detail || "")}</td><td>${step.elapsed_ms}ms</td>`;
    steps.appendChild(tr);
  });
  if (h.status) {
    const note = $("brain-note");
    note.className = "status-line " + (h.healthy ? "ok" : "warn");
    note.textContent = `health: ${h.status} · ${h.message} · neuPrint ${h.neuprint_live ? "reachable" : "unreachable"}`;
  }
}

$("brain-reconnect").onclick = async () => {
  line("brain-note", "reconnecting…", "warn");
  const res = await getJSON("/api/brain/reconnect", { method: "POST" });
  line("brain-note", res.error ? `❌ ${res.error}` : `✅ ${res.status} — ${res.detail}`, res.error ? "err" : "ok");
  refreshBrain();
};

$("brain-health").onclick = async () => {
  const res = await getJSON("/api/brain/health");
  if (res.error) return line("brain-note", `❌ ${res.error}`, "err");
  const ok = res.status === "HEALTHY";
  line("brain-note",
    `${ok ? "✅" : "❌"} ${res.status} · ${res.message} · output magnitude ${res.output_magnitude}` +
    ` · checksum ${res.matrix_checksum}`,
    ok ? "ok" : "err");
};

/* ----------------------------------------------------------------- news */
async function refreshNews() {
  const res = await getJSON("/api/news");
  if (res.error) return line("news-status", res.error, "err");
  const s = res.status || {};
  line("news-status",
    `coverage: ${s.coverage} · CryptoPanic ${s.cryptopanic} · NewsAPI ${s.newsapi} · RSS ${s.rss}` +
    ` · ${res.cache_size} cached headlines · NIV ${Number(res.niv).toFixed(3)}`,
    s.coverage === "full" ? "ok" : "warn");
}

$("news-poll").onclick = async () => {
  const res = await getJSON("/api/news/poll", { method: "POST" });
  line("news-note", res.error ? `❌ ${res.error}` : `polled: ${JSON.stringify(res.polled)} · NIV ${res.niv}`, res.error ? "err" : "ok");
  refreshNews();
};

$("test-rss").onclick = async () => {
  const res = await getJSON("/api/agents/rss/test", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
  });
  line("news-note", res.error ? `❌ ${res.error}` : `${res.detail}`, res.valid ? "ok" : "err");
};

$("test-emergency").onclick = async () => {
  const res = await getJSON("/api/news/emergency", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      headline: "Manual test event — exchange reports security breach (simulated)",
      reason: "Manual trigger from Settings",
    }),
  });
  line("news-note", res.error ? `❌ ${res.error}` : "⚡ emergency override broadcast to the dashboard", res.error ? "err" : "warn");
};

/* --------------------------------------------------------------- system */
async function refreshSystem() {
  const res = await getJSON("/api/system/config");
  if (res.error) return;
  $("system-info").innerHTML =
    `cycle <b>${res.cycle_period_seconds}s</b> (time scale ${res.time_scale}) · ` +
    `lock deadline ${res.lock_deadline_seconds}s · formula refresh ${res.formula_refresh_seconds}s<br>` +
    `signal threshold ±${res.signal_threshold} · min fusion confidence ${res.min_fusion_confidence} · ` +
    `emergency ${res.emergency_duration_seconds}s<br>` +
    `weights: drosophila ${res.weights.drosophila} / gemini ${res.weights.gemini} / ` +
    `local ${res.weights.local} / github ${res.weights.github}<br>` +
    `configured: ${Object.entries(res.configured).map(([k, v]) => `${k}=${v ? "yes" : "no"}`).join(" · ")}`;
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

(async function boot() {
  await refreshLocal();
  await refreshBrain();
  await refreshNews();
  await refreshSystem();
  setInterval(refreshLocal, 10000);
  setInterval(refreshBrain, 20000);
  setInterval(refreshNews, 20000);
})();
