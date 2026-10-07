/* Settings screen: key rings (3 slots + failover), up to 3 local models, brain control, news sources. */

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
    const provider = btn.dataset.test;
    const slot = btn.dataset.slot || "1";
    const key = $(btn.dataset.input).value.trim();
    line(`status-${provider}`, `testing slot ${slot}…`, "warn");
    btn.disabled = true;
    const res = await getJSON(`/api/agents/${provider}/test`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    });
    btn.disabled = false;
    if (res.error) return line(`status-${provider}`, `❌ slot ${slot}: ${res.error}`, "err");
    if (res.valid) return line(`status-${provider}`, `✅ slot ${slot}: ${res.detail || "valid"}`, "ok");
    line(`status-${provider}`, `❌ slot ${slot}: ${res.error || "invalid"}`, "err");
  };
});

const RING_PROVIDERS = ["gemini", "github", "cryptopanic", "newsapi"];

/* Which slot each ring is using right now, and why the others are resting. */
async function refreshRings() {
  const res = await getJSON("/api/settings/keys");
  if (res.error) return;
  for (const provider of RING_PROVIDERS) {
    const ring = (res.rings || {})[provider];
    if (!ring || !ring.slots) continue;
    ring.slots.forEach((slot) => {
      const role = $(`role-${provider}-${slot.slot}`);
      const input = $(`key-${provider}-${slot.slot}`);
      if (!role || !input) return;
      role.className = "key-role" + (slot.state === "in use" ? " active" : slot.state === "cooling" ? " cooling" : "");
      role.textContent = slot.slot === 1 ? "primary" : `backup ${slot.slot - 1}`;
      role.title = slot.state === "cooling"
        ? `cooling ${slot.cooldown_seconds}s — ${slot.reason}`
        : slot.state;
      if (slot.configured && !input.value && !input.dataset.touched) input.placeholder = `${slot.masked} (saved)`;
    });
    if (!ring.configured) {
      line(`status-${provider}`, "no key saved", "");
    } else {
      const active = ring.slots.find((s) => s.slot === ring.active_slot);
      const cooling = ring.slots.filter((s) => s.state === "cooling")
        .map((s) => `slot ${s.slot} cooling ${s.cooldown_seconds}s (${s.reason})`).join(" · ");
      line(`status-${provider}`,
        (ring.all_cooling ? "⏸ all keys cooling down" : `▶ using slot ${ring.active_slot} (${active ? active.masked : ""})`) +
        (ring.on_primary ? " · on primary" : ring.all_cooling ? "" : " · primary will be retried automatically") +
        (cooling ? ` · ${cooling}` : "") +
        ` · ${ring.configured_slots}/3 configured`,
        ring.all_cooling ? "warn" : ring.on_primary ? "ok" : "warn");
    }
  }
}

document.querySelectorAll("input[data-provider]").forEach((input) => {
  input.addEventListener("input", () => { input.dataset.touched = "1"; });
});

/* Round AO: proof that a key is contributing - not just stored. */
async function refreshEffects() {
  const res = await getJSON("/api/settings/effects");
  const body = document.querySelector("#effects-table tbody");
  if (!body) return;
  if (res.error) { body.innerHTML = `<tr><td class="err">${escapeHtml(res.error)}</td></tr>`; return; }
  body.innerHTML = "";
  Object.entries(res).forEach(([name, row]) => {
    if (name.startsWith("_")) return;
    const ok = row.configured && /voting|flowing|live|feeding/.test(String(row.effect || ""));
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${ok ? "🟢" : row.configured ? "🟠" : "⚪"}</td><td><b>${escapeHtml(name)}</b></td>` +
      `<td>${escapeHtml(String(row.effect || ""))}</td>` +
      `<td class="muted">${escapeHtml([row.status, row.model, row.detail].filter(Boolean).join(" · "))}</td>`;
    body.appendChild(tr);
  });
  const p = res._persistence || {};
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${p.env_exists ? "💾" : "⚠️"}</td><td><b>persistence</b></td><td colspan="2" class="muted">${escapeHtml((p.env_exists ? ".env present · " : ".env not written yet · ") + (p.note || ""))}</td>`;
  body.appendChild(tr);
}
const effectsBtn = $("effects-refresh");
if (effectsBtn) effectsBtn.onclick = refreshEffects;
refreshEffects();

$("save-keys").onclick = async () => {
  const persist = $("persist").checked;
  const saved = [];
  // Three-slot providers: send all three boxes; an untouched box keeps the
  // stored key, an emptied (touched) box clears that slot.
  for (const provider of RING_PROVIDERS) {
    const boxes = [1, 2, 3].map((n) => $(`key-${provider}-${n}`));
    if (!boxes.some((b) => b && b.dataset.touched)) continue;
    for (const box of boxes) {
      if (!box || !box.dataset.touched) continue;
      const res = await getJSON(`/api/settings/keys/${provider}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: box.value.trim(), persist, slot: Number(box.dataset.slot) }),
      });
      if (!res.error) saved.push(`${provider} slot ${box.dataset.slot}`);
    }
  }
  for (const slot of ["neuprint", "cave", "glassnode", "twelvedata", "lunarcrush"]) {
    const input = $(`key-${slot}`);
    if (!input || !input.value.trim()) continue;
    const res = await getJSON(`/api/settings/keys/${slot}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: input.value.trim(), persist }),
    });
    if (!res.error) saved.push(slot);
  }
  line("keys-note", saved.length ? `saved: ${saved.join(", ")} · applying…` : "nothing to save", "ok");
  // Round AO: a saved key must DO something now, not at some later poll.
  if (saved.includes("neuprint") || saved.includes("cave")) {
    line("keys-note", `saved: ${saved.join(", ")} · re-verifying the brain with the new token…`, "ok");
    await getJSON("/api/brain/reconnect", { method: "POST" });
    await refreshBrain();
  }
  if (saved.some((s) => s.startsWith("newsapi") || s.startsWith("cryptopanic"))) {
    await getJSON("/api/news/poll", { method: "POST" });
  }
  for (const provider of ["gemini", "github"]) {
    if (saved.some((s) => s.startsWith(provider))) await getJSON(`/api/agents/${provider}/test`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key: "" }) });
  }
  await refreshRings();
  await refreshEffects();
  line("keys-note", saved.length
    ? `saved: ${saved.join(", ")} · ${persist ? "written to .env" : "memory only (persist was off)"} · see "What each key is doing right now" below; the dashboard is at /`
    : "nothing to save", "ok");
};

/* --------------------------------------------------------- local models */
async function refreshLocal() {
  const res = await getJSON("/api/agents/local/status");
  if (res.error) return line("local-status", res.error, "err");
  const mem = res.memory || {};
  const ready = ["ACTIVE", "STUB"].includes(res.status);
  const text =
    `${ready ? "✅" : "❌"} ${res.models_loaded || 0}/3 models loaded` +
    (res.model ? ` · ${res.model}` : "") +
    ` · status ${res.status}` +
    (mem.rss_human ? ` · RAM ${mem.rss_human}` : "") +
    (res.detail ? ` · ${res.detail}` : "");
  line("local-status", text, ready ? "ok" : "");
  (res.slots || []).forEach((slot) => {
    const el = $(`model-status-${slot.slot}`);
    if (!el) return;
    if (!slot.ready) {
      line(`model-status-${slot.slot}`,
        slot.status === "ERROR" ? `❌ ${slot.detail || slot.last_error || "crashed"}` : (slot.file ? `unloaded · ${slot.file}` : "empty"),
        slot.status === "ERROR" ? "err" : "");
      return;
    }
    line(`model-status-${slot.slot}`,
      `✅ ${slot.model || "model"}` +
      (slot.parameter_count ? ` · ${slot.parameter_count}` : "") +
      (slot.kind ? ` · ${slot.kind}` : "") +
      (slot.last_decision ? ` · last ${slot.last_decision} ${Math.round((slot.last_confidence || 0) * 100)}% in ${slot.last_latency_ms}ms` : ` · ${slot.detail || "ready"}`),
      slot.stub ? "warn" : "ok");
  });
}

document.querySelectorAll("[data-upload]").forEach((btn) => {
  btn.onclick = async () => {
    const slot = btn.dataset.upload;
    const file = $(`model-file-${slot}`).files[0];
    if (!file) return line(`model-status-${slot}`, "Pick a .gguf or .onnx file first", "warn");
    line(`model-status-${slot}`, `uploading ${file.name} (${(file.size / 1048576).toFixed(1)} MB) into slot ${slot}…`, "warn");
    btn.disabled = true;
    const form = new FormData();
    form.append("file", file);
    const res = await getJSON(`/api/agents/local/upload?slot=${slot}`, { method: "POST", body: form });
    btn.disabled = false;
    if (res.error) line(`model-status-${slot}`, `❌ ${res.error}`, "err");
    else if (res.success) {
      line(`model-status-${slot}`,
        `✅ ${res.model_name} (${res.parameter_count}) ready · test inference ${res.test_inference_ms}ms`, "ok");
    } else line(`model-status-${slot}`, `❌ ${res.error}`, "err");
    refreshLocal();
  };
});

document.querySelectorAll("[data-unload]").forEach((btn) => {
  btn.onclick = async () => {
    const slot = Number(btn.dataset.unload);
    const res = await getJSON("/api/agents/local/unload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ delete_file: false, slot }),
    });
    line(`model-status-${slot}`, res.error ? `❌ ${res.error}` : `slot ${slot} unloaded`, res.error ? "err" : "ok");
    refreshLocal();
  };
});

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
    const icon = step.state === "skip" ? "⏭️" : (step.ok ? "✅" : "❌");
    tr.innerHTML = `<td title="${step.state || (step.ok ? "pass" : "fail")}">${icon}</td><td>${step.step}</td>` +
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
    `local ${res.weights.local} / github ${res.weights.github} / formula consensus ${res.weights.formulas ?? 0} / physics layer ${res.weights.physics ?? 0} (renormalised; simulator tape counts half)<br>` +
    `configured: ${Object.entries(res.configured).map(([k, v]) => `${k}=${v ? "yes" : "no"}`).join(" · ")}`;
}

async function refreshAutostart() {
  const res = await getJSON("/api/system/autostart?lines=40");
  if (res.error) return line("autostart-summary", res.error, "err");
  $("autostart-verdict").textContent = res.in_codespace ? `codespace ${res.codespace}` : "not running in a Codespace";
  line("autostart-summary",
    `${res.healthy ? "✅" : "⚠️"} ${res.verdict} · provisioned ${res.provisioned ? "yes" : "no"}` +
    ` · failures ${res.failures.length} · warnings ${res.warnings.length}`,
    res.healthy ? "ok" : "warn");
  const body = $("autostart-journal");
  body.innerHTML = "";
  (res.journal || []).slice(-25).forEach((row) => {
    const tr = document.createElement("tr");
    const icon = row.level === "fail" ? "❌" : row.level === "warn" ? "⚠️" : row.level === "ok" ? "✅" : "·";
    tr.innerHTML = `<td>${icon}</td><td class="muted">${escapeHtml(row.at)}</td><td>${escapeHtml(row.hook)}</td><td>${escapeHtml(row.message)}</td>`;
    body.appendChild(tr);
  });
  const logs = res.logs || {};
  const blocks = [];
  (logs.setup_passes || []).forEach((f) => blocks.push(`── ${f.path} (${f.bytes} B)\n${f.tail.join("\n")}`));
  ["pip", "server"].forEach((k) => {
    const f = logs[k];
    if (f) blocks.push(`── ${f.path} ${f.exists ? `(${f.bytes} B, ${f.age_seconds}s old)` : "(not present)"}\n${f.tail.join("\n")}`);
  });
  $("autostart-logs").textContent = blocks.join("\n\n");
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

(async function boot() {
  await refreshRings();
  await refreshLocal();
  await refreshBrain();
  await refreshNews();
  await refreshSystem();
  await refreshAutostart();
  setInterval(refreshLocal, 10000);
  setInterval(refreshRings, 10000);
  setInterval(refreshBrain, 20000);
  setInterval(refreshNews, 20000);
  setInterval(refreshAutostart, 30000);
})();
