#!/usr/bin/env node
/* Proves the prediction lock in a real DOM.
 *
 * Loads the served dashboard into jsdom, runs the real app.js against the
 * live engine (WebSocket and REST, exactly like a browser), and samples the
 * text of every element inside the PREDICTION cell four times a second.
 * Within a window nothing may change; at the boundary everything may.
 *
 *     npm install --no-save jsdom ws      (once; ~20 s)
 *     node tools/lock_watch.js [seconds=90] [base=http://localhost:8000]
 *
 * Exit code 0 = the cell was static inside every window observed.
 */
const path = require("path");
const fs = require("fs");
let JSDOM, WebSocket;
try { ({ JSDOM } = require("jsdom")); WebSocket = require("ws"); }
catch { console.error("run:  npm install --no-save jsdom ws"); process.exit(2); }

const SECONDS = Number(process.argv[2] || 90);
const BASE = process.argv[3] || "http://localhost:8000";
const CELL_IDS = [
  "w-prediction", "w-prediction-sub", "w-window-label", "w-fresh", "w-horizon-line",
  "w-micro-line", "w-crowd-text", "w-reasoning", "w-detail", "w-entry", "w-tp", "w-sl",
  "w-rr", "w-vol", "w-levels", "w-risk-note", "confidence-value", "reasoning", "weights",
];

(async () => {
  const html = await (await fetch(BASE + "/")).text();
  const dom = new JSDOM(html, { url: BASE + "/", runScripts: "outside-only", pretendToBeVisual: true });
  const w = dom.window;
  w.fetch = (u, o) => fetch(String(u).startsWith("http") ? u : BASE + u, o);
  w.WebSocket = WebSocket;
  w.navigator.vibrate = () => true;
  w.requestAnimationFrame = (cb) => setTimeout(() => cb(Date.now()), 16);
  w.cancelAnimationFrame = clearTimeout;
  w.performance = performance;
  const js = fs.readFileSync(path.join(__dirname, "..", "backend", "web", "app.js"), "utf8");
  w.eval(js);

  const last = {};
  let windowLabel = null;
  let violations = 0;
  let boundaries = 0;
  const t0 = Date.now();
  const text = (id) => (w.document.getElementById(id)?.textContent || "").replace(/\s+/g, " ").trim();

  const timer = setInterval(() => {
    const label = text("w-window-label");
    const stamp = ((Date.now() - t0) / 1000).toFixed(1).padStart(5) + "s";
    const boundary = label !== windowLabel;
    if (boundary) {
      if (windowLabel !== null) boundaries += 1;
      console.log(`${stamp}  ── ${label || "(no window yet)"} · countdown ${text("w-countdown")} · ${text("w-prediction")} ${text("confidence-value")}`);
      windowLabel = label;
      for (const id of CELL_IDS) last[id] = text(id);
      return;
    }
    if (!label || label === "—" || label === "starting up") return;
    for (const id of CELL_IDS) {
      const now = text(id);
      if (last[id] === undefined) { last[id] = now; continue; }
      // The first paint of a window may complete over two frames (the
      // boundary snapshot and the crowd line right behind it); give it 1 s.
      if (now !== last[id]) {
        const sinceBoundary = (Date.now() - boundaryAt) / 1000;
        if (sinceBoundary > 1.0) {
          violations += 1;
          console.log(`${stamp}  ✗ ${id} changed mid-window: ${JSON.stringify(last[id].slice(0, 70))} → ${JSON.stringify(now.slice(0, 70))}`);
        }
        last[id] = now;
      }
    }
  }, 250);
  let boundaryAt = Date.now();
  const origLog = console.log;
  console.log = (...a) => { if (String(a[0]).includes("──")) boundaryAt = Date.now(); origLog(...a); };

  setTimeout(() => {
    clearInterval(timer);
    origLog(`\n${boundaries} boundary(ies) crossed · ${violations} mid-window change(s) inside the prediction cell`);
    origLog(violations ? "FAIL: the prediction cell moved inside a window" : "OK: the prediction cell was static inside every window");
    process.exit(violations ? 1 : 0);
  }, SECONDS * 1000);
})().catch((e) => { console.error(e); process.exit(2); });
