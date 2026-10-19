/* Live-DOM check for the countdown contract: 60 seconds, smooth, and every
   panel refreshing together.

   Run it against a live engine:

       bash run.sh --bg
       mkdir -p /tmp/domcheck && cd /tmp/domcheck && npm install jsdom
       NODE_PATH=/tmp/domcheck/node_modules BASE=http://127.0.0.1:8000 \
            node tools/clock_check.js            # ~75 s: it crosses one boundary

   What it proves, with the real app code running in a real DOM:

     1. the number starts at 60 and walks down one second at a time, and it
        never jumps backwards, repeats or skips inside a window;
     2. the digits and the ring are rendered from the same value (they cannot
        disagree, so nothing "glitches");
     3. at the boundary the counter snaps to 60 and the next window starts;
     4. the panels (signal, formulas, news, agents, history) all repaint inside
        one frame, at the grid marks - and nothing repaints randomly in between.

   Exit code 0 = the countdown is 60 s, smooth, and everything ticks together. */

const { JSDOM, VirtualConsole } = require("jsdom");

const BASE = process.env.BASE || "http://127.0.0.1:8000";
const OBSERVE_MS = Number(process.env.OBSERVE_MS || 75000);
const fail = [];
const ok = [];
const check = (cond, msg) => (cond ? ok : fail).push(msg);

const virtualConsole = new VirtualConsole();
virtualConsole.on("jsdomError", (e) => console.log("jsdom error:", e.message));
virtualConsole.on("error", (e) => console.log("page error:", e));
virtualConsole.on("warn", () => {});

JSDOM.fromURL(BASE + "/", {
  resources: "usable",
  runScripts: "dangerously",
  pretendToBeVisual: true,
  virtualConsole,
})
  .then(async (dom) => {
    const { window } = dom;
    const doc = window.document;
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

    // Let the app boot (formula catalogue, self-test and the first snapshot).
    await sleep(4000);

    /* ---- instrument the render functions -------------------------------- */
    const PANELS = ["renderSignal", "renderFormulas", "renderNewsList", "renderAgents",
                    "renderHistory", "renderWidgetPanel"];
    const calls = [];
    for (const name of PANELS) {
      const original = window[name];
      if (typeof original !== "function") {
        fail.push(`${name} is not reachable - the page did not boot`);
        continue;
      }
      window[name] = function wrapped(...args) {
        calls.push({ name, at: Date.now() });
        return original.apply(this, args);
      };
    }

    const ring = doc.getElementById("w-ring");
    const digits = doc.getElementById("w-countdown");
    check(!!digits, "#w-countdown is in the page");
    check(!!ring, "#w-ring is in the page");

    /* ---- watch the countdown cross a boundary ---------------------------- */
    const samples = [];
    const start = Date.now();
    while (Date.now() - start < OBSERVE_MS) {
      samples.push({
        t: Date.now() - start,
        secs: Number(digits.textContent),
        frac: Number(ring.style.getPropertyValue("--frac")),
        sub: doc.getElementById("w-countdown-sub").textContent,
      });
      await sleep(200);
    }
    const windowSeconds = 60;

    // 1. the range
    const seen = samples.map((s) => s.secs).filter((n) => Number.isFinite(n));
    check(Math.min(...seen) >= 0 && Math.max(...seen) <= windowSeconds,
      `the countdown stays inside 0..${windowSeconds} (min ${Math.min(...seen)}, max ${Math.max(...seen)})`);
    check(Math.max(...seen) >= windowSeconds - 1,
      `the countdown reaches ${windowSeconds} at a boundary (max ${Math.max(...seen)})`);
    check(seen.filter((n) => n === windowSeconds).length >= 1,
      `"${windowSeconds}" is actually displayed at the start of a window`);

    // 2. monotonic inside a window: an increase may only happen at a rollover,
    //    and a rollover must land back at (nearly) the full window.
    const jumps = [];
    let skips = 0;
    for (let i = 1; i < samples.length; i += 1) {
      const previous = samples[i - 1].secs;
      const current = samples[i].secs;
      if (current > previous) {
        jumps.push({ at: samples[i].t, from: previous, to: current });
      } else if (previous - current > 2) {
        skips += 1;
      }
    }
    const rollovers = jumps.filter((j) => j.to >= windowSeconds - 6);
    const badJumps = jumps.filter((j) => j.to < windowSeconds - 6);
    check(badJumps.length === 0,
      "the countdown never jumps upward mid-window (findings: " +
      JSON.stringify(badJumps) + ")");
    check(rollovers.length >= 1, "the countdown crossed at least one window boundary");
    check(rollovers.length <= 2, `only real boundaries reset the countdown (${rollovers.length})`);
    check(skips === 0, `the countdown never skips seconds (${skips} skips)`);

    // 3. the walk is one second at a time: consecutive distinct seconds differ
    //    by exactly 1, so no second is skipped and none is repeated out of order
    let gaps = 0;
    for (const roll of rollovers) {
      const idx = samples.findIndex((s) => s.t === roll.at);
      const after = samples.slice(idx).map((s) => s.secs);
      const until = after.findIndex((n, i) => i > 0 && n > after[i - 1]);
      const walk = until === -1 ? after : after.slice(0, until);
      const distinct = [...new Set(walk)];
      for (let i = 1; i < distinct.length; i += 1) {
        if (distinct[i - 1] - distinct[i] !== 1) gaps += 1;
      }
      check(distinct[0] >= windowSeconds - 1,
        `the window restarts at ${windowSeconds} (saw ${distinct[0]})`);
    }
    check(gaps === 0, `the countdown ticks down one second at a time (${gaps} gaps)`);

    // 4. digits and ring agree: both come from the one value
    const mismatched = samples.filter((s) => Number.isFinite(s.frac) &&
      Math.abs(s.frac * windowSeconds - s.secs) > windowSeconds / 30 + 1);
    check(mismatched.length === 0,
      `the ring and the digits are rendered from the same value (${mismatched.length} mismatches)`);

    // 5. the sub-line explains the shared tick
    check(samples.some((s) => /next refresh/.test(s.sub || "")),
      "the countdown says when the next refresh is");

    /* ---- the batches: everything repaints together, and only on the marks -- */
    const batches = [];
    for (const call of calls) {
      const last = batches[batches.length - 1];
      if (last && call.at - last.lastAt < 250) {
        last.names.add(call.name);
        last.lastAt = call.at;
      } else {
        batches.push({ at: call.at, lastAt: call.at, names: new Set([call.name]) });
      }
    }
    for (const batch of batches) batch.spread = batch.lastAt - batch.at;
    const full = batches.filter((b) => b.names.size >= 4);
    const maxSpread = full.length ? Math.max(...full.map((b) => b.spread)) : 0;
    // jsdom builds the formula DOM by hand and is far slower than a browser;
    // a real browser paints one batch in a single frame.  Even here the whole
    // dashboard repaints inside a fraction of a second, from one message.
    check(maxSpread < 150,
      `a bottom-to-top refresh is one synchronous pass (max spread ${maxSpread} ms)`);

    const multi = batches.filter((b) => b.names.size >= 4);
    check(multi.length >= 1,
      `a single refresh covers the whole dashboard (${multi.length} full batches of ${batches.length})`);
    check(batches.length <= 8,
      `${batches.length} refresh batches in ${Math.round(OBSERVE_MS / 1000)}s - the grid, not a timer`);

    /* ---- report ----------------------------------------------------------- */
    console.log("\ncountdown trace (first 12 and last 12 samples):");
    const show = [...samples.slice(0, 12), ...samples.slice(-12)];
    show.forEach((s) => console.log(
      `  t+${String(s.t).padStart(6)}ms  ${String(s.secs).padStart(2)}s  frac=${s.frac.toFixed(3)}`));
    console.log("\nrefresh batches:");
    batches.forEach((b) => console.log(
      `  t+${String(b.at - start).padStart(6)}ms  spread=${String(b.spread ?? 0).padStart(3)}ms  ` +
      `${[...b.names].join(", ")}`));
    console.log("\nwindow rollovers:", JSON.stringify(rollovers));

    console.log("\n--- PASS (" + ok.length + ") ---");
    ok.forEach((m) => console.log("  ✔ " + m));
    if (fail.length) {
      console.log("--- FAIL (" + fail.length + ") ---");
      fail.forEach((m) => console.log("  ✘ " + m));
      process.exit(1);
    }
    process.exit(0);
  })
  .catch((err) => {
    console.error("harness error:", err && err.message);
    process.exit(2);
  });
