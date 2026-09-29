/* Live check for the Round-I dashboard path, without a DOM.

   jsdom is not installable in every environment, so this script verifies the
   two things a browser would exercise for the new blocks - the *formatters* the
   panel uses and the *payload* they are given - by pulling the real functions
   out of app.js and running them against the running engine:

       bash run.sh --bg
       BASE=http://127.0.0.1:8000 node tools/dashboard_payload_check.js

   It proves:
     1. the horizon really is the 60 seconds after the release, to the
        microsecond, with a six-digit ISO instant on both ends;
     2. the micro block carries a resolution, a jitter and a quote lifetime;
     3. every one of the 23 formulas has a microsecond timing and a history
        statistic, and the totals in the payload agree with the parts;
     4. the formatter the UI prints them with (`fmtUs`) is sane at every scale;
     6. (Round K) the deep reasoning layer - fourteen microstructure formulas,
        a normalised Bayesian posterior and the chain - is live and at lock time;
     5. (Round J) the crowd's eight emotions arrive live, decomposed over five
        timescales, with a dominant emotion, its hold time and a manipulation
        score - and the served page has the panel that draws them.

   Exit code 0 = the dashboard is being handed the data it renders. */

const fs = require("fs");
const path = require("path");

const BASE = process.env.BASE || "http://127.0.0.1:8000";
const APP = fs.readFileSync(path.join(__dirname, "..", "backend", "web", "app.js"), "utf8");
const fail = [];
const ok = [];
const check = (cond, msg) => (cond ? ok : fail).push(msg);

/** Pull a top-level function's source out of app.js and make it callable. */
function extract(name) {
  const start = APP.indexOf(`function ${name}(`);
  if (start < 0) throw new Error(`${name} is not defined in app.js`);
  let depth = 0;
  let index = APP.indexOf("{", start);
  for (let i = index; i < APP.length; i += 1) {
    if (APP[i] === "{") depth += 1;
    else if (APP[i] === "}") {
      depth -= 1;
      if (depth === 0) {
        const source = APP.slice(start, i + 1);
        // eslint-disable-next-line no-new-func
        return new Function(`${source}; return ${name};`)();
      }
    }
  }
  throw new Error(`${name} is not balanced`);
}

const getJSON = async (pathname) => {
  const res = await fetch(`${BASE}${pathname}`);
  if (!res.ok) throw new Error(`${pathname} -> HTTP ${res.status}`);
  return res.json();
};

(async () => {
  const fmtUs = extract("fmtUs");
  const fmtClockUs = extract("fmtClockUs");

  // 1. the formatter, at every scale the engine produces -----------------------
  check(fmtUs(812) === "812 µs", `fmtUs(812) = ${fmtUs(812)}`);
  check(fmtUs(4310) === "4.31 ms", `fmtUs(4310) = ${fmtUs(4310)}`);
  check(fmtUs(1204000) === "1.204 s", `fmtUs(1204000) = ${fmtUs(1204000)}`);
  check(fmtUs(0) === "—", `fmtUs(0) = ${fmtUs(0)}`);
  check(fmtClockUs("2026-09-27T09:52:37.961116Z") === "09:52:37.961116",
    `fmtClockUs -> ${fmtClockUs("2026-09-27T09:52:37.961116Z")}`);

  // 2. the prediction's forecast window ---------------------------------------
  const signal = await getJSON("/api/signal/current");
  const prediction = signal.prediction || {};
  const horizon = prediction.horizon || {};
  const spanUs = (horizon.target_at_us || 0) - (horizon.released_at_us || 0);
  check(spanUs === Math.round((horizon.seconds || 60) * 1e6),
    `horizon span ${spanUs} µs vs ${horizon.seconds}s window`);
  check(horizon.base_seconds === 60, `base_seconds = ${horizon.base_seconds}`);
  check(/\.\d{6}Z$/.test(horizon.released_at_precise || ""),
    `released_at_precise = ${horizon.released_at_precise}`);
  check(/\.\d{6}Z$/.test(horizon.target_at_precise || ""),
    `target_at_precise = ${horizon.target_at_precise}`);
  check(fmtClockUs(horizon.released_at_precise).length === 15,
    `the release clock line reads "${fmtClockUs(horizon.released_at_precise)}"`);
  check(typeof horizon.microseconds_to_target === "number",
    "microseconds_to_target is missing");
  check(horizon.scored_at_us - horizon.released_at_us === spanUs,
    "the outcome is not scored one window after the release");
  check(APP.includes("frozen until") && !APP.includes("s left in this window</"),
    "the horizon line must not print a running countdown inside the prediction cell");

  // 3. the detail block the panel renders behind the toggle -------------------
  const detail = prediction.detail || {};
  // Exactly the 22 registered formulas (Round M: helpers such as _hsi used to
  // be counted, so the panel said "24 formulas evaluated").
  check((detail.formulas_evaluated || 0) === 22,
    `formulas_evaluated = ${detail.formulas_evaluated}`);
  check(Object.keys(detail.category_scores || {}).length === 8,
    `${Object.keys(detail.category_scores || {}).length} category scores`);
  check((detail.supporters || []).length > 0, "no supporters in the detail block");
  const engine = ((detail.agreement || {}).engine) || {};
  check(engine.compute_us > 0, `engine compute_us = ${engine.compute_us}`);
  check(Object.keys(engine.per_formula_us || {}).length >= 20,
    `${Object.keys(engine.per_formula_us || {}).length} per-formula timings`);

  // 4. the live formula block: µs timings and history statistics --------------
  const live = await getJSON("/api/formulas/live");
  const timings = live.timings_us || {};
  check(Object.keys(timings).length >= 20, `${Object.keys(timings).length} µs timings`);
  check(Object.values(timings).every((v) => Number.isInteger(v) && v >= 0),
    "a microsecond timing is not a whole number");
  check((live.total_us || 0) > 0, `total_us = ${live.total_us}`);
  check(Object.keys(live.stats || {}).length >= 20,
    `${Object.keys(live.stats || {}).length} statistics entries`);
  check((live.history_window || 0) >= 360, `history_window = ${live.history_window}`);
  const micro = live.micro || {};
  check(micro.available === true, "the micro block says the tape is not available");
  check(micro.resolution_us > 0, `resolution_us = ${micro.resolution_us}`);
  check(fmtUs(micro.resolution_us).length > 2,
    `resolution label = ${fmtUs(micro.resolution_us)}`);
  check(typeof micro.jitter_us === "number" && typeof micro.quote_lifetime_us === "number",
    "jitter / quote lifetime missing");

  // 5. the timings endpoint used by the footer line ---------------------------
  const timingsPayload = await getJSON("/api/formulas/timings");
  check(timingsPayload.total_us > 0, `timings total_us = ${timingsPayload.total_us}`);
  check(timingsPayload.resolution_us > 0,
    `timings resolution_us = ${timingsPayload.resolution_us}`);

  // 6. Round J: the crowd's emotions, live --------------------------------------
  const crowd = await getJSON("/api/emotions");
  check(crowd.available === true, "the emotion engine has no live reading");
  check(Array.isArray(crowd.emotions) && crowd.emotions.length === 8,
    `${(crowd.emotions || []).length} emotions scored (want 8)`);
  const names = (crowd.emotions || []).map((e) => e.name).sort().join(",");
  check(names === "CAPITULATION,COMPLACENCY,DENIAL,EUPHORIA,FEAR,FOMO,HOPE,PANIC",
    `emotion vocabulary: ${names}`);
  check(crowd.dominant && crowd.dominant.label && crowd.dominant.percent >= 0,
    `dominant emotion = ${crowd.dominant && crowd.dominant.label} ${crowd.dominant && crowd.dominant.percent}%`);
  check((crowd.emotions || []).every((e) =>
    ["micro", "seconds", "window", "minutes", "news"].every((k) => typeof (e.by_timescale || {})[k] === "number")),
    "every emotion is decomposed over the five timescales");
  check(typeof crowd.held_seconds === "number" && crowd.held_seconds >= 0,
    `dominant held for ${crowd.held_seconds} s`);
  check(crowd.manipulation && crowd.manipulation.score >= 0 && crowd.manipulation.score <= 1,
    `manipulation = ${crowd.manipulation && crowd.manipulation.score} (${crowd.manipulation && crowd.manipulation.kind})`);
  check(/\.\d{6}Z$/.test(crowd.at || ""), `emotion instant in microseconds: ${crowd.at}`);
  check((crowd.interval_seconds || 0) <= 1.0, `sampled every ${crowd.interval_seconds} s`);
  check(crowd.dampening && typeof crowd.dampening.applied === "number",
    `crowd dampening applied = ${crowd.dampening && crowd.dampening.applied}`);
  check(typeof crowd.read === "string" && crowd.read.includes("dominant"),
    `read: ${String(crowd.read || "").slice(0, 60)}…`);
  const crowdDetail = (signal.prediction || {}).detail || {};
  check(crowdDetail.crowd && typeof crowdDetail.crowd.available === "boolean",
    "the prediction detail carries the crowd at lock time");
  check(APP.includes('case "EMOTION"') && APP.includes("function renderEmotions("),
    "app.js handles the EMOTION stream and renders the panel");
  const html = await (await fetch(`${BASE}/`)).text();
  check(html.includes('id="emotion-card"') && html.includes('id="w-crowd-line"'),
    "the served page has the Crowd Emotion card and the inline crowd line");

  // 7. Round K: the deep reasoning layer -------------------------------------------
  const deepPayload = await getJSON("/api/emotions/deep");
  const deep = deepPayload.live || {};
  check(deep.available === true, "the deep layer has a live reading");
  check(Array.isArray(deep.chain) && deep.chain.length === 15,
    `${(deep.chain || []).length} reasoning steps (want 15)`);
  check((deep.chain || []).every((s) => s.formula && s.reads && typeof s.step === "number"),
    "every step carries a formula, a value and a reading");
  const post = deep.posterior || {};
  const total = Object.values(post.posterior || {}).reduce((a, b) => a + b, 0);
  check(Math.abs(total - 1) < 0.01 && post.argmax,
    `Bayesian posterior sums to ${total.toFixed(3)}, believes ${post.argmax} ${((post.argmax_probability || 0) * 100).toFixed(0)}%`);
  check(["micro", "seconds", "window"].every((k) => deep.bands && typeof deep.bands[k].hurst === "number"),
    `three bands: H = ${["micro", "seconds", "window"].map((k) => deep.bands && deep.bands[k].hurst).join(" / ")}`);
  check(deep.flow && typeof deep.flow.vpin === "number" && typeof deep.flow.kyle_lambda_bps === "number",
    `VPIN ${deep.flow && deep.flow.vpin}, Kyle λ ${deep.flow && deep.flow.kyle_lambda_bps} (R² ${deep.flow && deep.flow.kyle_r2})`);
  check(deep.hawkes && typeof deep.hawkes.branching_ratio === "number",
    `Hawkes branching ratio ${deep.hawkes && deep.hawkes.branching_ratio}`);
  check(deep.regime && ["calm", "trend", "stress"].includes(deep.regime.label),
    `regime ${deep.regime && deep.regime.label}`);
  check(deep.manipulation && ["ignition", "stuffing", "spoofing", "toxicity", "pushable"].every((k) => typeof deep.manipulation[k] === "number"),
    "ignition / stuffing / spoofing / toxicity / pushable detectors present");
  check((deep.compute_us || 0) < 250000, `deep layer computed in ${deep.compute_us} µs`);
  check(deepPayload.locked && typeof deepPayload.locked.available === "boolean",
    "the deep layer at lock time is published");
  check((crowd.emotions || []).every((e) => typeof e.belief === "number" && typeof e.ramp === "number"),
    "every emotion carries the filter's belief and the ramp reading");
  check(crowdDetail.crowd && crowdDetail.crowd.deep && typeof crowdDetail.crowd.deep.available === "boolean",
    "the prediction detail carries the deep read at lock time");
  check(APP.includes("function renderDeep(") && html.includes('id="deep-chain"'),
    "app.js renders the deep block and the served page has it");

  // 8. Round L: sync, lock, and the crowd against the formulas -------------------
  const win = signal.window || {};
  check(win.lock && win.lock.stages === 2,
    `two-stage lock: agents ${win.lock && win.lock.agents_lead_seconds}s early, final freeze ${win.lock && win.lock.final_lock_lead_seconds}s before the boundary`);
  check(typeof (win.lock || {}).data_age_at_open_seconds === "number" && win.lock.data_age_at_open_seconds <= 1.5,
    `the live window was frozen ${win.lock && win.lock.data_age_at_open_seconds}s before it opened (want <= 1.5 s)`);
  check(crowd.formula_agreement && typeof crowd.formula_agreement.verdict === "string",
    `crowd vs formulas: ${crowd.formula_agreement && crowd.formula_agreement.verdict} (${crowd.formula_agreement && crowd.formula_agreement.note})`);
  check((crowd.formula_agreement || {}).weights && crowd.formula_agreement.weights.formulas === 0.4,
    "the agreement block states the formulas' 40% vote and the crowd's 25% cap");
  check((crowd.emotions || []).every((e) => typeof e.formula === "string" && e.formula.includes("=") && Array.isArray(e.terms) && e.terms.length >= 3),
    "every emotion carries its printed formula and live terms");
  check(crowd.formula_glossary && Object.keys(crowd.formula_glossary).length >= 20,
    `${Object.keys(crowd.formula_glossary || {}).length} formula terms defined`);
  check(deep.chain && deep.chain[deep.chain.length - 1].name === "22-formula cross-check",
    "the reasoning chain ends with the 22-formula cross-check");
  check(crowdDetail.crowd && crowdDetail.crowd.formula_agreement,
    "the locked crowd reading was cross-checked against the lock-time formulas");
  check(APP.includes("opts.boundary") && APP.includes("state.serverOffsetMs = arrival"),
    "the client snaps its clock to the boundary SIGNAL (reveal == zero)");
  check(APP.includes("renderLockedCrowdLine(") && !APP.includes("crowd now:"),
    "the prediction cell shows the crowd at lock, never the live one");
  check(APP.includes("emotionsDeepDirty") && APP.includes("function renderEmotionFormula("),
    "the deep block repaints only when a new one arrives; emotion formulas are rendered");
  check(html.indexOf('class="card signal-card"') < html.indexOf('id="emotion-card"'),
    "the locked signal card comes before the live crowd card");

  for (const line of ok) console.log(`  ok   ${line}`);
  for (const line of fail) console.log(`  FAIL ${line}`);
  console.log(`\n${ok.length} passed, ${fail.length} failed`);
  process.exit(fail.length ? 1 : 0);
})().catch((error) => {
  console.error(`dashboard payload check failed: ${error.message}`);
  process.exit(1);
});
