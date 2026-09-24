/* Live-DOM check: load the running dashboard in a real DOM, let the app's own
   polling code paint the emergency override, then answer the only question that
   matters: does any element carrying page content get pinned to the viewport?

   jsdom does not do layout, so this cannot measure pixels.  It does have a CSS
   cascade, so it CAN answer "which stylesheets rules end up applying to the
   HOLD box while an override is active" - which is exactly what broke.

   How to run it (optional; the committed pytest guard covers the same ground
   without a browser):

       bash run.sh --bg
       curl -s -X POST localhost:8000/api/news/emergency \
            -H 'Content-Type: application/json' \
            -d '{"headline":"manual test","source":"test"}' -o /dev/null
       mkdir -p /tmp/domcheck && cd /tmp/domcheck && npm install jsdom
       BASE=http://127.0.0.1:8000 node /path/to/tools/ui_override_check.js

   Exit code 0 = every widget stayed in the page while the override was live. */

const { JSDOM } = require("jsdom");

const BASE = process.env.BASE || "http://127.0.0.1:8000";
const fail = [];
const ok = [];

function check(cond, msg) {
  (cond ? ok : fail).push(msg);
}

const virtualConsole = new (require("jsdom").VirtualConsole)();
virtualConsole.on("jsdomError", () => {});
virtualConsole.on("error", () => {});
virtualConsole.on("warn", () => {});

JSDOM.fromURL(BASE + "/", {
  resources: "usable",
  runScripts: "dangerously",
  pretendToBeVisual: true,
  virtualConsole,
})
  .then(async (dom) => {
    const { window } = dom;
    // The dashboard polls /api/signal/status every couple of seconds.  An
    // override is armed on the server for this run, so waiting is the same as
    // being a real browser during a news emergency.
    await new Promise((r) => setTimeout(r, 6000));

    const doc = window.document;
    const hold = doc.getElementById("w-hold");
    check(!!hold, "the HOLD box is in the page");
    check(!!doc.getElementById("w-emergency"), "the inline emergency chip is in the page");
    check(!!doc.getElementById("prediction") || !!doc.getElementById("w-prediction"),
      "the prediction element is in the page");

    console.log("hold box classes :", hold ? hold.className : "(missing)");
    console.log("hold box title   :", doc.getElementById("w-hold-title")?.textContent);
    console.log("hold box text    :", doc.getElementById("w-hold-text")?.textContent?.slice(0, 90));
    console.log("chip hidden?     :", doc.getElementById("w-emergency")?.classList.contains("hidden"));
    console.log("chip text        :", doc.getElementById("w-emergency-text")?.textContent?.slice(0, 90));

    const cls = hold ? hold.className.split(/\s+/) : [];
    check(cls.includes("override"), "the HOLD box carries the scoped `override` state class");
    check(!cls.includes("emergency"), "the HOLD box does NOT carry a bare `emergency` class");
    check(doc.getElementById("w-hold-title")?.textContent.includes("HOLD"),
      "the HOLD box announces the forced HOLD");
    check(doc.getElementById("w-emergency") && !doc.getElementById("w-emergency").classList.contains("hidden"),
      "the inline chip is visible");

    // ---- which rules actually pin something to the viewport? -------------
    const pinned = [];
    for (const sheet of Array.from(doc.styleSheets)) {
      let rules = [];
      try { rules = Array.from(sheet.cssRules || []); } catch { continue; }
      for (const rule of rules) {
        if (!rule.selectorText || !rule.style) continue;
        if (!/position\s*:\s*fixed/.test(rule.style.cssText || "")) continue;
        pinned.push(rule.selectorText);
      }
    }
    console.log("full-screen rules in the live stylesheet:", pinned);

    const contentHits = [];
    for (const sel of pinned) {
      for (const part of sel.split(",")) {
        let matched = [];
        try { matched = Array.from(doc.querySelectorAll(part.trim())); } catch { }
        for (const el of matched) {
          if (el.closest(".modal")) continue;               // dialogs are allowed
          contentHits.push(`${part.trim()} -> <${el.tagName.toLowerCase()} id="${el.id}" class="${el.className}">`);
        }
      }
    }
    check(contentHits.length === 0,
      "no page element is pinned over the viewport (findings: " + (contentHits.join(" | ") || "none") + ")");

    // ---- computed style of the widget itself ------------------------------
    const computed = window.getComputedStyle(hold);
    console.log("computed position of #w-hold:", computed.position, "| max-height:", computed.maxHeight);
    check(computed.position !== "fixed" && computed.position !== "absolute",
      "the HOLD box stays in the document flow (computed position: " + computed.position + ")");

    // ---- the rest of the dashboard must still be reachable ----------------
    const wanted = ["w-prediction", "w-hold", "w-tp", "w-sl", "w-countdown", "price", "news-list"];
    for (const id of wanted) {
      const el = doc.getElementById(id);
      check(!!el, `#${id} survived the override (still in the DOM)`);
    }

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
