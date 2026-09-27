#!/usr/bin/env python
"""Find definitions that nothing uses, across every language in the repo.

The user's rule: *"remove unnecessary code that isn't in use but present."*  That
is easy to say and easy to get wrong, so the check is a tool with an explicit
allowlist rather than a one-off sweep:

* **Python** - functions, methods and classes in ``backend/`` and ``tools/`` that
  no other line in the repository mentions.  Decorated definitions are skipped
  (a FastAPI handler is registered by its decorator), as is anything in the
  allowlist below (public API surface, entry points, the scanner itself).
* **JavaScript** - functions in ``backend/web/*.js`` that are declared but never
  called, and element ids the script looks up that no page renders.
* **CSS** - classes in ``styles.css`` that no page or script uses.
* **Dart** - private classes and top-level functions in ``frontend/lib`` with no
  reference anywhere.

Run it directly for a report::

    python tools/dead_code.py            # report, exit 1 if anything is unused
    python tools/dead_code.py --list      # the same, with locations

``backend/tests/test_dead_code.py`` runs it with the allowlists and fails the
build when something new becomes unreachable.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Definitions that are deliberately kept although nothing calls them here.
#: Every entry needs a reason: this list is the only thing between "unused" and
#: "public", and it is exactly what a reviewer should be arguing about.
ALLOW: dict[str, str] = {
    # Entry points and lifecycle hooks
    "main": "process entry point / FastAPI lifespan hook",
    "run": "documented entry point",
    "boot": "browser entry point",
    # FastAPI / ASGI surface that is documented in README and /docs
    "health": "public endpoint + used by the supervisor",
    "matrix_payload": "public endpoint payload builder",
    "summary": "public introspection helper used from the CLI",
    "to_dict": "serialisation protocol used by many callers",
    "__init__": "constructor",
    "metadata": "public catalogue endpoint",
    # Brain/matrix generation scripts (run by hand, not imported)
    "build": "named constructor used by tests and tools",
    "generate": "offline generator entry point",
    "explain": "public brain explainer (imported by routes and manager)",
    "search": "documented CLI",
    # Accent classes applied by composing a string at runtime
    "buy": "accent on .signal-badge/.prediction-value, applied via String(s.signal).toLowerCase()",
    "sell": "same composition as 'buy'",
    "set": "written inside a longer className string (\"key-state\" + \" set\")",
    # Anything the scanner itself defines
    "scan_python": "this tool",
    "scan_js": "this tool",
    "scan_css": "this tool",
    "scan_dart": "this tool",
}

PY_PATTERN = re.compile(
    r"^(?:    )?(?:async )?def ([a-zA-Z_][a-zA-Z0-9_]*)\(|^class ([A-Za-z_][a-zA-Z0-9_]*)",
    re.M,
)
JS_FUNC = re.compile(r"^(?:async )?function ([a-zA-Z_][a-zA-Z0-9_]*)", re.M)
JS_CONST = re.compile(r"^(?:const|let|var) ([A-Za-z_][a-zA-Z0-9_]*) = (?:\(|function|async)", re.M)
DART_PRIVATE = re.compile(r"^(?:class|mixin|enum) (_[A-Za-z0-9_]+)|^[A-Za-z<>, ?]+ (_[a-zA-Z0-9_]+)\(", re.M)


def _python_files() -> list[Path]:
    """Production Python only: tests are collected by pytest, not called."""
    return [
        p
        for p in sorted((ROOT / "backend").rglob("*.py")) + sorted((ROOT / "tools").rglob("*.py"))
        if "__pycache__" not in str(p)
        and "tests" not in p.parts
        and p.name != "dead_code.py"
    ]


def _count(name: str, files: list[Path]) -> int:
    pattern = re.compile(r"\b" + re.escape(name) + r"\b")
    total = 0
    for path in files:
        try:
            total += len(pattern.findall(path.read_text(errors="ignore")))
        except OSError:
            continue
    return total


def scan_python() -> list[tuple[str, str]]:
    """(name, location) for every definition nothing references."""
    files = _python_files()
    extra = [ROOT / "run.sh", ROOT / "frontend"] + sorted((ROOT / "docs").glob("*.md"))
    findings: list[tuple[str, str]] = []
    for path in files:
        text = path.read_text(errors="ignore")
        for match in PY_PATTERN.finditer(text):
            name = match.group(1) or match.group(2)
            if name.startswith("__") or name in ALLOW:
                continue
            head = text[: match.start()].rstrip().splitlines()
            decorated = bool(head) and head[-1].strip().startswith("@")
            if decorated:
                continue
            hits = _count(name, files + [p for p in extra if p.exists() or p.is_dir()])
            if hits <= 1:
                line = text[: match.start()].count("\n") + 1
                findings.append((name, f"{path.relative_to(ROOT)}:{line}"))
    return findings


def scan_js() -> list[str]:
    """Functions declared in the browser bundle that nothing ever calls."""
    findings: list[str] = []
    for path in sorted((ROOT / "backend" / "web").glob("*.js")):
        text = path.read_text(errors="ignore")
        for match in list(JS_FUNC.finditer(text)) + list(JS_CONST.finditer(text)):
            name = match.group(1)
            calls = len(re.findall(r"\b" + re.escape(name) + r"\s*\(", text))
            uses = len(re.findall(r"\b" + re.escape(name) + r"\b", text))
            if name in ALLOW:
                continue
            if calls == 0 and uses <= 1:
                line = text[: match.start()].count("\n") + 1
                findings.append(f"{path.name}:{line} {name}")
    return findings


def _css_consumers() -> set[str]:
    """Class names the pages and scripts can actually put on an element.

    A plain substring search calls ``.buy`` used because the word "buy" appears
    somewhere in ``app.js``.  This collects the real places a class name is
    written: ``class="…"`` attributes, ``classList.add/toggle/…``, ``className =
    …``, quoted tokens, and the families that scripts build with a template
    literal (``sig-${kind}`` -> ``sig-BUY``).  Test files count too, so a class a
    guard asserts on can never be deleted by accident.
    """
    web = ROOT / "backend" / "web"
    names: set[str] = set()
    dynamic: list[str] = []
    for page in ("index.html", "matrix.html", "settings.html"):
        path = web / page
        if not path.exists():
            continue
        for attr in re.findall(r'class="([^"]+)"', path.read_text(errors="ignore")):
            names.update(attr.split())
    for script in ("app.js", "matrix.js", "settings.js"):
        path = web / script
        if not path.exists():
            continue
        text = path.read_text(errors="ignore")
        for found in re.findall(r'classList\.(?:add|remove|toggle|contains)\(\s*"([^"]+)"', text):
            names.add(found)
        for found in re.findall(r'className\s*=\s*"([^"\n]*)"', text):
            names.update(found.split())
        for found in re.findall(r'class="([^"]+)"', text):
            names.update(found.split())
        for found in re.findall(r'"([a-z][a-z0-9_-]{1,})"', text):
            names.add(found)
        for found in re.findall(r'class="[^"]*\$\{([a-z-]+)', text):
            dynamic.append(found)
        for found in re.findall(r'`([a-z][a-z0-9_-]*)-\$\{', text):
            dynamic.append(found)
    for path in sorted((ROOT / "backend" / "tests").glob("*.py")):
        names.update(re.findall(r'"\.?([a-z][a-zA-Z0-9_-]{2,})"', path.read_text(errors="ignore")))
    names.update(dynamic)
    names.update({f"sig-{d}" for d in ("BUY", "SELL", "EMERGENCY", "HOLD", "neutral")})
    names.update({f"st-{d}" for d in ("active", "warn", "error", "stub", "disabled")})
    return names


def scan_css() -> list[str]:
    """Classes defined in the stylesheet that no page or script uses.

    Comments are stripped first (they mention class names from older layouts),
    and compound selectors are split so an accent class used only next to its
    base class (``.signal-badge.buy``) still counts as used.
    """
    web = ROOT / "backend" / "web"
    css = (web / "styles.css").read_text(errors="ignore")
    css = re.sub(r"/\*.*?\*/", " ", css, flags=re.S)
    defined: set[str] = set()
    for selector in re.findall(r"([^{}]+)\{", css):
        for part in selector.split(","):
            defined.update(re.findall(r"\.([a-zA-Z][a-zA-Z0-9_-]+)", part))
    consumers = _css_consumers()
    return sorted(c for c in defined if c not in consumers and c not in ALLOW)


def scan_dart() -> list[str]:
    """Private Dart members with no other reference in the client."""
    files = sorted((ROOT / "frontend" / "lib").rglob("*.dart"))
    joined = "\n".join(p.read_text(errors="ignore") for p in files)
    findings: list[str] = []
    for path in files:
        text = path.read_text(errors="ignore")
        for match in DART_PRIVATE.finditer(text):
            name = match.group(1) or match.group(2)
            if name in ALLOW:
                continue
            if len(re.findall(r"\b" + re.escape(name) + r"\b", joined)) <= 1:
                findings.append(f"{path.relative_to(ROOT)} {name}")
    return findings


def report() -> dict[str, list[str]]:
    return {
        "python": [f"{name}  ({where})" for name, where in scan_python()],
        "javascript": scan_js(),
        "css": scan_css(),
        "dart": scan_dart(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print every finding with its location")
    args = parser.parse_args(argv)

    findings = report()
    total = sum(len(v) for v in findings.values())
    for language, items in findings.items():
        if not items:
            continue
        print(f"unused {language}: {len(items)}")
        for item in items if args.list else items[:10]:
            print("   ", item)
        if not args.list and len(items) > 10:
            print(f"    … {len(items) - 10} more (run with --list)")
    print(f"\ntotal unused definitions: {total}")
    return 1 if total else 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
