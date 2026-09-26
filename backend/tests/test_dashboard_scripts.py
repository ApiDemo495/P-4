"""Guard: the dashboard scripts must be *loadable*, not merely parseable.

Round-F regression.  When the signal panel was rewritten, ``renderHoldBox`` was
renamed to ``renderConvictionBox`` - but two call sites kept the old name.  The
file still passed ``node --check`` (the calls are syntactically fine), every
Python test stayed green, and the *page* was dead: ``renderAll`` threw
``ReferenceError: renderHoldBox is not defined`` on boot, so the dashboard
rendered nothing at all.  "The page is not working" again.

Two guards, both dependency-free:

1.  every function the scripts *call* must be defined in the same file (or be a
    browser/JS global) - this is what catches a rename that missed a caller;
2.  every element id the scripts look up must exist in the HTML that serves
    them - this catches a panel that was renamed in one file only.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"
PAGES = {
    "app.js": "index.html",
    "matrix.js": "matrix.html",
    "settings.js": "settings.html",
}

#: JS + browser globals that may legitimately appear in call position.
GLOBALS = {
    "if", "for", "while", "switch", "catch", "return", "typeof", "function",
    "new", "await", "async", "delete", "void", "in", "of", "do", "else", "try",
    "Number", "String", "Boolean", "Math", "JSON", "Object", "Array", "Date",
    "Set", "Map", "Promise", "WeakMap", "WeakSet", "Symbol", "BigInt", "Intl",
    "parseInt", "parseFloat", "isNaN", "isFinite", "setTimeout", "setInterval",
    "clearTimeout", "clearInterval", "requestAnimationFrame",
    "cancelAnimationFrame", "queueMicrotask", "structuredClone", "fetch",
    "WebSocket", "Error", "TypeError", "RangeError", "RegExp", "document",
    "window", "console", "alert", "confirm", "prompt", "URL", "URLSearchParams",
    "FormData", "Blob", "File", "FileReader", "Event", "CustomEvent",
    "performance", "localStorage", "sessionStorage", "navigator",
    "getComputedStyle", "MutationObserver", "ResizeObserver", "Image", "Audio",
    "CanvasRenderingContext2D", "require", "module", "exports", "define",
    "ApexCharts", "Chart", "jsPDF",
}


#: A ``/`` that follows one of these is the start of a regex literal, not a
#: division (the usual heuristic; it keeps a regex like ``/["']/`` from being
#: mistaken for an unterminated string, which would swallow the rest of a file).
_REGEX_PRECEDERS = set("(,=:[!&|?{};\n+*%~^<>")


def _strip_noise(source: str) -> str:
    """Remove comments and string / template / regex literals before scanning."""
    out: list[str] = []
    i, n = 0, len(source)
    previous = ""
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            end = source.find("\n", i)
            if end == -1:
                break
            i = end
            continue
        if ch == "/" and nxt == "*":
            end = source.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if ch == "/" and previous in _REGEX_PRECEDERS:
            i += 1
            in_class = False
            while i < n:
                if source[i] == "\\":
                    i += 2
                    continue
                if source[i] == "[":
                    in_class = True
                elif source[i] == "]":
                    in_class = False
                elif source[i] == "/" and not in_class:
                    i += 1
                    break
                elif source[i] == "\n":
                    break
                i += 1
            out.append(" ")
            previous = " "
            continue
        if ch in "'\"`":
            quote = ch
            i += 1
            while i < n:
                if source[i] == "\\":
                    i += 2
                    continue
                if source[i] == quote:
                    i += 1
                    break
                i += 1
            out.append(" ")  # placeholder keeps token boundaries honest
            previous = " "
            continue
        out.append(ch)
        if not ch.isspace():
            previous = ch
        i += 1
    return "".join(out)


def _defined_names(code: str) -> set[str]:
    names = set(re.findall(r"function\s+([A-Za-z_$][\w$]*)", code))
    names |= set(
        re.findall(
            r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function|\(|\{)",
            code,
        )
    )
    names |= set(re.findall(r"class\s+([A-Za-z_$][\w$]*)", code))
    # destructured and multiple declarations: `const a = ..., b = ...`
    names |= set(re.findall(r",\s*([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function|\()", code))
    # method definitions inside object/class literals
    names |= set(re.findall(r"^\s*([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{", code, re.M))
    return names


@pytest.mark.parametrize("script,page", sorted(PAGES.items()))
def test_every_called_function_is_defined(script: str, page: str) -> None:
    """A rename that misses a caller must fail here, not in the browser."""
    path = WEB / script
    if not path.exists():
        pytest.skip(f"{script} not present")
    code = _strip_noise(path.read_text())
    defined = _defined_names(code)
    called = set(re.findall(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\(", code))
    missing = sorted(
        name
        for name in called
        if name not in defined and name not in GLOBALS and not name.startswith("_")
    )
    # dotted properties (this.render()) are excluded by the lookbehind above.
    assert not missing, (
        f"{script} calls {missing} but never defines them - "
        f"the dashboard would throw a ReferenceError on boot"
    )


@pytest.mark.parametrize("script,page", sorted(PAGES.items()))
def test_every_looked_up_element_exists(script: str, page: str) -> None:
    """``$("w-fresh")`` must have a matching ``id="w-fresh"`` in the page."""
    js_path, html_path = WEB / script, WEB / page
    if not js_path.exists() or not html_path.exists():
        pytest.skip(f"{script} / {page} not present")
    html = html_path.read_text()
    html_ids = set(re.findall(r'id="([^"]+)"', html))
    code = _strip_noise(js_path.read_text())
    looked_up = set(re.findall(r'\$\(\s*"([\w-]+)"\s*\)', code))
    looked_up |= set(re.findall(r'getElementById\(\s*"([\w-]+)"\s*\)', code))
    missing = sorted(looked_up - html_ids)
    # Scripts legitimately query elements other scripts created at runtime, but
    # an id that exists in neither is a typo.
    runtime_created = set(re.findall(r'\.id\s*=\s*"([\w-]+)"', code))
    missing = [name for name in missing if name not in runtime_created]
    assert not missing, f"{script} looks up {missing}, which {page} never renders"


@pytest.mark.parametrize("script", sorted(PAGES))
def test_the_scripts_parse_with_node(script: str) -> None:
    """``node --check`` - the cheapest possible syntax gate, when node exists."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    path = WEB / script
    if not path.exists():
        pytest.skip(f"{script} not present")
    result = subprocess.run(
        [node, "--check", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
