"""The repository must stay free of code nothing calls (Round I).

The user's rule: *"remove unnecessary code that isn't in use but present."*  The
sweep in ``tools/dead_code.py`` is how that stays true - it walks Python, the
browser bundle, the stylesheet and the Flutter client and reports definitions
with no callers.  These tests are the gate: they fail the moment a new orphan
appears, and they fail if someone deletes the tool instead of keeping the tree
clean.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "dead_code.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_dead_code_scanner_reports_nothing_unused() -> None:
    result = _run("--list")
    assert result.returncode == 0, (
        "unused definitions found:\n" + result.stdout
    )
    assert "total unused definitions: 0" in result.stdout


def test_the_scanner_still_finds_its_own_fixtures() -> None:
    """A scanner that cannot detect anything is worse than no scanner."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("dead_code", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # the Python pattern sees a definition ...
    match = module.PY_PATTERN.search("def orphaned_helper(x):\n    return x\n")
    assert match and match.group(1) == "orphaned_helper"
    # ... the CSS consumer collector knows where a class is really applied
    # (``hold-box`` from index.html, ``matrix-cell`` from matrix.html) and never
    # mistakes a word in a comment for usage.
    consumers = module._css_consumers()
    assert "hold-box" in consumers
    assert "matrix-cell" in consumers
    # a class name that appears in no page and no script is not a consumer
    # (built at run time so this test file cannot make its own name "used")
    assert ("not" + "-a-real-class") not in consumers
    # and the report covers every language the repository ships
    assert set(module.report()) == {"python", "javascript", "css", "dart"}


def test_python_sources_have_no_unused_imports() -> None:
    """pyflakes is the second net: imports and locals the sweep cannot see."""
    result = subprocess.run(
        [sys.executable, "-m", "pyflakes", "backend"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if "No module named pyflakes" in result.stderr:
        return  # optional dev dependency; the dead-code sweep above still runs
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_removed_symbols_stay_removed() -> None:
    """The Round-I removals, by name, so a revert cannot creep back unnoticed."""
    gone = (
        ("backend/api/state.py", "local_agent"),
        ("backend/api/routes_formulas.py", "_live_payload"),
        ("backend/brain/brain.py", "def load_matrix"),
        ("backend/brain/brain.py", "def save_to"),
        ("backend/brain/health_check.py", "health_loop"),
        ("backend/brain/matrix_builder.py", "def set_edge"),
        ("backend/core/config.py", "reload_settings"),
        ("backend/core/cycle_manager.py", "run_cycle_manager"),
        ("backend/core/errors.py", "DataUnavailable"),
        ("backend/core/errors.py", "BrainUnavailable"),
        ("backend/core/signal_lock.py", "def formula_dict"),
        ("backend/data/cross_asset_sync.py", "def interpolate_ticks"),
        ("backend/data/simulator.py", "def synthetic_flash_move"),
        ("backend/formulas/_util.py", "def safe_div"),
        ("backend/formulas/_util.py", "def hurst_rs"),
        ("backend/formulas/engine.py", "def directional_values"),
        ("backend/formulas/synthetic.py", "def all_scenarios"),
        ("backend/news/news_engine.py", "report_price_event"),
        ("backend/web/app.js", "function windowElapsed"),
    )
    for relative, symbol in gone:
        text = (ROOT / relative).read_text()
        assert symbol not in text, f"{symbol} came back in {relative}"


def test_the_long_dead_panels_are_gone_from_the_stylesheet() -> None:
    """``#timer``/``#progress``/the spinner belonged to the old 8-second layout."""
    css = (ROOT / "backend" / "web" / "styles.css").read_text()
    for class_name in (".timer ", ".timer-card", ".timer-row", ".timer-side",
                       ".spinner", ".computing-text", ".computing-sub", ".steps "):
        assert class_name not in css, f"{class_name} is unused and should be gone"
    app = (ROOT / "backend" / "web" / "app.js").read_text()
    assert "$(\"timer\")" not in app
    assert "$(\"progress\").style" not in app
