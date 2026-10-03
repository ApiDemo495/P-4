"""Round M guards.

1.  The prediction cell is *static* inside a window.  The lock was always
    real server-side, but three pieces of text inside the cell repainted
    every second ("updated 7s ago", "43.0s left", the live tape line), so the
    user saw "the prediction changing every few seconds".  Nothing in the
    cell may depend on wall-clock time or on the live formula pass any more.
2.  Brain verification must not print an absent optional token, or an empty
    first-run cache, as a failure - and the cache step passes on the next run.
3.  The Flutter client is supervised by the engine: a status endpoint, a
    (re)start endpoint and a status page instead of a bare 404.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.brain import startup_verification as sv
from backend.core.redis_bus import Store

ROOT = Path(__file__).resolve().parents[2]
APP_JS = (ROOT / "backend" / "web" / "app.js").read_text(encoding="utf-8")
PANEL_DART = (ROOT / "frontend" / "lib" / "widgets" / "signal_widget_panel.dart").read_text(encoding="utf-8")
APP_STATE_DART = (ROOT / "frontend" / "lib" / "state" / "app_state.dart").read_text(encoding="utf-8")


def _js_function(name: str) -> str:
    match = re.search(rf"function {name}\((.|\n)*?\n}}\n", APP_JS)
    assert match, f"{name} missing from app.js"
    return match.group(0)


# ---------------------------------------------------------------------------
# 1. nothing inside the prediction cell moves during a window
# ---------------------------------------------------------------------------
def test_the_freshness_chip_no_longer_counts_seconds() -> None:
    body = _js_function("renderFreshness")
    assert "locked" in body
    code = "\n".join(line for line in body.splitlines() if not line.strip().startswith("//"))
    assert "s ago" not in code, "the chip must not print a running age"
    # Only the STALE text and the tooltip may carry the age.
    fresh_branch = body.split("chip.textContent = override")[1].split(";")[0].split(": ")[-1]
    assert "age" not in fresh_branch and "locked" in fresh_branch


def test_the_horizon_line_has_no_running_numbers() -> None:
    body = _js_function("renderHorizon")
    visible = body.split("el.innerHTML =")[1].split(";")[0]
    assert "left" not in visible and "scored" not in visible and "toFixed" not in visible
    assert "frozen until" in visible
    assert "s left" in body.split("el.title =")[1], "the live number belongs in the tooltip"


def test_the_micro_line_reads_the_lock_not_the_live_tape() -> None:
    body = _js_function("renderMicro")
    assert "state.liveMicro" not in body
    assert "prediction?.detail?.micro" in body
    live = _js_function("renderLiveFormulas")
    assert "renderMicro(" not in live, "the live pass must not repaint the prediction cell"


def test_the_flutter_prediction_cell_is_static_too() -> None:
    assert "updated ${(state.prediction.ageSeconds" not in PANEL_DART
    assert "s left · " not in PANEL_DART
    assert "'locked ${prediction.horizon.releaseClock" in PANEL_DART
    assert "MicroReading get micro => prediction.detail.micro.hasData" in APP_STATE_DART
    assert "lastMicro.hasData ? lastMicro : (prediction" not in APP_STATE_DART


def test_the_lock_watch_tool_exists() -> None:
    tool = (ROOT / "tools" / "lock_watch.js").read_text(encoding="utf-8")
    assert "w-prediction" in tool and "mid-window" in tool


# ---------------------------------------------------------------------------
# 2. brain verification: skipped is not failed, cache passes after first run
# ---------------------------------------------------------------------------
def _settings(**over):
    base = dict(
        brain_force_fallback=False,
        neuprint_token="",
        cave_token="",
        neuprint_server="http://127.0.0.1:9",  # nothing listens: connectivity fails fast
        neuprint_dataset="hemibrain:v1.2.1",
        cave_dataset="flywire_fafb_production",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _run(settings, cache):
    return asyncio.run(sv.run_verification(settings=settings, cache=cache))


def test_missing_tokens_are_skips_not_failures() -> None:
    cache = Store("memory://")
    result = _run(_settings(), cache)
    by_name = {s["step"]: s for s in result.steps}
    assert by_name["0-cache"]["state"] == "skip" and by_name["0-cache"]["ok"] is True
    assert by_name["4-flywire"]["state"] == "skip" and by_name["4-flywire"]["ok"] is True
    assert "CAVE_TOKEN" in by_name["4-flywire"]["detail"]
    assert by_name["5-fallback"]["ok"] is True
    assert not any(s["state"] == "fail" for s in result.steps if s["step"] != "1-connectivity")


def test_the_cache_step_passes_on_the_second_run() -> None:
    cache = Store("memory://")
    first = _run(_settings(), cache)
    assert first.cache_hit is False
    second = _run(_settings(), cache)
    assert second.cache_hit is True
    assert second.steps[0]["step"] == "0-cache"
    assert second.steps[0]["ok"] is True and second.steps[0]["state"] == "pass"
    assert second.status.value == first.status.value  # still honest: fallback
    assert np.array_equal(first.matrix, second.matrix)


def test_a_token_added_later_bypasses_the_cached_fallback() -> None:
    cache = Store("memory://")
    _run(_settings(), cache)
    result = _run(_settings(cave_token="not-a-real-token"), cache)
    names = [s["step"] for s in result.steps]
    assert "1-connectivity" in names, "with a token the live steps must run again"
    assert result.steps[0]["state"] == "pass"


def test_auth_is_skipped_only_without_a_token() -> None:
    # Reachable server, no token -> skip (never a failure).
    async def reachable(_settings):
        return True

    original = sv._step1_connectivity
    sv._step1_connectivity = reachable
    try:
        result = _run(_settings(), None)
    finally:
        sv._step1_connectivity = original
    by_name = {s["step"]: s for s in result.steps}
    assert by_name["2-auth"]["state"] == "skip" and by_name["2-auth"]["ok"] is True
    assert "NEUPRINT_APPLICATION_CREDENTIALS" in by_name["2-auth"]["detail"]


def test_the_settings_page_has_fields_for_both_brain_tokens() -> None:
    html = (ROOT / "backend" / "web" / "settings.html").read_text(encoding="utf-8")
    js = (ROOT / "backend" / "web" / "settings.js").read_text(encoding="utf-8")
    assert 'id="key-neuprint"' in html and 'id="key-cave"' in html
    assert '"neuprint", "cave"' in js
    assert "skip" in js, "the step table must render skipped steps distinctly"


# ---------------------------------------------------------------------------
# 3. the engine supervises the Flutter build
# ---------------------------------------------------------------------------
def test_flutter_status_reports_without_a_build(monkeypatch) -> None:
    from backend.api import flutter_build as fb

    monkeypatch.delenv("CODESPACE_NAME", raising=False)
    monkeypatch.setenv("AUTO_FLUTTER", "0")
    status = fb.status()
    for key in ("built", "running", "wanted", "stage", "log_tail", "retry", "manual"):
        assert key in status
    assert status["wanted"] is False
    assert fb.ensure_started() is None, "AUTO_FLUTTER=0 must never start a download"
    page = fb.status_page(8000)
    assert "/api/flutter/build" in page and "flutter-setup.log" in page


def test_flutter_is_wanted_in_a_codespace(monkeypatch) -> None:
    from backend.api import flutter_build as fb

    monkeypatch.delenv("AUTO_FLUTTER", raising=False)
    monkeypatch.setenv("CODESPACE_NAME", "demo")
    assert fb.wanted() is True
    monkeypatch.setenv("AUTO_FLUTTER", "0")
    assert fb.wanted() is False


def test_the_flutter_routes_are_registered() -> None:
    from backend.api.main import app

    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/api/flutter/status" in paths and "/api/flutter/build" in paths


def test_the_autostart_hook_uses_its_own_session() -> None:
    text = (ROOT / "tools" / "codespace_autostart.sh").read_text(encoding="utf-8")
    block = text.split("start_flutter()", 1)[1].split("\n}", 1)[0]
    assert "setsid" in block


# ---------------------------------------------------------------------------
# 4. self-update: the checkout fast-forwards its own branch, never a checkout
# ---------------------------------------------------------------------------
def test_the_self_update_script_never_switches_branches() -> None:
    text = (ROOT / "tools" / "self_update.sh").read_text(encoding="utf-8")
    code = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))
    assert "git checkout" not in code and "git switch" not in code
    assert "git reset" not in code and "git stash" not in code
    assert "--ff-only" in code, "fast-forward only"
    assert "main" not in code.replace("domain", "").replace("remain", ""), "never touches main"
    assert '+refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}' in code, "explicit refspec fetch"


def test_the_self_update_check_returns_json() -> None:
    import json
    import subprocess

    out = subprocess.run(
        ["bash", str(ROOT / "tools" / "self_update.sh"), "--check"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
    ).stdout.strip().splitlines()
    data = json.loads(out[-1])
    assert "behind" in data and "branch" in data


def test_the_update_routes_and_hook_are_wired() -> None:
    from backend.api.main import app

    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/api/update/status" in paths and "/api/update/apply" in paths
    hook = (ROOT / "tools" / "codespace_autostart.sh").read_text(encoding="utf-8")
    assert "self_update.sh" in hook
    for mode in ("start)", "attach)"):
        block = hook.split(mode, 1)[1].split(";;", 1)[0]
        assert "self_update" in block, f"{mode} must fast-forward before provisioning"
    html = (ROOT / "backend" / "web" / "index.html").read_text(encoding="utf-8")
    assert 'id="update-apply"' in html


def test_auto_update_is_on_in_a_codespace_only(monkeypatch) -> None:
    from backend.api import self_update as su

    monkeypatch.delenv("AUTO_UPDATE", raising=False)
    monkeypatch.delenv("CODESPACE_NAME", raising=False)
    assert su.auto_enabled() is False
    monkeypatch.setenv("CODESPACE_NAME", "demo")
    assert su.auto_enabled() is True
    monkeypatch.setenv("AUTO_UPDATE", "0")
    assert su.auto_enabled() is False


# ---------------------------------------------------------------------------
# 5. the page proves its own lock, and says which build it is
# ---------------------------------------------------------------------------
def test_the_lock_watchdog_and_build_id_are_wired() -> None:
    js = APP_JS
    html = (ROOT / "backend" / "web" / "index.html").read_text(encoding="utf-8")
    assert 'id="w-lock-proof"' in html
    assert "function lockWatchdog" in js and "lockWatchdog();" in js.split("async function safetyNet")[1]
    assert "LOCK BROKEN" in js
    assert "state.accuracyWindowId" in js, "the accuracy row must be frozen with the panel"
    assert "build ${state.config?.build" in js


def test_health_and_config_carry_the_build_id() -> None:
    from backend.api import main as m

    assert m.BUILD_ID and m.BUILD_ID != ""
    src = (ROOT / "backend" / "api" / "main.py").read_text(encoding="utf-8")
    assert src.count('"build": BUILD_ID') + src.count('payload["build"] = BUILD_ID') == 2
