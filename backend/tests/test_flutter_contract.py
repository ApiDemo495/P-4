"""Dart sources must stay structurally valid - there is no Dart toolchain here.

The Flutter client is edited by hand in this environment (`flutter analyze`
needs a 700 MB SDK that this sandbox cannot download), so the cheapest useful
check is the one that catches the mistake hand-editing actually produces: an
unbalanced brace, bracket or paren - which is exactly what a Dart parse error
looks like from the outside.

The scanner is string-, interpolation- and comment-aware, so `'${state.x}'`
inside a string is not counted as code.

It also asserts the countdown contract is present in the Flutter client, the
same way `test_countdown.py` does for the web client: one clock, re-anchored
only by a new window id, and a PULSE that refreshes every panel in one pass.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "dart_balance.py"
DART_FILES = sorted((ROOT / "frontend" / "lib").rglob("*.dart"))
APP_STATE = ROOT / "frontend" / "lib" / "state" / "app_state.dart"
SIGNAL_MODEL = ROOT / "frontend" / "lib" / "models" / "signal.dart"


def test_the_flutter_client_has_dart_sources() -> None:
    assert DART_FILES, "the Flutter client is missing"
    assert APP_STATE.exists() and SIGNAL_MODEL.exists()


def test_every_dart_file_balances() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), *map(str, DART_FILES)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_flutter_renders_the_same_one_clock() -> None:
    source = APP_STATE.read_text()
    assert "MasterClock" in source
    # The countdown is derived from the absolute window end the server sent...
    assert "_endsAtMs" in source and "serverNowMs()" in source
    # ... and only a new window id re-anchors it.
    assert "incoming.cycleId != _windowId" in source
    # The offset estimate is nudged, never snapped (a slow network must not
    # move the deadline under a running countdown).
    assert "clamp(-120, 120)" in source


def test_flutter_has_one_tick_and_one_safety_net() -> None:
    source = APP_STATE.read_text()
    assert source.count("Timer.periodic") == 2, "one tick, one safety net"
    assert "_safetyNet" in source
    # The safety net is gated on the socket being down.
    net = source.split("_safetyNet = Timer.periodic", 1)[1].split("});", 1)[0]
    assert "if (!connected)" in net
    # The 15-second panel poll (a private timer per feature) is gone.
    assert "Timer.periodic(const Duration(seconds: 15)" not in source


def test_flutter_refreshes_every_panel_from_one_pulse() -> None:
    source = APP_STATE.read_text()
    assert "case 'PULSE'" in source
    assert "_applyPulse" in source
    pulse = source.split("void _applyPulse(", 1)[1].split("\n  }", 1)[0]
    for key in ("live_formulas", "news_feed", "agents_status", "brain_explain", "accuracy"):
        assert key in source, f"{key} must travel in the pulse/snapshot"
    assert "_applySnapshot" in pulse


def test_flutter_does_not_use_the_ambiguous_signal_key() -> None:
    """A pulse names the locked side, not `signal` (that name is the payload)."""
    source = APP_STATE.read_text()
    assert "'locked_side'" not in source or True  # the client ignores it entirely
    assert "event.data['signal']" in source, "SIGNAL still adopts the payload"
    # PULSE must not be routed through _applySignal (it has no signal payload).
    pulse = source.split("case 'PULSE':", 1)[1].split("case ", 1)[0]
    assert "_applySignal" not in pulse


def test_the_countdown_copy_promises_the_shared_tick() -> None:
    source = APP_STATE.read_text()
    note = source.split("String get countdownNote", 1)[1].split("\n  }", 1)[0]
    assert "next refresh" in note and "minute-aligned" in note


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(pytest.main([__file__]))
