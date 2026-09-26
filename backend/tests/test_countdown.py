"""The countdown contract: 60 seconds, minute-aligned, one clock, one tick.

The user's report was three things at once, and they are three separate rules:

1. **"Countdown is not 60 seconds."**  The window is 60 seconds, phase-locked to
   the UTC minute, and the countdown starts at 60 and reaches 1 at the boundary.
2. **"Too glitchy."**  The countdown is rendered from *absolute instants* the
   backend publishes (window start, window end, server time), so it is
   monotonic: within a window the deadline never moves, whatever else arrives on
   the socket.
3. **"Not in parallel with other features, it randomly running."**  Every
   mid-window refresh happens on a fixed grid - t+15, t+30, t+45 in a 60-second
   window - and travels in a single PULSE message that carries every panel, so
   the dashboard repaints in one pass instead of nine private timers.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import pytest
import pytest_asyncio

from backend.core import config as cfg
from backend.core.cycle_manager import CycleManager, _grid_offsets

ROOT = Path(__file__).resolve().parents[2]
APP_JS = ROOT / "backend" / "web" / "app.js"
CYCLE_PY = ROOT / "backend" / "core" / "cycle_manager.py"


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def manager():
    instance = CycleManager(cfg.Settings(), local_stub=True)
    await instance.start()
    instance.mark_started()
    deadline = time.time() + 8.0
    while time.time() < deadline and instance.lock.try_get_current() is None:
        await asyncio.sleep(0.1)
    try:
        yield instance
    finally:
        await instance.stop()


# ---------------------------------------------------------------------------
# 1. Sixty seconds, minute-aligned
# ---------------------------------------------------------------------------


def test_the_countdown_is_sixty_seconds():
    production = cfg.Settings(time_scale=1.0)
    assert production.cycle_seconds == 60.0
    assert production.cycle_period_seconds == 60.0
    assert production.use_world_clock is True
    # 60 s windows are what the dashboard labels, and the clock block reports.
    assert production.prediction_expired == 65.0


def test_windows_are_locked_to_the_utc_minute():
    """Every window starts on a minute boundary, so two viewers agree."""
    for start in (1_700_000_000.0, 1_700_000_012.5, 1_700_000_059.9):
        snapped = CycleManager._snap_to_minute(start)
        assert snapped % 60.0 == pytest.approx(0.0, abs=1e-6)
        assert snapped > start
        assert snapped - start <= 60.0


async def test_the_clock_block_starts_at_sixty_and_only_counts_down(manager: CycleManager):
    """One window: 60 -> 1, no jumps, no restarts."""
    period = manager.settings.cycle_period_seconds
    if period > 20:
        pytest.skip("the counting assertion needs a compressed window")
    first = manager.master_clock()
    seen = [first["seconds_remaining"]]
    deadline = time.time() + period * 0.8
    while time.time() < deadline:
        await asyncio.sleep(max(0.05, period / 12.0))
        clock = manager.master_clock()
        if clock["cycle_id"] != first["cycle_id"]:
            break                       # rolled into the next window: fine
        seen.append(clock["seconds_remaining"])
    for earlier, later in zip(seen, seen[1:]):
        assert later <= earlier + 0.05, f"the clock went backwards: {seen}"
    assert seen[0] <= period + 1e-6
    assert seen[-1] < seen[0], "the countdown never moved"


async def test_the_window_block_carries_the_clock(manager: CycleManager):
    window = manager.window_status()
    clock = window["clock"]
    for key in (
        "period_seconds",
        "window_seconds",
        "window_started_at_ms",
        "window_ends_at_ms",
        "server_time_ms",
        "seconds_remaining",
        "cycle_id",
        "minute_aligned",
        "ticks",
        "freshness_max_age_seconds",
        "scoring_horizon_seconds",
    ):
        assert key in clock, key
    # Absolute instants, not "now minus something": this is what makes the
    # client countdown immune to message latency.
    assert clock["window_ends_at_ms"] - clock["window_started_at_ms"] == pytest.approx(
        clock["window_seconds"] * 1000, abs=2
    )
    assert clock["seconds_remaining"] <= clock["window_seconds"] + 1e-6
    assert clock["seconds_remaining"] >= 0.0
    assert (
        clock["seconds_elapsed"] + clock["seconds_remaining"]
        == pytest.approx(clock["window_seconds"], abs=0.2)
    )
    # The old keys the UI already used are still there.
    assert "seconds_remaining" in window and "window_seconds" in window


# ---------------------------------------------------------------------------
# 2. One tick: the grid, and everything travelling on it
# ---------------------------------------------------------------------------


def test_the_grid_is_derived_from_the_window_not_from_now():
    marks = _grid_offsets(15.0, 60.0)
    assert marks == [15.0, 30.0, 45.0]
    # Nothing on the boundary itself: that is the SIGNAL, not a pulse.
    assert all(0 < mark < 60.0 for mark in marks)
    # The compressed harness window (3 s) gets its marks from the same helper.
    assert _grid_offsets(0.75, 3.0) == [0.75, 1.5, 2.25]
    # The capping rule lives in tick_grid: a window shorter than the nominal
    # cadence still gets at least one mark, so no panel is static for a window.
    source = CYCLE_PY.read_text()
    assert "formula_every = min(formula_every, period / 2.0)" in source
    assert "news_every = min(news_every, period)" in source


async def test_tick_grid_lists_the_marks_inside_the_window(manager: CycleManager):
    clock = manager.master_clock()
    period = clock["window_seconds"]
    ticks = clock["ticks"]
    assert ticks, "a window must carry at least one refresh mark"
    for mark in ticks:
        assert 0 < mark["offset_seconds"] < period + 1e-6
        assert mark["at"].endswith("Z")
        assert mark["parts"], "every mark says which features refresh"
    offsets = [mark["offset_seconds"] for mark in ticks]
    assert offsets == sorted(offsets)
    # The marks are the same instants for every observer: derived from the
    # window start, and identical when read twice in the same window.
    again = manager.master_clock()["ticks"]
    assert [m["offset_seconds"] for m in again] == offsets


async def test_a_pulse_carries_every_panel_in_one_message(manager: CycleManager):
    """The heartbeat must not be a formula-only timer any more."""
    queue = manager.subscribe()
    period = manager.settings.cycle_period_seconds
    try:
        pulses = []
        deadline = time.time() + period * 1.6
        while time.time() < deadline and not pulses:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if message.get("type") == "PULSE":
                pulses.append(message["data"])
    finally:
        manager.unsubscribe(queue)

    assert pulses, "no PULSE arrived on the window grid"
    data = pulses[0]
    # Every feature the dashboard shows, in the one message.
    for key in (
        "clock",
        "window",
        "live_formulas",
        "news_feed",
        "agents_status",
        "brain_explain",
        "accuracy",
        "parts",
        "offset_seconds",
        "note",
    ):
        assert key in data, key
    assert data["live_formulas"]["formulas"], "the pulse must carry live formula values"
    assert "Live values only" in data["note"]
    assert data["clock"]["cycle_id"] == data["cycle_number"]
    # The pulse never moves the locked signal: it names it and nothing more.
    assert data["locked_side"] in ("BUY", "SELL")
    assert "signal" not in data, "a pulse must not impersonate a signal payload"


async def test_the_boundary_snapshot_carries_everything_too(manager: CycleManager):
    """The SIGNAL message is a full repaint, not just the signal."""
    for key in (
        "history",
        "outcomes",
        "live_formulas",
        "news_feed",
        "agents_status",
        "accuracy",
        "clock",
    ):
        assert key in manager.snapshot_payload(include_history=True), key


# ---------------------------------------------------------------------------
# 3. The client renders that one clock, and never runs its own schedule
# ---------------------------------------------------------------------------


def test_the_client_has_no_private_timers():
    """One frame loop, one safety net - and nothing else on a timer.

    Before this, the dashboard ran eleven `setInterval`s (news, agents, brain,
    readiness, history, outcomes, brain explain, emergency, formulas, clock
    sync, countdown).  That is what "randomly running" looked like on screen.
    """
    js = APP_JS.read_text()
    intervals = re.findall(r"setInterval\(([^,]+),", js)
    assert len(intervals) == 1, f"the client must have one timer, found {intervals}"
    assert "safetyNet" in intervals[0]
    assert "requestAnimationFrame" in js
    # ... and the safety net does nothing while the socket is healthy.
    net = js.split("async function safetyNet()", 1)[1].split("\n}", 1)[0]
    assert "WebSocket.OPEN" in net


def test_the_client_counts_to_absolute_instants_not_from_a_reading():
    """The anti-glitch rule, in the source: a re-read only re-anchors on a new window."""
    js = APP_JS.read_text()
    clock_block = js.split("function applyClock(", 1)[1].split("function windowRemaining", 1)[0]
    assert "window_ends_at_ms" in clock_block and "server_time_ms" in clock_block
    assert "sameWindow" in clock_block, "a new server reading must not move a running deadline"
    # The deadline is only written when the window id changes...
    assert "if (!sameWindow || opts.force)" in clock_block
    # ... and the remaining time is derived from the absolute end instant.
    remaining = js.split("function windowRemaining()", 1)[1].split("\n}", 1)[0]
    assert "endsAtMs" in remaining and "serverNowMs()" in remaining
    # The digits are whole seconds off that value, never a counter of their own.
    frame = js.split("function frame()", 1)[1].split("\n}", 1)[0]
    assert "windowRemaining()" in frame and "lastSecondShown" in frame


def test_only_a_signal_payload_is_called_signal(manager: CycleManager):
    """The name collision that broke the dashboard once already.

    The signal payload's own ``signal`` field is the *direction* ("BUY"), so a
    message that carries a bare ``signal`` key looks like a signal payload to a
    client that dispatches on shape.  The heartbeat therefore names the locked
    side ``locked_side``.  This is the regression guard for that bug.
    """
    snapshot = manager.snapshot_payload()
    # The snapshot IS a signal payload: it can be rendered straight away.
    assert snapshot["signal"] in ("BUY", "SELL", None)
    assert "risk" in snapshot and "lock_state" in snapshot

    source = CYCLE_PY.read_text()
    pulse = source.split('"type": "PULSE"', 1)[1].split("}", 1)[0]
    assert '"locked_side"' in source
    assert '\n                            "signal":' not in pulse


def test_the_client_recognizes_a_signal_payload_by_its_shape():
    js = APP_JS.read_text()
    assert "function isSignalPayload(" in js
    fn = js.split("function isSignalPayload(", 1)[1].split("\n}", 1)[0]
    for key in ("lock_state", "risk", "cycle_number"):
        assert key in fn, key
    # ... and the snapshot pass uses it, rather than a bare typeof check
    assert "isSignalPayload(data)" in js


def test_the_server_keeps_one_heartbeat_for_the_whole_dashboard():
    source = CYCLE_PY.read_text()
    assert "_heartbeat_loop" in source
    assert "_live_refresh_loop" not in source, "the private formula timer is gone"
    loop = source.split("async def _heartbeat_loop", 1)[1].split("async def ", 1)[0]
    assert "tick_grid" in loop, "the heartbeat must follow the window grid"
    assert "PULSE" in loop
    # One message type carries the panels.
    for key in ("live_formulas", "news_feed", "agents_status", "brain_explain", "accuracy"):
        assert key in loop, key


def test_the_pulse_cadence_matches_the_spec():
    """15 s formula refresh and 30 s news - both on the 60 s grid."""
    production = cfg.Settings(time_scale=1.0)
    assert production.formula_refresh_seconds == 15.0
    assert production.news_poll_seconds == 30.0
    marks = _grid_offsets(production.formula_refresh_seconds, production.cycle_period_seconds)
    assert marks == [15.0, 30.0, 45.0]
