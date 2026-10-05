"""Round AA window clock - the ONE description of a prediction window.

Every countdown (dashboard ring, Flutter ring, the `¶gn` phase stamp on each
formula and emotion) is rendered from the block built here and from nothing
else.  The block is *absolute*: it names the instants the window started and
ends (epoch ms and µs) plus the server's own time, so a client measures its
offset once and then counts down to a fixed instant - the number on screen
cannot drift, restart or run ahead when an unrelated message arrives.

    describe(started, ends, now, ...)   -> the clock block (JSON-ready)
    grid_marks(started, ends, ...)      -> the mid-window pulse instants
    grid_offsets(every, period)         -> every, 2*every, ... inside period
    phase(started, ends, now)           -> {progress, remaining, label}

Nothing in here touches the engine: it is pure arithmetic on wall-clock
seconds, which is what makes it the same clock for every consumer.
"""
from __future__ import annotations

import time


def _iso(timestamp: float) -> str:
    """Wall-clock ISO-8601 (UTC) stamp, second precision - what the UI renders."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def grid_offsets(every: float, period: float) -> list[float]:
    """Marks at every, every*2, ... inside ``period`` (never at the boundary)."""
    if every <= 0 or period <= 0:
        return []
    offsets: list[float] = []
    # Guard against a pathological cadence producing thousands of marks.
    for index in range(1, 65):
        offset = every * index
        if offset >= period - 0.05:
            break
        offsets.append(offset)
    return offsets


def grid_marks(started: float, ends: float, formula_every: float, news_every: float) -> list[dict]:
    """The instants inside the window at which every panel refreshes together.

        t+0    the signal for this window (published at the boundary)
        t+15   formulas + news + agents + brain + accuracy   (one PULSE)
        t+30   the same, plus a fresh news poll
        t+45   the same

    Derived from the window start, never from "now", so two clients - and the
    backend - always agree on when the next one is.
    """
    period = max(0.05, ends - started)
    # A window shorter than the nominal cadence still gets marks: the grid is
    # capped so the panel is never static for a whole window.
    formula_every = min(formula_every, period / 2.0)
    news_every = min(news_every, period)
    offsets: list[tuple[float, str]] = []
    for offset in grid_offsets(formula_every, period):
        offsets.append((offset, "formulas"))
    for offset in grid_offsets(news_every, period):
        offsets.append((offset, "news"))
    marks: list[dict] = []
    for offset in sorted({round(o, 3) for o, _ in offsets}):
        if offset <= 0.05 or offset >= period - 0.05:
            continue
        kinds = sorted({kind for o, kind in offsets if abs(o - offset) < 1e-6})
        marks.append({
            "kind": "tick",
            "parts": kinds,
            "offset_seconds": round(offset, 3),
            "at": _iso(started + offset),
            "at_wall": started + offset,
        })
    return marks


def phase(started: float, ends: float, now: float | None = None) -> dict:
    """Where inside the window ``now`` is: the `¶gn` stamp."""
    moment = time.time() if now is None else now
    period = max(1e-6, ends - started)
    elapsed = min(max(0.0, moment - started), period)
    progress = elapsed / period
    label = ("opening" if progress < 0.1 else "early" if progress < 0.4 else
             "mid" if progress < 0.7 else "late" if progress < 0.9 else "closing")
    return {
        "progress": round(progress, 4),
        "elapsed_seconds": round(elapsed, 3),
        "remaining_seconds": round(max(0.0, ends - moment), 3),
        "label": label,
    }


def describe(
    started: float,
    ends: float,
    *,
    now: float | None = None,
    period_seconds: float,
    cycle_id: int,
    marks: list[dict],
    minute_aligned: bool,
    freshness_max_age_seconds: float,
    scoring_horizon_seconds: float,
    publish_latency_us: int = 0,
    engine_compute_us: int = 0,
    snapshot_us: int = 0,
) -> dict:
    """The clock block both frontends count down from."""
    moment = time.time() if now is None else now
    remaining = max(0.0, ends - moment)
    return {
        "period_seconds": round(period_seconds, 3),
        "window_seconds": round(max(0.0, ends - started), 3),
        "window_started_at_ms": int(round(started * 1000)),
        "window_ends_at_ms": int(round(ends * 1000)),
        "server_time_ms": int(round(moment * 1000)),
        # Microsecond truth for the same instants: a 60-second window with a
        # 0.9 ms publication latency is not the same thing as one that
        # started late, and only the µs fields can tell the difference.
        "window_started_at_us": int(round(started * 1e6)),
        "window_ends_at_us": int(round(ends * 1e6)),
        "server_time_us": int(round(moment * 1e6)),
        "remaining_us": int(round(remaining * 1e6)),
        "elapsed_us": int(round(max(0.0, moment - started) * 1e6)),
        "publish_latency_us": int(publish_latency_us),
        "engine_compute_us": int(engine_compute_us),
        "snapshot_us": int(snapshot_us),
        "seconds_remaining": round(remaining, 3),
        "seconds_elapsed": round(max(0.0, moment - started), 3),
        "cycle_id": cycle_id,
        "minute_aligned": bool(minute_aligned),
        "aligned_to": _iso(started),
        "next_boundary_at": _iso(ends),
        "phase": phase(started, ends, moment),
        "ticks": [
            {
                "parts": mark["parts"],
                "offset_seconds": mark["offset_seconds"],
                "at": mark["at"],
                "seconds_until": round(mark["at_wall"] - moment, 3),
                "done": mark["at_wall"] <= moment,
            }
            for mark in marks
        ],
        "next_tick": next(
            (
                {"parts": mark["parts"], "at": mark["at"], "seconds_until": round(mark["at_wall"] - moment, 3)}
                for mark in marks
                if mark["at_wall"] > moment
            ),
            None,
        ),
        "freshness_max_age_seconds": round(freshness_max_age_seconds, 1),
        "scoring_horizon_seconds": round(scoring_horizon_seconds, 1),
    }
