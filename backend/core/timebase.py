"""Microsecond timebase.

The engine used to think in whole seconds: windows were counted in seconds, ages
were rounded to 0.1 s, compute times were reported in milliseconds, and a tick
was a float of seconds.  Everything downstream inherited that coarseness, so
"analysis in microseconds" was impossible even though the tick stream carries
sub-millisecond timestamps.

This module is the one place where the resolution is defined:

``now_us()``
    Monotonic-ish wall clock in microseconds since the epoch.  ``time.time() * 1e6``
    is accurate to ~0.24 µs at current epoch values (float64 has 52 mantissa
    bits), which is finer than any venue timestamp.
``now_ns()`` / ``perf_us()``
    Nanosecond wall clock, and a *monotonic* microsecond counter for measuring
    how long something took (never a wall clock - that can jump with NTP).
``us_to_iso()`` / ``iso_us()``
    ISO-8601 with six fractional digits, so a timestamp on screen is never
    rounded to the second.
``format_us()``
    Human units for a duration given in microseconds (µs / ms / s).
``span_us()``
    The microsecond span of a tick array whose time column is in *milliseconds*
    (the format the ring buffers use).

Time columns inside market arrays stay in milliseconds: they are compared with
``time.time() * 1000`` all over the data layer, and the precision of a float64
millisecond value at current epoch values is already ~0.24 µs.  Conversions
happen here, once.
"""

from __future__ import annotations

import time

__all__ = [
    "US_PER_SECOND",
    "now_us",
    "now_ns",
    "perf_us",
    "perf_ns",
    "us_to_iso",
    "iso_us",
    "ms_to_us",
    "us_to_ms",
    "format_us",
    "span_us",
    "interval_us",
]

US_PER_SECOND = 1_000_000


def now_us() -> int:
    """Wall-clock microseconds since the epoch (UTC), NTP-independent."""
    return time.time_ns() // 1_000


def now_ns() -> int:
    """Wall-clock nanoseconds since the epoch."""
    return time.time_ns()


def perf_us() -> int:
    """Monotonic microsecond counter - use this to measure durations."""
    return time.perf_counter_ns() // 1_000


def perf_ns() -> int:
    """Monotonic nanosecond counter."""
    return time.perf_counter_ns()


def ms_to_us(milliseconds: float) -> int:
    """Convert a millisecond timestamp (the market data convention) to µs."""
    return int(round(float(milliseconds) * 1000.0))


def us_to_ms(microseconds: float) -> float:
    return float(microseconds) / 1000.0


def us_to_iso(microseconds: float | int) -> str:
    """ISO-8601 UTC with microsecond precision: ``2026-01-01T00:00:00.123456Z``."""
    seconds = float(microseconds) / US_PER_SECOND
    whole = int(seconds)
    fraction = int(round((seconds - whole) * US_PER_SECOND))
    if fraction >= US_PER_SECOND:            # rounding can carry
        whole += 1
        fraction -= US_PER_SECOND
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(whole)) + f".{fraction:06d}Z"


def iso_us(microseconds: float | int | None = None) -> str:
    """``us_to_iso`` for now (or for a given stamp)."""
    return us_to_iso(now_us() if microseconds is None else microseconds)


def format_us(microseconds: float | int) -> str:
    """Human duration: 812 µs · 4.31 ms · 1.204 s."""
    value = float(microseconds)
    if value < 1000.0:
        return f"{value:.0f} µs"
    if value < 1_000_000.0:
        return f"{value / 1000.0:.2f} ms"
    return f"{value / 1_000_000.0:.3f} s"


def span_us(times_ms, count: int | None = None) -> int:
    """Microseconds between the first and last of a millisecond time column."""
    import numpy as np

    times = np.asarray(times_ms, dtype=float).reshape(-1)
    if times.size < 2:
        return 0
    if count is not None:
        count = int(min(count, times.size))
        times = times[-count:]
        if times.size < 2:
            return 0
    return ms_to_us(times[-1] - times[0])


def interval_us(times_ms, count: int | None = None):
    """Inter-tick intervals in microseconds (empty array when there is no diff)."""
    import numpy as np

    times = np.asarray(times_ms, dtype=float).reshape(-1)
    if count is not None and times.size > count:
        times = times[-int(count):]
    if times.size < 2:
        return np.zeros(0, dtype=float)
    return np.diff(times) * 1000.0
