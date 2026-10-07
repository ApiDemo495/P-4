"""Microsecond-resolution market analysis.

The engine's *decisions* are made once per 60-second window, but the *analysis*
behind them is not second-granular: the tick stream carries sub-millisecond
timestamps, and every quantity that describes the tape - how fast quotes arrive,
how jittery that arrival is, how far price moves per microsecond - is measured
here, in microseconds.

Two things read this module:

* the 22 formulas, which get the derivatives they need from the same frozen
  snapshot (the micro readings below do not replace them, they explain them);
* the prediction block, which publishes ``micro`` so the UI can show the
  tape's actual resolution instead of a rounded "1 s"-style summary.

Everything is a pure function of the frozen snapshot plus a small rolling state,
so the Signal Lock Protocol still holds: a micro reading can never change after
the snapshot is taken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from backend.core.timebase import format_us, now_us, span_us


def _safe(value: float, fallback: float = 0.0) -> float:
    return float(value) if value is not None and math.isfinite(float(value)) else fallback


@dataclass
class MicroState:
    """Rolling microsecond statistics, kept per asset between windows."""

    tick_rate_hz: float = 0.0
    mean_interval_us: float = 0.0
    windows: int = 0

    def observe(self, rate: float, mean_us: float) -> None:
        if rate <= 0 or mean_us <= 0:
            return
        # EMA so the reading is stable across windows but still tracks the tape.
        alpha = 0.35 if self.windows else 1.0
        self.tick_rate_hz = (1 - alpha) * self.tick_rate_hz + alpha * rate
        self.mean_interval_us = (1 - alpha) * self.mean_interval_us + alpha * mean_us
        self.windows += 1


@dataclass
class MicroReport:
    """The microsecond picture of one asset at one instant."""

    asset: str = ""
    available: bool = False
    reason: str = ""
    ticks: int = 0
    span_us: int = 0
    analyzed_us: int = 0
    resolution_us: float = 0.0
    mean_interval_us: float = 0.0
    median_interval_us: float = 0.0
    p95_interval_us: float = 0.0
    min_interval_us: float = 0.0
    max_interval_us: float = 0.0
    jitter_us: float = 0.0
    tick_rate_hz: float = 0.0
    ticks_per_second: float = 0.0
    burstiness: float = 0.0
    last_tick_age_us: int = 0
    micro_momentum: float = 0.0
    micro_trend_bps_per_s: float = 0.0
    micro_vol_bps: float = 0.0
    micro_range_bps: float = 0.0
    micro_vol_per_us: float = 0.0
    aggression: float = 0.0
    quote_lifetime_us: float = 0.0
    spread_us_changes: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "asset": self.asset,
            "available": self.available,
            "reason": self.reason,
            "ticks": self.ticks,
            "span_us": self.span_us,
            "span_label": format_us(self.span_us) if self.span_us else "—",
            "analyzed_us": self.analyzed_us,
            "resolution_us": round(self.resolution_us, 2),
            "resolution_label": format_us(self.resolution_us) if self.resolution_us else "—",
            "mean_interval_us": round(self.mean_interval_us, 2),
            "median_interval_us": round(self.median_interval_us, 2),
            "p95_interval_us": round(self.p95_interval_us, 2),
            "min_interval_us": round(self.min_interval_us, 2),
            "max_interval_us": round(self.max_interval_us, 2),
            "jitter_us": round(self.jitter_us, 2),
            "tick_rate_hz": round(self.tick_rate_hz, 3),
            "ticks_per_second": round(self.ticks_per_second, 3),
            "burstiness": round(self.burstiness, 3),
            "last_tick_age_us": self.last_tick_age_us,
            "micro_momentum": round(self.micro_momentum, 6),
            "micro_trend_bps_per_s": round(self.micro_trend_bps_per_s, 4),
            "micro_vol_bps": round(self.micro_vol_bps, 4),
            "micro_range_bps": round(self.micro_range_bps, 4),
            "micro_vol_per_us": self.micro_vol_per_us,
            "aggression": round(self.aggression, 6),
            "quote_lifetime_us": round(self.quote_lifetime_us, 2),
            "spread_us_changes": self.spread_us_changes,
            "extra": dict(self.extra),
        }


def analyze(
    snapshot,
    asset: str,
    *,
    state: MicroState | None = None,
    now: int | None = None,
    recent: int = 60,
    trend_ticks: int = 30,
) -> MicroReport:
    """Measure the tape in microseconds.

    ``recent`` bounds the window used for volatility and range (60 ticks is
    enough for a stable realised variance without dragging in stale prices), and
    ``trend_ticks`` the recency-weighted regression for momentum.
    """
    report = MicroReport(asset=asset.upper())
    if snapshot is None:
        report.reason = "no snapshot"
        return report

    ticks = snapshot.ticks(asset)
    ticks = np.asarray(ticks, dtype=float) if ticks is not None else np.zeros((0, 4))
    if ticks.ndim != 2 or ticks.shape[0] < 3:
        report.reason = "not enough ticks to resolve microseconds"
        return report

    times_ms = ticks[:, 0]
    prices = ticks[:, 1]
    quantities = ticks[:, 2] if ticks.shape[1] > 2 else np.zeros(ticks.shape[0])
    sides = ticks[:, 3] if ticks.shape[1] > 3 else np.zeros(ticks.shape[0])

    now_us_value = now if now is not None else now_us()
    report.ticks = int(ticks.shape[0])
    report.span_us = span_us(times_ms)
    report.analyzed_us = max(0, now_us_value - int(round(times_ms[0] * 1000.0)))
    report.last_tick_age_us = max(0, now_us_value - int(round(times_ms[-1] * 1000.0)))

    intervals = np.diff(times_ms) * 1000.0            # ms -> µs
    intervals = intervals[np.isfinite(intervals)]
    intervals = intervals[intervals > 0]
    if intervals.size == 0:
        report.reason = "tick timestamps are identical - no microsecond structure"
        return report

    report.available = True
    report.mean_interval_us = _safe(np.mean(intervals))
    report.median_interval_us = _safe(np.median(intervals))
    report.min_interval_us = _safe(np.min(intervals))
    report.max_interval_us = _safe(np.max(intervals))
    report.p95_interval_us = _safe(np.percentile(intervals, 95))
    # Jitter: the median absolute deviation, scaled to a sigma-equivalent, of the
    # inter-arrival interval.  Mean is the rate, jitter is the texture.
    report.jitter_us = _safe(1.4826 * np.median(np.abs(intervals - report.median_interval_us)))
    if report.mean_interval_us > 0:
        report.tick_rate_hz = 1e6 / report.mean_interval_us
        report.burstiness = _safe(np.std(intervals) / report.mean_interval_us)
    report.resolution_us = report.median_interval_us
    if report.analyzed_us > 0:
        report.ticks_per_second = report.ticks * 1e6 / report.analyzed_us

    # --- micro momentum: recency-weighted price velocity over the last ticks --
    span = min(int(trend_ticks), ticks.shape[0])
    if span >= 3:
        local_t = times_ms[-span:] * 1000.0            # µs
        local_p = prices[-span:]
        weights = np.linspace(0.25, 1.0, span)
        try:
            slope, _ = np.polyfit(local_t - local_t[0], local_p, 1, w=weights)
        except Exception:  # noqa: BLE001 - degenerate input must not break a window
            slope = 0.0
        mid = float(np.mean(local_p)) or 1.0
        report.micro_trend_bps_per_s = _safe(slope * 1e6 * 1e4 / mid)
        recent_move = float(local_p[-1] - local_p[0])
        sigma = float(np.std(np.diff(local_p))) or 1e-9
        report.micro_momentum = _safe(math.tanh(recent_move / (sigma * math.sqrt(span))))

    # --- volatility, per microsecond and in basis points ---------------------
    recent_prices = prices[-int(recent):]
    if recent_prices.size >= 3:
        mid = float(np.mean(recent_prices)) or 1.0
        returns = np.diff(np.log(np.maximum(recent_prices, 1e-12)))
        report.micro_vol_bps = _safe(float(np.std(returns)) * math.sqrt(recent_prices.size) * 1e4)
        report.micro_range_bps = _safe(
            (float(np.max(recent_prices)) - float(np.min(recent_prices))) / mid * 1e4
        )
        local_span_us = span_us(times_ms[-int(recent):])
        if local_span_us > 0:
            report.micro_vol_per_us = _safe(report.micro_vol_bps / local_span_us, 0.0)

    # --- aggression: signed volume over the analysed window ------------------
    if quantities.size and np.sum(np.abs(quantities)) > 0:
        report.aggression = _safe(
            float(np.sum(quantities * sides)) / float(np.sum(np.abs(quantities)))
        )

    # --- how long a quote lives: the spread history is also timestamped -------
    spreads = snapshot.spread_history(asset)
    spreads = np.asarray(spreads, dtype=float) if spreads is not None else np.zeros((0, 2))
    if spreads.ndim == 2 and spreads.shape[0] >= 3:
        spread_values = spreads[:, 1]
        changes = np.where(np.diff(spread_values) != 0)[0]
        report.spread_us_changes = int(changes.size)
        if changes.size >= 2:
            change_times_us = spreads[changes, 0] * 1000.0
            report.quote_lifetime_us = _safe(np.mean(np.diff(change_times_us)))

    report.extra = {
        "window_ticks": int(min(recent, recent_prices.size)),
        "trend_ticks": int(span),
        "vol_scale": "per window · annualisation not applied",
    }

    if state is not None:
        state.observe(report.tick_rate_hz, report.mean_interval_us)
        report.extra["rolling_tick_rate_hz"] = round(state.tick_rate_hz, 3)
        report.extra["rolling_mean_interval_us"] = round(state.mean_interval_us, 2)
        report.extra["windows_observed"] = state.windows

    return report
