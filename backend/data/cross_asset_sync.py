"""Cross-asset synchronisation for the hedge formulas (Section 5.3).

HRDD, SHRP, GCDV and HSI need BTC and PAXG on a **common time grid**.  PAXG is
much less liquid than BTC, so a naive alignment would compare a BTC tick from
200 ms ago with a PAXG tick from 9 seconds ago.  The spec resolves this with a
1-second LOCF (last observation carried forward) grid:

    1. take both tick series
    2. build a 60-point grid of 1-second timestamps
    3. for each grid point use the most recent tick price (LOCF)
    4. compute log returns on the synchronised grid
"""

from __future__ import annotations

import numpy as np

from backend.core import config as cfg

EPS = 1e-12


def locf_resample(times: np.ndarray, prices: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Last-observation-carried-forward resample onto ``grid``.

    Points earlier than the first observation take that first observation
    (flat-fill), which keeps the series the same length as the grid.
    """
    if times.size == 0 or prices.size == 0:
        return np.zeros_like(grid, dtype=np.float64)

    order = np.argsort(times, kind="stable")
    times, prices = times[order], prices[order]

    idx = np.searchsorted(times, grid, side="right") - 1
    idx = np.clip(idx, 0, prices.size - 1)
    return prices[idx].astype(np.float64)


def build_grid(now: float, window_seconds: int = cfg.SYNC_WINDOW_SECONDS) -> np.ndarray:
    """Ascending 1-second grid ending at ``now`` (inclusive)."""
    end = np.floor(now)
    return end - np.arange(window_seconds - 1, -1, -1, dtype=np.float64)


def log_returns(prices: np.ndarray) -> np.ndarray:
    if prices.size < 2:
        return np.zeros(0, dtype=np.float64)
    safe = np.maximum(prices, EPS)
    return np.diff(np.log(safe))


class SyncedSeries:
    """Synchronised BTC/PAXG prices and returns for one signal cycle."""

    __slots__ = ("grid", "btc_prices", "paxg_prices", "btc_returns", "paxg_returns", "valid")

    def __init__(
        self,
        grid: np.ndarray,
        btc_prices: np.ndarray,
        paxg_prices: np.ndarray,
        valid: bool = True,
    ) -> None:
        self.grid = grid
        self.btc_prices = btc_prices
        self.paxg_prices = paxg_prices
        self.btc_returns = log_returns(btc_prices)
        self.paxg_returns = log_returns(paxg_prices)
        self.valid = valid


def synchronise(
    btc_ticks: np.ndarray,
    paxg_ticks: np.ndarray,
    now: float | None = None,
    window_seconds: int = cfg.SYNC_WINDOW_SECONDS,
) -> SyncedSeries:
    """Build the shared 1-second grid for BTC and PAXG.

    Tick rows are ``[time_ms, price, qty, side]``.
    """
    import time as _time

    now = now if now is not None else _time.time()
    grid = build_grid(now, window_seconds)

    def _prepare(ticks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if ticks is None or ticks.size == 0:
            return np.zeros(0), np.zeros(0)
        times = ticks[:, 0] / 1000.0
        prices = ticks[:, 1]
        mask = np.isfinite(times) & np.isfinite(prices) & (prices > 0)
        return times[mask], prices[mask]

    b_times, b_prices = _prepare(btc_ticks)
    p_times, p_prices = _prepare(paxg_ticks)

    valid = b_prices.size >= 2 and p_prices.size >= 2
    btc_synced = locf_resample(b_times, b_prices, grid)
    paxg_synced = locf_resample(p_times, p_prices, grid)
    return SyncedSeries(grid, btc_synced, paxg_synced, valid=valid)


def interpolate_ticks(ticks: np.ndarray, target_count: int) -> np.ndarray:
    """Linearly interpolate a short tick series onto ``target_count`` rows.

    Used for PAXG when the buffer holds 15-29 ticks: formulas that require more
    than 60 ticks (VSD, VSS) then run on interpolated data and the UI shows a
    warning (Section 5.2).
    """
    if ticks is None or ticks.size == 0:
        return np.zeros((target_count, 2), dtype=np.float64)
    src = ticks[:, :2].astype(np.float64)
    if src.shape[0] >= target_count:
        return src[-target_count:]
    src_t = src[:, 0]
    src_t = src_t - src_t[0]
    dst_t = np.linspace(src_t[0], src_t[-1] if src_t[-1] > 0 else 1.0, target_count)
    out = np.empty((target_count, 2), dtype=np.float64)
    out[:, 0] = dst_t
    out[:, 1] = np.interp(dst_t, src_t, src[:, 1])
    return out
