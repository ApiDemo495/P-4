"""Take-profit / stop-loss geometry (Section 10.5).

A locked signal is only actionable if it comes with an exit plan, so every
directional signal carries a take-profit and a stop-loss **derived from the
market's own volatility** rather than from fixed pip targets:

    sigma    = realised volatility of the window, in basis points
    sl_bps   = clip(sigma_mult * sigma, min_sl_bps, max_sl_bps)   # the bounded loss
    tp_bps   = clip(sl_bps * rr_target, min_tp_bps, max_tp_bps)   # rr_target = 1.5

    BUY :  tp = entry * (1 + tp_bps/1e4)    sl = entry * (1 - sl_bps/1e4)
    SELL:  tp = entry * (1 - tp_bps/1e4)    sl = entry * (1 + sl_bps/1e4)

Round Y: the risk is asymmetric by rule - the stop is the volatility-sized,
bounded loss; the target is 1.5x further ("truncate the downside, reach for the
tail").  The distances are also reported in bps, because the price alone does
not say whether a level is tight or wide.

Since the signal layer became binary (BUY or SELL only), *every* window has a
position plan.  What changes instead is the sizing instruction: a LOW conviction
window carries QUIETER levels - the same volatility geometry, but the block says
so, so the panel can print "quarter size" next to it rather than inventing an
invisible half-signal.
"""

from __future__ import annotations

import numpy as np

from backend.core import config as cfg
from backend.core.risk_engine import size_levels
from backend.formulas._util import EPS


def realized_volatility_bps(snapshot, asset: str, horizon_seconds: float = 60.0) -> float:
    """Realised 1-minute volatility in basis points.

    Prefers the 1-minute candle closes (exactly the horizon we care about).
    Falls back to tick returns, rescaled to one minute using the observed tick
    span, which is what a scalper actually has in the first seconds of a tape.
    """
    closes = np.asarray(snapshot.candles(asset), dtype=np.float64)
    if closes.size >= 8:
        rets = np.diff(closes) / np.where(np.abs(closes[:-1]) > EPS, closes[:-1], np.nan)
        rets = rets[np.isfinite(rets)]
        if rets.size >= 6:
            sigma = float(np.std(rets, ddof=1)) * 1e4
            if sigma > 0:
                return sigma

    ticks = snapshot.ticks(asset)
    if ticks is None or len(ticks) < 20:
        return 0.0
    prices = np.asarray(ticks[:, 1], dtype=np.float64)
    try:
        times = np.asarray(ticks[:, 0], dtype=np.float64) / 1000.0
        span = float(times[-1] - times[0])
    except (TypeError, IndexError, ValueError):
        times, span = None, 0.0

    # Time-bucketed closes: the tape is resampled on a fixed grid (5 s buckets,
    # or coarser when the tape is short) and the bucket returns are scaled to
    # the horizon by sqrt(time).  Per-tick returns cannot be used directly: a
    # tick is not a unit of time, so sigma_tick * sqrt(horizon / span) read a
    # 100 bps/min tape as ~1.7 bps and pinned every stop on the floor.
    if times is not None and span >= 10.0:
        bucket = max(1.0, min(5.0, span / 12.0))
        edges = np.floor((times - times[0]) / bucket).astype(np.int64)
        change = np.flatnonzero(np.diff(edges)) if edges.size > 1 else np.zeros(0, dtype=np.int64)
        closes = np.concatenate([prices[change], prices[-1:]]) if change.size else prices[-1:]
        rets = np.diff(closes) / np.where(np.abs(closes[:-1]) > EPS, closes[:-1], np.nan)
        rets = rets[np.isfinite(rets)]
        if rets.size >= 4:
            sigma_bucket = float(np.std(rets, ddof=1)) * 1e4
            return sigma_bucket * float(np.sqrt(max(horizon_seconds, 1.0) / bucket))

    # No usable timestamps: per-tick returns summed over the ticks one horizon
    # is expected to hold (dense live tape, ~10 ticks/s).
    rets = np.diff(prices) / np.where(np.abs(prices[:-1]) > EPS, prices[:-1], np.nan)
    rets = rets[np.isfinite(rets)]
    if rets.size < 10:
        return 0.0
    sigma_tick = float(np.std(rets, ddof=1)) * 1e4
    ticks_per_horizon = max(1.0, min(float(rets.size), 10.0 * max(horizon_seconds, 1.0)))
    return sigma_tick * float(np.sqrt(ticks_per_horizon))


#: Conviction -> position size hint, printed with the levels.
SIZE_HINT = {
    "HIGH": "full size",
    "MEDIUM": "half size",
    "LOW": "quarter size",
}


def quoted_spread_bps(snapshot, asset: str) -> float:
    """Latest quoted spread in bps from the frozen spread history (0 if none)."""
    hist = np.asarray(snapshot.spread_history(asset), dtype=np.float64)
    prices = np.asarray(snapshot.prices(asset), dtype=np.float64)
    if hist.ndim != 2 or hist.shape[0] == 0 or prices.size == 0:
        return 0.0
    mid = float(prices[-1])
    spread = float(hist[-1, 1])
    if not np.isfinite(spread) or not np.isfinite(mid) or mid <= 0 or spread < 0:
        return 0.0
    return spread / mid * 1e4


def risk_levels(
    asset: str,
    signal: str,
    entry: float,
    volatility_bps: float,
    settings=None,
    horizon_seconds: float = 60.0,
    conviction: str = "HIGH",
    emergency_exit: bool = False,
    edge: float = 0.0,
    spread_bps: float = 0.0,
) -> dict:
    """Build the risk block embedded in every locked signal."""
    settings = settings or cfg.SETTINGS

    sigma = float(volatility_bps)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = settings.default_volatility_bps

    # Round AA - excursion-quantile risk engine (backend/core/risk_engine.py):
    # the stop is the 80% quantile of the maximum adverse excursion over the
    # window, the target reaches further the stronger the edge.
    levels = size_levels(
        asset, sigma, edge=edge, spread_bps=spread_bps, horizon_seconds=horizon_seconds, settings=settings
    )
    rr = float(getattr(settings, "rr_target", 1.5) or 1.5)
    sl_bps = levels.sl_bps
    tp_bps = levels.tp_bps

    entry = float(entry or 0.0)
    block = {
        "tradeable": signal in ("BUY", "SELL"),
        "direction": signal,
        "entry": round(entry, 6),
        "volatility_bps": round(sigma, 3),
        "tp_bps": round(tp_bps, 2),
        "sl_bps": round(sl_bps, 2),
        "rr": round(tp_bps / sl_bps, 3) if sl_bps > 0 else 0.0,
        "rr_target": rr,
        "engine": levels.as_dict(),
        "horizon_seconds": round(horizon_seconds, 1),
        "take_profit": None,
        "stop_loss": None,
        "conviction": conviction,
        "size_hint": SIZE_HINT.get(conviction, "full size"),
        "emergency_exit": bool(emergency_exit),
        "note": "waiting for a usable price to set the levels",
    }

    if entry > 0 and block["tradeable"]:
        sign = 1.0 if signal == "BUY" else -1.0
        block["take_profit"] = round(entry * (1.0 + sign * tp_bps / 1e4), 6)
        block["stop_loss"] = round(entry * (1.0 - sign * sl_bps / 1e4), 6)
        block["note"] = (
            f"{tp_bps:.0f} bps target / {sl_bps:.0f} bps stop = "
            f"{block['rr']:.2f}:1 reward:risk - {levels.method}"
            f" - {block['size_hint']} ({conviction.lower()} conviction)"
        )
        if emergency_exit:
            block["note"] = (
                f"EMERGENCY EXIT: {signal} flattens the position that is open. "
                f"{block['note']}"
            )
    return block
