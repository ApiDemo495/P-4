"""Take-profit / stop-loss geometry (Section 10.5).

A locked signal is only actionable if it comes with an exit plan, so every
directional signal carries a take-profit and a stop-loss **derived from the
market's own volatility** rather than from fixed pip targets:

    sigma_1m   = realised 1-minute volatility, in basis points
    tp_bps     = clip(tp_sigma_mult * sigma_1m, min_tp, max_tp)
    sl_bps     = clip(sl_sigma_mult * sigma_1m, min_sl, max_sl)

    BUY :  tp = entry * (1 + tp_bps/1e4)    sl = entry * (1 - sl_bps/1e4)
    SELL:  tp = entry * (1 - tp_bps/1e4)    sl = entry * (1 + sl_bps/1e4)
    HOLD:  no position - levels are None, and the UI says so instead of
           inventing a trade the engine did not recommend.

The ratio `tp_bps / sl_bps` is the reward:risk, which is reported to the UI so
the user can see whether the geometry of the trade is worth taking.
"""

from __future__ import annotations

import numpy as np

from backend.core import config as cfg
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
    rets = np.diff(prices) / np.where(np.abs(prices[:-1]) > EPS, prices[:-1], np.nan)
    rets = rets[np.isfinite(rets)]
    if rets.size < 10:
        return 0.0
    sigma_tick = float(np.std(rets, ddof=1)) * 1e4
    try:
        span = float(ticks[-1, 0] - ticks[0, 0]) / 1000.0
    except (TypeError, IndexError, ValueError):
        span = 0.0
    if span <= 1.0:
        # Unknown spacing: assume the dense tape of a live exchange (~10 ticks/s)
        span = max(1.0, rets.size / 10.0)
    return sigma_tick * float(np.sqrt(max(horizon_seconds, 1.0) / span))


def risk_levels(
    asset: str,
    signal: str,
    entry: float,
    volatility_bps: float,
    settings=None,
    horizon_seconds: float = 60.0,
) -> dict:
    """Build the risk block embedded in every locked signal."""
    settings = settings or cfg.SETTINGS
    params = cfg.risk_params(asset)

    sigma = float(volatility_bps)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = settings.default_volatility_bps

    tp_bps = float(np.clip(params["tp_sigma_mult"] * sigma, settings.min_tp_bps, settings.max_tp_bps))
    sl_bps = float(np.clip(params["sl_sigma_mult"] * sigma, settings.min_sl_bps, settings.max_sl_bps))

    entry = float(entry or 0.0)
    block = {
        "tradeable": signal in ("BUY", "SELL"),
        "direction": signal,
        "entry": round(entry, 6),
        "volatility_bps": round(sigma, 3),
        "tp_bps": round(tp_bps, 2),
        "sl_bps": round(sl_bps, 2),
        "rr": round(tp_bps / sl_bps, 3) if sl_bps > 0 else 0.0,
        "horizon_seconds": round(horizon_seconds, 1),
        "take_profit": None,
        "stop_loss": None,
        "note": "no position while the signal is HOLD",
    }

    if entry > 0 and block["tradeable"]:
        sign = 1.0 if signal == "BUY" else -1.0
        block["take_profit"] = round(entry * (1.0 + sign * tp_bps / 1e4), 6)
        block["stop_loss"] = round(entry * (1.0 - sign * sl_bps / 1e4), 6)
        block["note"] = (
            f"{tp_bps:.0f} bps target / {sl_bps:.0f} bps stop = "
            f"{block['rr']:.2f}:1 reward:risk, sized on {sigma:.0f} bps realised 1-minute volatility"
        )
    return block
