"""Round AA risk engine - excursion-quantile take-profit / stop-loss.

Replaces the "1.5 sigma stop, 1.5x target" rule with levels derived from the
distribution of price *excursions* over the window rather than the close:

* The stop is the q-quantile of the maximum adverse excursion (MAE) of a
  Brownian path over the horizon.  For a driftless path
  P(MAE >= a) = 2 (1 - Phi(a / sigma_T)), so a_q = sigma_T * Phi^-1(1 - (1-q)/2).
  With q = 0.80 the stop sits at 1.2816 sigma_T - it is hit by noise alone in
  only one window out of five.
* The target is the stop multiplied by a reward:risk that grows with the
  edge of the call (2P-1): 1.5:1 at zero edge, up to 2.5:1 at full edge,
  never below 1.2:1.  A call we barely believe reaches less far.
* Both levels are floored by the quoted spread (a stop inside the spread is
  not a stop) and by the per-asset bps clamps in settings.

`sigma_T` is the realised volatility scaled to the horizon with the square
root of time, from the per-second realised volatility the cycle measured.
Everything is deterministic from the inputs so a frozen signal keeps its
levels for the whole window.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from backend.core import config as cfg

MAE_QUANTILE = 0.80
RR_FLOOR = 1.2
RR_CEIL = 2.5
RR_EDGE_GAIN = 1.0          # rr = rr_target + gain * |edge|
SPREAD_MULT = 3.0           # stop >= 3 x quoted spread


def _norm_ppf(p: float) -> float:
    """Acklam's rational approximation of the normal quantile (|err| < 4.5e-4)."""
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - plow:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)




@dataclass(frozen=True)
class RiskLevels:
    sl_bps: float
    tp_bps: float
    rr: float
    sigma_window_bps: float
    mae_z: float
    spread_floor_bps: float
    method: str

    def as_dict(self) -> dict:
        return {
            "sl_bps": round(self.sl_bps, 2),
            "tp_bps": round(self.tp_bps, 2),
            "rr": round(self.rr, 3),
            "sigma_window_bps": round(self.sigma_window_bps, 2),
            "mae_z": round(self.mae_z, 3),
            "spread_floor_bps": round(self.spread_floor_bps, 2),
            "method": self.method,
        }


def size_levels(
    asset: str,
    volatility_bps: float,
    *,
    edge: float = 0.0,
    spread_bps: float = 0.0,
    horizon_seconds: float = 60.0,
    settings=None,
    moves: dict | None = None,
) -> RiskLevels:
    """Compute stop / target in bps from realised vol, edge and spread.

    `volatility_bps` is the realised volatility of the window (the cycle
    already scales its per-second estimate to the horizon); the horizon only
    matters when the caller sizes for a different window than it measured.
    """
    settings = settings or cfg.SETTINGS
    quantile = float(cfg.risk_params(asset).get("mae_quantile", MAE_QUANTILE))
    mae_z = _norm_ppf(1.0 - (1.0 - quantile) / 2.0)
    sigma = float(volatility_bps)
    if not math.isfinite(sigma) or sigma <= 0:
        sigma = float(settings.default_volatility_bps)
    horizon = max(float(horizon_seconds or 60.0), 1.0)
    base_horizon = float(getattr(settings, "cycle_seconds", 60) or 60)
    sigma_t = sigma * math.sqrt(horizon / base_horizon)

    spread_floor = SPREAD_MULT * max(float(spread_bps or 0.0), 0.0)
    moves = moves or {}
    measured = int(moves.get("n") or 0) >= 5 and float(moves.get("q80") or 0.0) > 0
    if measured and float(moves.get("exc50") or 0.0) > 0:
        # Round AP - sized for a ONE-MINUTE horizon from what the last
        # minutes actually did, one-sided:
        #   target = median one-sided excursion (the minute reaches it about
        #            half the time before any edge), floored at 2 x spread;
        #   stop   = 80 % one-sided excursion, capped at 1.25 x target so a
        #            stop is never a whole minute's range away.
        # Reward:risk is therefore reported, not targeted - a 1-minute
        # window cannot deliver a 1.5:1 geometry and a reachable target at
        # the same time, and pretending otherwise is what put the stop
        # 300 points away.
        exc50 = float(moves["exc50"])
        exc80 = float(moves.get("exc80") or moves.get("mae80") or exc50 * 1.6)
        tp = max(exc50, 2.0 * max(float(spread_bps or 0.0), 0.0), float(settings.min_tp_bps))
        tp = min(tp, float(settings.max_tp_bps))
        raw_stop = max(exc80, spread_floor)
        sl = min(max(raw_stop, tp, float(settings.min_sl_bps)), 1.25 * tp, float(settings.max_sl_bps))
        rr = tp / sl if sl > 0 else 1.0
        method = (f"1-minute sizing from {moves.get('source')}: target = median one-sided excursion "
                  f"{exc50:.1f} bps, stop = 80 % one-sided excursion {exc80:.1f} bps"
                  + (" capped at 1.25 x target" if exc80 > 1.25 * tp else "")
                  + (f", spread floor {spread_floor:.1f} bps" if spread_floor > 0 else "")
                  + f"; reward:risk {rr:.2f} (reported, not targeted)")
        return RiskLevels(sl, tp, rr, sigma_t, mae_z, spread_floor, method)

    if measured:
        raw_stop = max(float(moves.get("mae80") or moves["q80"]), spread_floor)
    else:
        raw_stop = max(mae_z * sigma_t, spread_floor)
    sl = min(max(raw_stop, float(settings.min_sl_bps)), float(settings.max_sl_bps))

    rr_target = float(getattr(settings, "rr_target", 1.5) or 1.5)
    rr = min(max(rr_target + RR_EDGE_GAIN * abs(float(edge or 0.0)), RR_FLOOR), RR_CEIL)
    tp = min(max(sl * rr, float(settings.min_tp_bps)), float(settings.max_tp_bps))
    if tp < sl:
        # the 1-minute target cap binds: the stop is never wider than the target
        sl = tp
    capped = ""
    if measured:
        ceiling = max(float(moves.get("q95") or 0.0), sl * RR_FLOOR, float(settings.min_tp_bps))
        if tp > ceiling:
            tp = ceiling
            capped = f"; target capped at the measured 95 % minute move ({ceiling:.1f} bps)"
        rr = tp / sl if sl > 0 else rr
    method = (
        (f"stop = measured 80 % adverse excursion over {moves.get('source')} ({raw_stop:.1f} bps"
         if measured else
         f"stop = {quantile:.0%} quantile of max adverse excursion ({mae_z:.2f} x {sigma_t:.0f} bps window sigma")
        + (f", spread floor {spread_floor:.0f} bps" if spread_floor > 0 else "")
        + f"); target = {rr:.2f} x stop (edge {abs(float(edge or 0.0)):.2f}){capped}"
    )
    return RiskLevels(sl, tp, tp / sl if sl > 0 else 0.0, sigma_t, mae_z, spread_floor, method)
