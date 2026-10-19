"""Round BM - Hawkes order-flow intensity at the boundary.

Trades on Binance do not arrive at random: a buy begets buys, a sell begets
sells, for a few seconds.  A self-exciting (Hawkes) process describes that
exactly - the arrival intensity of buyer-initiated trades at time ``t`` is

    lambda_b(t) = mu_b + sum_{t_i < t, buys} alpha * exp(-beta (t - t_i))

and the same for sells.  What matters for a 70 s sign bet made at second 0
is not the price, it is the *expected excess flow* still to come out of the
cascade that is running at the boundary.  With branching ratio
``n = alpha / beta`` (how many children one trade spawns, on average) the
excess intensity above the baseline decays at rate ``beta (1 - n)``, so the
expected net (buy - sell) volume the current cascade will still push through
the next ``H`` seconds is

    F_H = (lambda_b - lambda_s) * (1 - exp(-beta (1 - n) H)) / (beta (1 - n))

Everything is measured on the tape frozen at the lock:

* ``lambda_b / lambda_s`` - the kernel sums at the boundary, in volume per
  second, with ``beta = 1 / 5 s`` (the memory of a trade cascade on BTC);
* ``n`` - from the Fano factor of 1 s trade counts over the tape
  (``n = 1 - 1 / sqrt(F)`` for an exponential-kernel Hawkes process, the
  closed-form moment estimator), clipped to [0, 0.9];
* ``scale`` - the dispersion of signed 70 s volume over the tape, so the
  score ``tanh(F_H / scale)`` is in the same units as every other input.

The result is one signed input (``feature:hawkes``) for the meta-combiner and
the hindsight reviewer - graded walk-forward like everything else - plus the
two intensities and the branching ratio for the dashboard.
"""

from __future__ import annotations

import math

import numpy as np

BETA = 1.0 / 5.0        # kernel decay (1/s): a trade's excitement halves in ~3.5 s
LOOKBACK_S = 120.0      # kernel sums are taken over this much tape
BIN_S = 1.0             # bins for the branching-ratio estimate


def hawkes_flow(ticks: np.ndarray, horizon_s: float = 70.0, now_ms: float | None = None) -> dict:
    """Never raises; ``covered`` is False when the tape cannot support it."""
    out = {"covered": False, "lambda_buy": 0.0, "lambda_sell": 0.0, "branching": 0.0, "fano": 1.0,
           "projected_flow": 0.0, "scale": 0.0, "score": 0.0, "trades": 0, "logic": []}
    try:
        if ticks is None or ticks.ndim != 2 or ticks.shape[0] < 50:
            out["logic"] = ["fewer than 50 prints - no cascade to measure"]
            return out
        tr = ticks[ticks[:, 2] > 0]
        if tr.shape[0] < 50:
            out["logic"] = ["fewer than 50 trade prints - no cascade to measure"]
            return out
        t = tr[:, 0] / 1000.0
        q = tr[:, 2]
        side = np.sign(tr[:, 3])
        # prints without an aggressor flag are classified by the tick rule
        if np.all(side == 0):
            dp = np.sign(np.diff(tr[:, 1], prepend=tr[0, 1]))
            side = dp
            for i in range(1, side.size):
                if side[i] == 0:
                    side[i] = side[i - 1]
        now = (now_ms / 1000.0) if now_ms else float(t[-1])
        span = now - float(t[0])
        if span < 2.0 * horizon_s:
            out["logic"] = [f"tape spans {span:.0f} s - needs two horizons to calibrate"]
            return out
        # kernel sums at the boundary
        recent = t > now - LOOKBACK_S
        age = now - t[recent]
        k = np.exp(-BETA * age)
        lam_b = float(np.sum(k * q[recent] * (side[recent] > 0))) * BETA
        lam_s = float(np.sum(k * q[recent] * (side[recent] < 0))) * BETA
        # branching ratio from the Fano factor of 1 s trade counts
        edges = np.arange(t[0], now + BIN_S, BIN_S)
        counts, _ = np.histogram(t, bins=edges)
        m = float(counts.mean())
        fano = float(counts.var() / m) if m > 0 else 1.0
        n = 1.0 - 1.0 / math.sqrt(max(fano, 1.0))
        n = float(min(max(n, 0.0), 0.9))
        # expected excess net flow over the horizon
        beta_eff = BETA * (1.0 - n)
        f_h = (lam_b - lam_s) * (1.0 - math.exp(-beta_eff * horizon_s)) / beta_eff
        # scale: dispersion of signed volume over horizon-length windows
        w_edges = np.arange(t[0], now, horizon_s)
        if w_edges.size >= 3:
            idx = np.searchsorted(t, w_edges)
            signed = np.add.reduceat(q * side, idx)[: len(w_edges) - 1]
            scale = float(np.sqrt(np.mean(signed ** 2))) if signed.size else 0.0
        else:
            scale = 0.0
        scale = max(scale, 1e-9)
        score = math.tanh(f_h / scale)
        out.update({"covered": True, "lambda_buy": round(lam_b, 5), "lambda_sell": round(lam_s, 5),
                    "branching": round(n, 3), "fano": round(fano, 2), "projected_flow": round(f_h, 4),
                    "scale": round(scale, 4), "score": round(score, 4), "trades": int(tr.shape[0])})
        out["logic"] = [
            f"lambda_buy = {lam_b:.4f}, lambda_sell = {lam_s:.4f} (volume/s at the boundary, beta = {BETA:g}/s over the last {LOOKBACK_S:.0f} s)",
            f"Fano factor of 1 s counts = {fano:.2f} -> branching ratio n = 1 - 1/sqrt(F) = {n:.2f} (each trade spawns {n:.2f} more)",
            f"excess flow still to come over {horizon_s:.0f} s: F_H = (lambda_b - lambda_s) x (1 - e^(-beta(1-n)H)) / (beta(1-n)) = {f_h:+.3f}",
            f"scale = rms signed {horizon_s:.0f} s volume = {scale:.3f} -> score tanh(F_H / scale) = {score:+.3f}",
        ]
    except Exception as exc:  # noqa: BLE001
        out["logic"] = [f"hawkes abstained: {exc}"]
    return out
