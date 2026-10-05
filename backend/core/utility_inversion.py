"""Round AA inverse reinforcement learning - the crowd's utility, read off the tape.

The emotions above this layer *name* a state (fear, greed, ...).  This layer
recovers the **preferences** that would make the crowd's observed actions
optimal - the inverse-RL question "what utility are these traders
maximising?" - and reports three parameters of a prospect-theory decision
maker, each measured from live ticks and the live book:

* **lambda_t - loss aversion.**  In 1-second buckets the signed aggressive
  flow of bucket *t* is regressed on the previous bucket's return, split
  into its negative and positive parts: flow_t = b- * min(r, 0) + b+ * max(r, 0).
  A crowd that sells harder after a down-tick than it buys after an equal
  up-tick has |b-| > |b+|; lambda = |b-| / |b+| (Kahneman-Tversky's
  population value is ~2.25).  lambda > 2.5 is a crowd that pays anything to
  stop losing - the fingerprint of forced liquidation.
* **gamma_t - risk aversion.**  Participation (aggressive notional per
  second) in the calmest third of recent seconds versus the most volatile
  third.  gamma = log(participation_calm / participation_volatile): positive
  means the crowd steps back when volatility rises (risk-averse), negative
  means it chases it (risk-seeking, late-cycle greed).
* **alpha_t - probability weighting.**  The share of resting depth placed
  beyond 2 sigma_window from the mid is what the book *pays* for the tails.
  An indifferent book spreads depth evenly over its span, so its tail share
  is p_unif = (span - 2 sigma) / span.  With the power weighting function
  w(p) = p^alpha, alpha = ln w / ln p_unif: alpha < 1 means the book
  overweights the tails (braced for a jump), alpha > 1 that it neglects
  them (complacency).  The objective 2-sigma tail mass 2(1 - Phi(2)) = 4.6 %
  is reported alongside for the panel.

Everything is in the open: each parameter returns its inputs, a 0..1
*intensity* for the emotion filter (loss_averse / risk_seeking / tail_fear),
and a one-line reading.  Never raises; `available` says what was measurable.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

BUCKET_MS = 1000.0
MIN_BUCKETS = 24
TAIL_SIGMA = 2.0
P_OBJ_TAIL = 2.0 * (1.0 - 0.9772498680518208)   # 2(1 - Phi(2)) = 0.0455


def _ramp(v: float, lo: float, hi: float) -> float:
    if not math.isfinite(v):
        return 0.0
    return float(max(0.0, min(1.0, (v - lo) / (hi - lo))))


def _buckets(ticks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(returns_bps, signed_flow, abs_flow) per 1-second bucket."""
    t = ticks[:, 0]
    px = ticks[:, 1]
    qty = np.abs(ticks[:, 2])
    side = np.sign(ticks[:, 3]) if ticks.shape[1] >= 4 else np.sign(np.r_[0.0, np.diff(px)])
    idx = np.floor((t - t[0]) / BUCKET_MS).astype(int)
    n = int(idx[-1]) + 1
    if n < 2:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    last_px = np.full(n, np.nan)
    flow = np.zeros(n)
    notional = np.zeros(n)
    for i in range(ticks.shape[0]):
        b = idx[i]
        last_px[b] = px[i]
        flow[b] += side[i] * qty[i] * px[i]
        notional[b] += qty[i] * px[i]
    # forward-fill prices of empty buckets
    for b in range(1, n):
        if not np.isfinite(last_px[b]):
            last_px[b] = last_px[b - 1]
    good = np.isfinite(last_px) & (last_px > 0)
    if good.sum() < 3:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    first = int(np.argmax(good))
    last_px, flow, notional = last_px[first:], flow[first:], notional[first:]
    rets = np.diff(np.log(last_px)) * 1e4
    return rets, flow[1:], notional[1:]


def loss_aversion(rets: np.ndarray, flow: np.ndarray) -> dict:
    """lambda = |b-| / |b+| from flow_t ~ b- * r-_{t-1} + b+ * r+_{t-1}."""
    if rets.size < MIN_BUCKETS:
        return {"available": False, "reason": "fewer than 24 one-second buckets"}
    r_prev = rets[:-1]
    f_now = flow[1:]
    scale = float(np.std(f_now)) or 1.0
    f_now = f_now / scale
    neg = np.minimum(r_prev, 0.0)
    pos = np.maximum(r_prev, 0.0)
    if np.count_nonzero(neg) < 4 or np.count_nonzero(pos) < 4:
        return {"available": False, "reason": "not enough down and up seconds to compare"}
    X = np.column_stack([neg, pos])
    beta, *_ = np.linalg.lstsq(X, f_now, rcond=None)
    b_neg, b_pos = float(beta[0]), float(beta[1])
    # the response to a loss is selling after a down move: b_neg > 0 means
    # flow goes negative when r is negative (flow = b_neg * r, r < 0)
    resp_loss = max(b_neg, 0.0)
    resp_gain = max(b_pos, 0.0)
    if resp_gain < 1e-9 and resp_loss < 1e-9:
        lam = 1.0
    else:
        lam = (resp_loss + 1e-3) / (resp_gain + 1e-3)
    lam = float(min(6.0, max(0.15, lam)))
    return {
        "available": True,
        "lambda": round(lam, 3),
        "beta_loss": round(b_neg, 4),
        "beta_gain": round(b_pos, 4),
        "buckets": int(rets.size),
        "loss_averse": round(_ramp(lam, 1.5, 3.5), 4),
        "gain_chasing": round(_ramp(1.0 / lam, 1.5, 3.5), 4),
        "read": ("crowd sells losses far harder than it buys gains" if lam > 2.5 else
                 "crowd chases gains harder than it cuts losses" if lam < 0.5 else
                 "symmetric response to gains and losses"),
    }


def risk_aversion(rets: np.ndarray, notional: np.ndarray) -> dict:
    """gamma = log(participation in calm seconds / participation in volatile seconds)."""
    if rets.size < MIN_BUCKETS:
        return {"available": False, "reason": "fewer than 24 one-second buckets"}
    vol = np.abs(rets)
    lo_q, hi_q = np.quantile(vol, [0.34, 0.66])
    calm = notional[vol <= lo_q]
    wild = notional[vol >= hi_q]
    if calm.size < 4 or wild.size < 4:
        return {"available": False, "reason": "volatility terciles too thin"}
    p_calm = float(np.mean(calm)) + 1e-9
    p_wild = float(np.mean(wild)) + 1e-9
    gamma = float(np.clip(math.log(p_calm / p_wild), -3.0, 3.0))
    return {
        "available": True,
        "gamma": round(gamma, 3),
        "participation_calm": round(p_calm, 2),
        "participation_volatile": round(p_wild, 2),
        "risk_seeking": round(_ramp(-gamma, 0.3, 1.5), 4),
        "risk_averse": round(_ramp(gamma, 0.3, 1.5), 4),
        "read": ("crowd steps back when the tape gets wild (risk-averse)" if gamma > 0.3 else
                 "crowd chases volatility (risk-seeking)" if gamma < -0.3 else
                 "participation indifferent to volatility"),
    }


def probability_weighting(book: Any, mid: float, sigma_window_bps: float) -> dict:
    """Prelec alpha from the share of depth resting beyond 2 sigma of the mid."""
    try:
        arr = np.asarray(book, dtype=np.float64)
    except (TypeError, ValueError):
        return {"available": False, "reason": "no book"}
    if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] < 5 or mid <= 0 or sigma_window_bps <= 0:
        return {"available": False, "reason": "book too shallow for a tail read"}
    bids, asks = arr[0], arr[1]
    bids = bids[(bids[:, 0] > 0) & (bids[:, 1] > 0)]
    asks = asks[(asks[:, 0] > 0) & (asks[:, 1] > 0)]
    if bids.shape[0] < 3 or asks.shape[0] < 3:
        return {"available": False, "reason": "book too shallow for a tail read"}
    tail_bps = TAIL_SIGMA * sigma_window_bps
    dist_b = (mid - bids[:, 0]) / mid * 1e4
    dist_a = (asks[:, 0] - mid) / mid * 1e4
    span = float(max(dist_b.max(), dist_a.max()))
    if span <= tail_bps * 0.5:
        return {"available": False, "reason": f"book spans {span:.0f} bps, tail at {tail_bps:.0f} bps"}
    total = float(bids[:, 1].sum() + asks[:, 1].sum()) + 1e-12
    tail = float(bids[dist_b >= tail_bps, 1].sum() + asks[dist_a >= tail_bps, 1].sum())
    w = min(0.98, max(0.005, tail / total))
    # an indifferent book spends depth in proportion to span
    p_unif = min(0.98, max(0.02, (span - tail_bps) / span))
    alpha = math.log(w) / math.log(p_unif)
    alpha = float(np.clip(alpha, 0.2, 3.0))
    return {
        "available": True,
        "alpha": round(alpha, 3),
        "tail_share": round(w, 4),
        "uniform_tail": round(p_unif, 4),
        "objective_tail": round(P_OBJ_TAIL, 4),
        "tail_bps": round(tail_bps, 1),
        "levels": int(bids.shape[0] + asks.shape[0]),
        "tail_fear": round(_ramp(1.0 - alpha, 0.1, 0.6), 4),
        "complacent": round(_ramp(alpha - 1.0, 0.2, 1.0), 4),
        "read": ("book overweights the tails - braced for a jump" if alpha < 0.85 else
                 "book ignores the tails - complacent" if alpha > 1.25 else
                 "book prices the tails about right"),
    }


def analyze(tape: Any, asset: str, sigma_window_bps: float = 0.0) -> dict:
    """lambda, gamma, alpha for one asset on one frozen tape; never raises."""
    out: dict = {"available": False}
    try:
        ticks = np.asarray(tape.ticks(asset.upper()), dtype=np.float64)
        if ticks.ndim != 2 or ticks.shape[0] < 30 or ticks.shape[1] < 3:
            return {"available": False, "reason": "fewer than 30 ticks"}
        rets, flow, notional = _buckets(ticks)
        lam = loss_aversion(rets, flow)
        gam = risk_aversion(rets, notional)
        mid = float(ticks[-1, 1])
        sigma = float(sigma_window_bps) if sigma_window_bps and sigma_window_bps > 0 else (
            float(np.std(rets)) * math.sqrt(60.0) if rets.size > 2 else 0.0)
        alpha = probability_weighting(tape.book(asset.upper()), mid, sigma)
        live = sum(1 for d in (lam, gam, alpha) if d.get("available"))
        bits = [d["read"] for d in (lam, gam, alpha) if d.get("available")]
        out = {
            "available": live > 0,
            "live_measurements": live,
            "lambda": lam, "gamma": gam, "alpha": alpha,
            "intensity": {
                "loss_averse": lam.get("loss_averse", 0.0) if lam.get("available") else 0.0,
                "gain_chasing": lam.get("gain_chasing", 0.0) if lam.get("available") else 0.0,
                "risk_seeking": gam.get("risk_seeking", 0.0) if gam.get("available") else 0.0,
                "risk_averse": gam.get("risk_averse", 0.0) if gam.get("available") else 0.0,
                "tail_fear": alpha.get("tail_fear", 0.0) if alpha.get("available") else 0.0,
                "complacent": alpha.get("complacent", 0.0) if alpha.get("available") else 0.0,
            },
            "read": "; ".join(bits) or "no utility read yet",
            "method": "inverse RL: the prospect-theory agent (loss aversion λ, risk aversion γ, probability weighting α) whose optimal actions reproduce the observed flow and book",
        }
    except Exception as exc:  # noqa: BLE001
        out = {"available": False, "reason": str(exc)[:160]}
    return out
