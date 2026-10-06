"""Hedge outcomes - what the BTC/PAXG pair is expected to *do* this window.

Round AJ replaced the "Hedge Status" tiles (four bare formula numbers) with
an outcome system.  Given the locked side and its confidence, the synchronised
BTC/PAXG return grid and the four hedge formulas, it produces:

* the pair's covariance structure for the horizon: sigma of each leg, the
  correlation rho, the dollar hedge ratio beta (BTC on PAXG and PAXG on BTC)
  and the variance reduction a hedge buys (rho squared);
* a **joint outcome matrix** - the probability of each of the four
  (BTC up/down x PAXG up/down) results from a bivariate normal whose mean on
  the locked leg is the drift implied by the lock's probability and whose
  mean on the other leg follows the correlation plus the flow rotation (SHRP);
* the **pair actions** that are consistent with the lock (outright, hedged
  with PAXG at beta, pair-neutral) each with expected return in bp, the
  standard deviation of that return and the probability it ends positive;
  the best one by expected return per unit of risk is the recommendation;
* **scenarios**: what the other leg does if BTC (or PAXG) moves one or two
  sigma, from beta;
* the **regime** read from HSI (hedge working / breaking), HRDD (hedge ratio
  drifting), GCDV (paths diverging) and SHRP (rotation of flow).

Everything is derived from the frozen snapshot, so it is lock-safe; nothing
here is a promise - it is the distribution the engine is actually trading on,
written out instead of hidden.
"""
from __future__ import annotations

import math

import numpy as np

SQRT2 = math.sqrt(2.0)


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def _phi_inv(p: float) -> float:
    """Acklam's rational approximation of the probit (|error| < 1.2e-9)."""
    p = min(1.0 - 1e-9, max(1e-9, p))
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - 0.02425:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def bvn_upper(h: float, k: float, rho: float) -> float:
    """P(X > h, Y > k) for a standard bivariate normal with correlation rho
    (Drezner-Wesolowsky Gauss-Legendre quadrature, 20 nodes)."""
    rho = max(-0.999, min(0.999, rho))
    x, w = np.polynomial.legendre.leggauss(20)
    # integrate over the correlation: P = Phi(-h)Phi(-k) + (1/2pi) ∫0^rho exp(-(h²+k²-2rhk)/(2(1-r²)))/sqrt(1-r²) dr
    r = 0.5 * rho * (x + 1.0)
    integrand = np.exp(-(h * h + k * k - 2.0 * h * k * r) / (2.0 * (1.0 - r * r))) / np.sqrt(1.0 - r * r)
    integral = 0.5 * rho * float(np.dot(w, integrand))
    return max(0.0, min(1.0, _phi(-h) * _phi(-k) + integral / (2.0 * math.pi)))


def _joint(mu_b: float, mu_p: float, s_b: float, s_p: float, rho: float) -> dict:
    """Four-cell matrix of P(BTC dir, PAXG dir) over the horizon."""
    h = -mu_b / max(s_b, 1e-12)        # P(BTC up) = P(Z_b > h)
    k = -mu_p / max(s_p, 1e-12)
    up_up = bvn_upper(h, k, rho)
    p_b_up = _phi(-h)
    p_p_up = _phi(-k)
    up_dn = max(0.0, p_b_up - up_up)
    dn_up = max(0.0, p_p_up - up_up)
    dn_dn = max(0.0, 1.0 - up_up - up_dn - dn_up)
    return {"btc_up_paxg_up": up_up, "btc_up_paxg_down": up_dn, "btc_down_paxg_up": dn_up,
            "btc_down_paxg_down": dn_dn, "p_btc_up": p_b_up, "p_paxg_up": p_p_up}


def build(snapshot, formulas: dict, side: str, confidence: float, asset: str = "BTC",
          horizon_s: float = 60.0) -> dict:
    """The outcome report (never raises; returns ``valid: False`` with a reason)."""
    f = {k: float(v) for k, v in (formulas or {}).items() if isinstance(v, (int, float))}
    hsi, hrdd, shrp, gcdv = f.get("HSI", 0.0), f.get("HRDD", 0.0), f.get("SHRP", 0.0), f.get("GCDV", 0.0)
    rot = shrp if asset.upper() == "PAXG" else -shrp              # + ⇒ flow into PAXG (SHRP is signed per asset)
    regime = _regime(hsi, hrdd, gcdv, rot)
    base = {"valid": False, "asset": asset, "side": side, "confidence": round(float(confidence), 4),
            "horizon_s": horizon_s, "regime": regime,
            "formulas": {"hsi": round(hsi, 4), "hrdd": round(hrdd, 4), "shrp": round(shrp, 4), "gcdv": round(gcdv, 4)}}
    synced = getattr(snapshot, "synced", None)
    if synced is None or not getattr(synced, "valid", False):
        base["reason"] = (getattr(synced, "reason", "") or "BTC/PAXG grid not available yet")
        return base
    rb = np.asarray(synced.btc_returns, dtype=float)
    rp = np.asarray(synced.paxg_returns, dtype=float)
    ok = np.isfinite(rb) & np.isfinite(rp)
    rb, rp = rb[ok], rp[ok]
    n = int(rb.size)
    window = float(getattr(synced, "window_seconds", 60) or 60)
    if n < 10 or rb.std() <= 0 or rp.std() <= 0:
        base["reason"] = f"{n} grid returns - need 10 with movement on both legs"
        return base
    step = window / max(1, n)                       # seconds per grid step
    scale = math.sqrt(horizon_s / max(step, 1e-9))  # grid-step sigma → horizon sigma
    s_b = float(rb.std()) * scale
    s_p = float(rp.std()) * scale
    rho = float(np.corrcoef(rb, rp)[0, 1])
    rho = 0.0 if not math.isfinite(rho) else max(-0.999, min(0.999, rho))
    beta_b_on_p = rho * s_b / s_p          # dollar move of BTC per dollar move of PAXG
    beta_p_on_b = rho * s_p / s_b
    # Mean of the locked leg: the drift that makes P(side) equal the lock's
    # confidence over the horizon.  The other leg follows rho, then the flow
    # rotation (SHRP > 0 for PAXG means flow into gold) tilts it.
    # The lock's confidence is a conviction in [0, 1], not a probability:
    # map it to P(side) in [0.5, 0.95] (full conviction = 95 %).
    p_side = min(0.95, max(0.5, 0.5 + 0.45 * float(confidence)))
    z = _phi_inv(p_side)
    sign = 1.0 if side == "BUY" else -1.0
    if asset.upper() == "BTC":
        mu_b = sign * z * s_b
        mu_p = beta_p_on_b * mu_b + 0.25 * rot * s_p
    else:
        mu_p = sign * z * s_p
        mu_b = beta_b_on_p * mu_p - 0.25 * rot * s_b
    joint = _joint(mu_b, mu_p, s_b, s_p, rho)
    # Pair actions consistent with the lock (BTC leg sign is the lock when
    # the asset is BTC; the mirror when it is PAXG).
    b_leg = sign if asset.upper() == "BTC" else -sign
    actions = []

    def action(name: str, wb: float, wp: float, note: str) -> None:
        mu = wb * mu_b + wp * mu_p
        var = (wb * s_b) ** 2 + (wp * s_p) ** 2 + 2.0 * wb * wp * rho * s_b * s_p
        sd = math.sqrt(max(var, 1e-18))
        actions.append({"action": name, "btc_weight": round(wb, 3), "paxg_weight": round(wp, 3),
                        "expected_bps": round(mu * 1e4, 2), "sigma_bps": round(sd * 1e4, 2),
                        "p_profit": round(_phi(mu / sd), 4), "ratio": round(mu / sd, 4), "note": note})

    hedge = abs(beta_b_on_p) if abs(beta_b_on_p) < 5 else 5.0
    action("outright BTC", b_leg, 0.0, f"{'long' if b_leg > 0 else 'short'} BTC, no hedge")
    action("BTC hedged with PAXG at β", b_leg, -b_leg * beta_p_on_b, f"PAXG leg sized β = {beta_p_on_b:+.3f} per $1 of BTC")
    action("pair-neutral BTC vs PAXG", b_leg, -b_leg, "equal dollars, opposite sides")
    action("outright PAXG", 0.0, -b_leg, f"{'long' if -b_leg > 0 else 'short'} PAXG alone (the mirror leg)")
    best = max(actions, key=lambda a: a["ratio"])
    scenarios = []
    for mult, label in ((1.0, "1σ"), (2.0, "2σ")):
        for leg_sign, word in ((1.0, "up"), (-1.0, "down")):
            shock = leg_sign * mult * s_b
            pnl = best["btc_weight"] * shock + best["paxg_weight"] * beta_p_on_b * shock
            scenarios.append({"if": f"BTC {word} {mult * s_b * 1e4:.1f} bp ({label})",
                              "then": f"PAXG {beta_p_on_b * shock * 1e4:+.1f} bp expected",
                              "best_action_pnl_bps": round(pnl * 1e4, 1)})
    ratio_now = float(snapshot.last_price("BTC")) / max(1e-9, float(snapshot.last_price("PAXG")))
    bp = np.asarray(synced.btc_prices, dtype=float)
    pp = np.asarray(synced.paxg_prices, dtype=float)
    okp = np.isfinite(bp) & np.isfinite(pp) & (pp > 0)
    spread = np.log(bp[okp] / pp[okp]) if okp.sum() >= 5 else np.zeros(0)
    spread_z = float((spread[-1] - spread.mean()) / spread.std()) if spread.size >= 5 and spread.std() > 0 else 0.0
    logic = [
        f"grid {window:.0f} s, {n} returns; σ_BTC({horizon_s:.0f} s) = {s_b * 1e4:.1f} bp, σ_PAXG = {s_p * 1e4:.1f} bp, ρ = {rho:+.3f}",
        f"β(BTC on PAXG) = ρ·σ_B/σ_P = {beta_b_on_p:+.3f}; β(PAXG on BTC) = {beta_p_on_b:+.3f}; hedging removes ρ² = {rho * rho:.0%} of variance",
        f"lock {side} {asset}, conviction {float(confidence):.0%} ⇒ P(side) = ½ + 0.45·conviction = {p_side:.0%} ⇒ μ_{asset} = Φ⁻¹({p_side:.2f})·σ = {(mu_b if asset.upper() == 'BTC' else mu_p) * 1e4:+.1f} bp; "
        f"other leg = β·μ + ¼·rotation(SHRP {shrp:+.2f})·σ",
        f"P(BTC↑,PAXG↑) = {joint['btc_up_paxg_up']:.0%}, P(BTC↑,PAXG↓) = {joint['btc_up_paxg_down']:.0%}, "
        f"P(BTC↓,PAXG↑) = {joint['btc_down_paxg_up']:.0%}, P(BTC↓,PAXG↓) = {joint['btc_down_paxg_down']:.0%} (bivariate normal, Drezner quadrature)",
        f"best action by μ/σ: {best['action']} ({best['expected_bps']:+.2f} bp ± {best['sigma_bps']:.2f}, P(profit) {best['p_profit']:.0%})",
        f"regime: {regime['label']} - {regime['detail']}",
    ]
    base.update({
        "valid": True, "window_seconds": window, "samples": n,
        "sigma_btc_bps": round(s_b * 1e4, 2), "sigma_paxg_bps": round(s_p * 1e4, 2), "rho": round(rho, 4),
        "beta_btc_on_paxg": round(beta_b_on_p, 4), "beta_paxg_on_btc": round(beta_p_on_b, 4),
        "variance_reduction": round(rho * rho, 4), "hedge_ratio": round(hedge, 4),
        "mu_btc_bps": round(mu_b * 1e4, 2), "mu_paxg_bps": round(mu_p * 1e4, 2),
        "joint": {k: round(v, 4) for k, v in joint.items()},
        "actions": actions, "best": best, "scenarios": scenarios,
        "spread": {"ratio": round(ratio_now, 4), "z": round(spread_z, 3),
                   "read": ("BTC rich vs PAXG - reversion favours PAXG" if spread_z > 1.5 else
                            "BTC cheap vs PAXG - reversion favours BTC" if spread_z < -1.5 else "inside its band")},
        "logic": logic,
    })
    return base


def _regime(hsi: float, hrdd: float, gcdv: float, rot: float) -> dict:
    if hsi > 0.8:
        label, detail = "hedge breaking", f"HSI {hsi:.2f} > 0.80: the pair moves as one risk asset - PAXG is not a hedge right now"
    elif hsi > 0.5:
        label, detail = "hedge weakening", f"HSI {hsi:.2f}: correlation rising, hedge only partly effective"
    elif abs(hrdd) > 0.5:
        label, detail = "hedge ratio drifting", f"HRDD {hrdd:+.2f}: β has moved from its hour baseline - re-size the hedge"
    elif abs(gcdv) > 0.5:
        label, detail = "paths diverging", f"GCDV {gcdv:+.2f}: the two legs are separating fast - pair trade, not hedge"
    else:
        label, detail = "hedge working", f"HSI {hsi:.2f}, HRDD {hrdd:+.2f}, GCDV {gcdv:+.2f}: PAXG offsets BTC as expected"
    rotation = ("flow rotating into PAXG" if rot > 0.15 else "flow rotating into BTC" if rot < -0.15 else "no rotation")
    return {"label": label, "detail": detail, "rotation": rotation, "hsi": round(hsi, 4), "hrdd": round(hrdd, 4),
            "gcdv": round(gcdv, 4), "rotation_value": round(rot, 4)}
