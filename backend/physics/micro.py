"""Section 8 - the perpetual microstructural signals on the BTC/PAXG tape.

All five read only the frozen snapshot (tape) - never a live buffer - so they
are lock-safe like the 22 formulas.  Direction convention throughout:
``+1`` = BTC strengthens against PAXG (buy BTC / sell PAXG), ``-1`` = the
opposite.  Non-directional mechanisms return ``direction = 0`` and only an
edge estimate in basis points.
"""
from __future__ import annotations

import math

import numpy as np

from backend.physics import constants as K


def _ratio_series(snapshot, max_points: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """BTC/PAXG ratio sampled on the BTC tick clock (PAXG held at its last
    print before each BTC tick).  Returns (times_s, ratio)."""
    btc = snapshot.ticks("BTC")
    paxg = snapshot.ticks("PAXG")
    if btc.shape[0] < 10 or paxg.shape[0] < 2:
        return np.zeros(0), np.zeros(0)
    btc = btc[-max_points:]
    t_b = btc[:, 0] / 1000.0
    t_p = paxg[:, 0] / 1000.0
    idx = np.searchsorted(t_p, t_b, side="right") - 1
    valid = idx >= 0
    if valid.sum() < 10:
        return np.zeros(0), np.zeros(0)
    p = paxg[idx[valid], 1]
    ratio = btc[valid, 1] / np.where(p > 0, p, np.nan)
    ok = np.isfinite(ratio)
    return t_b[valid][ok], ratio[ok]


# --------------------------------------------------------------- 8.1 VPIN
def vpin(snapshot) -> dict:
    ticks = snapshot.ticks("BTC")
    if ticks.shape[0] < 20:
        return {"key": "vpin", "section": "8.1", "name": "VPIN flow toxicity", "value": 0.0,
                "direction": 0, "edge_bps": 0.0, "source": "tape", "logic": "fewer than 20 ticks"}
    vol = np.abs(ticks[:, 2])
    side = np.sign(ticks[:, 3])
    total = float(vol.sum())
    if total <= 0:
        return {"key": "vpin", "section": "8.1", "name": "VPIN flow toxicity", "value": 0.0,
                "direction": 0, "edge_bps": 0.0, "source": "tape", "logic": "no volume"}
    bucket = total / K.VPIN_BUCKETS
    buys = np.zeros(K.VPIN_BUCKETS)
    sells = np.zeros(K.VPIN_BUCKETS)
    cum = np.cumsum(vol)
    which = np.minimum((cum / bucket).astype(int), K.VPIN_BUCKETS - 1)
    np.add.at(buys, which, np.where(side > 0, vol, 0.0))
    np.add.at(sells, which, np.where(side < 0, vol, 0.0))
    v = float(np.abs(buys - sells).sum() / max(1e-12, (buys + sells).sum()))
    imbalance = float((buys.sum() - sells.sum()) / total)
    toxic = v > K.VPIN_CRIT
    direction = int(np.sign(imbalance)) if toxic else 0
    edge = (v - K.VPIN_CRIT) / (1.0 - K.VPIN_CRIT) * 8.0 if toxic else 0.0   # ≤ 8 bp momentum impulse
    return {
        "key": "vpin", "section": "8.1", "name": "VPIN flow toxicity",
        "value": v, "direction": direction, "edge_bps": edge, "imbalance": imbalance, "active": True,
        "buckets": K.VPIN_BUCKETS, "bucket_volume": bucket, "toxic": toxic, "source": "tape",
        "logic": (f"{K.VPIN_BUCKETS} equal-volume buckets of {bucket:.4f}; VPIN = Σ|V_buy−V_sell| / ΣV = {v:.3f} "
                  f"{'>' if toxic else '≤'} {K.VPIN_CRIT} ⇒ {'toxic: follow dominant flow' if toxic else 'balanced: mean reversion regime'}; "
                  f"net flow {imbalance:+.3f}"),
    }


# ------------------------------------------------------ 8.2 fragmentation
def fragmentation(snapshot, venues: dict, pools: list[dict]) -> dict:
    """alpha_frag = S_composite - mean S_k across the venues we can see."""
    quotes: dict[str, tuple[float, float]] = {}
    book = snapshot.book("BTC")
    try:
        if book is not None and book[0, 0, 0] > 0 and book[1, 0, 0] > 0:
            quotes["binance"] = (float(book[0, 0, 0]), float(book[1, 0, 0]))
    except (IndexError, TypeError):
        pass
    for venue, assets in (venues or {}).items():
        q = assets.get("BTC")
        if q and q["bid"] > 0 and q["ask"] > 0:
            quotes[venue] = (q["bid"], q["ask"])
    for pool in pools or []:
        if pool["base"] in ("WBTC", "CBBTC", "TBTC") and pool["quote"].startswith("USD"):
            half = pool["fee"] / 2.0
            quotes[f"{pool['dex']}:{pool['base']}"] = (pool["price_usd"] * (1 - half), pool["price_usd"] * (1 + half))
    if len(quotes) < 2:
        mid = snapshot.last_price("BTC")
        spread = (quotes["binance"][1] - quotes["binance"][0]) if "binance" in quotes else 0.0
        return {"key": "fragmentation", "section": "8.2", "name": "Multi-venue fragmentation", "value": 0.0,
                "direction": 0, "edge_bps": 0.0, "venues": len(quotes), "source": "tape (one venue visible)", "active": False,
                "logic": f"inactive: only {len(quotes)} venue visible (needs 2+ live venue quotes); "
                         f"own spread {spread / max(mid, 1e-9) * 1e4:.2f} bp; α_frag undefined"}
    bids = {k: v[0] for k, v in quotes.items()}
    asks = {k: v[1] for k, v in quotes.items()}
    composite = max(asks.values()) - min(bids.values())
    mean_internal = float(np.mean([a - b for b, a in quotes.values()]))
    alpha = composite - mean_internal
    cross = max(bids.values()) - min(asks.values())       # > 0 ⇒ crossed books: riskless
    mid = float(np.mean([(b + a) / 2 for b, a in quotes.values()]))
    alpha_bps = alpha / mid * 1e4
    cross_bps = cross / mid * 1e4
    return {
        "key": "fragmentation", "section": "8.2", "name": "Multi-venue fragmentation",
        "value": alpha_bps, "direction": 0, "edge_bps": max(0.0, cross_bps),
        "composite_spread": composite, "mean_internal_spread": mean_internal,
        "best_bid_venue": max(bids, key=bids.get), "best_ask_venue": min(asks, key=asks.get),
        "venues": len(quotes), "source": "live" if len(quotes) > 1 else "tape", "active": True,
        "logic": (f"{len(quotes)} venues; S_composite = max ask − min bid = {composite:.2f}; "
                  f"mean S_k = {mean_internal:.2f}; α_frag = {alpha_bps:.2f} bp; "
                  f"crossed by {cross_bps:+.2f} bp (buy {min(asks, key=asks.get)}, sell {max(bids, key=bids.get)})"),
    }


# --------------------------------------------- 8.3 Ornstein-Uhlenbeck bridge
def ornstein_uhlenbeck(snapshot) -> dict:
    t, p = _ratio_series(snapshot)
    base = {"key": "ou", "section": "8.3", "name": "Ornstein–Uhlenbeck mean reversion", "source": "tape"}
    if p.size < 30:
        return {**base, "value": 0.0, "direction": 0, "edge_bps": 0.0, "sharpe": 0.0,
                "logic": "fewer than 30 aligned BTC/PAXG prints"}
    now = t[-1]
    recent = p[t >= now - K.OU_EQUILIBRIUM_WINDOW_S]
    p_bar = float(recent.mean()) if recent.size else float(p.mean())
    # AR(1) on the deviation: x_{k+1} = φ x_k + e,  φ = e^{-κ Δt}
    x = p - p_bar
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    dt = max(dt, 1e-3)
    x0, x1 = x[:-1], x[1:]
    denom = float(np.dot(x0, x0))
    phi = float(np.dot(x0, x1) / denom) if denom > 0 else 0.0
    phi = min(0.999, max(1e-4, phi))
    kappa_s = -math.log(phi) / dt                     # per second
    resid = x1 - phi * x0
    sigma_s = float(resid.std()) / math.sqrt(dt) if resid.size > 2 else 0.0
    p0 = float(p[-1])
    gap = p_bar - p0
    expected = gap * (1.0 - math.exp(-60.0 * kappa_s))
    var60 = sigma_s ** 2 * (1.0 - math.exp(-120.0 * kappa_s)) / (2.0 * kappa_s) if kappa_s > 0 else 0.0
    sharpe = abs(expected) / math.sqrt(var60) if var60 > 0 else 0.0
    # Round AI: a reversion is only tradeable inside the window if its
    # half-life sits between 5 s and 10 min (faster is bid/ask bounce, slower
    # never arrives within 60 s) and the expected move clears the cost of
    # crossing the spread.  The edge is capped at one 60-second sigma - the
    # old 20 bp cap let a thin-tape fit dominate the Kelly blend.
    half_life_s = math.log(2.0) / kappa_s if kappa_s > 0 else float("inf")
    tradeable = 5.0 <= half_life_s <= 600.0
    try:
        book = snapshot.book("BTC")
        cost_bps = (float(book[1, 0, 0]) - float(book[0, 0, 0])) / max(1e-9, float(book[0, 0, 0])) * 1e4
    except (IndexError, TypeError, ZeroDivisionError):
        cost_bps = 0.0
    cost_bps = max(0.5, cost_bps)
    expected_bps = abs(expected) / p0 * 1e4 if p0 > 0 else 0.0
    sigma60_bps = math.sqrt(var60) / p0 * 1e4 if (var60 > 0 and p0 > 0) else 0.0
    execute = sharpe > K.OU_MIN_SHARPE and tradeable and expected_bps > cost_bps
    direction = int(np.sign(gap)) if execute else 0
    edge_bps = min(expected_bps - cost_bps, max(1.0, sigma60_bps)) if execute else 0.0
    why = ("execute" if execute else
           f"half-life {half_life_s:.0f} s outside 5-600 s" if not tradeable else
           f"E[ΔP] {expected_bps:.2f} bp ≤ spread cost {cost_bps:.2f} bp" if expected_bps <= cost_bps else
           "stand aside")
    return {
        **base, "value": gap / p0 if p0 > 0 else 0.0, "direction": direction, "edge_bps": edge_bps, "active": True,
        "half_life_s": half_life_s, "cost_bps": cost_bps, "expected_bps": expected_bps, "sigma60_bps": sigma60_bps,
        "p0": p0, "p_bar": p_bar, "kappa_per_min": kappa_s * 60.0, "sigma_per_s": sigma_s,
        "expected_move": expected, "sharpe": sharpe, "execute": execute, "samples": int(p.size),
        "logic": (f"dP = κ(P̄−P)dt + σdW on {p.size} prints; P̄ (VWAP-300 s) = {p_bar:.4f}, P₀ = {p0:.4f}; "
                  f"κ = −ln φ/Δt = {kappa_s * 60:.3f}/min, σ = {sigma_s:.5f}/√s; "
                  f"half-life {half_life_s:.0f} s; E[ΔP₆₀] = (P̄−P₀)(1−e^(−60κ)) = {expected:+.5f} = {expected_bps:.2f} bp vs cost {cost_bps:.2f} bp; "
                  f"SR₆₀ = {sharpe:.2f} {'>' if sharpe > K.OU_MIN_SHARPE else '≤'} {K.OU_MIN_SHARPE} ⇒ {why}"),
    }


# ------------------------------------------------------------ 8.4 pendulum
def pendulum(snapshot) -> dict:
    base = {"key": "pendulum", "section": "8.4", "name": "BTC/PAXG risk pendulum (REI)", "source": "tape"}
    btc = snapshot.ticks("BTC")
    paxg = snapshot.ticks("PAXG")
    if btc.shape[0] < 30 or paxg.shape[0] < 30:
        return {**base, "value": 0.5, "direction": 0, "edge_bps": 0.0, "logic": "not enough prints"}
    t_end = max(btc[-1, 0], paxg[-1, 0]) / 1000.0
    t_start = min(btc[0, 0], paxg[0, 0]) / 1000.0
    n = int(max(4, min(40, (t_end - t_start) // K.PENDULUM_BINS_S)))
    edges = np.linspace(t_start, t_end + 1e-6, n + 1)

    def notional(ticks):
        return np.histogram(ticks[:, 0] / 1000.0, bins=edges, weights=np.abs(ticks[:, 1] * ticks[:, 2]))[0]

    vb, vp = notional(btc), notional(paxg)
    # Each leg is normalised by its own mean notional, so REI measures the
    # *relative* flow between the two assets, not BTC's larger turnover.
    vb = vb / max(float(vb.mean()), 1e-12)
    vp = vp / max(float(vp.mean()), 1e-12)
    tot = vb + vp
    rei = np.where(tot > 0, vb / np.where(tot > 0, tot, 1.0), np.nan)
    rei = rei[np.isfinite(rei)]
    if rei.size < 4:
        return {**base, "value": 0.5, "direction": 0, "edge_bps": 0.0, "logic": "not enough bins"}
    y = rei - 0.5
    dt = K.PENDULUM_BINS_S
    if rei.size >= 6:
        # AR(2) fit  y_{k+1} = a y_k + b y_{k-1}  ⇔  discretised  ÿ + 2βẏ + ω₀²y = 0
        A = np.column_stack([y[1:-1], y[:-2]])
        coef, *_ = np.linalg.lstsq(A, y[2:], rcond=None)
        a, b = float(coef[0]), float(coef[1])
        omega0_sq = max(0.0, (1.0 - a - b) / dt ** 2)
        beta = max(0.0, (1.0 - b) / (2.0 * dt))
        fitted = True
    else:
        omega0_sq, beta, fitted = K.PENDULUM_OMEGA0 ** 2, 1.0 / 600.0, False
    y0, v0 = float(y[-1]), float((y[-1] - y[-2]) / dt)
    # Integrate 60 s ahead (semi-implicit Euler, 1 s steps)
    yy, vv = y0, v0
    for _ in range(60):
        acc = -2.0 * beta * vv - omega0_sq * yy
        vv += acc
        yy += vv
    rei_now, rei_next = 0.5 + y0, max(0.0, min(1.0, 0.5 + yy))
    delta = rei_next - rei_now
    direction = int(np.sign(delta)) if abs(delta) > 0.01 else 0
    return {
        **base, "value": rei_now, "direction": direction, "edge_bps": min(6.0, abs(delta) * 100.0), "active": True,
        "rei_next": rei_next, "omega0": math.sqrt(omega0_sq), "beta": beta, "fitted": fitted,
        "bins": int(rei.size),
        "logic": (f"REI = V_BTC/(V_BTC+V_PAXG) over {rei.size} bins of {dt:.0f} s = {rei_now:.3f}; "
                  f"R̈EI + 2β·ṘEI + ω₀²(REI−½) = F, {'AR(2) fit' if fitted else 'prior'} ω₀ = {math.sqrt(omega0_sq):.5f} rad/s, β = {beta:.5f}; "
                  f"REI(t+60) = {rei_next:.3f} ⇒ {'expansion: BTC' if direction > 0 else 'compression: PAXG' if direction < 0 else 'flat'}"),
    }


# ------------------------------------------------- 8.5 Avellaneda-Stoikov
def avellaneda_stoikov(snapshot, seconds_left: float = 60.0) -> dict:
    t, p = _ratio_series(snapshot)
    base = {"key": "as_spread", "section": "8.5", "name": "Avellaneda–Stoikov quote", "source": "tape",
            "direction": 0}
    if p.size < 10:
        return {**base, "value": 0.0, "edge_bps": 0.0, "logic": "not enough prints"}
    rets = np.diff(np.log(p))
    dt = max(1e-3, float(np.median(np.diff(t))))
    sigma = float(rets.std()) / math.sqrt(dt)               # per √s, in return units
    span = max(1.0, float(t[-1] - t[0]))
    arrival = p.size / span                                # prints per second (Poisson proxy)
    # κ of the fill intensity λ(δ) = A·e^(−κδ): the book's own half-spread
    # (return units) sets the decay scale, κ = 1 / half-spread.
    book = snapshot.book("BTC")
    try:
        half_spread_ret = (float(book[1, 0, 0]) - float(book[0, 0, 0])) / (2.0 * float(book[0, 0, 0]))
    except (IndexError, TypeError, ZeroDivisionError):
        half_spread_ret = 0.0
    half_spread_ret = max(half_spread_ret, 1e-6)
    kappa_book = 1.0 / half_spread_ret
    gamma = K.AS_RISK_AVERSION
    half = sigma ** 2 * seconds_left / 2.0 + (1.0 / gamma) * math.log(1.0 + gamma / kappa_book)
    half_bps = half * 1e4
    fill_rate = min(arrival, 2.0) * 0.1                    # a tenth of prints lift our quote (model)
    income_bps = fill_rate * 2.0 * half_bps * 60.0 / max(1.0, p.size)   # per unit of inventory
    kappa_arrival = arrival
    return {
        **base, "value": half_bps, "edge_bps": min(income_bps, 3.0), "active": True, "cost_bps": half_bps,
        "sigma_per_sqrt_s": sigma, "kappa_arrival": kappa_arrival, "gamma": gamma, "seconds_left": seconds_left,
        "logic": (f"δ = σ²(T−t)/2 + (1/γ)·ln(1+γ/κ) with σ = {sigma:.2e}/√s, T−t = {seconds_left:.0f} s, "
                  f"γ = {gamma}, κ_book = 1/half-spread = {kappa_book:.0f}, arrivals {kappa_arrival:.2f}/s ⇒ half-spread {half_bps:.3f} bp; "
                  f"inventory flattened at t = 59 s so the quote carries no direction"),
    }
