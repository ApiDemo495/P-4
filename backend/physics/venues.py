"""Sections 5 (AMM price surface) and 10 (peg drift) - the on-chain legs.

Both depend on live telemetry (DexScreener pools, gold spot, wrapped-BTC
price).  When a source is unreachable the mechanism says so and returns a
zero edge; it never invents a dislocation.
"""
from __future__ import annotations

import numpy as np

from backend.physics import constants as K


# ------------------------------------------------------------ Section 5
def amm_surface(pools: list[dict], btc_usd: float, paxg_usd: float, pool_source: str) -> dict:
    """x·y = k pools quoted through USD: P_k = y_k/x_k in PAXG per BTC."""
    base = {"key": "amm", "section": "5", "name": "AMM price surface (x·y = k)", "direction": 0}
    ref = btc_usd / paxg_usd if paxg_usd > 0 else 0.0
    btc_pools = [p for p in pools if p["base"] in ("WBTC", "CBBTC", "TBTC") and p["quote"].startswith("USD")]
    paxg_pools = [p for p in pools if p["base"] == "PAXG" and p["quote"].startswith("USD")]
    if not btc_pools or not paxg_pools or ref <= 0:
        return {**base, "value": 0.0, "edge_bps": 0.0, "pools": len(pools), "source": pool_source,
                "logic": f"{len(btc_pools)} BTC and {len(paxg_pools)} PAXG pools visible "
                         f"({'DexScreener blocked here' if not pools else 'need one of each'}); Δ_max undefined"}
    # Price surface: every (BTC pool, PAXG pool) pair gives one P_k.
    surface = []
    for b in btc_pools[:8]:
        for g in paxg_pools[:8]:
            price = b["price_usd"] / g["price_usd"]
            depth = min(b["liquidity_usd"], g["liquidity_usd"])
            fee = b["fee"] + g["fee"]
            surface.append((price, depth, fee, f"{b['dex']}:{b['base']}/{g['dex']}:PAXG"))
    prices = np.array([s[0] for s in surface])
    d_max = float(prices.max() - prices.min())
    d_max_bps = d_max / ref * 1e4
    # Best single leg against the reference, net of the two pool fees.
    best = max(surface, key=lambda s: abs(s[0] - ref) / ref - s[2])
    gross = (best[0] - ref) / ref                    # + ⇒ pools price BTC above reference
    net_bps = (abs(gross) - best[2]) * 1e4
    direction = int(-np.sign(gross)) if net_bps > 0 else 0   # pools rich in BTC ⇒ sell BTC there ⇒ −
    gamma = 0.02
    executable = gamma * best[1]
    return {
        **base, "value": gross * 1e4, "direction": direction, "edge_bps": max(0.0, net_bps),
        "reference": ref, "delta_max_bps": d_max_bps, "pools": len(pools), "pairs": len(surface),
        "best_pair": best[3], "best_depth_usd": best[1], "executable_usd": executable,
        "source": pool_source,
        "logic": (f"P_ref = {ref:.3f} PAXG/BTC; {len(surface)} pool pairs, Δ_max = {d_max_bps:.2f} bp; "
                  f"best {best[3]} at {best[0]:.3f} ({gross * 1e4:+.2f} bp), fees {best[2] * 1e4:.1f} bp ⇒ net {net_bps:+.2f} bp; "
                  f"γ·x_k depth {executable:,.0f} USD"),
    }


# ----------------------------------------------------------- Section 10
def peg_drift(paxg_usd: float, xau_usd: float | None, xau_source: str,
              wbtc_usd: float | None, btc_usd: float, wbtc_source: str) -> dict:
    base = {"key": "peg", "section": "10", "name": "PAXG / wBTC peg drift", "direction": 0}
    eps = K.PAXG_MINT_BURN_FEE + K.GAS_FEE_FRACTION
    rows = []
    direction = 0
    edge = 0.0
    if xau_usd and paxg_usd > 0:
        rho = paxg_usd / xau_usd
        dev = rho - 1.0
        act = abs(dev) > eps
        rows.append(f"ρ_PAXG = {paxg_usd:.2f}/{xau_usd:.2f} = {rho:.5f} ({dev * 1e4:+.1f} bp vs ε = {eps * 1e4:.0f} bp)"
                    + (" ⇒ PAXG rich: sell PAXG/buy BTC, mint" if act and dev > 0 else
                       " ⇒ PAXG cheap: buy PAXG/sell BTC, burn" if act else " ⇒ inside the band"))
        if act:
            direction = 1 if dev > 0 else -1
            edge = (abs(dev) - eps) * 1e4
    else:
        rho = None
        rows.append("ρ_PAXG unavailable (gold spot source unreachable here)")
    if wbtc_usd and btc_usd > 0:
        rho_b = wbtc_usd / btc_usd
        dev_b = rho_b - 1.0
        rows.append(f"ρ_wBTC = {rho_b:.5f} ({dev_b * 1e4:+.1f} bp)")
        if abs(dev_b) > eps and direction == 0:
            direction = -1 if dev_b > 0 else 1    # wBTC rich ⇒ native BTC cheap relative ⇒ sell wrapped, buy native (pair-neutral)
            edge = (abs(dev_b) - eps) * 1e4 * 0.5
    else:
        rho_b = None
        rows.append("ρ_wBTC unavailable")
    live = bool(xau_usd) or bool(wbtc_usd)
    return {
        **base, "value": (rho - 1.0) * 1e4 if rho else 0.0, "direction": direction, "edge_bps": max(0.0, edge),
        "rho_paxg": rho, "rho_wbtc": rho_b, "epsilon_bps": eps * 1e4,
        "source": f"live ({xau_source})" if live else "unavailable",
        "logic": "; ".join(rows),
    }
