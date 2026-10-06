"""Section 10 (peg drift) - the on-chain leg that survives Round AI.

It depends on live telemetry (gold spot, wrapped-BTC price).  When no source
is reachable the mechanism is *inactive* (excluded from the blend) and says
so; it never invents a dislocation.  Section 5 (AMM surface) was retired.
"""
from __future__ import annotations

from backend.physics import constants as K


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
        "source": f"live ({xau_source})" if live else "unavailable", "active": live,
        "logic": ("" if live else "inactive: needs a gold spot or wBTC quote; ") + "; ".join(rows),
    }
