"""Thermodynamic capital layer routes (Round T).

* ``/api/physics/current``   the report that was locked with the window on screen
* ``/api/physics/live``      a fresh pass on a live freeze (never touches the lock)
* ``/api/physics/telemetry`` every live source with provenance, age and last error
* ``/api/physics/spec``      which sections of the source document are implemented
"""
from __future__ import annotations

from fastapi import APIRouter

from backend.api.state import get_manager
from backend.physics import constants as K
from backend.physics.telemetry import get_telemetry

router = APIRouter()

SECTIONS = [
    {"section": "0", "title": "Ontology: BTC = consumed work, PAXG = inert rest mass", "status": "framing"},
    {"section": "1", "title": "Landauer thermal valve (Θ, w_thermal)", "status": "implemented", "inputs": "hashrate live/model, T_avg model"},
    {"section": "2", "title": "Planetary solar flux (Ω, w_solar, α blend)", "status": "implemented", "inputs": "exact geometry, hub table model"},
    {"section": "3", "title": "Multi-node ZK Nash swarm", "status": "not implemented", "why": "needs many nodes; one terminal"},
    {"section": "4", "title": "Work-to-rest-mass ratio, E=mc² price update, phase angle", "status": "implemented"},
    {"section": "5", "title": "AMM price surface x·y=k (5.1–5.2)", "status": "implemented (live pools via DexScreener)", "note": "5.3–5.5 routing/CLMM need on-chain execution"},
    {"section": "6", "title": "Relativistic spatial arbitrage", "status": "not implemented", "why": "needs geographically separate nodes"},
    {"section": "7", "title": "Lending-protocol liquidation sniping", "status": "not implemented", "why": "needs on-chain position data + execution"},
    {"section": "8", "title": "VPIN · fragmentation · O-U bridge · pendulum · Avellaneda-Stoikov", "status": "implemented (tape + venues)"},
    {"section": "9", "title": "Sovereign MEV block sequencing", "status": "not implemented", "why": "the app is not a block builder"},
    {"section": "10", "title": "PAXG / wBTC peg drift", "status": "implemented (gold spot + wBTC live)"},
    {"section": "11", "title": "Kelly blend, thermodynamic band, execution sequence", "status": "implemented as a weighted fusion layer", "deviation": "band = ±0.15 and acts as drag, not a hard clamp"},
    {"section": "12", "title": "TSR, phase angle (12.1, 12.4)", "status": "implemented", "note": "12.6 'cannot lose' is not claimed"},
    {"section": "13", "title": "60.000 s heartbeat", "status": "the existing minute-aligned lock"},
]


@router.get("/api/physics/current")
async def physics_current() -> dict:
    manager = get_manager()
    locked = (manager.last_fusion or {}).get("physics") or manager.last_physics
    return {"locked": bool((manager.last_fusion or {}).get("physics")), "report": locked,
            "weight": manager.settings.weight_physics}


@router.get("/api/physics/live")
async def physics_live() -> dict:
    manager = get_manager()
    snapshot = manager.market.freeze(news_items=(), drg_outcomes=manager.outcomes.array())
    # A separate engine instance would lose the trailing-hour histories, so
    # the shared one is used; it writes nothing the lock reads.
    report = manager.physics.compute(snapshot, manager.asset)
    return {"report": report}


@router.get("/api/physics/telemetry")
async def physics_telemetry(refresh: int = 0) -> dict:
    tele = get_telemetry()
    if refresh:
        await tele.refresh_once(force=True)
    return tele.status()


@router.get("/api/physics/spec")
async def physics_spec() -> dict:
    return {
        "sections": SECTIONS,
        "constants": {
            "k_B": K.K_B, "theta_star": K.THETA_STAR, "lambda_thermal": K.LAMBDA_THERMAL, "mu_solar": K.MU_SOLAR,
            "bits_erased_per_double_sha": K.BITS_ERASED_PER_DOUBLE_SHA, "vpin_crit": K.VPIN_CRIT,
            "ou_min_sharpe": K.OU_MIN_SHARPE, "delta_w": K.DELTA_W_DEFAULT, "mining_hubs": len(K.MINING_HUBS),
        },
    }
