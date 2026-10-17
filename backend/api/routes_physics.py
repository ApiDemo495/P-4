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
from backend.physics import kinetics
from backend.physics.telemetry import get_telemetry

router = APIRouter()

SECTIONS = [
    {"section": "0", "title": "Ontology: BTC = consumed work, PAXG = inert rest mass", "status": "framing"},
    {"section": "1", "title": "Landauer thermal valve (Θ, w_thermal)", "status": "retired", "why": "hashrate is constant inside a minute - the valve never moved (Round AJ)"},
    {"section": "2", "title": "Planetary solar flux (Ω, w_solar, α blend)", "status": "retired", "why": "a modelled hub table, not a measurement; constant inside a minute (Round AJ)"},
    {"section": "3", "title": "Multi-node ZK Nash swarm", "status": "not implemented", "why": "needs many nodes; one terminal"},
    {"section": "4", "title": "Work-to-rest-mass ratio, E=mc² price update, phase angle", "status": "retired", "why": "drift and phase angle printed 0.00 / 'balanced' every window (Round AJ)"},
    {"section": "5", "title": "AMM price surface x·y=k", "status": "retired", "why": "needs a DEX host that is unreachable from most networks (Round AJ)"},
    {"section": "6", "title": "Relativistic spatial arbitrage", "status": "not implemented", "why": "needs geographically separate nodes"},
    {"section": "7", "title": "Lending-protocol liquidation sniping", "status": "not implemented", "why": "needs on-chain position data + execution"},
    {"section": "8", "title": "VPIN · O-U bridge (cost-gated) · pendulum · Avellaneda-Stoikov · fragmentation", "status": "implemented (tape + venues)"},
    {"section": "8.6-8.10", "title": "Hawkes self-excitation · momentum flux · trade-sign entropy · cross-leg diffusion · tape temperature", "status": "implemented (tape only, Round AJ)"},
    {"section": "9", "title": "Sovereign MEV block sequencing", "status": "not implemented", "why": "the app is not a block builder"},
    {"section": "10", "title": "PAXG / wBTC peg drift", "status": "implemented when a gold spot or wBTC quote is live; inactive otherwise"},
    {"section": "11", "title": "Multi-mechanism Kelly, temperature drag, spread-cost gate", "status": "implemented as a weighted fusion layer"},
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
            "vpin_crit": K.VPIN_CRIT, "ou_min_sharpe": K.OU_MIN_SHARPE, "delta_w": K.DELTA_W_DEFAULT,
            "kelly_cap": K.KELLY_CAP, "hawkes_critical": kinetics.HAWKES_CRITICAL,
            "entropy_informed": kinetics.ENTROPY_INFORMED, "temperature_hot": kinetics.TEMPERATURE_HOT,
        },
    }
