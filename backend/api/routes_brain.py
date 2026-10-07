"""Brain routes - the Settings "Brain Connection" panel and the matrix viewer."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.api.state import get_manager
from backend.brain import explain as brain_explain

router = APIRouter()


@router.get("/api/brain/status")
async def brain_status() -> dict:
    return get_manager().brain_payload()


@router.get("/api/brain/connectome")
async def brain_connectome() -> dict:
    """Round AQ: the real FlyWire v783 connectome - download/build state, the
    mushroom-body and whole-brain populations, and the latest whole-brain
    read-out (balance, descending drive, activity per super-class, top MBONs)."""
    manager = get_manager()
    payload = manager.brain.connectome_payload()
    payload["last_fuse"] = manager.last_whole_brain
    return payload


@router.get("/api/brain/dopamine")
async def brain_dopamine() -> dict:
    """Round AR: the brain's own dopamine - reward prediction error, mood,
    over-confidence / tilt guards and the size appetite."""
    manager = get_manager()
    return {"dopamine": manager.dopamine.to_dict(), "brain_phasic_applied": manager.brain.dopamine_phasic}


@router.get("/api/triune")
async def triune() -> dict:
    """Round AR: the three-minds analyst - human / AI / data votes, their
    records, agreements and the verdict applied to the current window."""
    manager = get_manager()
    return {"current": manager.last_triune, "report": manager.triune.report()}


@router.get("/api/brain/matrix")
async def brain_matrix(limit: int = Query(0, ge=0, le=2000)) -> dict:
    """The 80x80 adjacency matrix as an edge list (for matrix_viewer)."""
    manager = get_manager()
    payload = manager.brain.matrix_payload(limit=limit or None)
    payload["source"] = manager.brain.status.value
    payload["gain"] = round(manager.brain.gain, 4)
    return payload


@router.post("/api/brain/reconnect")
async def reconnect() -> dict:
    manager = get_manager()
    status = await manager.brain.reconnect()
    await manager.brain.refresh_health()
    return {"status": status.value, "detail": manager.brain.message}


@router.get("/api/brain/health")
async def health() -> dict:
    manager = get_manager()
    result = await manager.brain.refresh_health()
    return result.to_dict()


@router.get("/api/brain/trace")
async def last_trace() -> dict:
    manager = get_manager()
    result = manager.last_formula_result
    if result is None or not result.brain_trace:
        return {"available": False}
    trace = result.brain_trace
    trace["available"] = True
    trace["cycle_number"] = manager.stats.cycle_number
    trace["asset"] = manager.asset
    return trace


@router.get("/api/brain/neurons")
async def brain_neurons(limit: int = 400) -> dict:
    """The named Drosophila circuit the brain runs on: every node's hemibrain
    cell type, compartment, transmitter and cell count; every edge's synapse
    count and sign (Round S).  ``live`` is true when neuPrint measured the
    counts, false when they are the coded typical hemibrain magnitudes."""
    from backend.brain import connectome

    payload = connectome.default().to_dict(edge_limit=max(1, min(int(limit), 5000)))
    brain = get_manager().brain
    payload["matrix_source"] = getattr(brain, "source", None)
    payload["matrix_live"] = bool(getattr(brain, "is_live", False))
    return payload


@router.get("/api/brain/wiring")
async def brain_wiring() -> dict:
    """How (and where) the Drosophila circuit is used, stage by stage."""
    return brain_explain.wiring()


@router.get("/api/brain/explain")
async def brain_explanation() -> dict:
    """Plain-language account of what the brain just did, for the locked window."""
    return get_manager().brain_explain_payload()
