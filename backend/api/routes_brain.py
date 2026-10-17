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


@router.get("/api/brain/matrix/flywire")
async def brain_matrix_flywire() -> dict:
    """Round AZ: the real brain matrix.  Population x population blocks of
    the measured FlyWire v783 connectome - the mushroom body (28 formula
    input channels -> Kenyon cells -> MBONs, with PAM/PPL1 dopamine and the
    lateral horn) and the whole brain by super-class - plus the latest
    whole-brain read-out.  ``loaded`` false means the connectome is still
    downloading/building and the 80x80 stand-in is in use."""
    brain = get_manager().brain
    out = {"loaded": brain.connectome_ready, "status": brain.status.value, "message": brain.message,
           "data": brain.data.to_dict(), "error": brain.connectome_error,
           "fallback_in_use": not brain.connectome_ready}
    if brain.mb is not None:
        out["mushroom_body"] = brain.mb.block_matrix()
        out["mushroom_body"]["passes"] = brain.mb_passes
        out["mushroom_body"]["last_pass_us"] = brain.mb_last_us
    if brain.whole is not None:
        out["whole_brain"] = brain.whole.block_matrix()
        out["whole_brain"]["passes"] = brain.whole_passes
        out["whole_brain"]["readout"] = brain.whole_readout.to_dict() if brain.whole_readout else None
    return out


@router.get("/api/brain/dopamine")
async def brain_dopamine() -> dict:
    """Round AR: the brain's own dopamine - reward prediction error, mood,
    over-confidence / tilt guards and the size appetite."""
    manager = get_manager()
    return {"dopamine": manager.dopamine.to_dict(), "brain_phasic_applied": manager.brain.dopamine_phasic}


@router.get("/api/competition")
async def competition() -> dict:
    """Round AS: the competition scoreboard (last-trade vs last-trade, tie =
    loss), the streak ladder and the boundary picker's learned weights."""
    manager = get_manager()
    return {"competition": manager.competition_payload(), "boundary_pick": manager.last_pick,
            "picker": manager.picker.to_dict()}


@router.get("/api/horizons")
async def horizons() -> dict:
    """Round BD: the 55 / 60 / 70 / 120 s picks of the current boundary and
    each lane's measured strict-sign record."""
    return {"horizons": get_manager().horizons_payload()}


@router.get("/api/temporal-arb")
async def temporal_arb() -> dict:
    """Round BL: continuous 70 s risk priced against its segmented twin
    (60 s rolled at :60 into 10 s) - the volatility decay spread, both
    structures' P(win) and the realised twin-minus-continuous ledger."""
    return {"temporal_arb": get_manager().temporal_arb_payload()}


@router.get("/api/hindsight")
async def hindsight() -> dict:
    """Round BI: the trailing hour's review of every input - formulas, agents,
    systems, features - with hits, misses, bps lost, the windows they lost
    in and the correction applied to the current window."""
    return {"hindsight": get_manager().hindsight_payload()}


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
