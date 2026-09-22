"""Brain routes - the Settings "Brain Connection" panel and the matrix viewer."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.api.state import get_manager

router = APIRouter()


@router.get("/api/brain/status")
async def brain_status() -> dict:
    manager = get_manager()
    payload = manager.brain.status_dict()
    payload["health"] = (
        manager.brain.last_health.to_dict() if manager.brain.last_health else None
    )
    return payload


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
