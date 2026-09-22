"""Formula routes for the Formula Explorer."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.api.state import get_manager
from backend.formulas.engine import CATEGORY_NAMES, FormulaEngine

router = APIRouter()


@router.get("/api/formulas")
async def formula_catalogue() -> dict:
    """All 22 formulas with metadata (used to build the explorer UI)."""
    return FormulaEngine.metadata()


@router.get("/api/formulas/current")
async def current_formulas() -> dict:
    """The frozen values that produced the locked signal for this cycle."""
    manager = get_manager()
    result = manager.last_formula_result
    if result is None:
        return {"cycle_number": manager.stats.cycle_number, "formulas": {}, "locked": False}
    payload = result.to_dict()
    payload["cycle_number"] = manager.stats.cycle_number
    payload["locked_values"] = True
    payload["signal"] = manager.lock.current_signal.signal if manager.lock.current_signal else None
    payload["degradation_level"] = int(manager.degradation)
    return payload


@router.get("/api/formulas/live")
async def live_formulas() -> dict:
    """Live values, refreshed every 15 s.  These do NOT change the signal."""
    manager = get_manager()
    return {
        "cycle_number": manager.stats.cycle_number,
        "note": "Live formula values only. Signal remains LOCKED.",
        "formulas": manager.last_live_formulas,
        "signal": manager.lock.current_signal.signal if manager.lock.current_signal else None,
    }


@router.get("/api/formulas/categories")
async def categories() -> dict:
    return {
        "categories": [{"key": key, "name": name} for key, name in CATEGORY_NAMES.items()]
    }


@router.get("/api/formulas/timings")
async def timings(limit: int = Query(1, ge=1, le=50)) -> dict:
    """Per-formula latency from the most recent pass - verifies the ~3 ms budget."""
    manager = get_manager()
    result = manager.last_formula_result
    if result is None:
        return {"timings_ms": {}, "total_ms": 0.0}
    ranked = sorted(result.timings_ms.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "cycle_number": manager.stats.cycle_number,
        "timings_ms": {k: round(v, 4) for k, v in ranked},
        "total_ms": round(result.total_ms, 4),
        "budget_ms": 3.0,
    }
