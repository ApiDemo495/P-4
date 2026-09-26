"""Formula routes for the Formula Explorer."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.api.state import get_manager
from backend.formulas.engine import CATEGORY_NAMES, FormulaEngine

router = APIRouter()


def _live_payload(result, manager) -> dict:
    """Values + readings + traces for the pass that produced them."""
    if result is None:
        return {"formulas": {}, "readings": {}, "traces": {}}
    return {
        "formulas": {k: round(float(v), 6) for k, v in result.values.items()},
        "readings": result.to_dict().get("readings", {}),
        "traces": result.traces,
        "timings_ms": {k: round(float(v), 4) for k, v in result.timings_ms.items()},
        "total_ms": round(result.total_ms, 4),
    }


@router.get("/api/formulas")
async def formula_catalogue() -> dict:
    """All 22 formulas with metadata + the logic registry (used by the explorer)."""
    return FormulaEngine.metadata()


@router.get("/api/formulas/self-test")
async def formula_self_test(refresh: bool = Query(False)) -> dict:
    """Does every formula do what it claims?  Runs the four synthetic tapes.

    This is the audit trail behind "the formulas work": each verdict carries the
    claim, the pass/fail, and the numbers observed on each tape.  Cached after
    the first call (four tapes x 40 windows of one engine each), ``refresh=true``
    re-runs it.
    """
    from backend.formulas import self_test as self_test_module

    report = self_test_module.cached_report(refresh=refresh)
    payload = report.to_dict()
    payload["summary"] = self_test_module.summary_line(report)
    return payload


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
    return manager.live_formulas_payload(
        note="Live formula values only. Signal remains LOCKED."
    )


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
