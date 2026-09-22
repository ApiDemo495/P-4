"""News routes (Section 4) plus the manual emergency trigger for testing."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.api.state import get_manager
from backend.news.critical_event_detector import CriticalEvent

router = APIRouter()


class EmergencyRequest(BaseModel):
    headline: str = "Manual test event"
    reason: str = "Triggered manually from the dashboard"
    severity: str = "CRITICAL"


@router.get("/api/news")
async def latest_news(limit: int = 8) -> dict:
    manager = get_manager()
    return {
        "items": manager.news.latest_items(limit),
        "status": manager.news.status.to_dict(),
        "niv": round(manager.news.current_niv(), 4),
        "cache_size": len(manager.news.cache.all()),
        "providers": manager.news.cache.providers,
    }


@router.get("/api/news/critical")
async def critical_history(limit: int = 20) -> dict:
    manager = get_manager()
    return {
        "events": manager.news.detector.history(limit),
        "active": manager.lock.emergency_active(),
        "remaining_seconds": manager.lock.emergency_remaining(),
    }


@router.post("/api/news/emergency")
async def trigger_emergency(request: EmergencyRequest) -> dict:
    """Force an emergency override - used to demo and test the override path."""
    manager = get_manager()
    event = CriticalEvent(
        headline=request.headline,
        severity=request.severity,
        reason=request.reason,
        source="manual",
        tier=1,
    )
    data = await manager.trigger_emergency(event)
    return {"triggered": True, "event": data}


@router.post("/api/news/emergency/clear")
async def clear_emergency() -> dict:
    manager = get_manager()
    manager.lock.clear_emergency()
    return {"cleared": True}


@router.post("/api/news/poll")
async def poll_now() -> dict:
    manager = get_manager()
    polled = await manager.news.poll_all(force=True)
    return {
        "polled": polled,
        "items": len(manager.news.cache.all()),
        "niv": round(manager.news.current_niv(), 4),
    }
