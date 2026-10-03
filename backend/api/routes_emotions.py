"""Crowd-emotion routes (the user's live "which emotion is dominant" ask).

The engine itself lives in :mod:`backend.core.emotions`; these routes only read
the monitor that the cycle manager keeps running, so the panel that renders them
sees exactly the numbers that dampened the locked signal.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.api.state import get_manager

router = APIRouter()


def _catalogue() -> dict:
    """Static descriptions, sent with the reading so the UI needs no second call."""
    from backend.core.emotions import CONVICTION_HINT, EMOTIONS, FAMILIES, TIMESCALES

    return {
        "timescales": list(TIMESCALES),
        "emotions": [
            {
                "name": name,
                "label": label,
                "tone": tone,
                "family": FAMILIES.get(name, ""),
                "hint": CONVICTION_HINT.get(name, ""),
            }
            for name, (label, tone) in EMOTIONS.items()
        ],
    }


@router.get("/api/emotions")
async def emotions(asset: str | None = Query(None)) -> dict:
    """The crowd's current emotions, at microsecond resolution.

    Returns the live reading (re-sampled several times a second), the reading
    taken on the frozen snapshot when the current signal was locked, how many
    seconds the dominant emotion has held, the manipulation score and the
    confidence dampening it produced.
    """
    manager = get_manager()
    payload = manager.emotions_payload()
    if asset:
        payload["requested_asset"] = asset.upper()
    payload["catalogue"] = _catalogue()
    payload["cycle_number"] = manager.stats.cycle_number
    return payload


@router.get("/api/emotions/history")
async def emotions_history(limit: int = Query(120, ge=2, le=720)) -> dict:
    """The recent emotion timeline (dominant emotion, tone and manipulation)."""
    manager = get_manager()
    history = list(getattr(manager.emotions.tracker, "history", []) or [])[-limit:]
    return {
        "asset": manager.asset,
        "interval_seconds": manager.settings.emotion_interval_seconds,
        "samples": len(history),
        "history": history,
        "catalogue": _catalogue(),
    }


@router.get("/api/emotions/deep")
async def emotions_deep() -> dict:
    """The deep-reasoning layer behind the reading (round K).

    Every microstructure formula with its inputs and value (Hawkes branching,
    VPIN, Kyle's lambda, variance ratio, Hurst, permutation entropy, sign
    memory, wavelet spectrum, regime filter, ignition / stuffing / spoofing
    detectors), the Bayesian filter's posterior with the evidence behind it,
    and the ordered reasoning chain - live and as it was at lock time.
    """
    from backend.core.deep_micro import BANDS, EVIDENCE_LABEL, LIKELIHOOD

    manager = get_manager()
    live = manager.emotions.payload()
    locked = manager.emotion_locked or {}
    return {
        "asset": manager.asset,
        "cycle_number": manager.stats.cycle_number,
        "interval_seconds": manager.settings.emotion_interval_seconds,
        "at_us": live.get("at_us"),
        "live": live.get("deep") or {"available": False},
        "locked": locked.get("deep") or {"available": False},
        "model": {
            "bands": [
                {"name": name, "label": label, "clock_ms": clock_ms, "q": q}
                for name, label, clock_ms, q in BANDS
            ],
            "evidence": EVIDENCE_LABEL,
            "likelihood": LIKELIHOOD,
        },
    }
