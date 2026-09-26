"""Signal routes: the WebSocket stream and the REST mirrors.

``/ws/signals`` is the single real-time channel.  Message types:

    HELLO             connection accepted + current state
    CYCLE_START       t=0: signal cleared, "Computing..."
    SIGNAL            t~8 s: the LOCKED signal (sent once per cycle)
    FORMULA_UPDATE    every 15 s: live formula values, signal stays locked
    EMERGENCY_OVERRIDE  critical news event -> the exit side (flip of the open
                        direction); the protocol is binary, so there is no HOLD
    OUTCOME           60 s later: win/loss for the DRG learner
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.api.state import get_manager
from backend.core import config as cfg

log = logging.getLogger("drosophila.api.signals")

router = APIRouter()


class AssetSwitch(BaseModel):
    asset: str


@router.websocket("/ws/signals")
async def signals_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    manager = get_manager()
    queue = manager.subscribe()
    log.info("WebSocket client connected (%d total)", len(manager._subscribers))

    try:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "HELLO",
                    "data": {
                        "asset": manager.asset,
                        "pending_asset": manager.pending_asset,
                        "lock": manager.lock.status(),
                        "status": manager.status(),
                        "signal": manager.signal_payload(),
                        "window": manager.window_status(),
                        "formulas": manager.last_live_formulas,
                        "conviction_note": manager.conviction_note,
                        "assets": list(cfg.ASSETS),
                    },
                }
            )
        )

        async def pump() -> None:
            while True:
                message = await queue.get()
                await websocket.send_text(json.dumps(message))

        async def listen() -> None:
            while True:
                raw = await websocket.receive_text()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                action = payload.get("action")
                if action == "switch_asset":
                    try:
                        result = manager.switch_asset(str(payload.get("asset", "")))
                        await websocket.send_text(
                            json.dumps({"type": "ASSET_SWITCH", "data": result})
                        )
                    except ValueError as exc:
                        await websocket.send_text(
                            json.dumps({"type": "ERROR", "data": {"message": str(exc)}})
                        )
                elif action == "ping":
                    await websocket.send_text(json.dumps({"type": "PONG", "data": {}}))

        sender = asyncio.create_task(pump())
        receiver = asyncio.create_task(listen())
        done, pending = await asyncio.wait(
            {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for task in done:
            exc = task.exception()
            if exc is not None and not isinstance(exc, WebSocketDisconnect):
                log.debug("websocket task ended: %s", exc)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        log.debug("websocket closed: %s", exc)
    finally:
        manager.unsubscribe(queue)
        with contextlib.suppress(Exception):
            await websocket.close()


@router.get("/api/signal/current")
async def current_signal() -> dict:
    manager = get_manager()
    # Freshness contract: a prediction may never be older than 15 seconds.  The
    # cycle loop refreshes every window; this is the safety net for a window
    # that was missed, and it is a no-op in the normal case.
    try:
        refreshed = await manager.ensure_fresh()
    except Exception as exc:  # noqa: BLE001
        log.warning("freshness check failed: %s", exc)
        refreshed = False
    signal = manager.lock.try_get_current()
    prediction = manager.prediction_payload(signal)
    if refreshed:
        # Re-read through the payload helper so the numbers the client renders
        # are the ones that were just computed.
        signal = manager.lock.try_get_current()
        prediction = manager.prediction_payload(signal)
    return {
        "lock_state": manager.lock.state.value,
        "lock_icon": manager.lock.state.icon,
        # Never ``null``: while the lock is COMPUTING this is the sentinel, so
        # clients can render the layout (and the countdown) immediately.
        "signal": signal.to_dict() if signal else manager.signal_payload(),
        "prediction": prediction,
        "prediction_age_seconds": prediction["age_seconds"],
        "prediction_stale": prediction["state"] == "STALE",
        "conviction_note": manager.conviction_note,
        "window": manager.window_status(),
        "asset": manager.asset,
        "pending_asset": manager.pending_asset,
        "degradation_level": int(manager.degradation),
    }


@router.get("/api/signal/history")
async def signal_history(limit: int = 20) -> dict:
    manager = get_manager()
    return {"history": manager.lock.recent_history(min(max(limit, 1), 240))}


@router.get("/api/signal/outcomes")
async def outcomes() -> dict:
    manager = get_manager()
    rows = manager.outcomes.array()
    return {
        "count": int(rows.shape[0]),
        "win_rate": round(manager.outcomes.win_rate(), 4),
        "rows": [{"outcome": float(o), "pnl_bps": float(p)} for o, p in rows],
    }


@router.post("/api/assets/switch")
async def switch_asset(payload: AssetSwitch) -> dict:
    """Queue an asset switch - it takes effect at the next cycle boundary."""
    manager = get_manager()
    try:
        return manager.switch_asset(payload.asset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/signal/status")
async def status() -> dict:
    return get_manager().status()
