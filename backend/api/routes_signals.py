"""Signal routes: the WebSocket stream and the REST mirrors.

``/ws/signals`` is the single real-time channel.  Message types:

    HELLO             connection accepted + current state + full snapshot
    SIGNAL            at the boundary: the LOCKED signal for the window that
                      just started, plus the same snapshot (one render pass)
    PULSE             on every mark of the window grid (t+15, t+30, t+45 of a
                      60 s window): live formulas, news, agents, brain, accuracy
                      - all in ONE message, so every panel refreshes together
    CYCLE_START       only with SIGNAL_PIPELINE=0: signal cleared, "Computing..."
    EMERGENCY_OVERRIDE  critical news event -> the exit side (flip of the open
                        direction); the protocol is binary, so there is no HOLD
    OUTCOME           60 s later: win/loss for the DRG learner

Every message that carries a window also carries ``clock``: the absolute
(window_started_at_ms, window_ends_at_ms, server_time_ms, cycle_id) block both
frontends render their countdown from.  The countdown can therefore never drift
or restart when an unrelated message arrives.
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
        # One snapshot on connect: the client paints every panel from this
        # single message instead of firing six REST requests on boot.
        snapshot = manager.snapshot_payload(include_history=True)
        snapshot.update(
            {
                "asset": manager.asset,
                "pending_asset": manager.pending_asset,
                "lock": manager.lock.status(),
                "status": manager.status(),
                "formulas": manager.last_live_formulas,
                "conviction_note": manager.conviction_note,
                "assets": list(cfg.ASSETS),
            }
        )
        await websocket.send_text(json.dumps({"type": "HELLO", "data": snapshot}))

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
        # The same authoritative clock the WebSocket pushes, so a client that
        # can only poll still counts down to the true boundary.
        "clock": manager.master_clock(),
        "asset": manager.asset,
        "pending_asset": manager.pending_asset,
        "degradation_level": int(manager.degradation),
    }


@router.get("/api/signal/history")
async def signal_history(limit: int = 20) -> dict:
    manager = get_manager()
    return manager.history_payload(limit)


@router.get("/api/signal/outcomes")
async def outcomes() -> dict:
    manager = get_manager()
    return manager.outcomes_payload()


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
