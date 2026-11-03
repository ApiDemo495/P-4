"""Formula Genesis Engine routes (Round AL).

* ``/api/genesis/current``            the composite that was locked with the window on screen
* ``/api/genesis/live``               the latest composite from the worker (never touches the lock)
* ``/api/genesis/status``             pool counts, lifecycle states, candle store, next re-score / genesis
* ``/api/genesis/formulas``           the pool, filterable by state / domain, fitness ordered
* ``/api/genesis/formulas/{fid}``     one formula: definition, params, 7 metrics, history, autopsy
* ``/api/genesis/domains``            the ten domains and their sub-categories with the mathematics
* ``/api/genesis/graveyard``          dead formulas with the autopsy's cause
* ``POST /api/genesis/rescore``       force a full re-evaluation now (background thread)
* ``POST /api/genesis/breed``         force a genesis generation now (background thread)
"""
from __future__ import annotations

import threading

from fastapi import APIRouter, HTTPException, Query

from backend.api.state import get_manager

router = APIRouter()


def _asset(asset: str | None) -> str:
    manager = get_manager()
    a = (asset or manager.asset or "BTC").upper()
    if a not in manager.genesis.assets:
        raise HTTPException(status_code=404, detail=f"unknown asset {a!r}")
    return a


@router.get("/api/genesis/current")
async def genesis_current() -> dict:
    return get_manager().genesis_payload()


@router.get("/api/genesis/live")
async def genesis_live(asset: str | None = None) -> dict:
    return get_manager().genesis.vote(_asset(asset))


@router.get("/api/genesis/status")
async def genesis_status() -> dict:
    manager = get_manager()
    out = manager.genesis.status()
    out["providers"] = manager.macro.status()
    out["weight"] = float(getattr(manager.settings, "weight_genesis", 0.0) or 0.0)
    return out


@router.get("/api/genesis/formulas")
async def genesis_formulas(asset: str | None = None, state: str | None = None, domain: int | None = None,
                           limit: int = Query(200, ge=1, le=2500), offset: int = Query(0, ge=0)) -> dict:
    return get_manager().genesis.list_formulas(_asset(asset), state=state, domain=domain, limit=limit, offset=offset)


@router.get("/api/genesis/formulas/{fid}")
async def genesis_formula(fid: str, asset: str | None = None) -> dict:
    found = get_manager().genesis.formula(_asset(asset), fid)
    if found is None:
        raise HTTPException(status_code=404, detail=f"no formula {fid!r}")
    return found


@router.get("/api/genesis/domains")
async def genesis_domains() -> dict:
    return {"domains": get_manager().genesis.domains()}


@router.get("/api/genesis/graveyard")
async def genesis_graveyard(asset: str | None = None, limit: int = Query(100, ge=1, le=300)) -> dict:
    a = _asset(asset)
    rows = get_manager().genesis.graveyard[a]
    return {"asset": a, "total": len(rows), "items": rows[-limit:][::-1]}


def _background(fn, *args) -> dict:
    manager = get_manager()
    if manager.genesis._busy:
        return {"started": False, "detail": f"worker is busy: {manager.genesis._busy}"}
    threading.Thread(target=fn, args=args, daemon=True, name="genesis-manual").start()
    return {"started": True}


@router.post("/api/genesis/rescore")
async def genesis_rescore(asset: str | None = None) -> dict:
    a = _asset(asset)
    manager = get_manager()
    if len(manager.genesis.stores[a]) < 240:
        return {"started": False, "detail": f"only {len(manager.genesis.stores[a])} candles; need 240"}
    return _background(manager.genesis.rescore, a)


@router.post("/api/genesis/breed")
async def genesis_breed(asset: str | None = None) -> dict:
    a = _asset(asset)
    manager = get_manager()
    if not manager.genesis.scored_at_len[a]:
        return {"started": False, "detail": "the pool has not been scored yet"}
    return _background(manager.genesis.genesis, a)
