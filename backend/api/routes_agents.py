"""Agent management routes (Section 7 + the Settings screen in Section 11.2).

Includes the endpoints that make the v1.0 fixes real:

* ``POST /api/agents/local/upload`` - multipart upload with validation, load and
  a mandatory test inference before the agent is declared ready
* ``POST /api/agents/{name}/test`` - "Test" buttons for the Gemini key, the
  GitHub PAT, the CryptoPanic key and the NewsAPI key
* ``POST /api/agents/local/unload`` - release memory, optionally delete the file
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.agents import keyring
from backend.api.state import get_manager

log = logging.getLogger("drosophila.api.agents")

router = APIRouter()

MAX_UPLOAD_BYTES = 16 * 1024 * 1024 * 1024  # 16 GB - sanity ceiling, not a promise


class KeyPayload(BaseModel):
    key: str = ""
    persist: bool = False
    #: 1 = primary (always preferred), 2-3 = temporary stand-ins used only
    #: while the primary is rejected / rate limited / erroring.
    slot: int = 1


class KeysPayload(BaseModel):
    """All three boxes at once; an empty string clears a slot."""
    keys: list[str] = []
    persist: bool = False


class TestPayload(BaseModel):
    key: str = ""


class UnloadPayload(BaseModel):
    delete_file: bool = False
    slot: int | None = None


@router.get("/api/agents")
async def agent_status() -> dict:
    manager = get_manager()
    return manager.agents.status_payload()


@router.get("/api/agents/{name}")
async def agent_detail(name: str) -> dict:
    manager = get_manager()
    agents = manager.agents.agents
    if name not in agents:
        raise HTTPException(status_code=404, detail=f"unknown agent {name!r}")
    agent = agents[name]
    payload = agent.health().to_dict()
    if getattr(agent, "last_result", None) is not None:
        payload["last_result"] = agent.last_result.to_dict()
    return payload


# ---------------------------------------------------------------------------
# Key testing and storage
# ---------------------------------------------------------------------------


@router.post("/api/agents/{name}/test")
async def test_agent(name: str, payload: TestPayload | None = None) -> dict:
    manager = get_manager()
    key = (payload.key if payload else "") or ""
    if name == "gemini":
        return await manager.agents.gemini.test_key(key)
    if name == "github":
        return await manager.agents.github.test_token(key)
    if name == "cryptopanic":
        from backend.news import cryptopanic_source

        return await cryptopanic_source.test_key(key or manager.settings.rings["cryptopanic"].current())
    if name == "newsapi":
        from backend.news import newsapi_source

        return await newsapi_source.test_key(key or manager.settings.rings["newsapi"].current())
    if name == "rss":
        from backend.news import rss_source

        feeds = await rss_source.check_feeds(manager.settings.rss_feeds)
        ok = sum(1 for f in feeds if f["ok"])
        return {"valid": ok > 0, "detail": f"{ok}/{len(feeds)} feeds reachable", "feeds": feeds}
    if name in ("glassnode", "twelvedata", "lunarcrush"):
        from backend.data import keyed_providers

        return await keyed_providers.test_key(name, key or getattr(manager.settings, f"{name}_key", ""))
    if name == "neuprint":
        from backend.brain import health_check

        settings = manager.settings
        result = await health_check.neuprint_test_token(
            key or settings.neuprint_token,
            server=settings.neuprint_server,
            dataset=settings.neuprint_dataset,
        )
        # A working token is only useful once the connectome has been rebuilt.
        if result.get("valid"):
            result["detail"] = (
                f"{result.get('detail', 'token accepted')} — press Rebuild connectome "
                "to use it now"
            )
        return result
    raise HTTPException(status_code=404, detail=f"unknown agent {name!r}")


@router.get("/api/settings/keys")
async def key_rings() -> dict:
    """Masked view of every key ring: which slot is in use, which are cooling
    down and why.  Never returns a full key."""
    manager = get_manager()
    rings = {name: ring.status() for name, ring in manager.settings.rings.items()}
    rings["neuprint"] = {"configured": bool(manager.settings.neuprint_token), "configured_slots": 1}
    rings["cave"] = {"configured": bool(manager.settings.cave_token), "configured_slots": 1}
    for name in ("glassnode", "twelvedata", "lunarcrush"):
        rings[name] = {"configured": bool(getattr(manager.settings, f"{name}_key", "")), "configured_slots": 1}
    return {"rings": rings, "max_slots": keyring.MAX_SLOTS}


def _store_key(manager, name: str, key: str, slot: int) -> None:
    slot = min(max(int(slot or 1), 1), keyring.MAX_SLOTS)
    if name == "gemini":
        manager.agents.gemini.set_key(key, slot)
    elif name == "github":
        manager.agents.github.set_token(key, slot)
    elif name in ("cryptopanic", "newsapi"):
        manager.settings.rings[name].set_slot(slot, key)
        setattr(manager.settings, f"{name}_key", manager.settings.rings[name].primary)
    elif name == "neuprint":
        manager.settings.neuprint_token = key
    elif name == "cave":
        manager.settings.cave_token = key
    elif name in ("glassnode", "twelvedata", "lunarcrush"):
        setattr(manager.settings, f"{name}_key", key)
    else:
        raise HTTPException(status_code=404, detail=f"unknown key slot {name!r}")


@router.post("/api/settings/keys/{name}")
async def set_key(name: str, payload: KeyPayload) -> dict:
    """Store one key (``slot`` 1-3) in the running process and optionally in
    ``.env``.  Slot 1 is the primary; 2 and 3 are the temporary stand-ins."""
    manager = get_manager()
    key = (payload.key or "").strip()
    _store_key(manager, name, key, payload.slot)
    written = False
    if payload.persist:
        written = _persist_env(name, key, payload.slot)
    ring = manager.settings.rings.get(name)
    return {"saved": True, "persisted": written, "slot": payload.slot,
            "ring": ring.status() if ring is not None else None}


@router.put("/api/settings/keys/{name}")
async def set_keys(name: str, payload: KeysPayload) -> dict:
    """All slots of one provider in a single call (the Settings page's
    three boxes).  Missing / empty entries clear the slot."""
    manager = get_manager()
    if name not in manager.settings.rings:
        raise HTTPException(status_code=404, detail=f"{name!r} has a single key box")
    keys = [(k or "").strip() for k in (payload.keys or [])][: keyring.MAX_SLOTS]
    keys += [""] * (keyring.MAX_SLOTS - len(keys))
    for index, key in enumerate(keys):
        _store_key(manager, name, key, index + 1)
    written = False
    if payload.persist:
        written = all(_persist_env(name, key, index + 1) for index, key in enumerate(keys))
    return {"saved": True, "persisted": written, "ring": manager.settings.rings[name].status()}


_SINGLE_ENV_NAMES = {
    "neuprint": "NEUPRINT_APPLICATION_CREDENTIALS",
    "cave": "CAVE_TOKEN",
    "glassnode": "GLASSNODE_API_KEY",
    "twelvedata": "TWELVEDATA_API_KEY",
    "lunarcrush": "LUNARCRUSH_API_KEY",
}


def _persist_env(provider: str, value: str, slot: int = 1) -> bool:
    """Write (or clear) a key in the repo ``.env`` (never committed)."""
    from backend.core import config as cfg

    if provider in keyring.ENV_NAMES:
        env_name = keyring.env_name(provider, slot)
    else:
        env_name = _SINGLE_ENV_NAMES.get(provider)
    if not env_name:
        return False
    path = cfg.REPO_ROOT / ".env"
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    updated = False
    for index, line in enumerate(lines):
        if line.strip().startswith(f"{env_name}="):
            lines[index] = f"{env_name}={value}"
            updated = True
            break
    if not updated and value:
        lines.append(f"{env_name}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Local model lifecycle (Section 7.2)
# ---------------------------------------------------------------------------


@router.post("/api/agents/local/upload")
async def upload_local_model(file: UploadFile = File(...), slot: int = 1) -> dict:
    """Upload into model slot 1, 2 or 3 - all loaded slots answer every
    cycle in parallel and are merged into the single local opinion."""
    manager = get_manager()
    agent = manager.agents.local
    if not 1 <= int(slot) <= 3:
        raise HTTPException(status_code=400, detail="slot must be 1, 2 or 3")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 16 GB upload ceiling")

    result = await agent.upload(file.filename or "model.gguf", payload, slot=int(slot))
    if not result.get("success"):
        # A validation failure is a 422 with a human-readable message, which is
        # what the v1.0 upload flow never produced.
        raise HTTPException(status_code=422, detail=result.get("error", "Model upload failed"))
    return result


@router.post("/api/agents/local/validate")
async def validate_local_model(file: UploadFile = File(...)) -> dict:
    """Step 2 only: validate without loading (used by the file picker preview)."""
    from backend.agents.local_model_agent import validate_file
    import tempfile
    from pathlib import Path

    payload = await file.read()
    suffix = Path(file.filename or "model.gguf").suffix or ".gguf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    try:
        result = validate_file(temp_path)
        return result.to_dict()
    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/api/agents/local/unload")
async def unload_local_model(payload: UnloadPayload | None = None) -> dict:
    manager = get_manager()
    delete = bool(payload.delete_file) if payload else False
    slot = payload.slot if payload else None
    return manager.agents.local.unload(delete_file=delete, slot=slot)


@router.get("/api/agents/local/status")
async def local_model_status() -> dict:
    manager = get_manager()
    agent = manager.agents.local
    health = agent.health().to_dict()
    health["memory"] = agent.check_memory()
    health["slots"] = agent.slots_payload()
    health["models_loaded"] = len(agent.loaded)
    health["max_models"] = 3
    if agent.validation is not None:
        health["validation"] = agent.validation.to_dict()
    return health


@router.post("/api/agents/local/stub")
async def toggle_stub(enabled: bool = True) -> dict:
    """Enable the clearly-labelled development stub (no real weights)."""
    manager = get_manager()
    agent = manager.agents.local
    agent.stub = bool(enabled)
    if enabled and not agent.slots[0].ready:
        result = await agent.load(agent.store_dir / "stub", slot=1)
        return {"stub": True, "load": result}
    if not enabled:
        agent.unload()
    return {"stub": agent.stub, "ready": agent.ready}
