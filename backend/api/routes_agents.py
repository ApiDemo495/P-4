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

from backend.api.state import get_manager

log = logging.getLogger("drosophila.api.agents")

router = APIRouter()

MAX_UPLOAD_BYTES = 16 * 1024 * 1024 * 1024  # 16 GB - sanity ceiling, not a promise


class KeyPayload(BaseModel):
    key: str = ""
    persist: bool = False


class TestPayload(BaseModel):
    key: str = ""


class UnloadPayload(BaseModel):
    delete_file: bool = False


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

        return await cryptopanic_source.test_key(key or manager.settings.cryptopanic_key)
    if name == "newsapi":
        from backend.news import newsapi_source

        return await newsapi_source.test_key(key or manager.settings.newsapi_key)
    if name == "rss":
        from backend.news import rss_source

        feeds = await rss_source.check_feeds(manager.settings.rss_feeds)
        ok = sum(1 for f in feeds if f["ok"])
        return {"valid": ok > 0, "detail": f"{ok}/{len(feeds)} feeds reachable", "feeds": feeds}
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


@router.post("/api/settings/keys/{name}")
async def set_key(name: str, payload: KeyPayload) -> dict:
    """Store a key in the running process (and optionally in ``.env``)."""
    manager = get_manager()
    key = (payload.key or "").strip()
    if name == "gemini":
        manager.agents.gemini.set_key(key)
    elif name == "github":
        manager.agents.github.set_token(key)
    elif name == "cryptopanic":
        manager.settings.cryptopanic_key = key
    elif name == "newsapi":
        manager.settings.newsapi_key = key
    elif name == "neuprint":
        manager.settings.neuprint_token = key
    elif name == "cave":
        manager.settings.cave_token = key
    else:
        raise HTTPException(status_code=404, detail=f"unknown key slot {name!r}")

    written = False
    if payload.persist and key:
        written = _persist_env(name, key)
    return {"saved": True, "persisted": written}


_ENV_NAMES = {
    "gemini": "GEMINI_API_KEY",
    "github": "GITHUB_MODELS_TOKEN",
    "cryptopanic": "CRYPTOPANIC_API_KEY",
    "newsapi": "NEWSAPI_API_KEY",
    "neuprint": "NEUPRINT_APPLICATION_CREDENTIALS",
    "cave": "CAVE_TOKEN",
}


def _persist_env(slot: str, value: str) -> bool:
    """Write a key into the repo ``.env`` (never committed - see .gitignore)."""
    from backend.core import config as cfg

    env_name = _ENV_NAMES.get(slot)
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
    if not updated:
        lines.append(f"{env_name}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Local model lifecycle (Section 7.2)
# ---------------------------------------------------------------------------


@router.post("/api/agents/local/upload")
async def upload_local_model(file: UploadFile = File(...)) -> dict:
    manager = get_manager()
    agent = manager.agents.local

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 16 GB upload ceiling")

    result = await agent.upload(file.filename or "model.gguf", payload)
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
    return manager.agents.local.unload(delete_file=delete)


@router.get("/api/agents/local/status")
async def local_model_status() -> dict:
    manager = get_manager()
    agent = manager.agents.local
    health = agent.health().to_dict()
    health["memory"] = agent.check_memory()
    if agent.validation is not None:
        health["validation"] = agent.validation.to_dict()
    return health


@router.post("/api/agents/local/stub")
async def toggle_stub(enabled: bool = True) -> dict:
    """Enable the clearly-labelled development stub (no real weights)."""
    manager = get_manager()
    agent = manager.agents.local
    agent.stub = bool(enabled)
    if enabled and not agent.ready:
        result = await agent.load(agent.store_dir / "stub")
        return {"stub": True, "load": result}
    if not enabled:
        agent.unload()
    return {"stub": agent.stub, "ready": agent.ready}
