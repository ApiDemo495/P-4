"""FastAPI entry point for DROSOPHILA TRADER v2.0.

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Serves three things:

* ``/ws/signals``  - the single real-time channel to the UI
* ``/api/*``       - REST for the Formula Explorer, agents, news and brain
* ``/``            - the dashboard (see ``backend/web``)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api import routes_agents, routes_brain, routes_formulas, routes_news, routes_signals, state
from backend.core import config as cfg
from backend.core.cycle_manager import CycleManager

logging.basicConfig(
    level=getattr(logging, cfg.SETTINGS.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
)
log = logging.getLogger("drosophila.main")

_START_TIME = time.time()


def _local_stub_setting() -> bool | None:
    raw = os.environ.get("LOCAL_AGENT_STUB", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off", ""):
        return None
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = CycleManager(cfg.SETTINGS, local_stub=_local_stub_setting())
    state.set_manager(manager)
    app.state.manager = manager
    manager.mark_started()
    await manager.start()
    log.info("=" * 78)
    log.info(" DROSOPHILA TRADER v2.0 ready")
    log.info("   brain      : %s", manager.brain.status.value)
    log.info("   market data: %s", manager.market.active_source)
    log.info("   news       : %s", manager.news.status.coverage)
    log.info("   cycle      : %.1fs (world clock: %s)", cfg.SETTINGS.cycle_period_seconds, cfg.SETTINGS.use_world_clock)
    log.info("   dashboard  : http://0.0.0.0:%d/", cfg.SETTINGS.port)
    log.info("=" * 78)
    try:
        yield
    finally:
        await manager.stop()


app = FastAPI(
    title="Drosophila Trader v2.0",
    description=(
        "1-minute scalp trading engine for BTC and PAXG. 22 formulas, a locked "
        "signal protocol, a continuous news sentiment engine and a Drosophila "
        "mushroom-body connectome consensus."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.SETTINGS.cors_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_signals.router)
app.include_router(routes_formulas.router)
app.include_router(routes_agents.router)
app.include_router(routes_news.router)
app.include_router(routes_brain.router)


@app.get("/api/health")
async def health() -> dict:
    manager = state.manager_or_none()
    if manager is None:
        return {"healthy": False, "detail": "starting"}
    payload = manager.health()
    payload["uptime_seconds"] = round(time.time() - _START_TIME, 1)
    payload["version"] = "2.0.0"
    return payload


@app.get("/api/system/degradation")
async def degradation() -> dict:
    manager = state.get_manager()
    return {
        "level": int(manager.degradation),
        "label": manager.degradation.label,
        "warnings": manager.warnings,
    }


@app.get("/api/system/config")
async def system_config() -> dict:
    """Non-secret configuration, so the UI can render sensible defaults."""
    settings = cfg.SETTINGS
    return {
        "assets": list(cfg.ASSETS),
        "asset_params": cfg.ASSET_PARAMS,
        "cycle_period_seconds": settings.cycle_period_seconds,
        "time_scale": settings.time_scale,
        "lock_deadline_seconds": settings.lock_deadline_seconds,
        "formula_refresh_seconds": settings.formula_refresh_seconds,
        "signal_threshold": settings.signal_threshold,
        "min_fusion_confidence": settings.min_fusion_confidence,
        "emergency_duration_seconds": settings.scaled(settings.emergency_duration_seconds),
        "news_poll_seconds": settings.news_poll_seconds,
        "weights": {
            "drosophila": settings.weight_drosophila,
            "gemini": settings.weight_gemini,
            "local": settings.weight_local,
            "github": settings.weight_github,
        },
        "configured": {
            "gemini": bool(settings.gemini_api_key),
            "github": bool(settings.github_models_token),
            "cryptopanic": bool(settings.cryptopanic_key),
            "newsapi": bool(settings.newsapi_key),
            "neuprint": bool(settings.neuprint_token),
            "cave": bool(settings.cave_token),
        },
        "news_enabled": settings.news_enabled,
        "market_data_mode": settings.market_data_mode,
        "simulator_allowed": settings.market_allow_simulator,
        "signal_pipeline": settings.signal_pipeline,
        # Live feed name + whether it is simulated, so the dashboard can label
        # the numbers honestly instead of claiming a live tape in CI.
        "market_source": state.get_manager().market.active_source,
        "simulated": state.get_manager().market.active_source == "simulator",
        # Lets the dashboard show/hide the "Flutter app" link and the first-run
        # key banner without guessing.
        "flutter_web": (cfg.REPO_ROOT / "frontend" / "build" / "web").exists(),
        "env_file": (cfg.REPO_ROOT / ".env").exists(),
    }


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
    log.exception("unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": f"internal error: {exc}"})


# ---------------------------------------------------------------------------
# Dashboard (static, no build step).  Mounted last so /api/* always wins.
# ---------------------------------------------------------------------------

_web_dir = cfg.WEB_DIR
if _web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_web_dir)), name="static")

    @app.get("/")
    async def dashboard() -> FileResponse:
        return FileResponse(str(_web_dir / "index.html"))

    @app.get("/matrix")
    async def matrix_viewer() -> FileResponse:
        return FileResponse(str(_web_dir / "matrix.html"))

    @app.get("/settings")
    async def settings_page() -> FileResponse:
        return FileResponse(str(_web_dir / "settings.html"))

else:  # pragma: no cover - only if the web assets were stripped
    @app.get("/")
    async def dashboard_missing() -> dict:
        return {"detail": "Dashboard assets not found", "api": "/docs"}


# ---------------------------------------------------------------------------
# Flutter client (served from the same origin as the API, so there is one port,
# one forwarded URL and no CORS).  Resolved per request, which means
# ``bash frontend/run_web.sh`` takes effect without restarting the server.
# ---------------------------------------------------------------------------

FLUTTER_BUILD_DIR = cfg.REPO_ROOT / "frontend" / "build" / "web"


@app.get("/flutter")
@app.get("/flutter/")
async def flutter_index():
    if not FLUTTER_BUILD_DIR.exists():
        return _flutter_missing()
    return FileResponse(str(FLUTTER_BUILD_DIR / "index.html"))


@app.get("/flutter/{path:path}")
async def flutter_asset(path: str):
    if not FLUTTER_BUILD_DIR.exists():
        return _flutter_missing()
    candidate = (FLUTTER_BUILD_DIR / path).resolve()
    root = FLUTTER_BUILD_DIR.resolve()
    if not str(candidate).startswith(str(root)) or not candidate.is_file():
        # Flutter's asset manifest and canvaskit paths are absolute under the
        # /flutter/ base href, so anything unknown falls back to index.html.
        return FileResponse(str(root / "index.html"))
    return FileResponse(str(candidate))


def _flutter_missing() -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "detail": "The Flutter web build has not been created yet.",
            "build": "bash frontend/run_web.sh",
            "web_dashboard": "/",
            "install_flutter": "INSTALL_FLUTTER=1 bash .devcontainer/setup.sh",
        },
    )


def main() -> None:
    """``python -m backend.api.main``"""
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host=cfg.SETTINGS.host,
        port=cfg.SETTINGS.port,
        log_level=cfg.SETTINGS.log_level,
    )


if __name__ == "__main__":
    main()
