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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api import (
    routes_agents,
    routes_brain,
    routes_emotions,
    routes_genesis,
    routes_physics,
    routes_formulas,
    routes_news,
    routes_signals,
    state,
)
from backend.api import self_update
from backend.core import config as cfg
from backend.core.cycle_manager import CycleManager

logging.basicConfig(
    level=getattr(logging, cfg.SETTINGS.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
)
log = logging.getLogger("drosophila.main")

_START_TIME = time.time()


def _build_id() -> str:
    """Short git hash of the running checkout ("unknown" outside git).

    Printed in the dashboard's Engine line and in /api/health so "is my
    Codespace on the new code?" has a one-glance answer.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(cfg.REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


BUILD_ID = _build_id()


def _local_stub_setting() -> bool | None:
    raw = os.environ.get("LOCAL_AGENT_STUB", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off", ""):
        return None
    return None


async def _warm_up(manager: CycleManager) -> None:
    """Bring the engine up and log the ready block (runs *after* the port opens)."""
    await manager.warm_up()
    if manager.start_error:
        log.error("=" * 78)
        log.error(" WARM-UP FAILED - the API and the dashboard are still serving")
        log.error("   reason: %s", manager.start_error)
        log.error("   the market/news panels fall back to degraded modes; see /api/health")
        log.error("=" * 78)
        return
    log.info("=" * 78)
    log.info(" DROSOPHILA TRADER v2.0 ready")
    log.info("   brain      : %s", manager.brain.status.value)
    log.info("   market data: %s", manager.market.active_source)
    log.info("   news       : %s", manager.news.status.coverage)
    log.info("   cycle      : %.1fs (world clock: %s)", cfg.SETTINGS.cycle_period_seconds, cfg.SETTINGS.use_world_clock)
    log.info("   dashboard  : http://0.0.0.0:%d/", cfg.SETTINGS.port)
    log.info("=" * 78)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the port FIRST, warm the engine up in the background.

    Why: uvicorn only accepts connections after the lifespan startup hook
    returns.  Brain verification and the market connect can take seconds (or
    hang on an unreachable host) - and a forwarded Codespaces port that nobody
    answers returns **502**, which looks like a broken app.  Warming up in a
    task means the first request always gets a page that says what is going on.
    """
    manager = CycleManager(cfg.SETTINGS, local_stub=_local_stub_setting())
    state.set_manager(manager)
    app.state.manager = manager
    manager.mark_started()

    warm_task = asyncio.create_task(_warm_up(manager), name="warm-up")
    # Self-update: in a Codespace the checkout fast-forwards its own branch on
    # its own (never a checkout, never main); AUTO_UPDATE=0 turns it off.
    update_task = asyncio.create_task(self_update.auto_loop(), name="auto-update") \
        if self_update.auto_enabled() else None
    # Round T: gold spot, other venues and wBTC for the
    # thermodynamic layer - public APIs, no keys, cached between cycles.
    from backend.physics.telemetry import get_telemetry

    telemetry_task = asyncio.create_task(get_telemetry().refresh_loop(), name="physics-telemetry")
    log.info("=" * 78)
    log.info(" DROSOPHILA TRADER v2.0 - port is live, engine warming up in the background")
    log.info("   dashboard  : http://0.0.0.0:%d/   (reload if it says 'warming up')", cfg.SETTINGS.port)
    log.info("   health     : /api/health  ->  ready / warming_up / start_error")
    log.info("=" * 78)
    try:
        yield
    finally:
        telemetry_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await telemetry_task
        if update_task is not None and not update_task.done():
            update_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await update_task
        if not warm_task.done():
            warm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await warm_task
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
app.include_router(routes_emotions.router)
app.include_router(routes_physics.router)
app.include_router(routes_genesis.router)


@app.get("/api/feeds/diagnose")
async def feeds_diagnose() -> dict:
    """Round AE - live network diagnosis (HTTP, WebSocket handshakes, RSS) plus
    the hub's own view (active source, per-feed delivery ages, rejected
    writes).  One ``curl`` answers "why does &b say offline here?"."""
    from backend.data import connectivity as _conn

    manager = state.manager_or_none()
    report = await _conn.diagnose(manager.settings if manager else None)
    if manager is not None:
        hub = manager.market.feeds_report()
        report["hub"] = {
            "active_source": manager.market.active_source,
            "btc_ticks": manager.market.tick_count("BTC"),
            "last_delivery_seconds_ago": hub.get("last_delivery_seconds_ago"),
            "rejected_writes": hub.get("rejected_writes"),
            "feeds": hub.get("feeds"),
            "boot_connectivity": (manager.market.connectivity or {}).get("summary"),
        }
        news = manager.news
        report["news"] = {"items": news.status.items, "niv": round(news.current_niv(), 4),
                          "providers": dict(news.cache.providers), "coverage": news.status.coverage}
    return report


@app.get("/api/health")
async def health() -> dict:
    manager = state.manager_or_none()
    if manager is None:
        return {"healthy": False, "detail": "starting"}
    payload = manager.health()
    payload["uptime_seconds"] = round(time.time() - _START_TIME, 1)
    payload["version"] = "2.0.0"
    payload["build"] = BUILD_ID
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
        "build": BUILD_ID,
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
            "physics": settings.weight_physics,
            "formulas": settings.weight_formulas,
            "genesis": settings.weight_genesis,
        },
        "configured": {
            "gemini": bool(settings.gemini_api_key),
            "github": bool(settings.github_models_token),
            "cryptopanic": settings.rings["cryptopanic"].configured,
            "newsapi": settings.rings["newsapi"].configured,
            "neuprint": bool(settings.neuprint_token),
            "cave": bool(settings.cave_token),
            "glassnode": bool(settings.glassnode_key),
            "twelvedata": bool(settings.twelvedata_key),
            "lunarcrush": bool(settings.lunarcrush_key),
        },
        "news_enabled": settings.news_enabled,
        "market_data_mode": settings.market_data_mode,
        "simulator_allowed": settings.market_allow_simulator,
        "signal_pipeline": settings.signal_pipeline,
        # Live feed name + whether it is simulated, so the dashboard can label
        # the numbers honestly instead of claiming a live tape in CI.
        "market_source": state.get_manager().market.active_source,
        "simulated": state.get_manager().market.active_source == "simulator",
        # Lets the dashboard show the first-run key banner without guessing.
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

    @app.get("/readme")
    @app.get("/readme/{name}")
    async def readme_page(name: str = "README") -> HTMLResponse:
        """README and docs rendered in the app (Round S) - no editor
        markdown preview needed."""
        from backend.api import docs_pages

        html, status = docs_pages.render(name)
        return HTMLResponse(html, status_code=status)

else:  # pragma: no cover - only if the web assets were stripped
    @app.get("/")
    async def dashboard_missing() -> dict:
        return {"detail": "Dashboard assets not found", "api": "/docs"}


@app.get("/api/update/status")
async def update_status(force: int = 0):
    """Is this checkout behind its own branch on origin?"""
    result = await self_update.check(force=bool(force))
    result["log_tail"] = self_update.log_tail()
    return result


@app.post("/api/update/apply")
async def update_apply():
    """Fetch + fast-forward + restart, detached.  Never a checkout or a merge."""
    return self_update.apply()


@app.get("/api/system/autostart")
async def autostart_status(lines: int = 60):
    """What the zero-command Codespace start did: provisioning passes, pip
    download, self-update - with the tail of every log, so a
    failed self-start or self-download is visible in the app, not just in a
    terminal banner (Round S)."""
    from backend.api import autostart_status as mod

    return mod.status(lines=max(5, min(int(lines), 400)))


@app.get("/api/system/health")
async def system_health():
    """Round AP: every subsystem - tape, emotions, physics, formulas, genesis,
    news, brain, websocket, event loop - running / waiting / failing, with
    the last error text.  The SYSTEM strip on the dashboard renders this."""
    from backend.api.state import get_manager

    return get_manager().health_payload()


def main() -> None:
    """``python -m backend.api.main``"""
    import uvicorn

    host = cfg.SETTINGS.host
    # A forwarded Codespaces port only reaches a socket bound to 0.0.0.0.  If
    # HOST was left as a loopback address in .env, every browser request would
    # get a 502 while curl inside the container kept working - so override it
    # loudly instead of failing mysteriously.
    if os.environ.get("CODESPACE_NAME") and host in ("127.0.0.1", "localhost", "::1"):
        log.warning("HOST=%s inside a Codespace cannot be reached from the browser - using 0.0.0.0", host)
        host = "0.0.0.0"

    uvicorn.run(
        "backend.api.main:app",
        host=host,
        port=cfg.SETTINGS.port,
        log_level=cfg.SETTINGS.log_level,
    )


if __name__ == "__main__":
    main()
