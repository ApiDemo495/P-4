"""Central configuration for DROSOPHILA TRADER v2.0.

Every tunable value in the specification lives here so that the rest of the
codebase never hard-codes a magic number.  Appendix C (PAXG parameter
overrides) is implemented as a per-asset override table rather than as inline
conditionals scattered through the formulas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
BRAIN_FALLBACK_DIR = BACKEND_DIR / "brain" / "fallback"
WEB_DIR = BACKEND_DIR / "web"
MODEL_STORE_DIR = Path(os.environ.get("MODEL_STORE_DIR", BACKEND_DIR / "models"))


def _load_dotenv() -> None:
    """Minimal .env loader (no hard dependency on python-dotenv)."""
    for candidate in (REPO_ROOT / ".env", BACKEND_DIR / ".env"):
        if not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key) or default)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(float(_env(key) or default))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = _env(key)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------
# Appendix C — per-asset parameter table
# --------------------------------------------------------------------------

#: Columns are exactly the rows of Appendix C in the specification.
ASSET_PARAMS: dict[str, dict[str, float]] = {
    "BTC": {
        "tai_ticks": 30,
        "afpr_power": 3.0,
        "vsd_baseline_window": 100,
        "vsd_recent_window": 20,
        "erc_tolerance_mult": 0.2,
        "dskd_q_fast": 1.0,
        "ccsv2_confidence_threshold": 0.55,
        "hedge_weight": 0.5,
        "forced_hold_ticks": 15,
    },
    "PAXG": {
        "tai_ticks": 20,
        "afpr_power": 2.0,
        "vsd_baseline_window": 50,
        "vsd_recent_window": 10,
        "erc_tolerance_mult": 0.3,
        "dskd_q_fast": 2.0,
        "ccsv2_confidence_threshold": 0.65,
        "hedge_weight": 0.5,
        "forced_hold_ticks": 10,
    },
}

ASSETS: tuple[str, ...] = ("BTC", "PAXG")
BINANCE_SYMBOLS: dict[str, str] = {"BTC": "btcusdt", "PAXG": "paxgusdt"}
COINGECKO_IDS: dict[str, str] = {"BTC": "bitcoin", "PAXG": "pax-gold"}


def asset_params(asset: str) -> dict[str, float]:
    return dict(ASSET_PARAMS.get(asset.upper(), ASSET_PARAMS["BTC"]))


# --------------------------------------------------------------------------
# Ring-buffer sizes (Section 5.1)
# --------------------------------------------------------------------------

TICK_BUFFER_SIZE = 600
L2_BUFFER_SNAPSHOTS = 2
L2_DEPTH_LEVELS = 20
CANDLE_BUFFER_SIZE = 60
NEWS_CACHE_SIZE = 20
OUTCOME_BUFFER_SIZE = 20
SYNC_WINDOW_SECONDS = 60


@dataclass(frozen=True)
class FormulaWeights:
    """Fusion weights for the 22nd formula's inputs (Section 8.1)."""

    formula_weight: float = 1.0


@dataclass
class Settings:
    """Runtime settings assembled from the environment."""

    # Server
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "info"))
    cors_origins: list[str] = field(
        default_factory=lambda: [o for o in _env("CORS_ORIGINS", "*").split(",") if o]
    )

    # Brain
    neuprint_token: str = field(default_factory=lambda: _env("NEUPRINT_APPLICATION_CREDENTIALS"))
    neuprint_server: str = field(
        default_factory=lambda: _env("NEUPRINT_SERVER", "https://neuprint.janelia.org")
    )
    neuprint_dataset: str = field(
        default_factory=lambda: _env("NEUPRINT_DATASET", "hemibrain:v1.2.1")
    )
    cave_token: str = field(default_factory=lambda: _env("CAVE_TOKEN"))
    cave_server: str = field(
        default_factory=lambda: _env("CAVE_SERVER", "https://global.daf-apis.com")
    )
    cave_dataset: str = field(
        default_factory=lambda: _env("CAVE_DATASET", "flywire_fafb_production")
    )
    brain_force_fallback: bool = field(
        default_factory=lambda: _env_bool("BRAIN_FORCE_FALLBACK", False)
    )
    brain_cache_ttl_seconds: int = 86_400
    brain_health_interval_seconds: int = 300

    # Agents
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-1.5-flash"))
    gemini_timeout_seconds: float = 7.0
    github_models_token: str = field(default_factory=lambda: _env("GITHUB_MODELS_TOKEN"))
    github_models_model: str = field(
        default_factory=lambda: _env("GITHUB_MODELS_MODEL", "gpt-4o-mini")
    )
    github_timeout_seconds: float = 7.0
    local_timeout_seconds: float = 7.0

    # Fusion weights (Section 8.1)
    weight_drosophila: float = 0.40
    weight_gemini: float = 0.25
    weight_local: float = 0.20
    weight_github: float = 0.15
    hsi_dampen_threshold: float = 0.80
    hsi_confidence_floor: float = 0.20
    min_fusion_confidence: float = 0.55

    # Decision thresholds (Section 10.1)
    signal_threshold: float = 0.25
    max_failed_formulas: int = 10  # >= 11 zeros => forced HOLD

    # News
    cryptopanic_key: str = field(default_factory=lambda: _env("CRYPTOPANIC_API_KEY"))
    newsapi_key: str = field(default_factory=lambda: _env("NEWSAPI_API_KEY"))
    news_enabled: bool = field(default_factory=lambda: _env_bool("NEWS_ENABLED", True))
    news_poll_seconds: float = field(
        default_factory=lambda: _env_float("NEWS_POLL_SECONDS", 30.0)
    )
    newsapi_poll_seconds: float = field(
        default_factory=lambda: _env_float("NEWSAPI_POLL_SECONDS", 60.0)
    )
    rss_poll_seconds: float = field(default_factory=lambda: _env_float("RSS_POLL_SECONDS", 60.0))
    rss_feeds: list[str] = field(
        default_factory=lambda: [
            f
            for f in _env(
                "RSS_FEEDS",
                "https://www.coindesk.com/arc/outboundfeeds/rss/,"
                "https://cointelegraph.com/rss,"
                "https://www.theblock.co/rss.xml",
            ).split(",")
            if f
        ]
    )
    news_decay_seconds: float = 300.0
    emergency_duration_seconds: float = 180.0
    flash_move_pct: float = 5.0
    flash_move_window_seconds: float = 30.0

    # Market data
    market_data_mode: str = field(default_factory=lambda: _env("MARKET_DATA_MODE", "auto").lower())
    market_allow_simulator: bool = field(
        default_factory=lambda: _env_bool("MARKET_ALLOW_SIMULATOR", True)
    )
    simulator_seed: int = field(default_factory=lambda: _env_int("SIMULATOR_SEED", 1337))
    binance_ws_base: str = field(
        default_factory=lambda: _env("BINANCE_WS_BASE", "wss://stream.binance.com:9443")
    )
    binance_rest_base: str = field(
        default_factory=lambda: _env("BINANCE_REST_BASE", "https://api.binance.com")
    )
    coingecko_base: str = field(
        default_factory=lambda: _env("COINGECKO_BASE", "https://api.coingecko.com/api/v3")
    )
    stale_tick_seconds: float = 30.0
    reconnect_max_seconds: float = 60.0

    # Persistence
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))

    # Cycle timing
    time_scale: float = field(default_factory=lambda: _env_float("TIME_SCALE", 1.0))
    lock_deadline_seconds: float = field(
        default_factory=lambda: _env_float("LOCK_DEADLINE_SECONDS", 8.0)
    )
    formula_refresh_seconds: float = field(
        default_factory=lambda: _env_float("FORMULA_REFRESH_SECONDS", 15.0)
    )
    outcome_horizon_seconds: float = 60.0

    @property
    def cycle_period_seconds(self) -> float:
        """Wall-clock length of one signal cycle.

        ``TIME_SCALE == 1`` gives the true 60-second world-clock cycle that is
        phase-locked to the UTC minute (Section 9).  Larger values compress the
        cycle for demos and automated tests.
        """
        return 60.0 / max(self.time_scale, 0.01)

    @property
    def use_world_clock(self) -> bool:
        return abs(self.time_scale - 1.0) < 1e-9

    def scaled(self, seconds: float) -> float:
        """Convert a spec (virtual, 60s-cycle) duration to wall-clock seconds."""
        return seconds / max(self.time_scale, 0.01)


SETTINGS = Settings()


def reload_settings() -> Settings:
    """Re-read the environment (used by the settings API at runtime)."""
    global SETTINGS
    SETTINGS = Settings()
    return SETTINGS
