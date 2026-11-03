"""Round AL - keyed macro providers for the Formula Genesis Engine.

Three optional feeds, each activated **only** when its key is present (the
Settings page has one box per provider; nothing here runs without a key):

* Glassnode   exchange net-flow of BTC (on-chain)      -> ``exch_flow``
* Twelve Data US-dollar index (DXY) 1-minute closes    -> ``dxy`` / ``dxy_ret``
* LunarCrush  social sentiment / galaxy score hourly   -> ``social``

Every provider exposes ``fetch(key) -> np.ndarray`` of ``(ts_ms, value)``
rows (oldest first) and ``test_key(key) -> {"valid", "detail"|"error"}`` for
the Settings page.  ``MacroFeed`` polls the configured providers in the
background and hands the genesis engine aligned columns through
``series()``; a provider that fails keeps its last good series and reports
the error in ``status()``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

import httpx
import numpy as np

log = logging.getLogger("drosophila.keyed")

GLASSNODE_URL = "https://api.glassnode.com/v1/metrics"
TWELVEDATA_URL = "https://api.twelvedata.com/time_series"
LUNARCRUSH_URL = "https://lunarcrush.com/api4/public/coins/{coin}/time-series/v2"
POLL_S = {"glassnode": 600, "twelvedata": 60, "lunarcrush": 300}
TIMEOUT = 12.0


# ------------------------------------------------------------------ Glassnode
async def glassnode_fetch(key: str, asset: str = "BTC", metric: str = "transactions/transfers_volume_exchanges_net",
                          interval: str = "1h") -> np.ndarray:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.get(f"{GLASSNODE_URL}/{metric}", params={"a": asset, "i": interval, "api_key": key})
        r.raise_for_status()
        rows = [(float(p["t"]) * 1000.0, float(p["v"])) for p in (r.json() or []) if p.get("v") is not None]
    return np.array(sorted(rows), dtype=np.float64) if rows else np.zeros((0, 2))


async def glassnode_test(key: str) -> dict:
    if not key:
        return {"valid": False, "error": "No key entered"}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(f"{GLASSNODE_URL}/market/price_usd_close",
                                 params={"a": "BTC", "i": "24h", "api_key": key,
                                         "s": int(time.time()) - 3 * 86400})
        if r.status_code == 200:
            return {"valid": True, "detail": f"{len(r.json() or [])} daily points returned"}
        if r.status_code in (401, 403):
            return {"valid": False, "error": "Glassnode rejected the key"}
        if r.status_code == 429:
            return {"valid": True, "detail": "rate limited, but the key was accepted"}
        return {"valid": False, "error": f"HTTP {r.status_code}: {r.text[:120]}"}
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "error": str(exc)}


# ---------------------------------------------------------------- Twelve Data
def _td_rows(payload: dict) -> list[tuple[float, float]]:
    rows = []
    for v in payload.get("values") or []:
        try:
            dt = datetime.strptime(v["datetime"], "%Y-%m-%d %H:%M:%S" if len(v["datetime"]) > 10 else "%Y-%m-%d")
            rows.append((dt.replace(tzinfo=timezone.utc).timestamp() * 1000.0, float(v["close"])))
        except (KeyError, ValueError):
            continue
    return sorted(rows)


async def twelvedata_fetch(key: str, symbol: str = "DXY", interval: str = "1min", outputsize: int = 500) -> np.ndarray:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for sym in (symbol, "EUR/USD"):
            r = await client.get(TWELVEDATA_URL, params={"symbol": sym, "interval": interval,
                                                         "outputsize": outputsize, "apikey": key, "timezone": "UTC"})
            data = r.json() if r.content else {}
            if r.status_code == 200 and data.get("status") != "error":
                rows = _td_rows(data)
                if sym == "EUR/USD":      # dollar index proxy: invert EUR/USD
                    rows = [(t, 100.0 / v) for t, v in rows if v > 0]
                return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 2))
            if int(data.get("code") or r.status_code) in (401, 403):
                raise PermissionError(data.get("message") or "Twelve Data rejected the key")
            if int(data.get("code") or 0) == 429:
                raise RuntimeError("Twelve Data rate limit")
    raise RuntimeError("Twelve Data returned no usable series")


async def twelvedata_test(key: str) -> dict:
    if not key:
        return {"valid": False, "error": "No key entered"}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(TWELVEDATA_URL, params={"symbol": "EUR/USD", "interval": "1min",
                                                         "outputsize": 2, "apikey": key})
        data = r.json() if r.content else {}
        if r.status_code == 200 and data.get("status") == "ok":
            return {"valid": True, "detail": f"EUR/USD 1-minute series reachable ({len(data.get('values') or [])} rows)"}
        code = int(data.get("code") or r.status_code)
        if code in (401, 403):
            return {"valid": False, "error": data.get("message") or "Twelve Data rejected the key"}
        if code == 429:
            return {"valid": True, "detail": "rate limited, but the key was accepted"}
        return {"valid": False, "error": data.get("message") or f"HTTP {r.status_code}"}
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "error": str(exc)}


# ----------------------------------------------------------------- LunarCrush
async def lunarcrush_fetch(key: str, coin: str = "BTC", field: str = "sentiment") -> np.ndarray:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.get(LUNARCRUSH_URL.format(coin=coin), params={"bucket": "hour", "interval": "1w"},
                             headers={"Authorization": f"Bearer {key}"})
        r.raise_for_status()
        rows = []
        for p in (r.json() or {}).get("data") or []:
            v = p.get(field, p.get("galaxy_score"))
            if v is not None and p.get("time") is not None:
                rows.append((float(p["time"]) * 1000.0, float(v)))
    return np.array(sorted(rows), dtype=np.float64) if rows else np.zeros((0, 2))


async def lunarcrush_test(key: str) -> dict:
    if not key:
        return {"valid": False, "error": "No key entered"}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(LUNARCRUSH_URL.format(coin="BTC"), params={"bucket": "hour", "interval": "1d"},
                                 headers={"Authorization": f"Bearer {key}"})
        if r.status_code == 200:
            n = len((r.json() or {}).get("data") or [])
            return {"valid": True, "detail": f"{n} hourly social points returned"}
        if r.status_code in (401, 403):
            return {"valid": False, "error": "LunarCrush rejected the key"}
        if r.status_code == 429:
            return {"valid": True, "detail": "rate limited, but the key was accepted"}
        return {"valid": False, "error": f"HTTP {r.status_code}: {r.text[:120]}"}
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "error": str(exc)}


PROVIDERS = {
    "glassnode": {"fetch": glassnode_fetch, "test": glassnode_test, "column": "exch_flow",
                  "label": "Glassnode exchange net-flow (BTC, hourly)"},
    "twelvedata": {"fetch": twelvedata_fetch, "test": twelvedata_test, "column": "dxy",
                   "label": "Twelve Data dollar index (1-minute)"},
    "lunarcrush": {"fetch": lunarcrush_fetch, "test": lunarcrush_test, "column": "social",
                   "label": "LunarCrush social sentiment (hourly)"},
}


async def test_key(provider: str, key: str) -> dict:
    spec = PROVIDERS.get(provider)
    if spec is None:
        return {"valid": False, "error": f"unknown provider {provider!r}"}
    return await spec["test"](key)


class MacroFeed:
    """Background poller for whichever keyed providers have keys."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.series_by_col: dict[str, np.ndarray] = {}
        self.state: dict[str, dict] = {name: {"configured": False, "rows": 0, "last_ok": 0.0, "error": ""}
                                       for name in PROVIDERS}
        self._task: asyncio.Task | None = None

    def key_for(self, provider: str) -> str:
        return str(getattr(self.settings, f"{provider}_key", "") or "")

    def configured(self) -> list[str]:
        return [p for p in PROVIDERS if self.key_for(p)]

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="macro-feed")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def poll_once(self) -> None:
        now = time.time()
        for name, spec in PROVIDERS.items():
            key = self.key_for(name)
            st = self.state[name]
            st["configured"] = bool(key)
            if not key or now - st["last_ok"] < POLL_S[name] and not st["error"]:
                continue
            try:
                rows = await spec["fetch"](key)
                self.series_by_col[spec["column"]] = rows
                st.update(rows=int(len(rows)), last_ok=now, error="")
            except Exception as exc:  # noqa: BLE001
                st["error"] = f"{type(exc).__name__}: {exc}"[:160]
                log.debug("macro provider %s failed: %s", name, exc)
                st["last_ok"] = now - POLL_S[name] + 60   # retry in a minute

    async def _loop(self) -> None:
        while True:
            try:
                await self.poll_once()
            except Exception as exc:  # noqa: BLE001
                log.debug("macro feed loop: %s", exc)
            await asyncio.sleep(30)

    def series(self) -> dict[str, np.ndarray]:
        """Column name -> (ts_ms, value) rows for ``build_frame(macro=...)``."""
        return {k: v for k, v in self.series_by_col.items() if len(v)}

    def status(self) -> dict:
        return {name: {**self.state[name], "label": spec["label"], "column": spec["column"]}
                for name, spec in PROVIDERS.items()}
