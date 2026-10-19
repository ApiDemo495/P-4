"""CoinGecko REST fallback (degradation level 5).

When Binance is down for more than 10 consecutive retries the app switches to
CoinGecko's public ``/simple/price`` endpoint.  Tick rate drops from ~10/s to
once every few seconds, and there is no order book at all, so DGW / LCS / BAR
are forced to 0 (Section 13.2, level 5) and the remaining 19 formulas run -
with interpolation for the ones that need more than 60 ticks.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from backend.core import config as cfg

log = logging.getLogger("drosophila.coingecko")

POLL_SECONDS = 3.0
RETRIES = 3
RETRY_GAP = 2.0


class CoinGeckoFeed:
    """Polls CoinGecko for BTC/PAXG spot prices and synthesises ticks."""

    def __init__(self, on_data, settings=None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.on_data = on_data
        self.connected = False
        self.last_price: dict[str, float] = {}
        self.last_error: str = ""
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def _fetch(self, client: httpx.AsyncClient) -> dict[str, float]:
        ids = ",".join(cfg.COINGECKO_IDS.values())
        url = f"{self.settings.coingecko_base}/simple/price"
        params = {"ids": ids, "vs_currencies": "usd", "include_last_updated_at": "true"}
        last_exc: Exception | None = None
        for attempt in range(RETRIES):
            try:
                resp = await client.get(url, params=params, timeout=6.0)
                resp.raise_for_status()
                payload = resp.json()
                out: dict[str, float] = {}
                for asset, cid in cfg.COINGECKO_IDS.items():
                    entry = payload.get(cid) or {}
                    price = entry.get("usd")
                    if price:
                        out[asset] = float(price)
                if out:
                    return out
                raise ValueError("empty price payload")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < RETRIES - 1:
                    await asyncio.sleep(RETRY_GAP)
        raise RuntimeError(f"CoinGecko failed after {RETRIES} attempts: {last_exc}")

    async def run(self) -> None:
        async with httpx.AsyncClient(headers={"accept": "application/json"}) as client:
            while not self._stop.is_set():
                try:
                    prices = await self._fetch(client)
                    now_ms = time.time() * 1000.0
                    for asset, price in prices.items():
                        prev = self.last_price.get(asset, price)
                        side = 1.0 if price >= prev else -1.0
                        qty = abs(price - prev) / max(price, 1e-9) * 1000.0 + 1.0
                        await self.on_data(asset, [(now_ms, price, qty, side)], None)
                        self.last_price[asset] = price
                    self.connected = True
                    self.last_error = ""
                except Exception as exc:  # noqa: BLE001
                    self.connected = False
                    self.last_error = str(exc)
                    log.warning("CoinGecko poll failed: %s", exc)
                await asyncio.sleep(POLL_SECONDS)
