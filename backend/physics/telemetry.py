"""Live telemetry for the thermodynamic layer - public APIs, no keys.

* gold spot (XAU/USD)   gold-api.com  -> CoinGecko (pax-gold / tether-gold)
* other venues          Coinbase, Kraken public tickers (BTC, PAXG)
* wrapped BTC           CoinGecko wrapped-bitcoin vs bitcoin

A background loop refreshes every source on its own cadence; the engine reads
the cache synchronously and never blocks on the network.  Each reading says
whether it is ``live`` (fetched, with its age) or ``model`` (fallback), and
the status endpoint lists every source with its last error, so a blocked
network reads as "blocked here", never as "the API is unavailable".
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx


log = logging.getLogger("drosophila.physics.telemetry")

TIMEOUT = 6.0


@dataclass
class Reading:
    value: object = None
    source: str = "model"          # live | model
    provider: str = ""
    fetched_at: float = 0.0
    error: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def age_seconds(self) -> float | None:
        return round(time.time() - self.fetched_at, 1) if self.fetched_at else None

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "source": self.source,
            "provider": self.provider,
            "age_seconds": self.age_seconds,
            "error": self.error,
            "detail": self.detail,
        }


class Telemetry:
    """Cached readings + the refresh loop."""

    CADENCE = {"xau": 60.0, "venues": 20.0, "wbtc": 120.0}

    def __init__(self) -> None:
        self.xau = Reading(value=None, provider="none")
        self.venues = Reading(value={}, provider="none")
        self.wbtc = Reading(value=None, provider="none")
        self._last_try: dict[str, float] = {}
        self.enabled = True

    # ------------------------------------------------------------ status
    def status(self) -> dict:
        return {
            "xau_usd": self.xau.to_dict(),
            "venues": {**self.venues.to_dict(), "value": sorted((self.venues.value or {}).keys())},
            "wbtc_usd": self.wbtc.to_dict(),
            "live_sources": sum(1 for r in (self.xau, self.venues, self.wbtc) if r.source == "live"),
        }

    # ------------------------------------------------------------- loop
    async def refresh_loop(self) -> None:
        await asyncio.sleep(2.0)
        while True:
            try:
                await self.refresh_once()
            except Exception as exc:  # noqa: BLE001
                log.debug("telemetry refresh failed: %s", exc)
            await asyncio.sleep(10.0)

    async def refresh_once(self, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        due = [k for k, c in self.CADENCE.items() if force or now - self._last_try.get(k, 0.0) >= c]
        if not due:
            return
        fetchers = {"xau": self._fetch_xau, "venues": self._fetch_venues, "wbtc": self._fetch_wbtc}
        async with httpx.AsyncClient(timeout=TIMEOUT, headers={"User-Agent": "drosophila-trader/2.0"}) as client:
            tasks = {k: fetchers[k](client) for k in due}
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for key, result in zip(tasks, results):
            self._last_try[key] = now
            if isinstance(result, Exception):
                reading: Reading = getattr(self, key)
                reading.error = f"{type(result).__name__}: {result}"[:160]

    # ---------------------------------------------------------- fetchers
    async def _fetch_xau(self, client: httpx.AsyncClient) -> None:
        try:
            r = await client.get("https://api.gold-api.com/price/XAU")
            r.raise_for_status()
            price = float(r.json()["price"])
            self.xau = Reading(price, "live", "gold-api.com", time.time())
            return
        except Exception as exc:  # noqa: BLE001
            first = f"gold-api.com: {type(exc).__name__}"
        r = await client.get("https://api.coingecko.com/api/v3/simple/price",
                             params={"ids": "pax-gold,tether-gold", "vs_currencies": "usd"})
        r.raise_for_status()
        data = r.json()
        # Two independent tokenised-gold prices: their mean is the best keyless
        # proxy for spot; the PAXG peg is then measured against it.
        prices = [float(v["usd"]) for v in data.values() if "usd" in v]
        if not prices:
            raise ValueError("coingecko returned no gold prices")
        self.xau = Reading(sum(prices) / len(prices), "live", "coingecko (PAXG+XAUT mean)", time.time(),
                           error=first, detail={"paxg_xaut": prices})

    async def _fetch_venues(self, client: httpx.AsyncClient) -> None:
        venues: dict[str, dict] = {}
        errors = []
        try:
            r = await client.get("https://api.exchange.coinbase.com/products/BTC-USD/ticker")
            r.raise_for_status()
            d = r.json()
            venues["coinbase"] = {"BTC": {"bid": float(d["bid"]), "ask": float(d["ask"])}}
        except Exception as exc:  # noqa: BLE001
            errors.append(f"coinbase: {type(exc).__name__}")
        try:
            r = await client.get("https://api.kraken.com/0/public/Ticker", params={"pair": "XBTUSD,PAXGUSD"})
            r.raise_for_status()
            result = r.json().get("result") or {}
            kraken: dict[str, dict] = {}
            for key, row in result.items():
                asset = "BTC" if "XBT" in key else "PAXG" if "PAXG" in key else None
                if asset:
                    kraken[asset] = {"bid": float(row["b"][0]), "ask": float(row["a"][0])}
            if kraken:
                venues["kraken"] = kraken
        except Exception as exc:  # noqa: BLE001
            errors.append(f"kraken: {type(exc).__name__}")
        if not venues:
            raise RuntimeError("; ".join(errors) or "no venue answered")
        self.venues = Reading(venues, "live", "+".join(sorted(venues)), time.time(), error="; ".join(errors))

    async def _fetch_wbtc(self, client: httpx.AsyncClient) -> None:
        r = await client.get("https://api.coingecko.com/api/v3/simple/price",
                             params={"ids": "wrapped-bitcoin,bitcoin", "vs_currencies": "usd"})
        r.raise_for_status()
        data = r.json()
        wbtc = float(data["wrapped-bitcoin"]["usd"])
        btc = float(data["bitcoin"]["usd"])
        self.wbtc = Reading(wbtc, "live", "coingecko", time.time(), detail={"btc_usd": btc})


_TELEMETRY: Telemetry | None = None


def get_telemetry() -> Telemetry:
    global _TELEMETRY
    if _TELEMETRY is None:
        _TELEMETRY = Telemetry()
    return _TELEMETRY
