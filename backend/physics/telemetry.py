"""Live telemetry for the thermodynamic layer - public APIs, no keys.

* global hashrate       mempool.space -> blockchain.info
* gold spot (XAU/USD)   gold-api.com  -> CoinGecko (pax-gold / tether-gold)
* DEX pools             DexScreener (PAXG and WBTC pairs, price + liquidity)
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

from backend.physics import constants as K

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

    CADENCE = {"hashrate": 300.0, "xau": 60.0, "pools": 60.0, "venues": 20.0, "wbtc": 120.0}

    def __init__(self) -> None:
        self.hashrate = Reading(value=K.MODEL_HASHRATE_H_PER_S, provider="model constant",
                                detail={"series": []})
        self.xau = Reading(value=None, provider="none")
        self.pools = Reading(value=[], provider="none")
        self.venues = Reading(value={}, provider="none")
        self.wbtc = Reading(value=None, provider="none")
        self._last_try: dict[str, float] = {}
        self.enabled = True

    # ------------------------------------------------------------ status
    def status(self) -> dict:
        return {
            "hashrate": self.hashrate.to_dict(),
            "xau_usd": self.xau.to_dict(),
            "dex_pools": {**self.pools.to_dict(), "value": len(self.pools.value or [])},
            "venues": {**self.venues.to_dict(), "value": sorted((self.venues.value or {}).keys())},
            "wbtc_usd": self.wbtc.to_dict(),
            "live_sources": sum(1 for r in (self.hashrate, self.xau, self.pools, self.venues, self.wbtc)
                                if r.source == "live"),
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
        fetchers = {"hashrate": self._fetch_hashrate, "xau": self._fetch_xau, "pools": self._fetch_pools,
                    "venues": self._fetch_venues, "wbtc": self._fetch_wbtc}
        async with httpx.AsyncClient(timeout=TIMEOUT, headers={"User-Agent": "drosophila-trader/2.0"}) as client:
            tasks = {k: fetchers[k](client) for k in due}
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for key, result in zip(tasks, results):
            self._last_try[key] = now
            if isinstance(result, Exception):
                reading: Reading = getattr(self, key)
                reading.error = f"{type(result).__name__}: {result}"[:160]

    # ---------------------------------------------------------- fetchers
    async def _fetch_hashrate(self, client: httpx.AsyncClient) -> None:
        try:
            r = await client.get("https://mempool.space/api/v1/mining/hashrate/1m")
            r.raise_for_status()
            data = r.json()
            current = float(data["currentHashrate"])
            series = [float(p["avgHashrate"]) for p in data.get("hashrates", [])][-30:]
            self.hashrate = Reading(current, "live", "mempool.space", time.time(),
                                    detail={"series": series, "difficulty": data.get("currentDifficulty")})
            return
        except Exception as exc:  # noqa: BLE001
            first = f"mempool.space: {type(exc).__name__}"
        r = await client.get("https://blockchain.info/q/hashrate")
        r.raise_for_status()
        current = float(r.text.strip()) * 1e9   # GH/s -> H/s
        self.hashrate = Reading(current, "live", "blockchain.info", time.time(),
                                error=first, detail={"series": self.hashrate.detail.get("series", [])})

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

    async def _fetch_pools(self, client: httpx.AsyncClient) -> None:
        pools: list[dict] = []
        for query in ("PAXG", "WBTC"):
            r = await client.get("https://api.dexscreener.com/latest/dex/search", params={"q": query})
            r.raise_for_status()
            for pair in (r.json().get("pairs") or [])[:40]:
                try:
                    liquidity = float((pair.get("liquidity") or {}).get("usd") or 0.0)
                    price = float(pair.get("priceUsd") or 0.0)
                except (TypeError, ValueError):
                    continue
                base = (pair.get("baseToken") or {}).get("symbol", "")
                quote = (pair.get("quoteToken") or {}).get("symbol", "")
                if liquidity < 50_000 or price <= 0 or base.upper() not in ("PAXG", "WBTC", "CBBTC", "TBTC"):
                    continue
                pools.append({
                    "dex": pair.get("dexId"), "chain": pair.get("chainId"),
                    "base": base.upper(), "quote": quote.upper(),
                    "price_usd": price, "liquidity_usd": liquidity,
                    "fee": 0.003 if "v2" in str(pair.get("labels") or "") else 0.0005,
                })
        self.pools = Reading(pools[:40], "live", "dexscreener", time.time())

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
