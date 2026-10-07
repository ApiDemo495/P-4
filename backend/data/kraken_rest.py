"""Kraken REST polling - a real tape over plain HTTPS (Round AA).

When neither WebSocket feed can be reached (corporate egress, blocked ports,
a WebSocket-hostile proxy) this feed still delivers **real trades and a real
order book** with nothing but ``GET`` on port 443:

    https://api.kraken.com/0/public/Trades?pair=XBTUSD&since=<ns>
    https://api.kraken.com/0/public/Depth?pair=XBTUSD&count=25

Trades are fetched incrementally (``since`` = the ``last`` cursor Kraken
returns), so every public print reaches the tape exactly once; the book is
polled every cycle.  Poll cadence 2 s (Kraken's public rate limit is ~1 req/s
per endpoint).  Emits the same ``(time_ms, price, qty, side)`` ticks and
``(2, L2_DEPTH_LEVELS, 2)`` books as the WebSocket feeds.
"""
from __future__ import annotations

import asyncio
import logging

import httpx
import numpy as np

from backend.core import config as cfg

log = logging.getLogger("drosophila.kraken.rest")

PAIRS: dict[str, str] = {"BTC": "XBTUSD", "PAXG": "PAXGUSD"}
BASE = "https://api.kraken.com/0/public"
POLL_SECONDS = 2.0


class KrakenRest:
    def __init__(self, on_data, settings=None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.on_data = on_data
        self.connected = False
        self.last_error = ""
        self.polls = 0
        self._since: dict[str, str | None] = {a: None for a in PAIRS}
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        async with httpx.AsyncClient(headers={"accept": "application/json"}, timeout=6.0) as client:
            while not self._stop.is_set():
                ok = False
                for asset, pair in PAIRS.items():
                    try:
                        ticks = await self._trades(client, asset, pair)
                        book = await self._depth(client, pair)
                        if ticks:
                            await self.on_data(asset, ticks, None)
                        if book is not None:
                            await self.on_data(asset, [], book)
                        ok = True
                    except Exception as exc:  # noqa: BLE001
                        self.last_error = f"{type(exc).__name__}: {str(exc)[:120]}"
                        log.debug("Kraken REST %s failed: %s", asset, exc)
                self.connected = ok
                if ok:
                    self.last_error = ""
                self.polls += 1
                await asyncio.sleep(POLL_SECONDS)

    async def _trades(self, client: httpx.AsyncClient, asset: str, pair: str) -> list:
        params = {"pair": pair}
        if self._since[asset]:
            params["since"] = self._since[asset]
        resp = await client.get(f"{BASE}/Trades", params=params)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("error"):
            raise RuntimeError(";".join(payload["error"]))
        result = payload.get("result") or {}
        rows = next((v for k, v in result.items() if k != "last"), [])
        first_poll = self._since[asset] is None
        self._since[asset] = str(result.get("last") or self._since[asset] or "")
        ticks = []
        for row in rows[-200:] if first_poll else rows:
            try:
                price, qty, ts, side = float(row[0]), float(row[1]), float(row[2]), str(row[3])
            except (TypeError, ValueError, IndexError):
                continue
            if price > 0 and qty > 0:
                ticks.append((ts * 1000.0, price, qty, 1.0 if side == "b" else -1.0))
        return ticks

    async def _depth(self, client: httpx.AsyncClient, pair: str) -> np.ndarray | None:
        resp = await client.get(f"{BASE}/Depth", params={"pair": pair, "count": 25})
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("error"):
            raise RuntimeError(";".join(payload["error"]))
        result = next(iter((payload.get("result") or {}).values()), None)
        if not result:
            return None
        book = np.zeros((2, cfg.L2_DEPTH_LEVELS, 2), dtype=np.float64)
        for side_idx, key in enumerate(("bids", "asks")):
            for level, entry in enumerate((result.get(key) or [])[: cfg.L2_DEPTH_LEVELS]):
                try:
                    book[side_idx, level, 0] = float(entry[0])
                    book[side_idx, level, 1] = float(entry[1])
                except (TypeError, ValueError, IndexError):
                    continue
        if book[0, 0, 0] <= 0 or book[1, 0, 0] <= 0:
            return None
        return book
