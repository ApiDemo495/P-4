"""Kraken WebSocket v2 feed - the second REAL tape (Round Z, ``&b``).

GitHub Codespaces run in US regions, where ``stream.binance.com`` answers 451.
Binance's market-data mirror usually works from there, but when it does not the
engine used to fall straight to CoinGecko's 10-second polls (no order book) and
then to the simulator.  Kraken's public v2 stream is reachable from the US,
needs no key, and carries both assets this engine trades:

    BTC/USD   trades + book (depth 25)
    PAXG/USD  trades + book (depth 25)

The book is maintained locally from Kraken's snapshot + deltas (a delta with
qty 0 removes the level) and emitted as the same ``(2, L2_DEPTH_LEVELS, 2)``
array the Binance feed produces, so every formula sees one shape regardless of
which exchange is behind it.  Prices are USD, like Binance's USDT pairs - the
few bps of basis between the two never matter because a source switch flushes
the tape (the two never share a history).

Reconnect policy mirrors ``binance_ws``: exponential backoff 1 -> 60 s, and a
watchdog that forces a reconnect when the stream is silent for
``stale_tick_seconds``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field

import numpy as np

from backend.core import config as cfg

log = logging.getLogger("drosophila.kraken")

KRAKEN_SYMBOLS: dict[str, str] = {"BTC": "BTC/USD", "PAXG": "PAXG/USD"}
WS_URL = "wss://ws.kraken.com/v2"
BOOK_DEPTH = 25


@dataclass
class KrakenStatus:
    connected: bool = False
    consecutive_failures: int = 0
    last_message_ts: float = 0.0
    messages: int = 0
    server: str = WS_URL
    last_error: str = ""

    def stale(self) -> bool:
        if not self.connected:
            return True
        return (time.time() - self.last_message_ts) > cfg.SETTINGS.stale_tick_seconds


@dataclass
class _LocalBook:
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)

    def apply(self, side: str, rows: list[dict], snapshot: bool) -> None:
        store = self.bids if side == "bids" else self.asks
        if snapshot:
            store.clear()
        for row in rows:
            try:
                price = float(row["price"])
                qty = float(row["qty"])
            except (KeyError, TypeError, ValueError):
                continue
            if qty <= 0:
                store.pop(price, None)
            else:
                store[price] = qty
        # Kraken keeps the book at BOOK_DEPTH; trim locally the same way.
        if len(store) > BOOK_DEPTH:
            keep = sorted(store, reverse=(side == "bids"))[:BOOK_DEPTH]
            for price in list(store):
                if price not in keep:
                    del store[price]

    def array(self) -> np.ndarray | None:
        if not self.bids or not self.asks:
            return None
        book = np.zeros((2, cfg.L2_DEPTH_LEVELS, 2), dtype=np.float64)
        for side_idx, (store, rev) in enumerate(((self.bids, True), (self.asks, False))):
            for level, price in enumerate(sorted(store, reverse=rev)[: cfg.L2_DEPTH_LEVELS]):
                book[side_idx, level, 0] = price
                book[side_idx, level, 1] = store[price]
        if book[0, 0, 0] <= 0 or book[1, 0, 0] <= 0 or book[1, 0, 0] < book[0, 0, 0]:
            return None
        return book


class KrakenWebSocket:
    """Public Kraken v2 stream: ``trade`` + ``book`` for BTC/USD and PAXG/USD."""

    def __init__(self, on_data, settings=None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.on_data = on_data
        self.status = KrakenStatus()
        self._stop = asyncio.Event()
        self._symbol_map = {v: k for k, v in KRAKEN_SYMBOLS.items()}
        self._books: dict[str, _LocalBook] = {a: _LocalBook() for a in cfg.ASSETS}
        self._last_book_emit: dict[str, float] = {a: 0.0 for a in cfg.ASSETS}

    @property
    def url(self) -> str:
        return str(getattr(self.settings, "kraken_ws_url", WS_URL) or WS_URL)

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    async def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._session()
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.status.connected = False
                self.status.consecutive_failures += 1
                self.status.last_error = str(exc)[:160]
                log.warning("Kraken WS error (%s). Failure %d, retry in %.0fs",
                            exc, self.status.consecutive_failures, backoff)
                await asyncio.sleep(backoff + random.uniform(0, 0.4))
                backoff = min(backoff * 2.0, self.settings.reconnect_max_seconds)

    async def _session(self) -> None:
        import websockets

        log.info("Connecting Kraken WS: %s", self.url)
        async with websockets.connect(self.url, ping_interval=20, ping_timeout=20,
                                      close_timeout=5, max_queue=512) as ws:
            symbols = [KRAKEN_SYMBOLS[a] for a in cfg.ASSETS]
            await ws.send(json.dumps({"method": "subscribe",
                                      "params": {"channel": "trade", "symbol": symbols, "snapshot": True}}))
            await ws.send(json.dumps({"method": "subscribe",
                                      "params": {"channel": "book", "symbol": symbols, "depth": BOOK_DEPTH,
                                                 "snapshot": True}}))
            self.status = KrakenStatus(connected=True, consecutive_failures=0,
                                       last_message_ts=time.time(), server=self.url)
            for book in self._books.values():
                book.bids.clear()
                book.asks.clear()
            log.info("Kraken WS connected (trade + book depth %d)", BOOK_DEPTH)
            watchdog = asyncio.create_task(self._watchdog(ws))
            try:
                async for raw in ws:
                    if self._stop.is_set():
                        break
                    await self._handle(raw)
            finally:
                watchdog.cancel()
                self.status.connected = False

    async def _watchdog(self, ws) -> None:
        while True:
            await asyncio.sleep(5)
            if self.status.stale():
                log.warning("Kraken stream stale (>%.0fs); forcing reconnect", self.settings.stale_tick_seconds)
                await ws.close()
                return

    # ------------------------------------------------------------------
    async def _handle(self, raw: str | bytes) -> None:
        self.status.last_message_ts = time.time()
        self.status.messages += 1
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(msg, dict):
            return
        channel = msg.get("channel")
        if channel == "trade":
            await self._handle_trades(msg.get("data") or [])
        elif channel == "book":
            await self._handle_book(msg.get("data") or [], snapshot=msg.get("type") == "snapshot")
        elif channel == "status" or msg.get("method") == "subscribe":
            if msg.get("success") is False:
                log.warning("Kraken subscribe refused: %s", msg.get("error"))

    async def _handle_trades(self, rows: list[dict]) -> None:
        by_asset: dict[str, list[tuple[float, float, float, float]]] = {}
        for row in rows:
            asset = self._symbol_map.get(str(row.get("symbol")))
            tick = self.parse_trade(row)
            if asset is None or tick is None:
                continue
            by_asset.setdefault(asset, []).append(tick)
        for asset, ticks in by_asset.items():
            await self.on_data(asset, ticks, None)

    async def _handle_book(self, rows: list[dict], snapshot: bool) -> None:
        for row in rows:
            asset = self._symbol_map.get(str(row.get("symbol")))
            if asset is None:
                continue
            book = self._books[asset]
            book.apply("bids", row.get("bids") or [], snapshot)
            book.apply("asks", row.get("asks") or [], snapshot)
            now = time.time()
            # Kraken sends every delta; the engine's book cadence is 100 ms.
            if now - self._last_book_emit[asset] < 0.1 and not snapshot:
                continue
            arr = book.array()
            if arr is not None:
                self._last_book_emit[asset] = now
                await self.on_data(asset, [], arr)

    @staticmethod
    def parse_trade(row: dict) -> tuple[float, float, float, float] | None:
        """Kraken v2 trade row -> (time_ms, price, qty, side) like Binance's."""
        try:
            price = float(row["price"])
            qty = float(row["qty"])
            side = 1.0 if str(row.get("side", "buy")).lower() == "buy" else -1.0
            ts = row.get("timestamp")
            if isinstance(ts, str):
                from datetime import datetime, timezone
                ts_ms = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp() * 1000.0
            else:
                ts_ms = float(ts) * (1000.0 if ts and float(ts) < 1e12 else 1.0) if ts else time.time() * 1000.0
        except (KeyError, TypeError, ValueError):
            return None
        if price <= 0 or qty <= 0:
            return None
        return (ts_ms, price, qty, side)
