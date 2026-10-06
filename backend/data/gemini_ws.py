"""Gemini exchange public market-data stream (Round AK, v3.0 spec: BTC/USD and
PAXG/USD *on Gemini*).

Gemini's v1 market-data socket needs no key and carries, per symbol, the full
order book (an ``initial`` snapshot followed by ``change`` events) and every
trade.  One socket per symbol:

    wss://api.gemini.com/v1/marketdata/btcusd?bids=true&offers=true&trades=true
    wss://api.gemini.com/v1/marketdata/paxgusd?...

The book is kept locally (same ``_LocalBook`` as the Kraken feed, so it trims
to the engine's depth the same way) and emitted as the ``(2, L2_DEPTH_LEVELS,
2)`` array every other feed produces.  Trades arrive with ``makerSide``; the
aggressor is the other side, so ``makerSide == "ask"`` is a BUY print (+1).

Reconnect policy mirrors the other sockets: exponential backoff 1 -> 60 s and
a watchdog that forces a reconnect on silence.  Gemini sends a heartbeat every
five seconds when ``heartbeat=true`` is requested, so silence really is a dead
socket.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass

import numpy as np

from backend.core import config as cfg
from backend.data.kraken_ws import _LocalBook

log = logging.getLogger("drosophila.gemini")

GEMINI_SYMBOLS: dict[str, str] = {"BTC": "btcusd", "PAXG": "paxgusd"}
WS_BASE = "wss://api.gemini.com/v1/marketdata"
REST_BASE = "https://api.gemini.com"


@dataclass
class GeminiStatus:
    connected: bool = False
    consecutive_failures: int = 0
    last_message_ts: float = 0.0
    messages: int = 0
    data_messages: int = 0
    server: str = WS_BASE
    last_error: str = ""
    last_error_at: float = 0.0
    sockets: int = 0

    def healthy(self) -> bool:
        return self.connected and self.data_messages > 0 and not self.stale()

    def stale(self) -> bool:
        if not self.connected:
            return True
        return (time.time() - self.last_message_ts) > cfg.SETTINGS.stale_tick_seconds


class GeminiWebSocket:
    """Public Gemini v1 market data: book + trades for btcusd and paxgusd."""

    def __init__(self, on_data, settings=None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.on_data = on_data
        self.status = GeminiStatus()
        self._stop = asyncio.Event()
        self._books: dict[str, _LocalBook] = {a: _LocalBook() for a in cfg.ASSETS}
        self._last_book_emit: dict[str, float] = {a: 0.0 for a in cfg.ASSETS}
        self._open: set[str] = set()

    @property
    def base_url(self) -> str:
        return str(getattr(self.settings, "gemini_ws_url", WS_BASE) or WS_BASE)

    def url_for(self, asset: str) -> str:
        return f"{self.base_url}/{GEMINI_SYMBOLS[asset]}?bids=true&offers=true&trades=true&heartbeat=true"

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        await asyncio.gather(*(self._run_symbol(a) for a in cfg.ASSETS if a in GEMINI_SYMBOLS))

    async def _run_symbol(self, asset: str) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._session(asset)
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._open.discard(asset)
                self.status.connected = bool(self._open)
                self.status.sockets = len(self._open)
                self.status.consecutive_failures += 1
                self.status.last_error = f"{type(exc).__name__}: {str(exc)[:140]}".strip(": ")
                self.status.last_error_at = time.time()
                log.warning("Gemini WS %s error (%s). Failure %d, retry in %.0fs",
                            asset, exc, self.status.consecutive_failures, backoff)
                await asyncio.sleep(backoff + random.uniform(0, 0.4))
                backoff = min(backoff * 2.0, self.settings.reconnect_max_seconds)

    async def _session(self, asset: str) -> None:
        import websockets

        url = self.url_for(asset)
        log.info("Connecting Gemini WS: %s", url.split("?")[0])
        async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=5,
                                      max_queue=1024) as ws:
            self._open.add(asset)
            self.status.connected = True
            self.status.sockets = len(self._open)
            self.status.last_message_ts = time.time()
            self._books[asset] = _LocalBook()
            watchdog = asyncio.create_task(self._watchdog(ws))
            had_data = self.status.data_messages
            try:
                async for raw in ws:
                    if self._stop.is_set():
                        break
                    await self._handle(asset, raw)
            finally:
                watchdog.cancel()
                self._open.discard(asset)
                self.status.connected = bool(self._open)
                self.status.sockets = len(self._open)
            if self.status.data_messages == had_data and not self._stop.is_set():
                raise RuntimeError("connected but no market data arrived (silent stream)")

    async def _watchdog(self, ws) -> None:
        while True:
            await asyncio.sleep(5)
            if self.status.stale():
                log.warning("Gemini stream stale (>%.0fs); forcing reconnect", self.settings.stale_tick_seconds)
                await ws.close()
                return

    async def _handle(self, asset: str, raw: str | bytes) -> None:
        self.status.last_message_ts = time.time()
        self.status.messages += 1
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(msg, dict) or msg.get("type") != "update":
            return
        events = msg.get("events") or []
        ts_ms = float(msg.get("timestampms") or time.time() * 1000.0)
        ticks: list[tuple[float, float, float, float]] = []
        book_changed = False
        snapshot = False
        book = self._books[asset]
        for ev in events:
            kind = ev.get("type")
            if kind == "trade":
                tick = self.parse_trade(ev, ts_ms)
                if tick is not None:
                    ticks.append(tick)
            elif kind == "change":
                side = "bids" if ev.get("side") == "bid" else "asks"
                if ev.get("reason") == "initial":
                    snapshot = True
                try:
                    book.apply(side, [{"price": ev.get("price"), "qty": ev.get("remaining")}], snapshot=False)
                except Exception:  # noqa: BLE001
                    continue
                book_changed = True
        if ticks or book_changed:
            self.status.data_messages += 1
            self.status.consecutive_failures = 0
        if ticks:
            await self.on_data(asset, ticks, None)
        if book_changed:
            now = time.time()
            if snapshot or now - self._last_book_emit[asset] >= 0.1:
                arr = book.array()
                if arr is not None:
                    self._last_book_emit[asset] = now
                    await self.on_data(asset, [], arr)

    @staticmethod
    def parse_trade(ev: dict, ts_ms: float) -> tuple[float, float, float, float] | None:
        """Gemini trade event -> (time_ms, price, qty, side); side = aggressor."""
        try:
            price = float(ev["price"])
            qty = float(ev["amount"])
        except (KeyError, TypeError, ValueError):
            return None
        if price <= 0 or qty <= 0:
            return None
        maker = str(ev.get("makerSide", "")).lower()
        side = 1.0 if maker == "ask" else -1.0 if maker == "bid" else 0.0
        if side == 0.0:
            return None
        return (ts_ms, price, qty, side)


async def fetch_candles(asset: str, limit: int = 1440, timeout: float = 10.0) -> np.ndarray:
    """Gemini public 1-minute candles -> rows ``[time_ms, open, high, low, close, volume]``
    oldest first (up to 1440).  Used to bootstrap the genesis candle store so the
    2100-formula pool can be scored on 500 candles the moment the engine starts.
    """
    import httpx

    symbol = GEMINI_SYMBOLS.get(asset)
    if symbol is None:
        return np.zeros((0, 6))
    url = f"{REST_BASE}/v2/candles/{symbol}/1m"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        rows = response.json()
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 6:
        return np.zeros((0, 6))
    arr = arr[np.argsort(arr[:, 0])]
    return arr[-limit:, :6]
