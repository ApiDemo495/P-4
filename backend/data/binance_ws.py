"""Binance public WebSocket consumer (Section 5.1).

Combined stream:
    wss://stream.binance.com:9443/stream?streams=
        btcusdt@aggTrade/paxgusdt@aggTrade/
        btcusdt@kline_1m/paxgusdt@kline_1m/
        btcusdt@depth20@100ms/paxgusdt@depth20@100ms

``@aggTrade`` is used instead of ``@trade`` because it aggregates fills at the
same price into one event, cutting message rate while preserving everything the
formulas need.

Resilience (Section 13.1):
  * reconnect with exponential backoff 1s -> 2s -> 4s -> ... -> 60s
  * after 10 consecutive failures the caller may switch to CoinGecko
  * a tick gap longer than ``STALE_TICK_SECONDS`` is treated as a disconnect
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import numpy as np

from backend.core import config as cfg

log = logging.getLogger("drosophila.binance")

TickCallback = Callable[[str, list[tuple[float, float, float, float]], np.ndarray | None], Awaitable[None]]


@dataclass
class BinanceStatus:
    connected: bool = False
    consecutive_failures: int = 0
    last_message_ts: float = 0.0
    messages: int = 0
    server: str = ""

    def stale(self) -> bool:
        if not self.connected:
            return True
        return (time.time() - self.last_message_ts) > cfg.SETTINGS.stale_tick_seconds


class BinanceWebSocket:
    """Async consumer for the Binance combined market stream."""

    def __init__(
        self,
        on_data: TickCallback,
        on_candle: Callable[[str, float], Awaitable[None]] | None = None,
        settings=None,
    ) -> None:
        self.settings = settings or cfg.SETTINGS
        self.on_data = on_data
        self.on_candle = on_candle
        self.status = BinanceStatus()
        self._stop = asyncio.Event()
        self._symbol_map = {v.upper(): k for k, v in cfg.BINANCE_SYMBOLS.items()}

    # ------------------------------------------------------------------
    @property
    def url(self) -> str:
        streams: list[str] = []
        for asset in cfg.ASSETS:
            sym = cfg.BINANCE_SYMBOLS[asset]
            streams += [f"{sym}@aggTrade", f"{sym}@kline_1m", f"{sym}@depth20@100ms"]
        return f"{self.settings.binance_ws_base}/stream?streams={'/'.join(streams)}"

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    async def run(self) -> None:
        """Connect-forever loop with exponential backoff."""
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._session()
                backoff = 1.0  # clean disconnect -> reset backoff
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.status.connected = False
                self.status.consecutive_failures += 1
                log.warning(
                    "Binance WS error (%s). Failure %d, retry in %.0fs",
                    exc,
                    self.status.consecutive_failures,
                    backoff,
                )
                await asyncio.sleep(backoff + random.uniform(0, 0.4))
                backoff = min(backoff * 2.0, self.settings.reconnect_max_seconds)

    async def _session(self) -> None:
        import websockets

        log.info("Connecting Binance WS: %s", self.url)
        async with websockets.connect(
            self.url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
            max_queue=512,
        ) as ws:
            self.status = BinanceStatus(
                connected=True, consecutive_failures=0, last_message_ts=time.time(), server=self.url
            )
            log.info("Binance WS connected (aggTrade + kline_1m + depth20@100ms)")
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
        """Force a reconnect if the stream goes silent (stale data check)."""
        while True:
            await asyncio.sleep(5)
            if self.status.stale():
                log.warning("Binance stream stale (>%.0fs); forcing reconnect", self.settings.stale_tick_seconds)
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

        stream = msg.get("stream", "")
        data = msg.get("data", msg)
        if "@" not in stream:
            return
        symbol, _, kind = stream.partition("@")
        asset = self._symbol_map.get(symbol.upper())
        if asset is None:
            return

        if kind.startswith("aggtrade"):
            tick = self._parse_agg_trade(data)
            if tick is not None:
                await self.on_data(asset, [tick], None)
        elif kind.startswith("kline"):
            close = self._parse_kline(data)
            if close is not None and self.on_candle is not None:
                await self.on_candle(asset, close)
        elif kind.startswith("depth"):
            book = self._parse_depth(data)
            if book is not None:
                await self.on_data(asset, [], book)

    @staticmethod
    def _parse_agg_trade(data: dict) -> tuple[float, float, float, float] | None:
        try:
            price = float(data["p"])
            qty = float(data["q"])
            ts = float(data.get("T") or time.time() * 1000)
            # "m" == True  -> the buyer was the maker -> the aggressor was the SELLER
            side = -1.0 if data.get("m") else 1.0
        except (KeyError, TypeError, ValueError):
            return None
        if price <= 0 or qty <= 0:
            return None
        return (ts, price, qty, side)

    @staticmethod
    def _parse_kline(data: dict) -> float | None:
        try:
            k = data["k"]
            if not k.get("x"):  # only act on closed candles
                return None
            return float(k["c"])
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _parse_depth(data: dict) -> np.ndarray | None:
        """Binance depth20 -> (2, 20, 2) float array."""
        try:
            bids = data["bids"]
            asks = data["asks"]
        except (KeyError, TypeError):
            return None
        book = np.zeros((2, cfg.L2_DEPTH_LEVELS, 2), dtype=np.float64)
        for side_idx, raw in enumerate((bids, asks)):
            for level, entry in enumerate(raw[: cfg.L2_DEPTH_LEVELS]):
                try:
                    book[side_idx, level, 0] = float(entry[0])
                    book[side_idx, level, 1] = float(entry[1])
                except (TypeError, ValueError, IndexError):
                    continue
        if book[0, 0, 0] <= 0 or book[1, 0, 0] <= 0:
            return None
        return book
