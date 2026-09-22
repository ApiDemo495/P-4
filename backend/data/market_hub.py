"""Market data hub - one owner for every live buffer (Section 5).

Responsibilities
----------------
* pick a feed (Binance -> CoinGecko -> simulator) according to
  ``MARKET_DATA_MODE`` and live health, reporting the active source
* keep the per-asset ring buffers up to date
* freeze a consistent snapshot at t=0 of every cycle (Signal Lock foundation)
* implement the PAXG tick-count rules from Section 5.2
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from backend.core import config as cfg
from backend.core.errors import ComponentStatus, DegradationLevel
from backend.data import cross_asset_sync as sync
from backend.data.binance_ws import BinanceWebSocket
from backend.data.coingecko_fallback import CoinGeckoFeed
from backend.data.ring_buffer import CandleBuffer, L2Buffer, TickBuffer
from backend.data.simulator import MarketSimulator, run_simulator

log = logging.getLogger("drosophila.market")


class AssetBuffers:
    def __init__(self, asset: str) -> None:
        self.asset = asset
        self.ticks = TickBuffer()
        self.book = L2Buffer()
        self.candles = CandleBuffer()


class MarketDataHub:
    """Owns the live market state and produces frozen snapshots."""

    def __init__(self, settings=None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.buffers: dict[str, AssetBuffers] = {a: AssetBuffers(a) for a in cfg.ASSETS}
        self.binance: BinanceWebSocket | None = None
        self.coingecko: CoinGeckoFeed | None = None
        self.simulator: MarketSimulator | None = None
        self.active_source = "none"
        self.warnings: list[str] = []
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()
        self._day_high: dict[str, float] = {}
        self._flash_marks: dict[str, list[tuple[float, float]]] = {a: [] for a in cfg.ASSETS}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        mode = self.settings.market_data_mode
        log.info("Starting market data hub (mode=%s)", mode)

        if mode in ("auto", "binance"):
            self.binance = BinanceWebSocket(self._on_data, self._on_candle, self.settings)
            self._tasks.append(asyncio.create_task(self.binance.run(), name="binance-ws"))
            self.active_source = "binance"
        if mode in ("auto", "coingecko"):
            self.coingecko = CoinGeckoFeed(self._on_data, self.settings)
            if mode == "coingecko":
                self._tasks.append(asyncio.create_task(self.coingecko.run(), name="coingecko"))
                self.active_source = "coingecko"
        if mode == "simulator" or self.settings.market_allow_simulator:
            self.simulator = MarketSimulator()
            self._warm_up()

        self._tasks.append(asyncio.create_task(self._supervisor(), name="feed-supervisor"))

    async def stop(self) -> None:
        self._stop.set()
        for client in (self.binance, self.coingecko):
            if client is not None:
                client.stop()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def _warm_up(self) -> None:
        """Pre-fill buffers so the very first cycle already computes all formulas."""
        assert self.simulator is not None
        warm = self.simulator.warm_up(seconds=180)
        for asset in cfg.ASSETS:
            rows = warm[asset]
            if rows.size:
                self.buffers[asset].ticks.seed(rows)
            self.buffers[asset].book.push(warm["book"][asset])
            for close in warm["candles"][asset]:
                self.buffers[asset].candles.push(float(close))
        if not self.buffers["BTC"].candles.closes().size:
            # ensure the DRG baseline has something to chew on
            for asset in cfg.ASSETS:
                px = float(warm["book"][asset][1, 0, 0] or 1.0)
                for _ in range(cfg.CANDLE_BUFFER_SIZE):
                    self.buffers[asset].candles.push(px)
        log.info(
            "Simulator warm-up complete: BTC %d ticks, PAXG %d ticks",
            self.buffers["BTC"].ticks.size,
            self.buffers["PAXG"].ticks.size,
        )

    # ------------------------------------------------------------------
    # Feed callbacks
    # ------------------------------------------------------------------
    async def _on_data(
        self,
        asset: str,
        ticks: list[tuple[float, float, float, float]],
        book: np.ndarray | None,
    ) -> None:
        buf = self.buffers.get(asset)
        if buf is None:
            return
        for row in ticks or []:
            buf.ticks.append(row[1], row[2], row[3], time_ms=row[0])
            self._track_flash(asset, row[1], row[0] / 1000.0)
        if book is not None:
            buf.book.push(book)

    async def _on_candle(self, asset: str, close: float) -> None:
        buf = self.buffers.get(asset)
        if buf is not None:
            buf.candles.push(close)

    def _track_flash(self, asset: str, price: float, ts: float) -> None:
        """Keep a short price history so the flash-move detector can fire."""
        marks = self._flash_marks[asset]
        marks.append((ts, price))
        cutoff = ts - max(self.settings.flash_move_window_seconds * 2, 60.0)
        while marks and marks[0][0] < cutoff:
            marks.pop(0)

    def recent_flash_move(self, asset: str) -> tuple[float, float]:
        """Return ``(max_abs_pct_move, direction)`` inside the flash window."""
        marks = self._flash_marks[asset]
        if len(marks) < 2:
            return 0.0, 0.0
        now = marks[-1][0]
        window = [p for t, p in marks if now - t <= self.settings.flash_move_window_seconds]
        if len(window) < 2:
            return 0.0, 0.0
        arr = np.asarray(window, dtype=np.float64)
        base = arr[0]
        if base <= 0:
            return 0.0, 0.0
        pct = float((arr[-1] / base - 1.0) * 100.0)
        return abs(pct), float(np.sign(pct))

    # ------------------------------------------------------------------
    # Supervisor: keeps exactly one healthy source alive
    # ------------------------------------------------------------------
    async def _supervisor(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(2.0)
            try:
                await self._reconcile_source()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("feed supervisor error: %s", exc)

    async def _reconcile_source(self) -> None:
        mode = self.settings.market_data_mode
        # 1) Is Binance healthy?
        if self.binance is not None and self.binance.status.connected and not self.binance.status.stale():
            self._set_source("binance")
            return

        # 2) Binance unhealthy -> CoinGecko?
        if mode in ("auto", "coingecko") and self.coingecko is not None:
            if mode == "coingecko" or self.binance is None or self.binance.status.consecutive_failures >= 10:
                if self.coingecko.connected:
                    self._set_source("coingecko")
                    return
                if mode == "coingecko":
                    self._set_source("coingecko")
                    return
                if self.binance.status.consecutive_failures >= 10:
                    self._ensure_coingecko_task()
                    return

        # 3) Simulator fallback
        if self.settings.market_allow_simulator:
            self._ensure_simulator_task()
            self._set_source("simulator")
            return

        self._set_source("none")

    def _set_source(self, source: str) -> None:
        if source == self.active_source:
            return
        log.warning("Market data source -> %s", source)
        self.active_source = source

    def _ensure_coingecko_task(self) -> None:
        if self.coingecko is None:
            return
        if any(t.get_name() == "coingecko" for t in self._tasks):
            return
        log.warning("Switching to CoinGecko fallback after Binance failures")
        self._tasks.append(asyncio.create_task(self.coingecko.run(), name="coingecko"))

    def _ensure_simulator_task(self) -> None:
        if self.simulator is None:
            return
        if any(t.get_name() == "simulator" for t in self._tasks):
            return
        log.warning("Market data unavailable or simulated mode: using built-in simulator")
        self._tasks.append(
            asyncio.create_task(run_simulator(self.simulator, self._on_data), name="simulator")
        )

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def tick_count(self, asset: str) -> int:
        return self.buffers[asset].ticks.size

    def last_price(self, asset: str) -> float:
        buf = self.buffers[asset]
        px = buf.book.mid()
        if px > 0:
            return px
        prices = buf.ticks.prices(1)
        return float(prices[0]) if prices.size else 0.0

    def freeze(self, news_items: tuple = (), drg_outcomes: np.ndarray | None = None):
        """Deep-copy the live buffers into the cycle's frozen inputs.

        Called at t=0 of every cycle.  Formulas never read the live buffers
        again until the next cycle, which is what makes the Signal Lock
        Protocol mechanical rather than a convention.
        """
        timestamp = time.time()
        warnings: list[str] = []
        payload: dict = {}

        for asset in cfg.ASSETS:
            buf = self.buffers[asset]
            key = asset.lower()
            ticks = buf.ticks.view()
            payload[f"{key}_ticks"] = ticks
            payload[f"{key}_book"] = np.array(buf.book.current, copy=True)
            payload[f"{key}_book_prev"] = np.array(buf.book.previous, copy=True)
            payload[f"{key}_candles"] = buf.candles.closes()
            payload[f"{key}_spreads"] = buf.book.spreads()
            payload[f"{key}_tick_count"] = buf.ticks.size

        # Section 5.2 - PAXG-specific tick thresholds
        paxg_ticks = payload["paxg_tick_count"]
        if paxg_ticks < 15:
            warnings.append("Insufficient PAXG data for reliable signal.")
        elif paxg_ticks < 30:
            warnings.append("PAXG tick count low: interpolating for VSD/VSS.")
        if self.active_source not in ("binance", "coingecko"):
            warnings.append("Live exchange feed unavailable - simulated tape in use.")
        if payload["btc_tick_count"] < 15:
            warnings.append("Insufficient BTC data for reliable signal.")

        synced = sync.synchronise(payload["btc_ticks"], payload["paxg_ticks"], now=timestamp)
        payload["synced"] = synced
        payload["warnings"] = tuple(warnings)
        payload["source"] = self.active_source
        payload["timestamp"] = timestamp
        if drg_outcomes is None:
            drg_outcomes = np.zeros((0, 2), dtype=np.float64)
        payload["news_items"] = tuple(news_items)
        payload["drg_outcomes"] = drg_outcomes

        from backend.core.frozen_snapshot import FrozenMarketSnapshot

        return FrozenMarketSnapshot(**payload)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def degradation_level(self) -> DegradationLevel:
        if self.active_source in ("none",) and self.tick_count("BTC") < 15:
            return DegradationLevel.MINIMAL
        if self.active_source == "coingecko":
            return DegradationLevel.COINGECKO
        return DegradationLevel.FULL

    def status(self) -> ComponentStatus:
        src = self.active_source
        details = {
            "binance": "aggTrade + depth20@100ms",
            "coingecko": "REST polling (no order book)",
            "simulator": "built-in simulator - simulated tape, not live market data",
        }
        # The simulator is a supported operating mode (offline / CI), not a
        # failure: data is flowing and every formula is computable.  It is
        # reported as healthy but flagged in `detail` and in the warnings.
        healthy = src != "none" and self.tick_count("BTC") >= 15
        extra = {
            "btc_ticks": self.tick_count("BTC"),
            "paxg_ticks": self.tick_count("PAXG"),
            "btc_price": round(self.last_price("BTC"), 2),
            "paxg_price": round(self.last_price("PAXG"), 2),
            "book_valid": self.buffers["BTC"].book.is_valid(),
        }
        if self.binance is not None:
            extra["binance_failures"] = self.binance.status.consecutive_failures
        return ComponentStatus(
            name="market_data",
            healthy=healthy,
            detail=details.get(src, "no feed"),
            mode=src,
            extra=extra,
        )
