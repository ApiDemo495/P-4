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
import functools
import logging
import time

import numpy as np

from backend.core import config as cfg
from backend.core.errors import ComponentStatus, DegradationLevel
from backend.data import cross_asset_sync as sync
from backend.data.binance_ws import BinanceWebSocket
from backend.data.coingecko_fallback import CoinGeckoFeed
from backend.data.kraken_rest import KrakenRest
from backend.data.kraken_ws import KrakenWebSocket
from backend.data.ring_buffer import CandleBuffer, L2Buffer, TickBuffer
from backend.data.simulator import MarketSimulator, run_simulator

log = logging.getLogger("drosophila.market")


class AssetBuffers:
    def __init__(self, asset: str) -> None:
        self.asset = asset
        self.ticks = TickBuffer()
        self.book = L2Buffer()
        self.candles = CandleBuffer()
        # Minute-candle roller for feeds without klines (CoinGecko, simulator):
        # the minute currently being built and its latest trade price.
        self.candle_minute: int = -1
        self.candle_close: float = 0.0
        self.candles_from_ticks: bool = False


class MarketDataHub:
    """Owns the live market state and produces frozen snapshots."""

    def __init__(self, settings=None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.buffers: dict[str, AssetBuffers] = {a: AssetBuffers(a) for a in cfg.ASSETS}
        self.binance: BinanceWebSocket | None = None
        self.coingecko: CoinGeckoFeed | None = None
        self.kraken: KrakenWebSocket | None = None
        self.kraken_rest: KrakenRest | None = None
        self.simulator: MarketSimulator | None = None
        self.active_source = "none"
        self.warnings: list[str] = []
        # Round Z: writes from a feed that is not the active source are
        # dropped and counted - one tape, one source, always.
        self.rejected: dict[str, int] = {}
        self.source_changed_at: float = time.time()
        self._started_at: float = time.time()
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

        self._started_at = time.time()
        if mode in ("auto", "binance"):
            self.binance = BinanceWebSocket(
                functools.partial(self.ingest, "binance"), self._on_candle, self.settings
            )
            self._tasks.append(asyncio.create_task(self.binance.run(), name="binance-ws"))
            self.active_source = "binance"
        if mode in ("auto", "kraken"):
            # Round Z: a second REAL tape with an order book, reachable from
            # the US regions Codespaces run in.  Always connected in auto mode
            # so the switch-over is instant when Binance drops.
            self.kraken = KrakenWebSocket(functools.partial(self.ingest, "kraken"), self.settings)
            self._tasks.append(asyncio.create_task(self.kraken.run(), name="kraken-ws"))
            if mode == "kraken":
                self.active_source = "kraken"
        if mode in ("auto", "krakenrest"):
            # Round AA: real trades + book over plain HTTPS GET, for networks
            # where WebSockets cannot get out.  Started on demand.
            self.kraken_rest = KrakenRest(functools.partial(self.ingest, "krakenrest"), self.settings)
            if mode == "krakenrest":
                self._tasks.append(asyncio.create_task(self.kraken_rest.run(), name="krakenrest"))
                self.active_source = "krakenrest"
        if mode in ("auto", "coingecko"):
            self.coingecko = CoinGeckoFeed(functools.partial(self.ingest, "coingecko"), self.settings)
            if mode == "coingecko":
                self._tasks.append(asyncio.create_task(self.coingecko.run(), name="coingecko"))
                self.active_source = "coingecko"
        if mode == "simulator" or self.settings.market_allow_simulator:
            self.simulator = MarketSimulator()
            if mode == "simulator":
                # Only a deliberately simulated engine is pre-filled with a
                # simulated history.  In auto mode the real feed gets a clean
                # tape; the simulator is a last resort and warms up its own
                # buffers if and when it is switched on.
                self.active_source = "simulator"
                self._warm_up()

        self._tasks.append(asyncio.create_task(self._supervisor(), name="feed-supervisor"))

    async def stop(self) -> None:
        self._stop.set()
        for client in (self.binance, self.kraken, self.kraken_rest, self.coingecko):
            if client is not None:
                client.stop()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def _warm_up(self) -> None:
        """Pre-fill buffers so the very first cycle already computes all formulas."""
        assert self.simulator is not None
        warm = self.simulator.warm_up(seconds=600)  # 10 one-minute candles for RSV / volatility from the first window
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
    @property
    def tape_is_simulated(self) -> bool:
        return self.active_source == "simulator"

    async def ingest(
        self,
        source: str,
        asset: str,
        ticks: list[tuple[float, float, float, float]] | None,
        book: np.ndarray | None,
    ) -> None:
        """The ONLY way data enters the buffers.

        Round Z: a feed writes only while it is the active source.  Before
        this guard a fresh Codespace ran the simulator *and* Binance into the
        same tape (the simulator started during Binance's connect, and nothing
        ever stopped it); the alternating 65k/real prices pinned every TP/SL at
        its ceiling and made every formula read noise - identically in every
        new Codespace, because the simulator is seeded.
        """
        if source != self.active_source:
            self.rejected[source] = self.rejected.get(source, 0) + 1
            return
        await self._on_data(asset, ticks, book)

    async def _on_data(
        self,
        asset: str,
        ticks: list[tuple[float, float, float, float]] | None,
        book: np.ndarray | None,
    ) -> None:
        buf = self.buffers.get(asset)
        if buf is None:
            return
        for row in ticks or []:
            buf.ticks.append(row[1], row[2], row[3], time_ms=row[0])
            self._track_flash(asset, row[1], row[0] / 1000.0)
            self._roll_candle(buf, float(row[1]), float(row[0]))
        if book is not None:
            buf.book.push(book)

    def _roll_candle(self, buf: AssetBuffers, price: float, time_ms: float) -> None:
        """Build 1-minute closes from the trade tape.

        Only Binance delivers klines; CoinGecko and the simulator used to leave
        the candle buffer at its warm-up contents forever, so the realised
        volatility (and with it every take-profit / stop-loss) sat on the 4 bps
        floor and never moved.  Every feed now rolls its own minute closes;
        when Binance klines are present they stay authoritative.
        """
        if self.active_source == "binance" and not buf.candles_from_ticks:
            return
        minute = int(time_ms // 60_000)
        if buf.candle_minute < 0:
            buf.candle_minute = minute
        elif minute > buf.candle_minute:
            buf.candles.push(buf.candle_close, buf.candle_minute * 60_000)
            buf.candles_from_ticks = True
            buf.candle_minute = minute
        buf.candle_close = price

    async def _on_candle(self, asset: str, close: float) -> None:
        buf = self.buffers.get(asset)
        if buf is not None:
            if buf.candles_from_ticks:
                # Binance is back: hand the buffer over to real klines again.
                buf.candles_from_ticks = False
                buf.candle_minute = -1
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
            self._stop_simulator_task()
            self._set_source("binance")
            return
        # 1b) Kraken healthy?  (second real tape, with a book)
        if self.kraken is not None and self.kraken.status.connected and not self.kraken.status.stale():
            if mode == "kraken" or self.binance is None or self.binance.status.stale():
                self._stop_simulator_task()
                self._set_source("kraken")
                return
        # A real feed gets a grace period to connect before anything simulated
        # is allowed near the tape (Round Z).  While waiting, the source is
        # "none" and the UI says "connecting to the market", not a fake price.
        waiting = (time.time() - self._started_at) < float(
            getattr(self.settings, "real_feed_grace_seconds", 45.0)
        )
        if waiting and mode in ("auto", "binance", "kraken") and self.active_source in ("none", "binance", "kraken"):
            self._set_source("none")
            return

        # 1c) Both sockets failing -> Kraken over HTTPS (real tape + book).
        sockets_down = (self.binance is None or self.binance.status.consecutive_failures >= 2) and (
            self.kraken is None or self.kraken.status.consecutive_failures >= 2)
        if self.kraken_rest is not None and mode in ("auto", "krakenrest") and (sockets_down or mode == "krakenrest"):
            if not any(t.get_name() == "krakenrest" for t in self._tasks):
                log.warning("WebSocket feeds unreachable - polling Kraken REST for a real tape")
                self._tasks.append(asyncio.create_task(self.kraken_rest.run(), name="krakenrest"))
            if self.kraken_rest.connected and self.buffers["BTC"].ticks.size > 0 or (
                    self.kraken_rest.connected and self.active_source != "krakenrest"):
                self._stop_simulator_task()
                self._set_source("krakenrest")
                return
            if self.active_source == "krakenrest":
                return
        # 2) Binance unhealthy -> CoinGecko?
        if mode in ("auto", "coingecko") and self.coingecko is not None:
            real_down = (self.binance is None or self.binance.status.consecutive_failures >= 3) and (
                self.kraken is None or self.kraken.status.consecutive_failures >= 3)
            if mode == "coingecko" or real_down:
                if self.coingecko.connected:
                    self._stop_simulator_task()
                    self._set_source("coingecko")
                    return
                if mode == "coingecko":
                    self._set_source("coingecko")
                    return
                if real_down:
                    self._ensure_coingecko_task()
                    if time.time() - self._started_at < float(
                            getattr(self.settings, "real_feed_grace_seconds", 45.0)) + 30.0:
                        # CoinGecko is polling but has no price yet: stay
                        # honest (none) a little longer rather than simulate.
                        self._set_source("none")
                        return

        # 3) Simulator fallback
        if self.settings.market_allow_simulator:
            self._ensure_simulator_task()
            self._set_source("simulator")
            return

        self._set_source("none")

    def _set_source(self, source: str) -> None:
        self.set_source(source)

    def set_source(self, source: str) -> None:
        """Switch the active source.  Any change of source flushes every
        buffer - one tape, one source, one history."""
        if source == self.active_source:
            return
        previous = self.active_source
        log.warning("Market data source -> %s (was %s)", source, previous)
        # Every change of source flushes: a simulated and a real tape must
        # never share a history, and two exchanges differ by a basis (USDT vs
        # USD) that would print as a fake jump in volatility.
        if previous != "none" or source == "simulator":
            for buf in self.buffers.values():
                buf.ticks.clear()
                buf.book = L2Buffer()
                buf.candles = CandleBuffer()
                buf.candle_minute = -1
                buf.candle_close = 0.0
                buf.candles_from_ticks = False
            self._flash_marks = {a: [] for a in cfg.ASSETS}
            self._day_high = {}
            log.warning("tape flushed: %s -> %s share no history", previous, source)
        self.active_source = source
        self.source_changed_at = time.time()
        if source == "simulator" and self.simulator is not None and not self.buffers["BTC"].ticks.size:
            self._warm_up()

    def _ensure_coingecko_task(self) -> None:
        if self.coingecko is None:
            return
        if any(t.get_name() == "coingecko" for t in self._tasks):
            return
        log.warning("Switching to CoinGecko fallback after Binance failures")
        self._tasks.append(asyncio.create_task(self.coingecko.run(), name="coingecko"))

    def _stop_simulator_task(self) -> None:
        """A real feed is healthy: the simulator must not touch the tape again."""
        for task in list(self._tasks):
            if task.get_name() == "simulator":
                task.cancel()
                self._tasks.remove(task)
                log.warning("simulator stopped: a real feed is healthy")

    def _ensure_simulator_task(self) -> None:
        if self.simulator is None:
            return
        if any(t.get_name() == "simulator" for t in self._tasks):
            return
        log.warning("Market data unavailable or simulated mode: using built-in simulator")
        self._tasks.append(
            asyncio.create_task(
                run_simulator(self.simulator, functools.partial(self.ingest, "simulator")),
                name="simulator",
            )
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
        if self.active_source == "simulator":
            warnings.append("Live exchange feed unavailable - simulated tape in use.")
        elif self.active_source == "none":
            warnings.append("No market feed connected yet - waiting for Binance / Kraken / CoinGecko.")
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
        if self.active_source == "none":
            return DegradationLevel.MINIMAL
        return DegradationLevel.FULL

    def feeds_report(self) -> dict:
        """Round AA: every feed's own story - connected, failures, last error.
        This is what the header chip and /api/health show so an 'offline' tape
        comes with the reason (451, DNS, timeout, refused…)."""
        out: dict = {"active": self.active_source, "simulated": self.tape_is_simulated,
                     "grace_seconds": float(getattr(self.settings, "real_feed_grace_seconds", 45.0)),
                     "uptime_seconds": round(time.time() - self._started_at, 1), "feeds": {}}
        for name, client in (("binance", self.binance), ("kraken", self.kraken)):
            if client is None:
                out["feeds"][name] = {"enabled": False}
                continue
            st = client.status
            out["feeds"][name] = {
                "enabled": True,
                "connected": bool(st.connected),
                "stale": bool(st.stale()),
                "failures": int(st.consecutive_failures),
                "messages": int(getattr(st, "messages", 0)),
                "server": str(getattr(st, "server", "")),
                "last_error": str(getattr(st, "last_error", "") or ""),
                "last_error_age_s": (round(time.time() - st.last_error_at, 1)
                                     if getattr(st, "last_error_at", 0.0) else None),
            }
        if self.kraken_rest is not None:
            out["feeds"]["krakenrest"] = {
                "enabled": True,
                "polling": any(t.get_name() == "krakenrest" for t in self._tasks),
                "connected": bool(self.kraken_rest.connected),
                "polls": int(self.kraken_rest.polls),
                "last_error": str(self.kraken_rest.last_error or ""),
            }
        else:
            out["feeds"]["krakenrest"] = {"enabled": False}
        if self.coingecko is not None:
            out["feeds"]["coingecko"] = {
                "enabled": True,
                "polling": any(t.get_name() == "coingecko" for t in self._tasks),
                "connected": bool(self.coingecko.connected),
                "last_error": str(getattr(self.coingecko, "last_error", "") or ""),
            }
        else:
            out["feeds"]["coingecko"] = {"enabled": False}
        out["feeds"]["simulator"] = {
            "enabled": self.simulator is not None,
            "running": any(t.get_name() == "simulator" for t in self._tasks),
        }
        out["rejected_writes"] = dict(self.rejected)
        return out

    def status(self) -> ComponentStatus:
        src = self.active_source
        details = {
            "binance": "aggTrade + depth20@100ms",
            "kraken": "trade + book depth 25 (Kraken v2, USD pairs)",
            "krakenrest": "Kraken REST polling - real trades + book over HTTPS (2 s)",
            "none": "connecting to a live feed - no tape yet",
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
            "tape_is_simulated": self.tape_is_simulated,
            "rejected_writes": dict(self.rejected),
            "source_age_seconds": round(time.time() - self.source_changed_at, 1),
        }
        if self.binance is not None:
            extra["binance_failures"] = self.binance.status.consecutive_failures
        if self.kraken is not None:
            extra["kraken_connected"] = self.kraken.status.connected
            extra["kraken_failures"] = self.kraken.status.consecutive_failures
        return ComponentStatus(
            name="market_data",
            healthy=healthy,
            detail=details.get(src, "no feed"),
            mode=src,
            extra=extra,
        )
