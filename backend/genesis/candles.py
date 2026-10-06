"""Layer 1 storage: one-minute candles with microstructure, per asset.

The lock engine's own ring buffers are sized for the 22 named formulas (60
closes).  The genesis pool scores formulas on a rolling 500-candle window and
breeds on more, so it keeps its own store:

    column  0 ts_ms      minute start (UTC, ms)
            1 open  2 high  3 low  4 close
            5 volume    base-asset volume
            6 trades    number of prints
            7 buy_vol   taker-buy volume     (NaN when the minute came from a REST bootstrap)
            8 sell_vol  taker-sell volume    (NaN likewise)
            9 bid_depth summed bid qty, top of the engine's book   (NaN likewise)
           10 ask_depth summed ask qty                            (NaN likewise)
           11 spread    mean spread over the minute, in bps         (NaN likewise)
           12 imbalance mean (bid - ask) / (bid + ask) over the minute (NaN likewise)
           13 vwap

Minutes are rolled from the accepted tape (every tick the hub lets through -
one source, one history) and the first 720 are bootstrapped from a public
OHLC endpoint (Kraken, which reports trade counts; Gemini otherwise) so the
pool has a scoring history the moment the engine starts.  Order-book
snapshots are kept every five seconds (ten minutes deep) for the transport
formulas.  Everything is persisted under ``.run/genesis`` and reloaded on
restart, so history survives the engine's own restarts.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque

import numpy as np

from backend.genesis import book_ot

log = logging.getLogger("drosophila.genesis.candles")

OT_COLUMNS = ("w1_bid", "w1_ask", "w2_bid", "w2_ask", "sink_bid", "sink_ask", "bid_centroid", "ask_centroid",
              "bary_bid", "bary_ask", "ot_extrap_imb", "tftc_w1", "tftc_dir")
COLUMNS = ("ts_ms", "open", "high", "low", "close", "volume", "trades", "buy_vol", "sell_vol",
           "bid_depth", "ask_depth", "spread_bps", "imbalance", "vwap") + OT_COLUMNS
N_COLS = len(COLUMNS)
CAPACITY = 4096
BOOK_SNAPSHOT_EVERY_S = 5.0
BOOK_SNAPSHOTS = 120  # 10 minutes


class _Minute:
    __slots__ = ("minute", "open", "high", "low", "close", "volume", "trades", "buy", "sell",
                 "notional", "bid_sum", "ask_sum", "spread_sum", "imb_sum", "book_n",
                 "first_book", "last_book", "books", "buy_px", "sell_px")

    def __init__(self, minute: int, price: float) -> None:
        self.minute = minute
        self.open = self.high = self.low = self.close = price
        self.volume = self.trades = self.buy = self.sell = self.notional = 0.0
        self.bid_sum = self.ask_sum = self.spread_sum = self.imb_sum = 0.0
        self.book_n = 0
        self.first_book = self.last_book = None
        self.books: list = []
        self.buy_px: list = []
        self.sell_px: list = []

    def row(self, with_ot: bool = False) -> np.ndarray:
        n = max(1, self.book_n)
        has_book = self.book_n > 0
        if with_ot:
            ot = book_ot.minute_readings(self.first_book, self.last_book, self.books)
            mid = 0.5 * (self.first_book[0, 0, 0] + self.first_book[1, 0, 0]) if self.first_book is not None else self.close
            ot["tftc_w1"], ot["tftc_dir"] = book_ot.trade_flow_transport(self.buy_px, self.sell_px, mid)
        else:
            ot = {}
        tail = [ot.get(k, math.nan) for k in OT_COLUMNS]
        return np.array([
            self.minute * 60_000.0, self.open, self.high, self.low, self.close, self.volume, self.trades,
            self.buy, self.sell,
            self.bid_sum / n if has_book else math.nan,
            self.ask_sum / n if has_book else math.nan,
            self.spread_sum / n if has_book else math.nan,
            self.imb_sum / n if has_book else math.nan,
            (self.notional / self.volume) if self.volume > 0 else self.close,
        ] + tail, dtype=np.float64)


class CandleStore:
    """Per-asset minute store + book snapshot history.  Thread-safe."""

    def __init__(self, asset: str, state_dir: str | None = None) -> None:
        self.asset = asset
        self.state_dir = state_dir
        self._rows = np.zeros((0, N_COLS), dtype=np.float64)
        self._cur: _Minute | None = None
        self._lock = threading.Lock()
        self.books: deque[tuple[float, np.ndarray]] = deque(maxlen=BOOK_SNAPSHOTS)
        self._last_book_snap = 0.0
        self.closed_minutes = 0
        self.bootstrapped_from = ""
        self.on_close = None  # callback(asset, row) when a minute closes
        self._load()

    # ------------------------------------------------------------------ IO
    def _path(self) -> str | None:
        if not self.state_dir:
            return None
        return os.path.join(self.state_dir, f"candles_{self.asset}.npy")

    def _load(self) -> None:
        path = self._path()
        if not path or not os.path.exists(path):
            return
        try:
            arr = np.load(path)
            if arr.ndim == 2 and arr.shape[1] == N_COLS:
                self._rows = arr[-CAPACITY:]
                log.info("%s: restored %d minutes from %s", self.asset, len(self._rows), path)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: could not restore candles (%s)", self.asset, exc)

    def save(self) -> None:
        path = self._path()
        if not path:
            return
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            with self._lock:
                rows = self._rows.copy()
            tmp = path + ".tmp"
            np.save(tmp, rows)
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001
            log.debug("%s: candle save failed (%s)", self.asset, exc)

    # ------------------------------------------------------------- ingest
    def bootstrap(self, rows: np.ndarray, source: str) -> int:
        """Merge REST OHLC rows ``[ts_ms, o, h, l, c, v(, trades)]`` oldest-first
        into the store, never overwriting minutes rolled from the live tape."""
        if rows is None or len(rows) == 0:
            return 0
        rows = np.asarray(rows, dtype=np.float64)
        out = np.full((len(rows), N_COLS), math.nan)
        out[:, :6] = rows[:, :6]
        out[:, 6] = rows[:, 6] if rows.shape[1] > 6 else math.nan
        out[:, 13] = rows[:, 4]
        out[:, 0] = (out[:, 0] // 60_000) * 60_000
        with self._lock:
            have = set(self._rows[:, 0].astype(np.int64).tolist()) if len(self._rows) else set()
            fresh = out[[int(t) not in have for t in out[:, 0]]]
            if len(fresh) == 0:
                return 0
            merged = np.vstack((self._rows, fresh)) if len(self._rows) else fresh
            merged = merged[np.argsort(merged[:, 0], kind="stable")]
            self._rows = merged[-CAPACITY:]
            self.bootstrapped_from = source
        return int(len(fresh))

    def on_tape(self, ticks, book: np.ndarray | None) -> None:
        """Hub listener: ``ticks`` rows are ``(time_ms, price, qty, side)``."""
        closed: list[np.ndarray] = []
        with self._lock:
            for row in ticks or []:
                t_ms, price, qty, side = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                if price <= 0:
                    continue
                minute = int(t_ms // 60_000)
                cur = self._cur
                if cur is None or minute > cur.minute:
                    if cur is not None:
                        closed.append(self._close_locked(cur))
                    cur = self._cur = _Minute(minute, price)
                elif minute < cur.minute:
                    continue  # late print from a previous minute: ignore
                cur.high = max(cur.high, price)
                cur.low = min(cur.low, price)
                cur.close = price
                cur.volume += qty
                cur.trades += 1
                cur.notional += price * qty
                if side > 0:
                    cur.buy += qty
                    cur.buy_px.append(price)
                elif side < 0:
                    cur.sell += qty
                    cur.sell_px.append(price)
            if book is not None and self._cur is not None:
                bids, asks = book[0], book[1]
                bid_qty = float(bids[:, 1].sum())
                ask_qty = float(asks[:, 1].sum())
                bb, ba = float(bids[0, 0]), float(asks[0, 0])
                if bb > 0 and ba > bb:
                    cur = self._cur
                    cur.bid_sum += bid_qty
                    cur.ask_sum += ask_qty
                    cur.spread_sum += (ba - bb) / ((ba + bb) / 2.0) * 1e4
                    tot = bid_qty + ask_qty
                    cur.imb_sum += (bid_qty - ask_qty) / tot if tot > 0 else 0.0
                    cur.book_n += 1
                    if cur.first_book is None:
                        cur.first_book = np.array(book, dtype=np.float64, copy=True)
                    cur.last_book = book
                now = time.time()
                if now - self._last_book_snap >= BOOK_SNAPSHOT_EVERY_S:
                    self._last_book_snap = now
                    snap = np.array(book, dtype=np.float64, copy=True)
                    self.books.append((now, snap))
                    if self._cur is not None:
                        self._cur.books.append(snap)
        for row in closed:
            if self.on_close is not None:
                try:
                    self.on_close(self.asset, row)
                except Exception as exc:  # noqa: BLE001
                    log.debug("on_close failed: %s", exc)

    def _close_locked(self, cur: _Minute) -> np.ndarray:
        if cur.last_book is not None:
            cur.last_book = np.array(cur.last_book, dtype=np.float64, copy=True)
            if len(cur.books) < 2:
                cur.books = [cur.first_book, cur.last_book]
        row = cur.row(with_ot=True)
        if len(self._rows) and self._rows[-1, 0] >= row[0]:
            # replace a bootstrapped copy of the same minute with the live one
            keep = self._rows[:, 0] < row[0]
            self._rows = np.vstack((self._rows[keep], row[None, :]))
        else:
            self._rows = np.vstack((self._rows, row[None, :])) if len(self._rows) else row[None, :]
        if len(self._rows) > CAPACITY:
            self._rows = self._rows[-CAPACITY:]
        self.closed_minutes += 1
        return row

    def flush(self) -> None:
        """Source switch: the minute being built belongs to the old tape."""
        with self._lock:
            self._cur = None
            self.books.clear()

    # --------------------------------------------------------------- views
    def rows(self, limit: int | None = None, include_open: bool = True) -> np.ndarray:
        """Chronological copy; the forming minute is appended when requested."""
        with self._lock:
            base = self._rows
            if include_open and self._cur is not None:
                base = np.vstack((base, self._cur.row()[None, :])) if len(base) else self._cur.row()[None, :]
            if limit is not None:
                base = base[-limit:]
            return base.copy()

    def __len__(self) -> int:
        with self._lock:
            return int(len(self._rows))

    def last_close(self) -> float:
        with self._lock:
            if self._cur is not None:
                return float(self._cur.close)
            return float(self._rows[-1, 4]) if len(self._rows) else 0.0

    def book_pair(self, seconds_apart: float) -> tuple[np.ndarray, np.ndarray] | None:
        """(book now, book ~``seconds_apart`` ago) from the snapshot history."""
        with self._lock:
            if len(self.books) < 2:
                return None
            now_t, now_b = self.books[-1]
            target = now_t - seconds_apart
            best = min(self.books, key=lambda tb: abs(tb[0] - target))
            if abs(best[0] - target) > max(BOOK_SNAPSHOT_EVERY_S * 1.5, seconds_apart * 0.5):
                return None
            return now_b, best[1]

    def recent_books(self, n: int) -> list[np.ndarray]:
        with self._lock:
            return [b for _, b in list(self.books)[-n:]]

    def status(self) -> dict:
        with self._lock:
            n = len(self._rows)
            live = int(np.sum(~np.isnan(self._rows[:, 7]))) if n else 0
            first = float(self._rows[0, 0]) if n else 0.0
            last = float(self._rows[-1, 0]) if n else 0.0
        return {"minutes": n, "live_minutes": live, "bootstrapped_from": self.bootstrapped_from,
                "book_snapshots": len(self.books), "closed_live": self.closed_minutes,
                "first_minute_ms": first, "last_minute_ms": last}


async def bootstrap_history(asset: str, timeout: float = 10.0) -> tuple[np.ndarray, str]:
    """Public 1-minute history: Kraken OHLC first (720 rows with trade counts),
    Gemini candles second (1440 rows).  Returns ``(rows, source)``."""
    import httpx

    pair = {"BTC": "XBTUSD", "PAXG": "PAXGUSD"}.get(asset)
    if pair:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get("https://api.kraken.com/0/public/OHLC", params={"pair": pair, "interval": 1})
                r.raise_for_status()
                result = (r.json() or {}).get("result") or {}
                key = next((k for k in result if k != "last"), None)
                data = result.get(key) if key else None
                if data:
                    arr = np.array([[float(x[0]) * 1000.0, float(x[1]), float(x[2]), float(x[3]), float(x[4]),
                                     float(x[6]), float(x[7])] for x in data], dtype=np.float64)
                    return arr, "kraken"
        except Exception as exc:  # noqa: BLE001
            log.info("%s: Kraken OHLC bootstrap unavailable (%s)", asset, exc)
    try:
        from backend.data.gemini_ws import fetch_candles
        arr = await fetch_candles(asset, 1440, timeout)
        if len(arr):
            return arr, "gemini"
    except Exception as exc:  # noqa: BLE001
        log.info("%s: Gemini candle bootstrap unavailable (%s)", asset, exc)
    return np.zeros((0, 7)), ""
