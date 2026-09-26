"""Fixed-size ring buffers for market data (Section 5.1).

Sizes are taken verbatim from the specification:

===============  ======  ===============================================
Buffer           Size    Reason
===============  ======  ===============================================
Tick (per asset) 600     VSD needs 120, VSS needs 300, 2x margin
L2 (per asset)   2       BAR needs current + previous
Candle           60      1-hour lookback for the DRG baseline
News cache       20      last 20 headlines from all sources
Outcome buffer   20      DRG needs the last 20 outcomes
===============  ======  ===============================================

The buffers are O(1) per append and never allocate during steady state.  A
``FrozenMarketSnapshot`` is produced by copying them, which is the foundation
of the Signal Lock Protocol: the formulas only ever see the frozen copy.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from backend.core import config as cfg

TICK_COLUMNS = ("time_ms", "price", "qty", "side")


class TickBuffer:
    """Circular buffer of aggregated trades: [time_ms, price, qty, side(+/-1)]."""

    def __init__(self, capacity: int = cfg.TICK_BUFFER_SIZE) -> None:
        self.capacity = capacity
        self._data = np.zeros((capacity, 4), dtype=np.float64)
        self._head = 0
        self._size = 0
        self._lock = threading.Lock()
        self.last_update: float = 0.0

    def append(self, price: float, qty: float, side: float, time_ms: float | None = None) -> None:
        if time_ms is None:
            time_ms = time.time() * 1000.0
        with self._lock:
            self._data[self._head] = (time_ms, price, qty, side)
            self._head = (self._head + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
            self.last_update = time.time()

    def extend(self, rows: np.ndarray) -> None:
        if rows.size == 0:
            return
        with self._lock:
            for row in rows:
                self._data[self._head] = row[:4]
                self._head = (self._head + 1) % self.capacity
                self._size = min(self._size + 1, self.capacity)
            self.last_update = time.time()

    @property
    def size(self) -> int:
        return self._size

    def view(self, limit: int | None = None) -> np.ndarray:
        """Return a chronological copy (oldest first)."""
        with self._lock:
            n = self._size if limit is None else min(limit, self._size)
            if n == 0:
                return np.zeros((0, 4), dtype=np.float64)
            start = (self._head - n) % self.capacity
            if start + n <= self.capacity:
                out = self._data[start : start + n].copy()
            else:
                first = self.capacity - start
                out = np.vstack((self._data[start:], self._data[: n - first]))
            return out

    def prices(self, limit: int | None = None) -> np.ndarray:
        return self.view(limit)[:, 1]

    def age_seconds(self) -> float:
        if self.last_update == 0.0:
            return float("inf")
        return time.time() - self.last_update

    def clear(self) -> None:
        with self._lock:
            self._data[:] = 0.0
            self._head = 0
            self._size = 0

    def seed(self, rows: np.ndarray) -> None:
        """Bulk-load initial history (used by the simulator warm-up)."""
        self.clear()
        self.extend(rows)


def empty_book() -> np.ndarray:
    """Shape (2, 20, 2) = [ [bids(price,qty)] , [asks(price,qty)] ]."""
    return np.zeros((2, cfg.L2_DEPTH_LEVELS, 2), dtype=np.float64)


class L2Buffer:
    """Keeps exactly two order-book snapshots - plus a spread history.

    BAR (Formula 7) needs current + previous, hence the two slots.  SED
    (Formula 3) needs the spread *at tick timestamps*, so a rolling
    ``(time_ms, spread)`` history is retained as well; see the note in
    ``core/frozen_snapshot.py`` for why that is necessary.
    """

    SPREAD_HISTORY = 900

    def __init__(self) -> None:
        self.current = empty_book()
        self.previous = empty_book()
        self.last_update: float = 0.0
        self.spread_history: deque[tuple[float, float]] = deque(maxlen=self.SPREAD_HISTORY)

    def push(self, book: np.ndarray) -> None:
        if book.shape != (2, cfg.L2_DEPTH_LEVELS, 2):
            raise ValueError(f"order book must be {(2, cfg.L2_DEPTH_LEVELS, 2)}, got {book.shape}")
        self.previous = self.current
        self.current = np.array(book, dtype=np.float64, copy=True)
        self.last_update = time.time()
        bid, ask = self.best_bid(), self.best_ask()
        if bid > 0 and ask > bid:
            self.spread_history.append((self.last_update * 1000.0, ask - bid))

    def spreads(self, min_points: int = 2) -> np.ndarray:
        if len(self.spread_history) < min_points:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray(self.spread_history, dtype=np.float64)

    @property
    def has_history(self) -> bool:
        return bool(self.previous.any()) and bool(self.current.any())

    def age_seconds(self) -> float:
        if self.last_update == 0.0:
            return float("inf")
        return time.time() - self.last_update

    def best_bid(self) -> float:
        book = self.current
        return float(book[0, 0, 0]) if book[0, 0, 0] > 0 else 0.0

    def best_ask(self) -> float:
        book = self.current
        return float(book[1, 0, 0]) if book[1, 0, 0] > 0 else 0.0

    def mid(self) -> float:
        bid, ask = self.best_bid(), self.best_ask()
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        return 0.0

    def spread(self) -> float:
        bid, ask = self.best_bid(), self.best_ask()
        return max(0.0, ask - bid) if bid > 0 and ask > 0 else 0.0

    def is_valid(self) -> bool:
        return self.best_bid() > 0 and self.best_ask() > self.best_bid()


class CandleBuffer:
    """Last N 1-minute closes (Section 5.1)."""

    def __init__(self, capacity: int = cfg.CANDLE_BUFFER_SIZE) -> None:
        self.capacity = capacity
        self._closes: deque[float] = deque(maxlen=capacity)
        self._open_time: deque[int] = deque(maxlen=capacity)

    def push(self, close: float, open_time_ms: int | None = None) -> None:
        self._closes.append(float(close))
        self._open_time.append(int(open_time_ms or 0))

    def closes(self) -> np.ndarray:
        return np.asarray(self._closes, dtype=np.float64)

    def __len__(self) -> int:
        return len(self._closes)


@dataclass(frozen=True)
class NewsItem:
    """Immutable news record shared by the news engine and the formulas."""

    headline: str
    source: str
    tier: int
    sentiment: float
    published_at: float
    url: str = ""
    provider: str = ""

    @property
    def credibility(self) -> float:
        return {1: 1.0, 2: 0.7, 3: 0.4}.get(self.tier, 0.2)

    def age_seconds(self, now: float | None = None) -> float:
        return max(0.0, (now or time.time()) - self.published_at)

    def to_dict(self) -> dict:
        return {
            "headline": self.headline,
            "source": self.source,
            "tier": self.tier,
            "credibility": self.credibility,
            "sentiment": round(self.sentiment, 4),
            "published_at": self.published_at,
            "age_seconds": round(self.age_seconds(), 1),
            "url": self.url,
            "provider": self.provider,
        }


class NewsCache:
    """Newest-first cache of the last N headlines from all providers."""

    def __init__(self, capacity: int = cfg.NEWS_CACHE_SIZE) -> None:
        self.capacity = capacity
        self._items: deque[NewsItem] = deque(maxlen=capacity)
        self.last_poll: float = 0.0
        self.providers: dict[str, str] = {}

    def add(self, items: list[NewsItem]) -> int:
        seen = {i.headline for i in self._items}
        added = 0
        for item in items:
            if item.headline in seen:
                continue
            seen.add(item.headline)
            self._items.appendleft(item)
            added += 1
        self.last_poll = time.time()
        return added

    def latest(self, n: int = 5) -> tuple[NewsItem, ...]:
        ordered = sorted(self._items, key=lambda i: i.published_at, reverse=True)
        return tuple(ordered[:n])

    def all(self) -> tuple[NewsItem, ...]:
        return tuple(sorted(self._items, key=lambda i: i.published_at, reverse=True))

    def seconds_since_poll(self) -> float:
        return float("inf") if self.last_poll == 0.0 else time.time() - self.last_poll


class OutcomeBuffer:
    """Last 20 signal outcomes used by the DRG reward learner."""

    def __init__(self, capacity: int = cfg.OUTCOME_BUFFER_SIZE) -> None:
        self.capacity = capacity
        self._rows: deque[tuple[float, float]] = deque(maxlen=capacity)

    def append(self, outcome: float, pnl_bps: float) -> None:
        self._rows.append((float(outcome), float(pnl_bps)))

    def array(self) -> np.ndarray:
        if not self._rows:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray(self._rows, dtype=np.float64)

    def win_rate(self, window: int = 20) -> float:
        arr = self.array()
        if arr.shape[0] == 0:
            return 0.0
        arr = arr[-window:]
        directional = arr[arr[:, 0] != 0]
        if directional.shape[0] == 0:
            return 0.0
        return float((directional[:, 0] > 0).mean())

    def __len__(self) -> int:
        return len(self._rows)
