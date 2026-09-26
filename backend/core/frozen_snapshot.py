"""FrozenMarketSnapshot - the immutable market state for one signal cycle.

At t=0 of every cycle the data hub deep-copies its ring buffers into this
object.  The 22 formulas then read **only** from here, which is the mechanical
guarantee behind the Signal Lock Protocol: even though the WebSocket keeps
appending to the live buffers throughout the minute, nothing the formulas see
can change after the snapshot is taken.

Two fields are additions to the specification's list, both required to make the
specified formulas computable:

``{asset}_spreads``
    Formula 3 (SED) is defined on "L2 snapshots interpolated to tick
    timestamps".  Keeping only the current and previous book (which is all BAR
    needs) makes that interpolation degenerate, because the only spread change
    available is the single step between the two snapshots.  A rolling
    (timestamp, spread) history - still sourced exclusively from L2 updates -
    makes SED meaningful.

``synced``
    The 1-second LOCF BTC/PAXG grid from Section 5.3, pre-computed once per
    cycle so the four hedge formulas share exactly the same alignment.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class FrozenMarketSnapshot(NamedTuple):
    """Immutable market data taken at cycle start."""

    timestamp: float
    btc_ticks: Any  # (600, 4): [time_ms, price, qty, side]
    paxg_ticks: Any  # (600, 4)
    btc_book: Any  # (2, 20, 2): [[bids], [asks]] x [price, qty]
    paxg_book: Any
    btc_book_prev: Any  # previous L2 snapshot, for BAR
    paxg_book_prev: Any
    btc_candles: Any  # last 60 1-minute closes
    paxg_candles: Any
    news_items: tuple
    drg_outcomes: Any  # (20, 2): [outcome, pnl_bps]
    btc_tick_count: int
    paxg_tick_count: int

    # -- extensions (see module docstring) ---------------------------------
    btc_spreads: Any = None  # (N, 2): [time_ms, spread]
    paxg_spreads: Any = None
    synced: Any = None
    warnings: tuple = ()
    source: str = ""

    # ------------------------------------------------------------------
    # Accessors - the formulas never index raw fields directly, so the BTC /
    # PAXG duality is handled in exactly one place.
    # ------------------------------------------------------------------
    def _pick(self, asset: str, btc_value, paxg_value):
        return btc_value if str(asset).upper() == "BTC" else paxg_value

    def ticks(self, asset: str):
        import numpy as np

        value = self._pick(asset, self.btc_ticks, self.paxg_ticks)
        return value if value is not None else np.zeros((0, 4))

    def prices(self, asset: str):
        ticks = self.ticks(asset)
        return ticks[:, 1] if ticks.size else ticks.reshape(-1)

    def book(self, asset: str):
        return self._pick(asset, self.btc_book, self.paxg_book)

    def book_prev(self, asset: str):
        return self._pick(asset, self.btc_book_prev, self.paxg_book_prev)

    def candles(self, asset: str):
        value = self._pick(asset, self.btc_candles, self.paxg_candles)
        import numpy as np

        return value if value is not None else np.zeros(0)

    def spread_history(self, asset: str):
        import numpy as np

        value = self._pick(asset, self.btc_spreads, self.paxg_spreads)
        return value if value is not None else np.zeros((0, 2))

    def tick_count(self, asset: str) -> int:
        return int(self._pick(asset, self.btc_tick_count, self.paxg_tick_count))

    def last_price(self, asset: str) -> float:
        """Mid from the frozen book, falling back to the last tick."""
        book = self.book(asset)
        try:
            if book is not None and book[0, 0, 0] > 0 and book[1, 0, 0] > 0:
                return float((book[0, 0, 0] + book[1, 0, 0]) / 2.0)
        except (IndexError, TypeError):
            pass
        ticks = self.ticks(asset)
        return float(ticks[-1, 1]) if ticks.size else 0.0

    # ------------------------------------------------------------------
    def age_seconds(self, now: float | None = None) -> float:
        import time

        return max(0.0, (now or time.time()) - self.timestamp)

    def summary(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "btc_tick_count": self.btc_tick_count,
            "paxg_tick_count": self.paxg_tick_count,
            "btc_price": round(self.last_price("BTC"), 2),
            "paxg_price": round(self.last_price("PAXG"), 2),
            "news_items": len(self.news_items or ()),
            "outcomes": int(self.drg_outcomes.shape[0]) if self.drg_outcomes is not None else 0,
            "warnings": list(self.warnings),
        }
