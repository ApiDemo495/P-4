"""Layer 1: one vectorised pre-computation pass over the candle store.

Every formula kernel reads a ``Frame`` instead of raw rows, so the rolling
statistics the 2,100 formulas share are computed once per candle close.  The
frame also carries the cross-asset leg (the *other* asset's closes aligned by
minute), the venue leg (a second exchange's closes when a second feed is
delivering) and the keyed macro columns (DXY, social sentiment, exchange
flow) when their providers are configured.  Missing columns are NaN arrays
of the right length - a kernel never has to test for presence.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from backend.genesis import ops
from backend.genesis.candles import COLUMNS, OT_COLUMNS


@dataclass
class Frame:
    asset: str
    ts_ms: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    trades: np.ndarray
    buy_vol: np.ndarray
    sell_vol: np.ndarray
    bid_depth: np.ndarray
    ask_depth: np.ndarray
    spread_bps: np.ndarray
    imbalance: np.ndarray
    vwap: np.ndarray
    ret: np.ndarray          # log return close/close[-1]
    bar_ret: np.ndarray      # log(close/open)
    rng: np.ndarray          # log(high/low)
    vol_ret: np.ndarray      # log volume ratio
    other_close: np.ndarray  # cross asset (PAXG for BTC, BTC for PAXG)
    other_ret: np.ndarray
    venue_close: np.ndarray  # second exchange (NaN when only one feed delivers)
    macro: dict = field(default_factory=dict)  # name -> array (dxy, social, flow, ...)
    ot: dict = field(default_factory=dict)     # per-minute order-book transport columns
    books: list = field(default_factory=list)   # recent book snapshots (2, L, 2)
    tick_size: float = 0.01
    built_at: float = 0.0
    #: heavy (layer 3/4) kernels only evaluate the last ``eval_tail`` positions
    eval_tail: int = 700
    cache: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(len(self.close))

    # memoised rolling stats - kernels share windows heavily
    def roll(self, name: str, n: int) -> np.ndarray:
        key = (name, n)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        src = {"ret_mean": (ops.rmean, self.ret), "ret_std": (ops.rstd, self.ret),
               "ret_skew": (ops.rskew, self.ret), "ret_kurt": (ops.rkurt, self.ret),
               "vol_mean": (ops.rmean, self.volume), "vol_std": (ops.rstd, self.volume),
               "close_mean": (ops.rmean, self.close), "close_std": (ops.rstd, self.close),
               "rv": (lambda x, k: np.sqrt(ops.rsum(x * x, k)), self.ret),
               "hurst": (ops.hurst_rs, self.ret)}[name]
        out = src[0](src[1], n)
        self.cache[key] = out
        return out

    def col(self, name: str) -> np.ndarray:
        if name in self.macro:
            return self.macro[name]
        if name in self.ot:
            return self.ot[name]
        return getattr(self, name)


def build_frame(asset: str, rows: np.ndarray, other_rows: np.ndarray | None = None,
                venue_rows: np.ndarray | None = None, macro: dict | None = None,
                books: list | None = None, tick_size: float = 0.01) -> Frame:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != len(COLUMNS):
        rows = np.zeros((0, len(COLUMNS)))
    n = len(rows)
    cols = {name: rows[:, i].copy() if n else ops.nan(0) for i, name in enumerate(COLUMNS)}
    close = cols["close"]
    with np.errstate(all="ignore"):
        ret = ops.logret(close)
        bar_ret = np.log(close) - np.log(cols["open"])
        rng = np.log(cols["high"]) - np.log(cols["low"])
        vol = cols["volume"]
        vol_ret = np.log(vol + 1e-9) - np.log(ops.shift(vol, 1) + 1e-9)
    other_close = _align(cols["ts_ms"], other_rows)
    venue_close = _align(cols["ts_ms"], venue_rows)
    macro_cols = {}
    for name, arr in (macro or {}).items():
        a = np.asarray(arr, dtype=np.float64)
        macro_cols[name] = a if len(a) == n else _align(cols["ts_ms"], a if a.ndim == 2 else None)
    return Frame(asset=asset, ts_ms=cols["ts_ms"], open=cols["open"], high=cols["high"], low=cols["low"],
                 close=close, volume=vol, trades=cols["trades"], buy_vol=cols["buy_vol"],
                 sell_vol=cols["sell_vol"], bid_depth=cols["bid_depth"], ask_depth=cols["ask_depth"],
                 spread_bps=cols["spread_bps"], imbalance=cols["imbalance"], vwap=cols["vwap"],
                 ret=ret, bar_ret=bar_ret, rng=rng, vol_ret=vol_ret,
                 other_close=other_close, other_ret=ops.logret(other_close),
                 venue_close=venue_close, macro=macro_cols, ot={k: cols[k] for k in OT_COLUMNS},
                 books=list(books or []),
                 tick_size=tick_size, built_at=time.time())


def _align(ts_ms: np.ndarray, other_rows) -> np.ndarray:
    """Other series' close at or before each of our minutes (forward fill)."""
    n = len(ts_ms)
    if other_rows is None or len(other_rows) == 0 or n == 0:
        return ops.nan(n)
    other = np.asarray(other_rows, dtype=np.float64)
    o_ts = other[:, 0]
    o_close = other[:, 4] if other.shape[1] > 4 else other[:, 1]
    idx = np.searchsorted(o_ts, ts_ms, side="right") - 1
    out = ops.nan(n)
    ok = idx >= 0
    out[ok] = o_close[idx[ok]]
    # stale beyond 10 minutes counts as missing
    out[ok & ((ts_ms - o_ts[np.clip(idx, 0, len(o_ts) - 1)]) > 10 * 60_000)] = np.nan
    return out
