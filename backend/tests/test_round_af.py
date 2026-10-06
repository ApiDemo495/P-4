"""Round AF - hedge pair on a thin PAXG tape, graded &b, Gemini discovery."""
from __future__ import annotations

import time

import numpy as np

from backend.agents.gemini_agent import choose_model
from backend.data.cross_asset_sync import synchronise
from backend.formulas.engine import FEED_STATE_WEIGHT, provenance


def _ticks(now, step_s, span_s, base, wiggle):
    rows = [[(now - i) * 1000.0, base + wiggle(i), 0.1, 1.0] for i in range(span_s, 0, -step_s)]
    return np.array(rows, dtype=np.float64)


def test_sync_widens_for_a_thin_paxg_tape():
    now = time.time()
    btc = _ticks(now, 1, 900, 100000.0, lambda i: 20 * np.sin(i / 7.0))
    paxg = _ticks(now, 90, 900, 4000.0, lambda i: 0.5 * (i // 90))   # one print every 90 s
    s = synchronise(btc, paxg, now=now)
    assert s.valid and s.widened and s.window_seconds >= 300 and s.paxg_updates >= 6
    assert s.paxg_returns.std() > 0, "the thin leg must carry a covariance after widening"
    assert "widened" in s.reason
    # a dense pair stays on the 60 s spec grid
    dense = synchronise(btc, _ticks(now, 1, 900, 4000.0, lambda i: np.cos(i / 5.0)), now=now)
    assert dense.valid and not dense.widened and dense.window_seconds == 60


def test_sync_explains_an_empty_leg():
    now = time.time()
    btc = _ticks(now, 1, 120, 100000.0, lambda i: np.sin(i))
    s = synchronise(btc, np.zeros((0, 4)), now=now)
    assert not s.valid and "PAXG" in s.reason


class _Snap:
    def __init__(self, source, ticks, book, candles, synced, news=()):
        self.source, self._t, self._b, self._c, self.synced, self.news_items = source, ticks, book, candles, synced, news

    def ticks(self, asset):
        return self._t

    def book(self, asset):
        return self._b

    def candles(self, asset):
        return self._c


def test_provenance_is_graded_not_binary():
    now = time.time()
    btc = _ticks(now, 1, 120, 100000.0, lambda i: np.sin(i))
    synced = synchronise(btc, _ticks(now, 1, 120, 4000.0, lambda i: np.cos(i)), now=now)
    # CoinGecko: real prices, no order book, one candle so far -> partial, never offline
    p = provenance(_Snap("coingecko", btc, None, np.zeros(1), synced), "BTC")
    assert p["real"] and p["grade"] == "partial"
    assert p["feeds"]["tape"]["state"] == "live"
    assert p["feeds"]["book"]["state"] == "derived" and "order book" in p["feeds"]["book"]["note"]
    assert p["feeds"]["candles"]["state"] == "warming"
    assert 0 < p["coverage"] < 1
    # the simulator is labelled, not "offline"
    assert provenance(_Snap("simulator", btc, None, np.zeros(1), synced), "BTC")["grade"] == "simulated"
    # no source at all is the only "offline"
    assert provenance(_Snap("none", np.zeros((0, 4)), None, np.zeros(0), None), "BTC")["grade"] == "offline"
    assert set(FEED_STATE_WEIGHT) >= {"live", "partial"} - {"partial"}


def test_gemini_model_discovery():
    # newest generation first (3.8 > 3.7 > 3.5 > 3 > 2.5), flash before pro,
    # never an embedding / image / tts model
    pool = ["models/gemini-2.5-flash", "models/gemini-2.5-pro", "models/gemini-3-pro-preview",
            "models/gemini-3.5-flash", "models/gemini-3.5-pro", "models/gemini-3.7-flash",
            "models/gemini-3.8-flash-image", "models/gemini-embedding-001", "models/gemini-flash-latest"]
    assert choose_model(pool) == "gemini-3.7-flash"
    assert choose_model(pool + ["models/gemini-3.8-flash"]) == "gemini-3.8-flash"
    assert choose_model(["models/gemini-3.8-flash", "models/gemini-3.8-pro"]) == "gemini-3.8-flash"
    assert choose_model(["models/gemini-3.6-flash-preview", "models/gemini-3.6-flash"]) == "gemini-3.6-flash"
    assert choose_model(["models/gemini-2.0-flash", "models/gemini-2.5-flash"]) == "gemini-2.5-flash"
    assert choose_model(pool, pinned="gemini-3.5-pro") == "gemini-3.5-pro"
    assert choose_model(["models/gemini-9.1-flash"]) == "gemini-9.1-flash", "future generations need no code change"
    assert choose_model([]) == "gemini-3.8-flash"
