"""Round Z: one tape, one source - the fresh-Codespace root cause.

In a fresh Codespace the engine starts, the supervisor sees Binance "not yet
connected" two seconds later and starts the built-in simulator; Binance then
connects and BOTH feeds write into the same buffers for ever.  A tape that
alternates between a seeded simulator (~65k) and the real market is pure
noise: realised volatility pins the TP/SL at their ceilings every window
(identical levels), every formula reads garbage, and - because the simulator
is seeded - every new Codespace fails the same way.  These tests pin the fix:
a feed may only write while it is the active source, a switch between a
simulated and a real tape flushes the buffers, and the simulator is stopped
once a real feed is healthy.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest

from backend.core import config as cfg
from backend.data.market_hub import MarketDataHub


def _book(mid: float) -> np.ndarray:
    levels = cfg.L2_DEPTH_LEVELS
    bids = np.column_stack([mid - np.arange(1, levels + 1) * 0.5, np.ones(levels)])
    asks = np.column_stack([mid + np.arange(1, levels + 1) * 0.5, np.ones(levels)])
    return np.stack([bids, asks])


def _run(coro):
    return asyncio.run(coro)


def test_a_feed_that_is_not_the_active_source_cannot_write_the_tape():
    hub = MarketDataHub(cfg.Settings())
    hub.active_source = "simulator"
    now_ms = time.time() * 1000.0
    _run(hub.ingest("simulator", "BTC", [(now_ms, 65_000.0, 1.0, 1.0)], _book(65_000.0)))
    _run(hub.ingest("binance", "BTC", [(now_ms + 1, 110_000.0, 1.0, 1.0)], _book(110_000.0)))
    prices = hub.buffers["BTC"].ticks.prices()
    assert prices.tolist() == [65_000.0], "the real feed wrote while the simulator was active"
    assert hub.last_price("BTC") == pytest.approx(65_000.0)
    assert hub.rejected["binance"] >= 1


def test_switching_from_a_simulated_tape_to_a_real_one_flushes_everything():
    hub = MarketDataHub(cfg.Settings())
    hub.active_source = "simulator"
    now_ms = time.time() * 1000.0
    for i in range(50):
        _run(hub.ingest("simulator", "BTC", [(now_ms + i, 65_000.0 + i, 1.0, 1.0)], None))
    _run(hub.ingest("simulator", "BTC", None, _book(65_000.0)))
    hub.buffers["BTC"].candles.push(65_000.0)
    hub.set_source("binance")
    assert hub.buffers["BTC"].ticks.size == 0
    assert len(hub.buffers["BTC"].candles) == 0
    assert not hub.buffers["BTC"].book.is_valid()
    assert hub.tape_is_simulated is False
    _run(hub.ingest("binance", "BTC", [(now_ms + 100, 110_000.0, 1.0, 1.0)], _book(110_000.0)))
    assert hub.buffers["BTC"].ticks.prices().tolist() == [110_000.0]
    # A switch between two exchanges flushes too (USDT vs USD basis would
    # otherwise print as a fake volatility jump).
    hub.set_source("kraken")
    assert hub.buffers["BTC"].ticks.size == 0


def test_the_simulator_is_a_last_resort_and_is_stopped_when_a_real_feed_is_healthy(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_MODE", "auto")  # the Codespace default (conftest forces simulator)
    hub = MarketDataHub(cfg.Settings())

    class FakeStatus:
        connected = False
        consecutive_failures = 0
        last_message_ts = 0.0
        data_messages = 0

        def stale(self):
            return not self.connected

        def healthy(self):
            return self.connected and self.data_messages > 0

    class FakeBinance:
        status = FakeStatus()

    hub.binance = FakeBinance()  # type: ignore[assignment]
    from backend.data.simulator import MarketSimulator
    hub.simulator = MarketSimulator(seed=1)
    started: list[str] = []
    stopped: list[str] = []
    hub._ensure_simulator_task = lambda: started.append("sim")  # type: ignore[method-assign]
    hub._stop_simulator_task = lambda: stopped.append("sim")  # type: ignore[method-assign]
    hub._started_at = time.time()
    # t+0: Binance not connected yet -> NO simulator (it gets a grace period).
    _run(hub._reconcile_source())
    assert started == [] and hub.active_source in ("none", "binance")
    # Grace period over, Binance still down -> simulator, honestly labelled.
    hub._started_at = time.time() - cfg.Settings().real_feed_grace_seconds - 1
    _run(hub._reconcile_source())
    assert started == ["sim"] and hub.active_source == "simulator" and hub.tape_is_simulated
    # Binance handshake alone is NOT health (Round AB: a silent stream stays down)
    FakeBinance.status.connected = True
    FakeBinance.status.last_message_ts = time.time()
    _run(hub._reconcile_source())
    assert stopped == [] and hub.active_source == "simulator"
    # ...real data arrives -> simulator stopped, tape flushed, source = binance.
    FakeBinance.status.data_messages = 5
    _run(hub._reconcile_source())
    assert stopped == ["sim"] and hub.active_source == "binance" and not hub.tape_is_simulated


def test_market_status_says_whether_the_tape_is_simulated():
    hub = MarketDataHub(cfg.Settings())
    hub.active_source = "simulator"
    assert hub.status().extra["tape_is_simulated"] is True
    hub.set_source("binance")
    assert hub.status().extra["tape_is_simulated"] is False
    assert "rejected_writes" in hub.status().extra


def test_kraken_feed_produces_the_same_shapes_as_binance():
    from backend.data.kraken_ws import KrakenWebSocket, _LocalBook

    tick = KrakenWebSocket.parse_trade({"symbol": "BTC/USD", "side": "sell", "price": "110000.5", "qty": "0.02",
                                        "timestamp": "2026-10-04T12:00:00.123456Z"})
    assert tick is not None and tick[1] == 110000.5 and tick[3] == -1.0 and tick[0] > 1.7e12
    book = _LocalBook()
    book.apply("bids", [{"price": 110000.0 - i, "qty": 1.0} for i in range(30)], snapshot=True)
    book.apply("asks", [{"price": 110001.0 + i, "qty": 1.0} for i in range(30)], snapshot=True)
    arr = book.array()
    assert arr is not None and arr.shape == (2, cfg.L2_DEPTH_LEVELS, 2)
    assert arr[0, 0, 0] == 110000.0 and arr[1, 0, 0] == 110001.0
    # a delta with qty 0 removes the level
    book.apply("asks", [{"price": 110001.0, "qty": 0}], snapshot=False)
    assert book.array()[1, 0, 0] == 110002.0


def test_the_hub_knows_kraken_as_a_real_source(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_MODE", "auto")
    hub = MarketDataHub(cfg.Settings())
    hub.active_source = "kraken"
    assert hub.tape_is_simulated is False
    assert "Kraken" in hub.status().detail


def test_news_impact_is_signed_per_asset_and_world_news_counts():
    from backend.news import impact

    war = impact.classify("Missile strikes hit capital as war escalates")
    assert war["theme"] == "war" and war["btc"] < 0 < war["paxg"] and war["scope"] == "world"
    cut = impact.classify("Fed signals rate cut in December")
    assert cut["theme"] == "dovish" and cut["btc"] > 0 and cut["paxg"] > 0
    etf = impact.classify("Spot ETF approval drives record inflows")
    assert etf["theme"] == "adoption" and etf["btc"] > 0
    assert impact.classify("Local bakery opens second shop")["theme"] == "neutral"

    class Item:
        def __init__(self, h, tier=1, age=60.0):
            self.headline, self.tier, self.published_at = h, tier, time.time() - age

    agg = impact.aggregate([Item("War escalates after overnight airstrikes"), Item("Gold price hits record on safe-haven demand", 2)])
    assert agg["BTC"] < 0 < agg["PAXG"] and agg["world_items"] == 2 and agg["drivers"]["PAXG"]
    # an hour-old headline weighs less than a fresh one
    fresh = impact.aggregate([Item("Exchange hacked, funds drained")])["BTC"]
    old = impact.aggregate([Item("Exchange hacked, funds drained", age=3 * 3600)])["BTC"]
    assert fresh < old < 0


def test_fusion_lets_the_news_wire_vote_per_asset():
    from backend.agents import fusion

    news = {"BTC": -0.6, "PAXG": 0.5, "weight": 2.0,
            "drivers": {"BTC": [{"theme": "war", "impact": -0.6}], "PAXG": [{"theme": "war", "impact": 0.5}]}}
    btc = fusion.fuse(agents={}, ccs_value=0.0, ccs_confidence=0.5, hsi=0.1, settings=cfg.SETTINGS,
                      news_impact=news, asset="BTC")
    paxg = fusion.fuse(agents={}, ccs_value=0.0, ccs_confidence=0.5, hsi=0.1, settings=cfg.SETTINGS,
                       news_impact=news, asset="PAXG")
    assert btc.contributions["news"]["value"] < 0 < paxg.contributions["news"]["value"]
    assert "news impact" in btc.reasoning
    quiet = fusion.fuse(agents={}, ccs_value=0.0, ccs_confidence=0.5, hsi=0.1, settings=cfg.SETTINGS,
                        news_impact={"BTC": 0.0, "weight": 0.0}, asset="BTC")
    assert "news" not in quiet.contributions


def test_world_feeds_are_part_of_the_default_wire(monkeypatch):
    monkeypatch.delenv("RSS_FEEDS", raising=False)
    feeds = " ".join(cfg.Settings().rss_feeds)
    example = (Path(__file__).resolve().parents[2] / ".env.example").read_text()
    assert "bbci" in example and "federalreserve" in example
    assert "bbci" in feeds and "federalreserve" in feeds and "news.google.com" in feeds


def test_geometry_layer_measures_real_things():
    from backend.core import geometry

    rng = np.random.default_rng(7)
    # A regular book has no cavities; pulling the levels 3..8 behind the top opens one.
    levels = cfg.L2_DEPTH_LEVELS
    bids = np.column_stack([100.0 - np.arange(1, levels + 1) * 0.1, np.ones(levels)])
    asks = np.column_stack([100.0 + np.arange(1, levels + 1) * 0.1, np.ones(levels)])
    regular = geometry.book_topology(np.stack([bids, asks]))
    torn_bids = bids.copy(); torn_bids[2:9, 1] = 0.0
    torn = geometry.book_topology(np.stack([torn_bids, asks]))
    assert regular["available"] and regular["cavities"] == 0 and regular["tearing"] == 0.0
    assert torn["cavities"] >= 1 and torn["tearing"] > regular["tearing"] and torn["max_persistence"] >= 3
    # Takens/Lyapunov: a logistic map in its chaotic regime diverges, a sine does not.
    x = [0.3]
    for _ in range(300):
        x.append(3.99 * x[-1] * (1 - x[-1]))
    chaotic = geometry.takens_lyapunov(np.asarray(x))
    periodic = geometry.takens_lyapunov(np.sin(np.linspace(0, 40, 300)))
    assert chaotic["available"] and periodic["available"]
    assert chaotic["lyapunov"] > periodic["lyapunov"]
    # Critical slowing down: an AR(1) whose memory and variance both rise.
    calm = rng.normal(0, 1, 30)
    ar = [0.0]
    for _ in range(30):
        ar.append(0.8 * ar[-1] + rng.normal(0, 2.0))
    csd = geometry.critical_slowing_down(np.concatenate([calm, ar[1:]]))
    assert csd["available"] and csd["csd"] > 0.3 and csd["variance_ratio"] > 1.1
    # Entropy production: BTC sold, PAXG bought, BTC falling vs PAXG -> heat BTC->PAXG.
    t = np.arange(200, dtype=float) * 200.0
    btc = np.column_stack([t, 100.0 - np.linspace(0, 0.5, 200), np.ones(200), -np.ones(200)])
    paxg = np.column_stack([t, 50.0 + np.linspace(0, 0.1, 200), np.ones(200), np.ones(200)])
    ent = geometry.entropy_production(btc, paxg)
    assert ent["available"] and ent["flux_J"] < 0 and "BTC→PAXG" in ent["heat_direction"] and ent["sigma_raw"] > 0
    # Quantum interference: a classical (independent) tape has ~0 interference.
    sides = rng.choice([-1.0, 1.0], 2000)
    px = 100 + np.cumsum(rng.normal(0, 0.01, 2000))
    qi = geometry.quantum_interference(np.column_stack([np.arange(2000.0), px, np.ones(2000), sides]))
    assert qi["available"] and abs(qi["interference"]) < 0.1


def test_geometry_is_part_of_the_deep_emotion_evidence():
    from backend.core import deep_micro

    for name in ("tearing", "chaotic", "csd", "cooling", "heating", "polarised"):
        assert name in deep_micro.EVIDENCE_LABEL
        assert any(name in table for table in deep_micro.LIKELIHOOD.values())


def test_kraken_rest_is_a_real_source_with_a_book(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_MODE", "auto")
    from backend.formulas.engine import REAL_SOURCES
    assert "krakenrest" in REAL_SOURCES
    hub = MarketDataHub(cfg.Settings())
    hub.active_source = "krakenrest"
    assert not hub.tape_is_simulated and "HTTPS" in hub.status().detail
    report = hub.feeds_report()
    assert set(report["feeds"]) >= {"binance", "kraken", "krakenrest", "coingecko", "simulator"}
    from backend.data.binance_ws import BinanceWebSocket
    hub.binance = BinanceWebSocket(lambda *a: None, None, cfg.Settings())
    hub.binance.status.last_error = "InvalidStatus: HTTP 451"
    assert hub.feeds_report()["feeds"]["binance"]["last_error"] == "InvalidStatus: HTTP 451"
