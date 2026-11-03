"""Round AE - a real wire must not read 0.00, and the hub must never sit on
"none" while a real feed is delivering."""
from __future__ import annotations

import asyncio
import time


from backend.core import config as cfg
from backend.data.market_hub import MarketDataHub
from backend.news.sentiment_lexicon import score_headline

WORLD_WIRE = [
    "Israel strikes Gaza as ceasefire talks stall",
    "Trump announces new tariffs on Chinese goods",
    "Ukraine says Russian drone attack hits Kyiv power grid",
    "Stocks fall as Treasury yields climb",
    "Oil prices surge after attack on tanker in Red Sea",
]


def test_world_headlines_carry_sentiment():
    scores = [score_headline(h) for h in WORLD_WIRE]
    assert all(s != 0.0 for s in scores), scores
    assert all(s < 0 for s in scores), "every one of these is risk-off for BTC"
    assert score_headline("Israel and Hamas agree ceasefire deal") > 0
    assert score_headline("Fed holds rates steady, signals caution") == 0.0


def test_hub_adopts_a_delivering_real_feed(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_MODE", "auto")
    hub = MarketDataHub(cfg.Settings())
    hub.active_source = "none"
    hub._started_at = time.time()          # still inside the grace period

    async def run():
        # Kraken REST delivers while both sockets are silent: the write is
        # rejected (not the active source) but the delivery is remembered...
        await hub.ingest("krakenrest", "BTC", [(time.time() * 1000, 100000.0, 0.1, 1.0)], None)
        assert hub.rejected.get("krakenrest") == 1
        assert "krakenrest" in hub.last_delivery
        # ...and the very next reconcile adopts it instead of waiting on "none".
        await hub._reconcile_source()
        return hub.active_source

    assert asyncio.run(run()) == "krakenrest"
    assert "last_delivery_seconds_ago" in hub.feeds_report()


def test_diagnose_shape():
    from backend.data import connectivity

    report = asyncio.run(connectivity.diagnose(None))
    assert {"http", "websocket", "rss", "verdict"} <= set(report)
    assert {"kraken_ws", "binance_ws"} == set(report["websocket"])
    assert isinstance(report["verdict"], str) and report["verdict"]
