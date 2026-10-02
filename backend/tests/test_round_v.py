"""Round V - the stops move, the dopamine gate can say "loss", the brain is
sign-neutral on a flat tape, and every feed builds its own minute candles."""

from __future__ import annotations

import asyncio

import numpy as np

from backend.core import config as cfg
from backend.core.frozen_snapshot import FrozenMarketSnapshot
from backend.core.risk import realized_volatility_bps, risk_levels
from backend.data.binance_ws import BinanceWebSocket
from backend.data.market_hub import MarketDataHub
from backend.formulas import drg
from backend.formulas.category_h_brain import ccsv2


def _tape(sigma_bps_per_minute: float, seconds: int = 180, rate: int = 10) -> np.ndarray:
    """A random-walk tape with a known one-minute volatility."""
    n = seconds * rate
    rng = np.random.default_rng(7)
    t = np.arange(n) * (1000.0 / rate)
    rets = rng.normal(0.0, sigma_bps_per_minute / 1e4 / np.sqrt(60 * rate), size=n)
    px = 60_000.0 * np.exp(np.cumsum(rets))
    return np.column_stack([t, px, np.ones(n), np.ones(n)])


def _snapshot(ticks: np.ndarray, candles=None) -> FrozenMarketSnapshot:
    book = np.zeros((2, 20, 2))
    return FrozenMarketSnapshot(
        timestamp=0.0, btc_ticks=ticks, paxg_ticks=ticks, btc_book=book, paxg_book=book,
        btc_book_prev=book, paxg_book_prev=book,
        btc_candles=np.asarray(candles if candles is not None else []), paxg_candles=np.zeros(0),
        news_items=(), drg_outcomes=np.zeros((0, 2)), btc_tick_count=len(ticks), paxg_tick_count=len(ticks),
    )


def test_realised_volatility_tracks_the_tape_not_the_tick_count():
    # Ticks are not time: the old sigma_tick * sqrt(horizon / span) read a
    # ~60 bps/min tape as ~2 bps and parked every stop on the 4 bps floor.
    calm = realized_volatility_bps(_snapshot(_tape(6.0)), "BTC")
    wild = realized_volatility_bps(_snapshot(_tape(60.0)), "BTC")
    assert 3.0 < calm < 12.0
    assert 30.0 < wild < 120.0
    assert wild > 5 * calm


def test_take_profit_and_stop_move_with_volatility():
    quiet = risk_levels("BTC", "BUY", 60_000.0, 6.0, cfg.SETTINGS)
    loud = risk_levels("BTC", "BUY", 60_000.0, 40.0, cfg.SETTINGS)
    assert loud["tp_bps"] > quiet["tp_bps"] >= cfg.SETTINGS.min_tp_bps
    assert loud["take_profit"] - 60_000.0 > quiet["take_profit"] - 60_000.0
    assert loud["tp_bps"] == loud["sl_bps"]  # the 1:1 rule is untouched


def test_drg_turns_negative_after_a_loss():
    # outcome -1 with a signed P&L of -40 bps is a loss, not "+40 x +1".
    wins = np.array([[1.0, 30.0]] * 6)
    loss = np.vstack([wins, [[-1.0, -40.0]]])
    state = drg.State()
    assert drg.compute(_snapshot(_tape(6.0))._replace(drg_outcomes=wins), state) > 0
    assert drg.compute(_snapshot(_tape(6.0))._replace(drg_outcomes=loss), state) < 0


def test_non_directional_formulas_do_not_push_the_brain():
    ctx = {name: 0.0 for name in ccsv2.FORMULA_ORDER}
    ctx["HSI"], ctx["ERC"] = 0.9, 0.9
    assert not ccsv2._vector_from_ctx(ctx).any()
    ctx["SED"] = -0.5
    assert ccsv2._vector_from_ctx(ctx).sum() == -0.5


def test_every_feed_builds_minute_candles_from_ticks():
    hub = MarketDataHub(cfg.SETTINGS)
    hub.active_source = "coingecko"
    buf = hub.buffers["BTC"]
    before = len(buf.candles)
    rows = [(m * 60_000.0 + 500.0, 100.0 + m, 1.0, 1.0) for m in range(5)]
    asyncio.run(hub._on_data("BTC", rows, None))
    assert len(buf.candles) == before + 4
    assert buf.candles.closes()[-1] == 103.0
    # Binance klines take the buffer back.
    asyncio.run(hub._on_candle("BTC", 250.0))
    assert buf.candles_from_ticks is False


def test_binance_rotates_to_the_public_market_data_host():
    ws = BinanceWebSocket(None, None, cfg.SETTINGS)
    assert ws.url.startswith(cfg.SETTINGS.binance_ws_base)
    ws.status.consecutive_failures = 1
    assert ws.url.startswith("wss://data-stream.binance.vision")


def test_the_brain_reads_every_formula_with_positive_as_approach():
    from backend.formulas import self_test
    from backend.brain import graph_convolution as gc

    brain = self_test._brain_stub()
    for i, name in enumerate(gc.PN_NAMES):
        if name in ccsv2.NON_DIRECTIONAL_PN:
            continue
        ctx = {n: 0.0 for n in gc.PN_NAMES}
        ctx.update({name: 0.7, "_brain": brain, "_drg": 0.3, "HSI": 0.3})
        up = ccsv2.compute(None, "BTC", ccsv2.State(), {}, ctx)
        ctx[name] = -0.7
        down = ccsv2.compute(None, "BTC", ccsv2.State(), {}, ctx)
        assert up > 0 > down, name                       # polarity (22-E)
        assert abs(up + down) < 1e-9, name               # odd read-out (22-D)


def test_stale_news_decays_instead_of_holding_its_sentiment():
    from backend.formulas.category_g_news import niv
    from backend.formulas.synthetic import NewsItem

    fresh = _snapshot(_tape(6.0))._replace(
        timestamp=1_700_000_000.0, news_items=(NewsItem("ETF inflows", sentiment=0.72),)
    )
    stale = fresh._replace(timestamp=1_700_000_000.0 + 1500.0)
    now_v = niv.compute(fresh, "BTC", niv.State(), {})
    old_v = niv.compute(stale, "BTC", niv.State(), {})
    assert 0.4 < now_v < 0.72
    assert 0.0 < old_v < 0.03
