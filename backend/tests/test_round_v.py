"""Round V - the stops move, the dopamine gate can say "loss", the brain is
sign-neutral on a flat tape, and every feed builds its own minute candles."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

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
    assert loud["tp_bps"] == pytest.approx(loud["sl_bps"] * cfg.SETTINGS.rr_target, rel=1e-3)


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


def test_critical_keywords_need_whole_words_and_a_market_context():
    import time as _t
    from backend.news import sentiment_lexicon as lex
    from backend.news.critical_event_detector import CriticalEventDetector

    assert lex.critical_keywords_in("Fed warns of software award forwarded") == []
    assert lex.critical_keywords_in("Exchange hacked: $200M drained") == ["hack"]
    det = CriticalEventDetector()
    # Tier-1 politics story with "war": not a market emergency.
    assert det.check_headline("Trade war rhetoric heats up before the vote", "Reuters") is None
    # Hours-old market story: a backlog item, not breaking news.
    assert det.check_headline("Binance exchange hack confirmed", "Reuters", published_at=_t.time() - 7200) is None
    # Fresh, Tier 1, market-relevant: fires exactly once.
    assert det.check_headline("Binance exchange hack confirmed", "Reuters", published_at=_t.time()) is not None
    assert det.check_headline("Binance exchange hack confirmed", "Reuters", published_at=_t.time()) is None


def test_emergency_override_keeps_the_window_and_is_stamped():
    from backend.core.signal_lock import SignalLockController, FrozenSignal

    lock = SignalLockController()
    base = FrozenSignal(cycle_number=9, timestamp="t", asset="BTC", signal="SELL", confidence=0.25,
                        reasoning="", formula_values=(), agent_results=(), is_emergency_override=False, computed_at="c",
                        valid_from="vf", valid_until="vu", window_seconds=60)
    lock.current_signal = base
    over = lock.emergency_override({"headline": "Exchange hack"}, duration_seconds=30, timestamp="2026-10-02T10:19:10Z")
    assert over.signal == "BUY" and over.is_emergency_override
    assert over.computed_at == "2026-10-02T10:19:10Z"
    assert (over.valid_from, over.valid_until, over.window_seconds) == ("vf", "vu", 60)
    assert over.to_dict()["lock_icon"] == "\u26a1"


def test_stall_defences_exist_on_every_layer():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    loop = (root / "backend/core/cycle_manager.py").read_text()
    assert "index += 1" in loop.split("Never spin on a boundary")[1][:900]       # never spin on a past boundary
    assert '"stalled": stalled' in loop
    run = (root / "run.sh").read_text()
    assert '"stalled":[[:space:]]*true' in run and "engine unresponsive" in run
    js = (root / "backend/web/app.js").read_text()
    assert "silentMs > 15000" in js and "state.lastMessageAt = Date.now()" in js
    dart = (root / "frontend/lib/services/signal_socket.dart").read_text()
    assert "SOCKET_SILENT" in dart and "silenceLimit" in dart
    auto = (root / "tools/codespace_autostart.sh").read_text()
    assert "nice -n 19" in auto


# ---------------------------------------------------------------------------
# Round Y: one logic - history ramps in, asymmetric risk, probability branches
# ---------------------------------------------------------------------------
def test_round_y_probability_branches_are_one_model_with_the_confidence():
    from backend.core.prediction import branches

    risk = {"entry": 68_000.0, "volatility_bps": 12.0, "tp_bps": 27.0, "sl_bps": 18.0}
    for side, conf in (("BUY", 0.62), ("SELL", 0.71)):
        b = branches(risk, side, conf, 60.0)
        assert b["available"] is True
        # The fan's drift is chosen so P(close on the called side) IS the confidence.
        assert b["p_close_for"] == pytest.approx(0.5 + 0.5 * conf, abs=2e-3)
        assert 0.0 <= b["p_tp_first"] <= 1.0
        assert len(b["fan"]) == 13 and b["fan"][0]["q50"] == pytest.approx(68_000.0)
        last = b["fan"][-1]
        assert last["q5"] < last["q25"] < last["q50"] < last["q75"] < last["q95"]
        if side == "BUY":
            assert last["q50"] > 68_000.0
        else:
            assert last["q50"] < 68_000.0
        assert sum(e["p"] for e in b["branches"]) == pytest.approx(1.0, abs=1e-3)
        # Deterministic (no sampling): byte-stable within a cycle.
        assert branches(risk, side, conf, 60.0) == b
    assert branches({}, "BUY", 0.6, 60.0)["available"] is False


def test_round_y_risk_is_asymmetric_by_rule():
    block = risk_levels("BTC", "BUY", 60_000.0, 12.0, cfg.SETTINGS)
    assert cfg.SETTINGS.rr_target == pytest.approx(1.5)
    assert block["tp_bps"] == pytest.approx(block["sl_bps"] * 1.5)
    assert "target = 1.50 x stop" in block["note"]


def test_round_y_dashboard_draws_the_branches_and_warns_on_stale_builds():
    root = Path(__file__).resolve().parents[2]
    html = (root / "backend/web/index.html").read_text()
    js = (root / "backend/web/app.js").read_text()
    assert 'id="w-branches"' in html and 'id="synesthesia-toggle"' in html
    assert "function renderBranches" in js and "recordPath(msg.data.live_price" in js
    assert "STALE BUILD" in js and 'classList.add("blocked")' in js
    # Still exactly one client timer (Round H/X rule).
    assert js.count("setInterval(") == 1
