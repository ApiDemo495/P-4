"""Round AM - flat windows and the live edge guard.

An 8 % "win rate" in an hour cannot come from a coin-flip scorer; it comes
from (a) windows that never moved being scored as losses, or (b) a side that
is genuinely anti-correlated with the tape.  (a) is now FLAT, (b) is now
inverted by the edge guard - and both are visible.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from backend.agents import fusion
from backend.core import config as cfg
from backend.core import edge_guard
from backend.core.cycle_manager import CycleManager

ROOT = Path(__file__).resolve().parents[2]


def test_wilson_interval_and_guard_thresholds():
    lo, hi = edge_guard.wilson(4, 50)
    assert 0.02 < lo < 0.05 and 0.15 < hi < 0.2
    # 1 win of 12 decided -> upper bound far below 50 %: invert
    g = edge_guard.assess([-1] * 11 + [1], False)
    assert g.inverted and g.decided == 12 and g.upper < 0.5 and "INVERTED" in g.note
    # 5 of 12 -> coin-flip territory: watch, do not invert
    g = edge_guard.assess([-1] * 7 + [1] * 5, False)
    assert not g.inverted and "watching" in g.note
    # not enough decided windows -> arming
    assert not edge_guard.assess([-1] * 11, False).inverted
    # hysteresis: once inverted, stays on while the raw side is below 50 %
    g = edge_guard.assess([-1] * 7 + [1] * 5, True)
    assert g.inverted
    g = edge_guard.assess([-1] * 6 + [1] * 6, True)
    assert not g.inverted and "released" in g.note


def test_fusion_publishes_the_opposite_side_when_the_guard_is_inverted():
    settings = cfg.Settings()
    base = dict(agents={}, ccs_value=0.6, ccs_confidence=0.7, hsi=0.0, settings=settings)
    plain = fusion.fuse(**base)
    inverted = fusion.fuse(**base, edge_guard={"inverted": True, "note": "INVERTED test"})
    assert plain.decision == "BUY"
    assert inverted.decision == "SELL"
    assert inverted.edge_guard["inverted"] and inverted.edge_guard["raw_side"] == "BUY"
    assert "INVERTED" in inverted.reasoning
    assert inverted.to_dict()["edge_guard"]["raw_side"] == "BUY"


class _Sig:
    def __init__(self, side, price, cycle):
        self.signal, self.price, self.cycle_number, self.asset = side, price, cycle, "BTC"


def _manager():
    m = CycleManager(cfg.Settings(), local_stub=True)
    m.settings.outcome_horizon_seconds = 0.01
    m.settings.time_scale = 1.0
    return m


def test_flat_windows_are_not_losses_and_the_guard_scores_the_raw_side(monkeypatch):
    m = _manager()
    prices = iter([100.0, 100.0, 100.0, 99.0, 101.0, 99.0, 99.0])

    async def price_at(asset, fallback):
        return next(prices)
    monkeypatch.setattr(m, "_price_at", price_at)
    monkeypatch.setattr(m, "_trading_cost_bps", lambda asset: 2.0)

    async def noop(*a, **k):
        return None
    monkeypatch.setattr(m, "broadcast", noop)
    monkeypatch.setattr(m.settings.__class__, "outcome_horizon", property(lambda self: 0.0))

    async def run():
        await m._evaluate_outcome(_Sig("BUY", 100.0, 1), None)      # exit == entry -> FLAT
        await m._evaluate_outcome(_Sig("SELL", 100.0, 2), None)     # exit == entry -> FLAT
        await m._evaluate_outcome(_Sig("BUY", 100.01, 3), None)     # 1 bp inside a 2 bp spread -> FLAT
        await m._evaluate_outcome(_Sig("BUY", 100.0, 4), None)      # -100 bp -> LOSS
        await m._evaluate_outcome(_Sig("BUY", 100.0, 5), None)      # +100 bp -> WIN
        m._inverted_cycles.add(6)
        await m._evaluate_outcome(_Sig("SELL", 100.0, 6), None)     # published SELL wins -> raw BUY lost
        await m._evaluate_outcome(_Sig("BUY", 100.0, 7), None)      # LOSS
    asyncio.run(run())
    acc = m.accuracy_block()
    assert acc["flat"] == 3 and acc["decided"] == 4 and acc["evaluated"] == 7
    assert acc["win_rate"] == pytest.approx(2 / 4)
    assert list(m._raw_outcomes) == [-1, 1, -1, -1]
    assert "edge_guard" in acc and acc["edge_guard"]["decided"] == 4
    payload = m.outcomes_payload()
    assert payload["flat"] == 3 and payload["decided"] == 4


def test_dashboard_shows_decided_flat_and_the_guard():
    html = (ROOT / "backend/web/index.html").read_text(encoding="utf-8")
    js = (ROOT / "backend/web/app.js").read_text(encoding="utf-8")
    assert 'id="w-guard"' in html and "flat (inside spread)" in html
    assert "decidedRows" in js and "INVERTED" in js


# ---------------------------------------------------------------- Round AN
def test_levels_are_bounded_by_the_measured_minute_moves():
    import numpy as np
    from backend.core.risk import minute_move_quantiles, risk_levels

    class Snap:
        def __init__(self, t, p):
            self._t, self._p = t, p

        def ticks(self, asset):
            return np.column_stack([self._t * 1000, self._p])

        def candles(self, asset):
            return np.zeros(0)

    rng = np.random.default_rng(0)
    t = np.arange(0, 1800, 0.5)
    p = 100000 * np.exp(np.cumsum(rng.normal(0, 0.4e-4, t.size)))     # ~4 bps per minute
    moves = minute_move_quantiles(Snap(t, p), "BTC")
    assert moves["n"] > 100 and 1 < moves["q50"] < moves["q80"] < moves["q95"] < 15
    settings = cfg.Settings()
    # a wildly inflated sigma (one bad print) no longer produces a 40/60 bps plan
    bad = risk_levels("BTC", "BUY", 100000.0, 45.0, settings, horizon_seconds=60, edge=0.5, spread_bps=0.1, moves=moves)
    assert bad["sl_bps"] <= moves["mae80"] + 1e-6 and bad["tp_bps"] <= moves["q95"] + 1e-6
    assert bad["tp_bps"] >= bad["sl_bps"] * 1.2 - 1e-6
    assert "measured" in bad["note"]
    unbounded = risk_levels("BTC", "BUY", 100000.0, 45.0, settings, horizon_seconds=60, edge=0.5, spread_bps=0.1)
    assert unbounded["tp_bps"] > 5 * bad["tp_bps"]
    # floors are one-minute floors now
    assert settings.min_tp_bps <= 2.0 and settings.min_sl_bps <= 1.5 and settings.max_tp_bps <= 60


# ---------------------------------------------------------------- Round AO
def test_saved_keys_persist_by_default_and_report_their_effect():
    """Keys were a show piece: the persist box was off by default, so every
    engine restart / self-update / fresh Codespace lost them, and the page
    bounced to the dashboard before anything could be seen.  Now persist is
    the default, the save applies the key immediately (brain re-verify, news
    poll, agent test) and /api/settings/effects says what each key is doing."""
    import re
    from pathlib import Path

    from backend.api import routes_agents

    root = Path(__file__).resolve().parents[2]
    html = (root / "backend/web/settings.html").read_text()
    js = (root / "backend/web/settings.js").read_text()
    assert re.search(r'id="persist"[^>]*checked', html)
    assert 'id="effects-table"' in html and "Codespaces" in html
    assert routes_agents.KeyPayload().persist is True
    assert routes_agents.KeysPayload().persist is True
    assert "/api/settings/effects" in js and "function refreshEffects" in js
    assert '/api/news/poll' in js and '/api/brain/reconnect' in js
    assert 'location.href = "/"' not in js.split("save-keys")[1].split("};")[0]


def test_neuprint_connection_survives_a_missing_version_route():
    """neuPrint's platform migration removed /api/version the same way it
    removed /api/databaseInfo; a 404 there must fall through to the dbmeta
    route instead of reporting the token as unusable."""
    import httpx

    from backend.brain.query_circuits import HttpNeuprintClient

    def handler(request):
        if request.url.path == "/api/version":
            return httpx.Response(404)
        if request.url.path == "/api/dbmeta/datasets":
            assert request.headers["authorization"] == "Bearer tok"
            return httpx.Response(200, json={"hemibrain:v1.2.1": {}, "male-cns:v0.9": {}})
        return httpx.Response(500)

    client = HttpNeuprintClient("https://neuprint.test", "hemibrain:v1.2.1", "tok")
    client._client = httpx.Client(base_url="https://neuprint.test", transport=httpx.MockTransport(handler),
                                  headers={"Authorization": "Bearer tok"})
    assert "hemibrain" in client.fetch_version()

    def rejected(request):
        return httpx.Response(401)

    client._client = httpx.Client(base_url="https://neuprint.test", transport=httpx.MockTransport(rejected))
    import pytest

    with pytest.raises(PermissionError):
        client.fetch_version()
