"""Round BT - "most of them are flat": quote prints (qty 0 / side 0) must not
dilute trade statistics, self-scale floors must be code constants, and the
flat watch must name the flat formulas with reasons."""
from __future__ import annotations

import numpy as np

from backend.core.frozen_snapshot import FrozenMarketSnapshot
from backend.formulas._util import SelfScale, real_trades
from backend.formulas.category_a_microstructure import afpr, sed, vsd
from backend.formulas.category_f_kalman import dskd


def _tape(n: int = 600, quote_share: float = 0.7, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t0 = 1_700_000_000_000
    times = t0 + np.arange(n) * 100.0
    prices = 65_000 + np.cumsum(rng.normal(0, 4, n))
    prices[n // 2:] += np.linspace(0, 120, n - n // 2)       # an impulse up
    qty = rng.lognormal(-2, 0.8, n)
    side = np.where(rng.random(n) < 0.68, 1.0, -1.0)
    quote = rng.random(n) < quote_share
    qty[quote] = 0.0
    side[quote] = 0.0
    return np.column_stack([times, prices, qty, side])


def _snapshot(btc: np.ndarray) -> FrozenMarketSnapshot:
    paxg = btc.copy()
    paxg[:, 1] = 2_400 + (paxg[:, 1] - 65_000) * 0.02
    book = np.zeros((2, 20, 2))
    spreads = np.column_stack([btc[:, 0], np.full(btc.shape[0], 6.5)])
    return FrozenMarketSnapshot(
        timestamp=float(btc[-1, 0]) / 1000.0, btc_ticks=btc, paxg_ticks=paxg,
        btc_book=book, paxg_book=book, btc_book_prev=book, paxg_book_prev=book,
        btc_candles=np.zeros(0), paxg_candles=np.zeros(0), news_items=(),
        drg_outcomes=np.zeros((0, 2)), btc_tick_count=len(btc), paxg_tick_count=len(paxg),
        btc_spreads=spreads, paxg_spreads=spreads,
    )


def test_snapshot_trades_accessor_drops_quote_prints():
    snap = _snapshot(_tape())
    ticks = snap.ticks("BTC")
    trades = snap.trades("BTC")
    assert ticks.shape[0] == 600
    assert 0 < trades.shape[0] < ticks.shape[0]
    assert np.all(trades[:, 2] > 0) and np.all(trades[:, 3] != 0)
    assert np.array_equal(real_trades(snap, "BTC"), trades)


def test_real_trades_falls_back_to_filtering_ticks():
    class Stand:
        def __init__(self, t):
            self._t = t

        def ticks(self, asset):
            return self._t

    t = _tape()
    out = real_trades(Stand(t), "BTC")
    assert np.all(out[:, 2] > 0)


def test_microstructure_formulas_are_not_flat_on_a_quote_heavy_tape():
    """70 % of the rows are quote prints - the volume formulas must still read
    the trades underneath (VSD used to see 'baseline median vol 0')."""
    snap = _snapshot(_tape())
    ctx: dict = {"traces": {}}
    v_state, s_state, a_state = vsd.State(), sed.State(), afpr.State()
    vals = {}
    for _ in range(3):
        vals["VSD"] = vsd.compute(snap, "BTC", v_state, {}, ctx)
        vals["SED"] = sed.compute(snap, "BTC", s_state, {}, ctx)
        vals["AFPR"] = afpr.compute(snap, "BTC", a_state, {}, ctx)
    assert all(np.isfinite(v) for v in vals.values()), vals
    assert abs(vals["AFPR"]) > 0.02, vals      # 68 % buyers -> clear aggressor lean
    assert abs(vals["VSD"]) > 0.0, vals
    # the trace must show a real baseline volume, never 0
    rows = {r["label"]: r["value"] for r in ctx["traces"].get("VSD", []) if isinstance(r, dict)}
    if "baseline median vol" in rows:
        assert rows["baseline median vol"] > 0


def test_self_scale_floor_is_a_code_constant_not_state():
    old = SelfScale(decay=0.98, floor=0.05)
    old.update(0.001)
    restored = SelfScale.from_dict(old.to_dict(), floor=1e-4)
    assert restored.floor == 1e-4
    assert restored.value == old.value
    assert sed.MEANINGFUL_EFFECT < 1e-3


def test_dskd_scale_no_longer_dominated_by_the_window_move():
    assert dskd.REALISED_SHARE < 0.75
    snap = _snapshot(_tape(quote_share=0.0, seed=3))
    state = dskd.State()
    ctx: dict = {"traces": {}}
    for _ in range(4):
        value = dskd.compute(snap, "BTC", state, {}, ctx)
    assert np.isfinite(value)


def test_emotions_runs_ignore_quote_prints():
    from backend.core import emotions

    helper = getattr(emotions, "_herding", None)
    if helper is None:
        import pytest
        pytest.skip("run helper renamed")
    t = _tape(quote_share=0.9)
    only_trades = t[t[:, 3] != 0]
    assert abs(helper(t) - helper(only_trades)) < 1e-9


def test_flat_watch_endpoint_shape():
    from backend.core.cycle_manager import CycleManager

    assert hasattr(CycleManager, "flat_watch") and hasattr(CycleManager, "_flat_watch_record")
    assert CycleManager.FLAT_EPS == 0.02


def test_dashboard_has_flat_watch_line():
    html = open("backend/web/index.html", encoding="utf-8").read()
    js = open("backend/web/app.js", encoding="utf-8").read()
    assert 'id="flat-watch"' in html
    assert "function renderFlatWatch" in js
    assert js.count("setInterval(") == 1
