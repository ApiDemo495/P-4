"""Round AJ - hedge outcomes, &b grade, physics kinetics on the engine report."""
from __future__ import annotations

import numpy as np

from backend.core import hedge_outcomes
from backend.formulas import synthetic
from backend.formulas.engine import FEED_STATE_WEIGHT, LIVE_EQUIVALENT, provenance


def test_hedge_outcomes_are_a_proper_joint_distribution():
    snap = synthetic.scenario("BEAR").snapshots[-1]
    f = {"HSI": 0.3, "HRDD": 0.1, "SHRP": -0.2, "GCDV": 0.05}
    out = hedge_outcomes.build(snap, f, "SELL", 0.6, "BTC")
    assert out["valid"]
    j = out["joint"]
    cells = [j["btc_up_paxg_up"], j["btc_up_paxg_down"], j["btc_down_paxg_up"], j["btc_down_paxg_down"]]
    assert all(0.0 <= c <= 1.0 for c in cells) and abs(sum(cells) - 1.0) < 1e-6
    assert abs(j["btc_up_paxg_up"] + j["btc_up_paxg_down"] - j["p_btc_up"]) < 1e-6
    # a SELL lock on BTC puts P(BTC up) below one half and the best action short
    assert j["p_btc_up"] < 0.5
    assert out["best"]["btc_weight"] <= 0 and len(out["actions"]) == 4
    assert all(0 <= a["p_profit"] <= 1 for a in out["actions"])
    assert -1 <= out["rho"] <= 1 and out["sigma_btc_bps"] > 0 and out["logic"]
    assert out["regime"]["label"] == "hedge working"
    # the mirror lock flips the BTC leg
    buy = hedge_outcomes.build(snap, f, "BUY", 0.6, "BTC")
    assert buy["joint"]["p_btc_up"] > 0.5 and buy["best"]["btc_weight"] >= 0
    # no grid -> an honest reason, never an exception
    class _NoSync:
        synced = None

        def last_price(self, a):
            return 1.0

    out = hedge_outcomes.build(_NoSync(), f, "BUY", 0.5, "BTC")
    assert out["valid"] is False and out["reason"]


def test_bivariate_normal_quadrature_matches_known_values():
    assert abs(hedge_outcomes.bvn_upper(0.0, 0.0, 0.0) - 0.25) < 1e-6
    assert abs(hedge_outcomes.bvn_upper(0.0, 0.0, 0.5) - (0.25 + np.arcsin(0.5) / (2 * np.pi))) < 1e-6
    assert abs(hedge_outcomes._phi_inv(0.975) - 1.959964) < 1e-5


class _Snap:
    def __init__(self, source, ticks, book, candles, synced, news=()):
        self.source, self._t, self._b, self._c, self.synced, self.news_items = source, ticks, book, candles, synced, news

    def ticks(self, asset):
        return self._t

    def book(self, asset):
        return self._b

    def candles(self, asset):
        return self._c


class _Synced:
    valid = True

    def describe(self):
        return {"valid": True, "window_seconds": 300, "widened": True,
                "reason": "grid widened to 300 s so PAXG (4 updates) carries a covariance"}


def test_amp_b_reads_live_on_a_real_feed_with_a_widened_grid_and_quiet_news():
    import time

    now = time.time()
    t = np.arange(120.0)
    ticks = np.column_stack([(now - 120 + t) * 1000.0, 100000.0 + np.sin(t), np.ones(120), np.where(t % 2 == 0, 1.0, -1.0)])
    book = np.zeros((2, 20, 2))
    book[0, 0] = (99999.0, 1.0)
    book[1, 0] = (100001.0, 1.0)
    p = provenance(_Snap("kraken", ticks, book, np.zeros(5), _Synced()), "BTC")
    assert p["real"] and p["grade"] == "live" and p["coverage"] == 1.0
    assert p["feeds"]["cross"]["state"] == "derived" and p["holding_back"] == []
    assert p["news_state"] == "offline"             # reported beside the grade, never demoting it
    assert FEED_STATE_WEIGHT["derived"] == 1.0 and "derived" in LIVE_EQUIVALENT
    # a quiet tape (prints every 12 s, last one 20 s ago) is still live
    slow = np.column_stack([(now - 20 - 12 * np.arange(50)[::-1]) * 1000.0, 100000.0 + np.arange(50), np.ones(50), np.ones(50)])
    p2 = provenance(_Snap("kraken", slow, book, np.zeros(5), _Synced()), "BTC")
    assert p2["feeds"]["tape"]["state"] == "live"
