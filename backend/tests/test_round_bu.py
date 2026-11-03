"""Round BU - Theory of Nearly Everything: unified evidence ledger + physics
estimators.  The ledger must weigh sources by what they have told us, remove
double counting, calibrate, and only take the side when its record earns it."""
from __future__ import annotations

import numpy as np

from backend.core.everything import MIN_FIT, TAKEOVER_RATE, UnifiedEvidence
from backend.core.hindsight import WindowRecord
from backend.core.physics_estimators import (ESTIMATORS, estimate_all, fluctuation_response, ising_flow,
                                             kramers_moyal, maxent_window)


def _tape(n=4000, drift=0.2, buy_share=0.6, seed=1, quote_share=0.4):
    rng = np.random.default_rng(seed)
    t = 1.7e12 + np.cumsum(rng.exponential(120, n))
    p = 65_000 + np.cumsum(rng.normal(drift, 3, n))
    q = rng.lognormal(-2, 0.8, n)
    s = np.where(rng.random(n) < buy_share, 1.0, -1.0)
    quote = rng.random(n) < quote_share
    q[quote] = 0.0
    s[quote] = 0.0
    return np.column_stack([t, p, q, s])


def _rows(n=300, seed=0, signal=0.3):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        y = 1 if rng.random() < 0.5 else -1
        a = y * signal + rng.normal()
        inp = {"f:A": a, "f:B": y * signal + 0.5 * rng.normal(), "f:C": 0.9 * a + 0.1 * rng.normal(),
               "f:noise": rng.normal(), "regime:vol": 1.0, "fusion:recipe": float(rng.choice([-1, 1]))}
        rows.append(WindowRecord(cycle=i, at=1e9 + 70 * i, side="BUY", confidence=0.5, inputs=inp, groups={},
                                 move_bps=float(5 * y), settled=True))
    return rows


# ---- physics estimators ----------------------------------------------------
def test_all_four_estimators_cover_a_normal_tape_and_print_their_logic():
    out = estimate_all(_tape(), 70.0)
    assert set(out["estimators"]) == set(ESTIMATORS)
    for key, res in out["estimators"].items():
        assert res["covered"], (key, res.get("why"))
        assert -1.0 <= res["score"] <= 1.0
        assert len(res["logic"]) >= 3
    assert len(out["values"]) == 4


def test_estimators_never_raise_on_a_short_or_empty_tape():
    for fn in (kramers_moyal, fluctuation_response, maxent_window):
        assert fn(np.zeros((0, 4)), 70.0)["covered"] is False
        assert fn(_tape(n=30), 70.0)["covered"] is False
    assert ising_flow(np.zeros((0, 4)))["covered"] is False
    assert ising_flow(_tape(n=20))["covered"] is False


def test_ising_follows_the_crowd_and_quote_prints_do_not_count():
    buyers = ising_flow(_tape(buy_share=0.8, quote_share=0.7))
    sellers = ising_flow(_tape(buy_share=0.2, quote_share=0.7))
    assert buyers["covered"] and sellers["covered"]
    assert buyers["magnetisation"] > 0.3 > -0.3 > sellers["magnetisation"]
    assert buyers["score"] > 0 > sellers["score"]
    assert buyers["trades"] <= 600


def test_kramers_moyal_measures_restoring_dynamics_not_a_functional_form():
    """An Ornstein-Uhlenbeck tape has a restoring D1 (slope < 0 against the
    state); the drift at the current state points back toward the mean."""
    rng = np.random.default_rng(11)
    n = 4000
    t = 1.7e12 + np.arange(n) * 250.0
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] - 0.08 * x[i - 1] + rng.normal(0, 3)
    p = 65_000 + x
    tape = np.column_stack([t, p, np.ones(n) * 0.1, np.where(rng.random(n) < 0.5, 1.0, -1.0)])
    res = kramers_moyal(tape, 70.0)
    assert res["covered"] and res["restoring_rate"] > 0
    assert len(res["logic"]) == 4
    if abs(res["state_sigma"]) > 0.5:
        assert np.sign(res["d1"]) == -np.sign(res["state_sigma"])


def test_maxent_distribution_meets_its_constraints():
    res = maxent_window(_tape(seed=5), 70.0)
    assert res["covered"] and res["converged"]
    assert 0.0 <= res["p_up"] <= 1.0
    assert abs(res["score"] - (2 * res["p_up"] - 1)) < 1e-9


# ---- unified ledger --------------------------------------------------------
def test_ledger_weighs_informative_sources_and_shrinks_correlated_ones():
    u = UnifiedEvidence()
    u.build_record(_rows())
    assert u.fit is not None
    w = dict(zip(u.fit.names, u.fit.slopes))
    assert w["f:B"] > w["f:A"] > 0            # cleaner source carries more evidence
    assert abs(w["f:noise"]) < 0.5 * w["f:A"]  # noise is shrunk toward zero
    assert u.fit.rho_bar > 0.1                 # A and C are correlated -> k_eff < k
    pred = u.predict(_rows()[-1].inputs)
    assert pred["available"] and pred["k_eff"] < pred["sources_used"]
    assert 0.0 <= pred["predictability"] <= 1.0
    assert abs(pred["predictability"] + pred["entropy_bits"] - 1.0) < 1e-9
    assert pred["sources"][0]["bits"] > 0


def test_ledger_takes_the_side_only_with_an_earned_record():
    u = UnifiedEvidence()
    assert u.may_decide()[0] is False
    u.build_record(_rows(signal=0.6))
    rate, n = u.trailing()
    assert n >= 30 and rate > TAKEOVER_RATE
    ok, why = u.may_decide(meta_rate=None)
    assert ok, why
    ok2, why2 = u.may_decide(meta_rate=0.99)
    assert not ok2 and "meta-combiner" in why2
    coin = UnifiedEvidence()
    coin.build_record(_rows(signal=0.0, seed=4))
    assert coin.may_decide()[0] is False or coin.trailing()[0] >= TAKEOVER_RATE


def test_ledger_state_round_trips():
    u = UnifiedEvidence()
    u.build_record(_rows())
    v = UnifiedEvidence()
    assert v.import_state(u.export_state()) == len(u.graded)
    assert v.fit.names == u.fit.names
    assert v.predict(_rows()[-1].inputs)["p_up"] == u.predict(_rows()[-1].inputs)["p_up"]
    assert v.to_dict()["fitted"]
    assert MIN_FIT == 40


def test_ledger_is_wired_into_the_lock_path_api_and_ui():
    cm = open("backend/core/cycle_manager.py", encoding="utf-8").read()
    assert "self._apply_everything(self._apply_meta(fusion))" in cm
    assert "self.everything.remember(" in cm and "self.everything.settle(" in cm
    assert 'self.vault.save("everything"' in cm
    assert '"physics"' in cm and "estimate_all(" in cm
    api = open("backend/api/routes_brain.py", encoding="utf-8").read()
    assert '"/api/everything"' in api
    html = open("backend/web/index.html", encoding="utf-8").read()
    js = open("backend/web/app.js", encoding="utf-8").read()
    assert 'id="everything-card"' in html and 'id="ev-estimators"' in html
    assert "function renderEverything" in js and js.count("setInterval(") == 1
    fusion = open("backend/agents/fusion.py", encoding="utf-8").read()
    assert "everything: dict | None = None" in fusion
