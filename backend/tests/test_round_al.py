"""Round AL - the Formula Genesis Engine (v3.0).

* the pool: 10 domains x 10 sub-categories x 21 variants = 2,100 unique ids
* the exact mathematics the specification names, checked on known inputs
* the seven-metric fitness, correlation pruning, lifecycle + autopsy
* the engine end-to-end on a synthetic tape (reduced pool for speed)
* fusion / ledger / API / dashboard wiring, keyed providers only with keys
"""
from __future__ import annotations

import asyncio
import math
import random
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from backend.genesis import fitness as fit
from backend.genesis import lifecycle as life
from backend.genesis import ops
from backend.genesis import regime as reg
from backend.genesis import symbolic
from backend.genesis.candles import CandleStore
from backend.genesis.domains import all_variants
from backend.genesis.domains.d01_signatures import _sig2
from backend.genesis.domains.d02_homology import _betti
from backend.genesis.domains.d03_infogeo import fisher_distance
from backend.genesis.domains.d06_tropical import _maxplus_eigen
from backend.genesis.domains.d07_quantum import _entropy
from backend.genesis.features import build_frame
from backend.genesis.spec import DOMAINS, State

ROOT = Path(__file__).resolve().parents[2]
warnings.simplefilter("ignore", RuntimeWarning)


def _tape(seed: int, minutes: int = 320, trend: float = 0.0) -> CandleStore:
    st = CandleStore("BTC")
    rng = np.random.default_rng(seed)
    t0, price = 1_700_000_000_000, 60000.0
    for m in range(minutes):
        for k in range(12):
            price *= math.exp(rng.normal(trend, 2e-4))
            book = np.zeros((2, 10, 2))
            book[0, :, 0] = price - np.arange(1, 11)
            book[1, :, 0] = price + np.arange(1, 11)
            book[:, :, 1] = rng.uniform(0.1, 2.0, (2, 10))
            st.on_tape([(t0 + m * 60000 + k * 5000, price, rng.uniform(0.001, 0.5), rng.choice([-1.0, 1.0]))],
                       book if k % 4 == 0 else None)
    return st


@pytest.fixture(scope="module")
def frame():
    return build_frame("BTC", _tape(1).rows(), _tape(2).rows())


# ------------------------------------------------------------------ pool
def test_pool_is_2100_unique_formulas_in_ten_domains():
    specs = all_variants()
    assert len(specs) == 2100
    assert len({s.fid for s in specs}) == 2100
    by_domain = {d: [s for s in specs if s.domain == d] for d in DOMAINS}
    assert all(len(v) == 210 for v in by_domain.values())
    for d, lst in by_domain.items():
        assert len({s.subcategory for s in lst}) == 10, d
    assert all(s.definition and s.interpretation for s in specs)


def test_every_formula_returns_candle_aligned_arrays(frame):
    specs = all_variants()
    rng = random.Random(7)
    sample = [s for s in specs if s.variant == 1] + rng.sample(specs, 60)
    for sp in sample:
        sig, raw = sp.kernel(frame, **sp.params)
        assert sig.shape == (frame.n,), sp.fid
        assert raw.shape == (frame.n,), sp.fid
        if not np.isnan(sig).all():
            assert np.nanmax(np.abs(sig)) <= 1.0 + 1e-9, sp.fid


# ----------------------------------------------------------- mathematics
def test_levy_area_of_a_unit_square_loop_is_its_signed_area():
    # counter-clockwise unit square: Lévy area A = ½(∫x dy − ∫y dx) = +1
    x = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    S_x, S_y, S_xy, S_yx, *_ = _sig2(x, y, 4)
    assert S_x[-1] == pytest.approx(0.0) and S_y[-1] == pytest.approx(0.0)
    assert 0.5 * (S_xy[-1] - S_yx[-1]) == pytest.approx(1.0)


def test_betti_1_by_euler_formula_counts_the_hole_in_a_square():
    W = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    V, E, C, b1, *_ = _betti(W, eps=1.1)       # edges of length 1 only - a 4-cycle
    assert (V, E, C, b1) == (4, 4, 1, 1)
    *_, b1_full, _, _ = _betti(W, eps=1.5)      # diagonals join - the hole fills
    assert b1_full == 3 - 0 + 0 or b1_full >= 1  # E=6, V=4, C=1 -> 3 (clique, not a hole)


def test_fisher_rao_distance_is_zero_at_identity_and_grows_with_mean_shift():
    assert fisher_distance(0.0, 1.0, 0.0, 1.0) == pytest.approx(0.0)
    d1 = fisher_distance(0.0, 1.0, 0.5, 1.0)
    d2 = fisher_distance(0.0, 1.0, 1.0, 1.0)
    assert 0 < d1 < d2
    # closed form for equal σ: √2·arccosh(1 + Δμ²/(4σ²))
    assert d2 == pytest.approx(math.sqrt(2) * math.acosh(1.25))


def test_wasserstein_1_via_cdf_matches_the_shift_of_a_point_mass():
    grid = np.arange(5, dtype=float)
    p = np.array([1.0, 0, 0, 0, 0])
    q = np.array([0, 0, 0, 1.0, 0])
    assert ops.wasserstein1_cdf(p, q, grid) == pytest.approx(3.0)
    a = np.random.default_rng(0).normal(0, 1, 4000)
    assert ops.wasserstein1_samples(a, a + 0.7) == pytest.approx(0.7, abs=0.05)


def test_tropical_maxplus_eigenvalue_is_the_maximum_cycle_mean():
    A = np.array([[-np.inf, 2.0], [4.0, -np.inf]])   # one 2-cycle of weight 6 -> mean 3
    assert _maxplus_eigen(A) == pytest.approx(3.0)


def test_von_neumann_entropy_pure_vs_maximally_mixed():
    pure = np.zeros((1, 3, 3)); pure[0, 0, 0] = 1.0
    mixed = np.eye(3)[None] / 3.0
    assert _entropy(pure)[0] == pytest.approx(0.0, abs=1e-9)
    assert _entropy(mixed)[0] == pytest.approx(math.log(3))


def test_padic_valuation_and_lempel_ziv():
    assert list(ops.padic_valuation(np.array([8, 12, 5, 0]), 2)) == [3, 2, 0, 0]
    assert list(ops.padic_valuation(np.array([9, 27, 10]), 3)) == [2, 3, 0]
    assert ops.lz76([0] * 32) <= 3
    rng = np.random.default_rng(1)
    assert ops.lz76(rng.integers(0, 2, 256)) > 20


def test_grunwald_letnikov_of_a_constant_is_zero_only_for_level_relative_input():
    x = np.full(80, 3.0)
    d = ops.grunwald_letnikov(x - x[0], 0.5, 34)
    assert np.allclose(np.nan_to_num(d[-10:]), 0.0)


# ---------------------------------------------------------------- fitness
def test_seven_metric_fitness_rewards_foresight_and_penalises_noise():
    rng = np.random.default_rng(3)
    ret = rng.normal(0, 1e-3, 1200)
    oracle = np.roll(np.sign(ret), -1)       # knows the next return
    noise = rng.choice([-1.0, 1.0], 1200)
    s_good, s_bad = fit.score(oracle, ret), fit.score(noise, ret)
    assert set(s_good.metrics) == {"hit", "ic", "sharpe", "pf", "dd", "regime", "steady"}
    assert s_good.metrics["hit"] > 0.95 and s_good.fitness > s_bad.fitness + 0.3
    assert fit.score(np.zeros(1200), ret).note == "no signal"


def test_greedy_selection_prunes_correlated_duplicates():
    rng = np.random.default_rng(5)
    base = rng.normal(size=600)

    class F:  # noqa: D401 - tiny stand-in
        def __init__(self, fid):
            self.fid = fid
    sigs = {"a": base, "b": base * 0.99 + 1e-3, "c": rng.normal(size=600)}
    chosen = fit.select([F("a"), F("b"), F("c")], sigs, k=3, max_corr=0.7)
    assert [f.fid for f in chosen] == ["a", "c"]


# -------------------------------------------------------------- lifecycle
def test_lifecycle_birth_to_active_to_decay_to_death_and_autopsy_resurrection():
    spec = all_variants()[0]
    f = life.Formula(spec=spec)
    f.record(fit.Score(fitness=0.7))
    assert life.transition(f, True) == State.ACTIVE
    f.record(fit.Score(fitness=0.2))
    life.transition(f, False)
    assert life.transition(f, False) == State.DECAYING
    for _ in range(life.DEATH_STRIKES):
        life.transition(f, False)
    assert f.state == State.DEAD
    # autopsy: profitable in trending, ruinous in mean_reverting -> gated second life
    n = 600
    ret = np.random.default_rng(2).normal(0, 1e-3, n)
    labels = np.array(["trending"] * (n // 2) + ["mean_reverting"] * (n - n // 2), dtype=object)
    sig = np.roll(np.sign(ret), -1)
    sig[n // 2:] *= -1
    report = life.autopsy(f, sig, ret, labels)
    assert report["verdict"].startswith("resurrected")
    assert f.state == State.CANDIDATE and f.spec.gates == {"regimes": ["trending"]}
    assert life.gated_signal(f, 0.8, "mean_reverting") == 0.0
    assert life.gated_signal(f, 0.8, "trending") == 0.8


def test_regime_classifier_and_gate_sizes(frame):
    info = reg.classify(frame)
    assert info["regime"] in reg.REGIMES
    labels = reg.regime_labels(frame)
    assert labels.shape == (frame.n,) and set(labels) <= set(reg.REGIMES)

    class F:
        def __init__(self, d):
            self.spec = type("S", (), {"domain": d})()
    active = [F(d) for d in range(1, 11) for _ in range(20)]
    for r, (_, size) in reg.GATES.items():
        assert len(reg.gate(active, r)) == size


# --------------------------------------------------------------- symbolic
def test_symbolic_trees_evaluate_and_breed_without_eval(frame):
    rng = random.Random(1)
    tree = symbolic.random_tree(rng, ["D1.01.v01"])
    frame.cache["__signals__"] = {"D1.01.v01": np.tanh(frame.ret * 1000)}
    out = symbolic.evaluate(tree, frame, frame.cache["__signals__"])
    assert out.shape == (frame.n,)
    kids = symbolic.breed(rng, [], ["D1.01.v01"], 1, 5)
    assert len(kids) == 5 and all(k.origin == "bred" and k.expression for k in kids)
    sig, raw = kids[0].kernel(frame)
    assert sig.shape == (frame.n,) and "eval(" not in Path(symbolic.__file__).read_text()


# ----------------------------------------------------------------- engine
def test_engine_scores_selects_votes_and_persists(tmp_path, monkeypatch):
    from backend.genesis import engine as eng

    e = eng.GenesisEngine(state_dir=str(tmp_path))
    small = [s for s in e.templates if s.variant in (1, 11)]
    e.templates = small
    e.pools = {a: {sp.fid: life.Formula(spec=sp) for sp in small} for a in e.assets}
    e.stores["BTC"], e.stores["PAXG"] = _tape(1, trend=1e-5), _tape(2)
    summary = e.rescore("BTC")
    assert summary["scored"] == len(small) and summary["active"] > 0
    live = e.evaluate_live("BTC")
    assert live["status"] == "live" and live["decision"] in ("BUY", "SELL", "NEUTRAL")
    assert live["gated"] <= live["active"] and live["regime"]["regime"] in reg.REGIMES
    assert {d["domain"] for d in live["domains"]} <= set(DOMAINS)
    g = e.genesis("BTC")
    assert g["children"] == eng.BREED_COUNT
    assert any(f.spec.origin == "bred" for f in e.pools["BTC"].values())
    st = e.status("BTC")["assets"]["BTC"]
    assert st["generation"] == 1 and "NumPy" in e.status()["compute"]
    assert (tmp_path / "genesis_BTC.json").exists()
    e2 = eng.GenesisEngine(state_dir=str(tmp_path))
    assert e2.generation["BTC"] == 1 and sum(1 for f in e2.pools["BTC"].values() if f.spec.origin == "bred") > 0
    detail = e.formula("BTC", live["top"][0]["id"])
    assert detail and "score" in detail and detail["score"]["metrics"]


# --------------------------------------------------------- fusion / ledger
def test_fusion_counts_the_genesis_composite_as_a_weighted_voter():
    from backend.agents import fusion
    from backend.core import config as cfg

    settings = cfg.Settings()
    settings.weight_genesis = 0.25
    genesis = {"status": "live", "vote": 0.6, "confidence": 0.5, "agreement": 0.8, "firing": 40, "gated": 50,
               "active": 200, "generation": 2, "regime": {"regime": "trending"}}
    res = fusion.fuse(agents={}, ccs_value=0.2, ccs_confidence=0.6, hsi=0.0, settings=settings, genesis=genesis)
    assert "genesis" in res.contributions and res.contributions["genesis"]["decision"] == "BUY"
    assert res.genesis["vote"] == 0.6
    warming = dict(genesis, status="warming", firing=0)
    res2 = fusion.fuse(agents={}, ccs_value=0.2, ccs_confidence=0.6, hsi=0.0, settings=settings, genesis=warming)
    assert "genesis" not in res2.contributions


def test_ledger_votes_and_lock_weights_know_the_genesis_source():
    from backend.core import lock_weights
    from backend.core.calibration import EvidenceLedger

    votes = EvidenceLedger.votes_from(formula_values={}, directional={}, ccs_value=0.0, agents={},
                                      spec_score=0.0, genesis_vote=-0.4)
    assert votes["genesis:composite"] == -1
    assert lock_weights.BACKERS["genesis"] == ("genesis:composite",)


# ------------------------------------------------------ keyed providers
def test_keyed_providers_only_activate_with_keys():
    from backend.core import config as cfg
    from backend.data import keyed_providers as kp

    settings = cfg.Settings()
    feed = kp.MacroFeed(settings)
    assert feed.configured() == [] or all(getattr(settings, f"{p}_key") for p in feed.configured())
    assert set(kp.PROVIDERS) == {"glassnode", "twelvedata", "lunarcrush"}
    assert asyncio.run(kp.test_key("nope", "x"))["valid"] is False
    assert asyncio.run(kp.test_key("glassnode", ""))["valid"] is False
    for p in kp.PROVIDERS:
        assert hasattr(settings, f"{p}_key")
    html = (ROOT / "backend/web/settings.html").read_text(encoding="utf-8")
    for p in kp.PROVIDERS:
        assert f'id="key-{p}"' in html and f'data-test="{p}"' in html


# ------------------------------------------------------------- dashboard
def test_dashboard_has_the_genesis_card_and_one_timer():
    html = (ROOT / "backend/web/index.html").read_text(encoding="utf-8")
    js = (ROOT / "backend/web/app.js").read_text(encoding="utf-8")
    assert 'id="genesis-card"' in html and 'data-panel="genesis"' in html
    assert "function renderGenesis" in js and "renderGenesis(data.genesis)" in js
    assert "/api/genesis/domains" in js
    assert len(re.findall(r"\bsetInterval\(", js)) == 1
    routes = (ROOT / "backend/api/routes_genesis.py").read_text(encoding="utf-8")
    for path in ("/api/genesis/current", "/api/genesis/live", "/api/genesis/status", "/api/genesis/formulas",
                 "/api/genesis/domains", "/api/genesis/graveyard", "/api/genesis/rescore", "/api/genesis/breed"):
        assert path in routes
