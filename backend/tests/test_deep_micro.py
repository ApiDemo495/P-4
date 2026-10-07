"""Round K - the deep microstructure reasoning layer.

Each formula is checked against a series whose answer is known analytically
(a random walk has VR = 1 and H = 0.5, a monotone path has zero permutation
entropy, price = lambda * flow gives R^2 = 1 ...), then the Bayesian filter is
checked for the properties a filter must have (normalised, sticky, moved by
the evidence in the right direction), and finally the layer is checked where
it is consumed: the emotion report, the manipulation read, the streamed
message, the routes and the served pages.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from backend.core import deep_micro as D
from backend.core import emotions as E
from backend.formulas import synthetic

ROOT = Path(__file__).resolve().parents[2]


def _snapshot(key: str):
    return synthetic.scenario(key).snapshots[-1]


def _report(key: str) -> dict:
    return E.analyze(_snapshot(key), "BTC").to_dict()


# ---------------------------------------------------------------------------
# 1. the formulas, on series with known answers
# ---------------------------------------------------------------------------
def test_variance_ratio_reads_random_walk_momentum_and_reversion() -> None:
    RNG = np.random.default_rng(1)
    noise = RNG.normal(size=4000)
    vr_rw, z_rw = D.variance_ratio(noise, 4)
    assert 0.85 < vr_rw < 1.15 and abs(z_rw) < 4
    # AR(1) with positive phi -> momentum (VR > 1); negative phi -> reversion.
    def ar(phi: float) -> np.ndarray:
        x = np.zeros(4000)
        for i in range(1, 4000):
            x[i] = phi * x[i - 1] + noise[i]
        return x
    assert D.variance_ratio(ar(0.5), 4)[0] > 1.5
    assert D.variance_ratio(ar(-0.5), 4)[0] < 0.7


def test_hurst_exponent_brackets_half_for_white_noise() -> None:
    RNG = np.random.default_rng(2)
    noise = RNG.normal(size=4000)
    assert 0.4 < D.hurst_exponent(noise) < 0.6
    persistent = np.cumsum(RNG.normal(size=4000))            # returns of an integrated path
    assert D.hurst_exponent(persistent) > 0.8
    alternating = np.diff(RNG.normal(size=4001))              # over-differenced -> anti-persistent
    assert D.hurst_exponent(alternating) < 0.3


def test_permutation_entropy_is_zero_for_order_and_high_for_noise() -> None:
    RNG = np.random.default_rng(3)
    assert D.permutation_entropy(np.arange(200.0)) == 0.0
    assert D.permutation_entropy(RNG.normal(size=2000)) > 0.95
    assert D.permutation_entropy(np.arange(5.0)) == 1.0        # too short -> "no information"


def test_hawkes_branching_separates_poisson_from_clustered_arrivals() -> None:
    RNG = np.random.default_rng(4)
    poisson = np.cumsum(RNG.exponential(100.0, size=1500))   # ~10 prints/s for 150 s
    n_poisson = D.hawkes_branching(poisson)["branching_ratio"]
    assert n_poisson < 0.35
    # Clustered: bursts of 20 prints 5 ms apart every ~2 s.
    bursts = []
    t = 0.0
    while t < 150_000:
        bursts.extend(t + np.arange(20) * 5.0)
        t += RNG.exponential(2000.0)
    n_burst = D.hawkes_branching(np.asarray(bursts))["branching_ratio"]
    assert n_burst > 0.6 and n_burst > n_poisson


def test_vpin_is_one_for_one_sided_flow_and_near_zero_when_balanced() -> None:
    RNG = np.random.default_rng(5)
    n = 480
    times = np.arange(n) * 100.0
    prices = 100.0 + np.cumsum(RNG.normal(scale=0.01, size=n))
    qty = np.ones(n)
    buys = np.column_stack([times, prices, qty, np.ones(n)])
    alternating = np.column_stack([times, prices, qty, np.where(np.arange(n) % 2 == 0, 1.0, -1.0)])
    assert D.vpin(buys)["vpin"] == pytest.approx(1.0)
    assert D.vpin(alternating)["vpin"] < 0.1
    assert D.vpin(buys[:8])["buckets"] == 0                    # too short -> empty, not an error


def test_kyle_lambda_recovers_a_planted_impact() -> None:
    RNG = np.random.default_rng(6)
    n = 60 * 10
    times = np.arange(n) * 100.0                              # 10 prints per second, 60 s
    flow = RNG.choice([-1.0, 1.0], size=n)
    qty = RNG.uniform(0.5, 1.5, size=n)
    # Each second's price change is exactly 0.5 bps per unit of that second's signed volume.
    lam = 0.5
    prices = np.empty(n)
    price = 100.0
    for k in range(0, n, 10):
        signed = float((flow[k:k + 10] * qty[k:k + 10]).sum())
        price *= 1 + lam * signed / 10_000.0          # the second's flow moves its close
        prices[k:k + 10] = price
    ticks = np.column_stack([times, prices, qty, flow])
    out = D.kyle_lambda(ticks)
    assert out["r2"] > 0.95
    assert out["lambda_bps"] == pytest.approx(lam, rel=0.15)


def test_sign_memory_sees_runs_and_not_alternation() -> None:
    RNG = np.random.default_rng(7)
    runs = np.repeat(RNG.choice([-1.0, 1.0], size=60), 10)
    alternating = np.where(np.arange(600) % 2 == 0, 1.0, -1.0)
    assert D.sign_memory(runs)["memory"] > 0.8
    assert D.sign_memory(alternating)["memory"] == 0.0
    assert set(D.sign_memory(runs)["acf"]) == {"1", "2", "3", "5", "8", "13"}


def test_haar_spectrum_shares_sum_to_one_and_find_the_active_scale() -> None:
    RNG = np.random.default_rng(8)
    n = 512
    slow = np.repeat(RNG.normal(size=n // 16), 16)           # energy at the 32-tick scale
    spec = D.haar_spectrum(slow, resolution_us=1000.0)
    assert sum(s["share"] for s in spec) == pytest.approx(1.0, abs=0.01)
    assert max(spec, key=lambda s: s["share"])["scale_ticks"] == 32
    assert spec[0]["scale_label"]


def test_regime_filter_is_a_distribution_and_finds_stress() -> None:
    RNG = np.random.default_rng(9)
    typical = 1.0
    calm = D.regime_filter(RNG.normal(scale=0.5, size=60), typical)
    stress = D.regime_filter(RNG.normal(scale=4.0, size=60), typical)
    for out in (calm, stress):
        assert out["calm"] + out["trend"] + out["stress"] == pytest.approx(1.0, abs=0.01)
    assert stress["label"] == "stress" and stress["stress"] > 0.7
    assert calm["label"] in ("calm", "trend") and calm["stress"] < 0.1


def test_microprice_leans_towards_the_heavier_side() -> None:
    bids = np.array([[99.0, 5.0], [98.0, 5.0]])
    asks = np.array([[101.0, 1.0], [102.0, 1.0]])
    out = D.microprice(np.stack([bids, asks]))
    assert out["microprice_bps"] > 0 and out["pressure_top5"] > 0.5
    assert out["spread_bps"] == pytest.approx(200.0)
    assert D.microprice(None)["microprice_bps"] == 0.0


def test_ignition_detects_a_burst_that_fades() -> None:
    n = 400
    times = np.arange(n) * 50.0                                # 20 prints/s, 20 s
    prices = np.full(n, 100.0)
    prices[200:220] = np.linspace(100.0, 100.3, 20)            # +30 bps in one second
    prices[220:320] = np.linspace(100.3, 100.05, 100)          # gives 83% of it back in 5 s
    prices[320:] = 100.05
    out = D.ignition(times, prices, typical_1s_bps=5.0)
    assert out["score"] > 0.5 and out["retrace"] > 0.6 and out["burst_bps"] > 20
    assert D.ignition(times, np.full(n, 100.0), 5.0)["score"] == 0.0


def test_quote_stuffing_needs_a_surge_with_no_price() -> None:
    times = np.concatenate([np.arange(500) * 100.0, 50_000.0 + np.arange(200) * 2.0])
    flat = np.full(times.size, 100.0)
    assert D.quote_stuffing(times, flat, tick_rate_hz=10.0) > 0.9
    moving = flat + np.arange(times.size) * 0.001
    assert D.quote_stuffing(times, moving, tick_rate_hz=10.0) == 0.0


# ---------------------------------------------------------------------------
# 2. the Bayesian filter
# ---------------------------------------------------------------------------
def _flat_evidence(**overrides: float) -> dict[str, float]:
    base = {key: 0.0 for key in D.EVIDENCE_LABEL}
    base.update(overrides)
    return base


def test_the_posterior_is_normalised_and_moved_by_the_evidence() -> None:
    panic = D.bayesian_update(None, _flat_evidence(down_fast=1.0, cascade=0.8, stress=0.9, toxicity=0.8))
    calm = D.bayesian_update(None, _flat_evidence(calm=0.9, disorder=0.9))
    chase = D.bayesian_update(None, _flat_evidence(up_fast=1.0, cascade=0.8, herd_memory=0.8))
    for out in (panic, calm, chase):
        assert sum(out["posterior"].values()) == pytest.approx(1.0, abs=1e-3)
        assert set(out["posterior"]) == set(D.EMOTION_NAMES)
        assert 0.0 <= out["entropy"] <= 1.0 and out["certainty"] == pytest.approx(1 - out["entropy"], abs=1e-3)
    assert panic["argmax"] == "PANIC"
    assert calm["argmax"] == "COMPLACENCY"
    assert chase["argmax"] == "FOMO"
    assert panic["evidence_for"]["PANIC"][0]["evidence"] in ("down_fast", "cascade")


def test_the_filter_is_sticky_but_not_stubborn() -> None:
    state = D.DeepState()
    for _ in range(12):
        D.bayesian_update(state, _flat_evidence(calm=0.9, disorder=0.9))
    settled = state.posterior["COMPLACENCY"]
    assert settled > 0.7
    # One conflicting sample moves the belief but does not flip it outright ...
    first = D.bayesian_update(state, _flat_evidence(down_fast=1.0, cascade=0.8, stress=0.9))
    assert first["surprise_kl"] > 0.1
    assert first["posterior"]["PANIC"] > 0.05
    # ... a sustained panic does, within a few samples.
    for _ in range(6):
        out = D.bayesian_update(state, _flat_evidence(down_fast=1.0, cascade=0.8, stress=0.9))
    assert out["argmax"] == "PANIC" and out["posterior"]["PANIC"] > settled * 0.8
    assert state.samples == 19


def test_the_likelihood_table_only_uses_known_evidence() -> None:
    for name, table in D.LIKELIHOOD.items():
        assert name in D.EMOTION_NAMES
        assert set(table) <= set(D.EVIDENCE_LABEL), (name, set(table) - set(D.EVIDENCE_LABEL))


# ---------------------------------------------------------------------------
# 3. the whole layer on a tape
# ---------------------------------------------------------------------------
def test_analyze_produces_every_block_and_a_fifteen_step_chain() -> None:
    snap = _snapshot("IMPULSE_DOWN")
    f = E.features(snap, "BTC")
    deep = D.analyze(snap, "BTC", f)
    assert deep["available"]
    for key in ("bands", "hawkes", "flow", "book", "spectrum", "regime", "manipulation",
                "evidence", "posterior", "chain", "compute_us"):
        assert key in deep, key
    assert set(deep["bands"]) == {"micro", "seconds", "window"}
    for band in deep["bands"].values():
        assert 0.0 <= band["hurst"] <= 1.0 and band["variance_ratio"] > 0 and 0.0 <= band["entropy"] <= 1.0
    assert len(deep["chain"]) == 15
    for step in deep["chain"]:
        assert {"step", "name", "formula", "inputs", "value", "unit", "reads", "timescale", "feeds"} <= set(step)
        assert step["reads"]
    assert [s["step"] for s in deep["chain"]] == list(range(1, 16))
    assert deep["compute_us"] < 200_000            # well under one emotion interval
    assert set(deep["manipulation"]) >= {"ignition", "stuffing", "spoofing", "toxicity", "pushable"}


def test_the_filter_agrees_with_the_tone_of_the_tape() -> None:
    negatives = {"FEAR", "PANIC", "CAPITULATION", "DENIAL"}
    positives = {"HOPE", "EUPHORIA", "FOMO"}
    def argmax(key: str) -> str:
        snap = _snapshot(key)
        return D.analyze(snap, "BTC", E.features(snap, "BTC"))["posterior"]["argmax"]
    assert argmax("IMPULSE_DOWN") in negatives
    assert argmax("BEAR") in negatives
    assert argmax("IMPULSE_UP") in positives
    assert argmax("BULL") in positives
    assert argmax("FLAT") == "COMPLACENCY"


def test_analyze_declines_politely_on_a_short_tape() -> None:
    class Tape:
        def ticks(self, asset):
            return np.zeros((10, 4))
        def book(self, asset):
            return None
    out = D.analyze(Tape(), "BTC", {})
    assert out["available"] is False and out["reason"]


def test_compact_keeps_the_chain_and_drops_the_raw_arrays() -> None:
    snap = _snapshot("BULL")
    deep = D.analyze(snap, "BTC", E.features(snap, "BTC"))
    small = D.compact(deep)
    assert len(small["chain"]) == 15 and "inputs" not in small["chain"][0]
    assert "energy" not in small["spectrum"][0] and "share" in small["spectrum"][0]
    assert "log_likelihood" not in small["posterior"] and "evidence_for" in small["posterior"]
    assert D.compact({"available": False, "reason": "x"}) == {"available": False, "reason": "x"}


# ---------------------------------------------------------------------------
# 4. where it is consumed
# ---------------------------------------------------------------------------
def test_the_emotion_report_carries_the_deep_layer_and_blends_the_belief() -> None:
    payload = _report("IMPULSE_UP")
    deep = payload["deep"]
    assert deep["available"] and deep["posterior"]["argmax"]
    for item in payload["emotions"]:
        assert 0.0 <= item["belief"] <= 1.0 and 0.0 <= item["ramp"] <= 1.0
        expected = min(1.0, 0.65 * item["ramp"] + 0.35 * min(1.0, 2.5 * item["belief"]))
        assert item["intensity"] == pytest.approx(expected, abs=1e-3)
    assert "Bayesian filter" in payload["read"]
    components = payload["manipulation"]["components"]
    assert {"ignition", "toxicity", "stuffing", "spoofing", "pushable"} <= set(components)


def test_the_tracker_carries_the_filter_between_samples() -> None:
    tracker = E.EmotionTracker(asset="BTC")
    snaps = synthetic.scenario("FLAT").snapshots[-4:]
    for snap in snaps:
        E.analyze(snap, "BTC", tracker=tracker)
    assert tracker.deep_state.samples == len(snaps)
    assert sum(tracker.deep_state.posterior.values()) == pytest.approx(1.0, abs=1e-3)


def test_the_streamed_reading_includes_a_compact_deep_block() -> None:
    payload = _report("BEAR")
    small = E.compact(payload)
    assert small["deep"]["available"] and len(small["deep"]["chain"]) == 15
    assert "inputs" not in small["deep"]["chain"][0]
    assert "features" not in small


def test_the_deep_route_is_mounted() -> None:
    from backend.api import routes_emotions
    paths = {getattr(route, "path", "") for route in routes_emotions.router.routes}
    assert "/api/emotions/deep" in paths


def test_the_dashboard_renders_the_deep_layer() -> None:
    html = (ROOT / "backend" / "web" / "index.html").read_text()
    for element_id in ("deep-block", "deep-belief", "deep-verdicts", "deep-bands",
                       "deep-detectors", "deep-spectrum", "deep-chain"):
        assert f'id="{element_id}"' in html, element_id
    js = (ROOT / "backend" / "web" / "app.js").read_text()
    assert "function renderDeep(" in js and "renderDeep(state.emotionsDeep, top)" in js
    assert "function deepLockedHtml(" in js and "deepLockedHtml(crowd.deep)" in js
    css = (ROOT / "backend" / "web" / "styles.css").read_text()
    assert ".deep-chain li::before" in css


def test_the_prediction_reasoning_carries_the_deep_read() -> None:
    from backend.core import prediction
    crowd = _report("IMPULSE_DOWN")
    reasoning = prediction.build_reasoning(
        side="SELL", confidence=0.7, conviction="HIGH",
        formula_values={"TAI": -0.4}, directional={"TAI": "A"},
        risk={"tradeable": False}, crowd=crowd,
    )
    bullet = next(b for b in reasoning["bullets"] if b["kind"] == "crowd")
    assert "deep read: Bayesian filter" in bullet["text"]
    assert "regime" in bullet["text"] and "VPIN" in bullet["text"]
    manager = (ROOT / "backend" / "core" / "cycle_manager.py").read_text()
    assert "_crowd_deep_summary" in manager and '"deep": self._crowd_deep_summary' in manager
    assert math.isfinite(float(crowd["deep"]["posterior"]["argmax_probability"]))


def test_formula_consensus_pulls_the_filter_toward_the_formulas_side():
    """Round L: the 22-formula vote is evidence for the crowd filter.  With the
    same tape, a bullish consensus must raise the buying emotions' likelihood
    and a bearish one the selling emotions' - and with fewer than three voters
    the formulas must not move the filter at all."""
    tape = _snapshot("FLAT")
    f = E.features(tape, "BTC")
    bull = D.analyze(tape, "BTC", f, formula_consensus=0.6, formula_voters=12)
    bear = D.analyze(tape, "BTC", f, formula_consensus=-0.6, formula_voters=12)
    silent = D.analyze(tape, "BTC", f, formula_consensus=0.9, formula_voters=2)
    assert bull["evidence"]["formula_up"] == 1.0 and bull["evidence"]["formula_down"] == 0.0
    assert bear["evidence"]["formula_down"] == 1.0
    assert silent["evidence"]["formula_up"] == 0.0 and silent["evidence"]["formula_conviction"] == 0.0
    ll_bull = bull["posterior"]["log_likelihood"]
    ll_bear = bear["posterior"]["log_likelihood"]
    assert ll_bull["HOPE"] > ll_bear["HOPE"] and ll_bull["EUPHORIA"] > ll_bear["EUPHORIA"]
    assert ll_bear["FEAR"] > ll_bull["FEAR"] and ll_bear["PANIC"] > ll_bull["PANIC"]
    last = bull["chain"][-1]
    assert last["name"] == "22-formula cross-check" and last["value"] == 0.6
    assert "12 formulas lean BUY" in last["reads"]
