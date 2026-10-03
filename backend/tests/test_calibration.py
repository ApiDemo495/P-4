"""Round N: the evidence ledger learns which sources predict."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pytest

from backend.core.calibration import EvidenceLedger, SourceStat


def _synthetic(ledger: EvidenceLedger, n: int, seed: int = 7) -> tuple[int, int]:
    """Three sources: good (62 %), inverted (38 %), noise (50 %).  The spec
    recipe follows the *inverted* one (the failure mode being fixed)."""
    rng = random.Random(seed)
    spec_hits = ledger_hits = 0
    for cycle in range(n):
        truth = 1 if rng.random() < 0.5 else -1
        good = truth if rng.random() < 0.62 else -truth
        bad = -truth if rng.random() < 0.62 else truth
        noise = 1 if rng.random() < 0.5 else -1
        votes = {"good": good, "bad": bad, "noise": noise, "spec:fusion": bad}
        verdict = ledger.evaluate("BTC", votes)
        side = verdict["side"] if verdict["active"] and verdict["side"] else ("BUY" if bad > 0 else "SELL")
        if (side == "BUY") == (truth > 0):
            ledger_hits += 1
        if (bad > 0) == (truth > 0):
            spec_hits += 1
        ledger.remember("BTC", cycle, votes, verdict["p_up"])
        ledger.score("BTC", cycle, truth > 0)
    return spec_hits, ledger_hits


def test_source_stat_weights_are_log_odds_and_clipped() -> None:
    s = SourceStat(hits=60, misses=40)
    assert 0.55 < s.reliability() < 0.60
    assert math.isclose(s.weight(), math.log(s.reliability() / (1 - s.reliability())))
    s2 = SourceStat(hits=1000, misses=0)
    assert s2.weight() == 1.5
    assert SourceStat(hits=40, misses=60).weight() < 0


def test_the_ledger_learns_to_follow_good_and_fade_bad_sources() -> None:
    ledger = EvidenceLedger(None, enabled=True, min_samples=30, half_life=200)
    spec_hits, ledger_hits = _synthetic(ledger, 600)
    report = ledger.report("BTC")["assets"]["BTC"]
    by = {r["source"]: r for r in report["sources"]}
    assert by["good"]["verdict"] == "follow" and by["good"]["weight"] > 0.2
    assert by["bad"]["verdict"] == "fade" and by["bad"]["weight"] < -0.2
    assert by["noise"]["verdict"] == "noise"
    assert report["active"] is True and report["scored"] == 600
    # The spec recipe followed the inverted source (~38 %); the ledger must be
    # clearly better than both the spec and a coin toss.
    assert spec_hits / 600 < 0.45
    assert ledger_hits / 600 > 0.55, ledger_hits / 600


def test_the_ledger_is_watch_only_until_min_samples() -> None:
    ledger = EvidenceLedger(None, enabled=True, min_samples=30)
    for cycle in range(10):
        votes = {"good": 1}
        v = ledger.evaluate("BTC", votes)
        assert v["active"] is False
        ledger.remember("BTC", cycle, votes, 0.5)
        ledger.score("BTC", cycle, True)
    assert ledger.evaluate("BTC", {"good": 1})["active"] is False
    for cycle in range(10, 40):
        ledger.remember("BTC", cycle, {"good": 1}, 0.6)
        ledger.score("BTC", cycle, True)
    v = ledger.evaluate("BTC", {"good": 1})
    assert v["active"] is True and v["side"] == "BUY" and v["p_up"] > 0.6


def test_disabled_ledger_never_takes_over() -> None:
    ledger = EvidenceLedger(None, enabled=False, min_samples=1)
    for cycle in range(5):
        ledger.remember("BTC", cycle, {"g": 1}, 0.5)
        ledger.score("BTC", cycle, True)
    assert ledger.evaluate("BTC", {"g": 1})["active"] is False


def test_flat_windows_are_not_scored() -> None:
    ledger = EvidenceLedger(None, enabled=True, min_samples=1)
    ledger.remember("BTC", 1, {"g": 1}, 0.5)
    assert ledger.score("BTC", 1, None) is None
    assert ledger.report("BTC")["assets"]["BTC"]["scored"] == 0


def test_calibration_buckets_track_claimed_vs_realised() -> None:
    ledger = EvidenceLedger(None, enabled=True, min_samples=1)
    for cycle in range(40):
        ledger.remember("BTC", cycle, {"g": 1}, 0.7)
        ledger.score("BTC", cycle, cycle % 10 != 0)  # 90 % up
    cal = ledger.report("BTC")["assets"]["BTC"]["calibration"]
    row = next(r for r in cal if r["claimed"] == "70-80%")
    assert row["windows"] == 40 and row["realised"] == 0.9


def test_state_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "cal.json"
    ledger = EvidenceLedger(path, enabled=True, min_samples=5)
    _synthetic(ledger, 50)
    reloaded = EvidenceLedger(path, enabled=True, min_samples=5)
    a, b = ledger.report("BTC")["assets"]["BTC"], reloaded.report("BTC")["assets"]["BTC"]
    assert a["scored"] == b["scored"] == 50
    assert [r["source"] for r in a["sources"]] == [r["source"] for r in b["sources"]]
    assert reloaded.evaluate("BTC", {"good": 1, "bad": 1})["active"] is True


def test_votes_from_reads_every_source_family() -> None:
    class Agent:
        def __init__(self, decision, available=True):
            self.decision, self.available = decision, available

    candles = np.array([100.0, 101.0, 100.5, 100.8, 100.2, 100.9, 101.5])
    ticks = np.column_stack([np.arange(30), np.full(30, 100.0), np.ones(30), np.where(np.arange(30) % 3 == 0, -1, 1)])
    votes = EvidenceLedger.votes_from(
        formula_values={"OFI": 0.3, "VPIN": -0.2, "HSI": 0.9, "ZERO": 0.0},
        directional={"OFI": "flow", "VPIN": "flow", "ZERO": "flow"},
        ccs_value=-0.1, agents={"gemini": Agent("BUY"), "github": Agent("SELL", available=False)},
        spec_score=0.05, niv=-0.3, crowd_tone=0.2, candles=candles, ticks=ticks,
    )
    assert votes["f:OFI"] == 1 and votes["f:VPIN"] == -1 and "f:HSI" not in votes and "f:ZERO" not in votes
    assert votes["brain:CCSv2"] == -1 and votes["agent:gemini"] == 1 and "agent:github" not in votes
    assert votes["spec:fusion"] == 1 and votes["news:NIV"] == -1 and votes["crowd:tone"] == 1
    assert votes["micro:ret_1m"] == 1 and votes["micro:ret_5m"] == 1 and votes["micro:order_flow"] == 1


def test_fusion_lets_an_active_ledger_decide_and_caps_confidence() -> None:
    from backend.agents import fusion
    from backend.core import config as cfg

    learned = {"active": True, "side": "SELL", "score": -0.6, "p_side": 0.58, "p_up": 0.42,
               "realised_at_this_confidence": 0.55, "scored": 120, "for": [{"source": "f:OFI", "reliability": 0.6}]}
    out = fusion.fuse(agents={}, ccs_value=0.8, ccs_confidence=0.9, hsi=0.1, settings=cfg.SETTINGS, learned=learned)
    assert out.decision == "SELL"
    assert out.spec_score > 0 and out.score < 0
    # confidence is the edge: 2 x 0.55 - 1 = 0.10, the realised edge of the bucket
    assert out.confidence == pytest.approx(2 * 0.55 - 1)
    assert "prediction history weighs 100%" in out.reasoning and "overrides the spec recipe" in out.reasoning
    # Round Y: the record ramps in - half the windows scored, half the say.
    half = fusion.fuse(agents={}, ccs_value=0.8, ccs_confidence=0.9, hsi=0.1, settings=cfg.SETTINGS,
                       learned={**learned, "scored": 10, "min_samples": 20})
    assert "weighs 50%" in half.reasoning
    assert half.score == pytest.approx(0.5 * half.spec_score + 0.5 * -0.6)
    off = fusion.fuse(agents={}, ccs_value=0.8, ccs_confidence=0.9, hsi=0.1, settings=cfg.SETTINGS,
                      learned={"active": False, "scored": 3})
    assert off.decision == "BUY" and off.learned["active"] is False
