"""Round O - three keys per provider with automatic failover, and up to three
local models answering in parallel."""

from __future__ import annotations

import asyncio
import time

import pytest

from backend.agents import keyring
from backend.agents.base import AgentResult, AgentStatus
from backend.agents.gemini_agent import GeminiAgent, RateLimited
from backend.agents.github_agent import GitHubAgent
from backend.agents.keyring import KeyRing
from backend.agents.local_pool import LocalModelPool, merge_results
from backend.core import config as cfg


def _settings():
    s = cfg.Settings()
    s.rings = {p: KeyRing(p) for p in keyring.ENV_NAMES}
    return s


# ---------------------------------------------------------------- ring
def test_primary_is_preferred_and_comes_back_after_its_cooldown():
    ring = KeyRing.from_keys("gemini", ["primary", "backup-a", "backup-b"])
    assert ring.current() == "primary"
    ring.report_failure("primary", "rate_limited")
    assert ring.current() == "backup-a"
    assert ring.status()["active_slot"] == 2
    # the moment the primary's cooldown ends it is preferred again
    ring.slots[0].cooldown_until = time.time() - 1
    assert ring.current() == "primary"
    assert ring.status()["on_primary"] is True


def test_rate_limit_cooldown_doubles_and_resets_on_success():
    ring = KeyRing.from_keys("gemini", ["k1"])
    assert ring.report_failure("k1", "rate_limited") == 10.0
    assert ring.report_failure("k1", "rate_limited") == 20.0
    assert ring.report_failure("k1", "rate_limited") == 40.0
    ring.report_success("k1")
    assert ring.report_failure("k1", "rate_limited") == 10.0
    assert ring.report_failure("k1", "rejected") == keyring.REJECTED_COOLDOWN


def test_every_slot_is_optional_and_env_accepts_suffixes(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "one")
    monkeypatch.setenv("GEMINI_API_KEY_3", "three")
    monkeypatch.delenv("GEMINI_API_KEY_2", raising=False)
    ring = KeyRing.from_env("gemini")
    assert ring.keys() == ["one", "three", ""]
    assert ring.status()["configured_slots"] == 2
    empty = KeyRing.from_keys("newsapi", [])
    assert not empty and empty.current() == "" and empty.candidates() == []
    # masked view never leaks the key
    assert "one" not in str(ring.status()) or ring.status()["slots"][0]["masked"] != "one"


def test_all_cooling_still_offers_the_soonest_key_when_close():
    ring = KeyRing.from_keys("github", ["a", "b"])
    ring.report_failure("a", "rejected")
    ring.report_failure("b", "error")
    assert ring.all_cooling()
    assert ring.candidates() == []           # 60 s away - do not hammer
    ring.slots[1].cooldown_until = time.time() + 2
    assert ring.candidates() == ["b"]        # about to recover: try it


# ------------------------------------------------------------- failover
def test_gemini_switches_to_the_second_key_inside_one_call(monkeypatch):
    settings = _settings()
    settings.rings["gemini"] = KeyRing.from_keys("gemini", ["bad", "good", ""])
    agent = GeminiAgent(settings)
    used: list[str] = []

    async def fake_call(prompt, key):
        used.append(key)
        if key == "bad":
            raise RateLimited("HTTP 429", 30)
        return "BUY", 0.7, "fine", "{}"

    monkeypatch.setattr(agent, "_call", fake_call)
    result = asyncio.run(agent.decide({"asset": "BTC"}, 0.25))
    assert used == ["bad", "good"]
    assert result.decision == "BUY" and result.status is AgentStatus.ACTIVE
    ring = agent.ring.status()
    assert ring["active_slot"] == 2 and ring["slots"][0]["state"] == "cooling"
    assert ring["slots"][0]["cooldown_seconds"] >= 29
    assert "rate limited" in agent.last_failover
    # next cycle starts straight on the backup, primary untouched while cooling
    used.clear()
    asyncio.run(agent.decide({"asset": "BTC"}, 0.25))
    assert used == ["good"]


def test_gemini_reports_rate_limited_only_when_every_key_is_cooling(monkeypatch):
    settings = _settings()
    settings.rings["gemini"] = KeyRing.from_keys("gemini", ["a", "b"])
    agent = GeminiAgent(settings)

    async def always_limited(prompt, key):
        raise RateLimited("HTTP 429")

    monkeypatch.setattr(agent, "_call", always_limited)
    result = asyncio.run(agent.decide({}, 0.25))
    assert result.status is AgentStatus.RATE_LIMITED
    assert agent.status() is AgentStatus.RATE_LIMITED
    assert "key 1" in result.error and "key 2" in result.error
    # A brand-new key pasted into slot 1 clears its cooldown immediately.
    agent.set_key("fresh", 1)
    assert agent.ring.current() == "fresh"


def test_github_rejected_pat_hands_over_to_the_backup(monkeypatch):
    settings = _settings()
    settings.rings["github"] = KeyRing.from_keys("github", ["revoked", "", "spare"])
    agent = GitHubAgent(settings)

    async def fake_call(prompt, token):
        if token == "revoked":
            raise PermissionError("rejected")
        return '{"decision": "SELL", "confidence": 0.6, "reasoning": "ok"}'

    monkeypatch.setattr(agent, "_call", fake_call)
    result = asyncio.run(agent.decide({}, 0.15))
    assert result.decision == "SELL"
    assert agent.ring.status()["active_slot"] == 3
    assert agent.ring.slots[0].reason.startswith("rejected")


# ---------------------------------------------------------------- pool
def test_three_local_models_answer_in_parallel_and_are_merged(tmp_path):
    settings = _settings()
    pool = LocalModelPool(settings, store_dir=tmp_path)
    assert len(pool.slots) == 3 and not pool.ready
    for agent in pool.slots:
        agent.stub = True
    asyncio.run(pool.load(tmp_path / "stub", slot=1))
    asyncio.run(pool.load(tmp_path / "stub", slot=2))
    asyncio.run(pool.load(tmp_path / "stub", slot=3))
    assert len(pool.loaded) == 3 and pool.status() is AgentStatus.STUB
    context = {"asset": "BTC", "formulas": {"MPS": 0.6, "SHRP": 0.4}, "ccs": 0.3}
    result = asyncio.run(pool.decide(context, 0.20))
    assert result.decision in ("BUY", "SELL")
    assert "3/3 models agree" in result.reasoning or "/3 models agree" in result.reasoning
    assert len(pool.last_results) == 3
    status = pool.slots_payload()
    assert [row["slot"] for row in status] == [1, 2, 3]
    assert all(row["ready"] for row in status)
    # one slot can be dropped without touching the others
    pool.unload(slot=2)
    assert len(pool.loaded) == 2
    assert pool.slots_payload()[1]["ready"] is False


def test_merge_is_a_confidence_weighted_majority_with_agreement_scaling():
    r = lambda n, d, c: AgentResult(agent="local", decision=d, confidence=c, status=AgentStatus.ACTIVE, model=f"m{n}", weight=0.2)  # noqa: E731
    merged = merge_results({1: r(1, "BUY", 0.9), 2: r(2, "SELL", 0.55), 3: r(3, "BUY", 0.6)}, 0.2, 5.0)
    assert merged.decision == "BUY"
    assert merged.confidence == pytest.approx(0.75 * (0.5 + 0.5 * 2 / 3), abs=1e-4)
    assert "2/3 models agree on BUY" in merged.reasoning
    # a model that did not answer is reported, not hidden
    missing = AgentResult(agent="local", status=AgentStatus.TIMEOUT, model="m3", weight=0.2, error="timed out")
    merged = merge_results({1: r(1, "SELL", 0.7), 3: missing}, 0.2, 5.0)
    assert merged.decision == "SELL" and "model 3: timed out" in merged.error
