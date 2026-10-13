"""Round AY - the local model cannot starve the engine: one inference at a
time, a deadline the token loop honours, and a vote budget sized to the CPU."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from backend.agents.local_model_agent import LocalModelAgent


class _SlowGGUF:
    """A fake llama: streams one token every `step` seconds, forever."""

    def __init__(self, step: float, text: str = '{"decision": "BUY", "confidence": 0.7, "reasoning": "x"}'):
        self.step = step
        self.text = text
        self.closed = 0

    def tokenize(self, data: bytes, add_bos: bool = True):
        return list(range(max(1, len(data) // 4)))

    def create_chat_completion(self, messages, max_tokens, temperature, stream=False):
        assert stream is True
        parent = self

        class _Stream:
            def __iter__(self_inner):
                for i, ch in enumerate(parent.text * 50):
                    if i >= max_tokens:
                        return
                    time.sleep(parent.step)
                    yield {"choices": [{"delta": {"content": ch}}]}

            def close(self_inner):
                parent.closed += 1

        return _Stream()


def _agent(model, budget: float) -> LocalModelAgent:
    agent = LocalModelAgent(settings=SimpleNamespace(local_timeout_seconds=budget, local_model_dir="/tmp"))
    agent.model = model
    agent.stub = False
    agent.model_kind = "gguf"
    agent.model_name = "fake"
    agent.loaded_at = time.time()
    agent.validation = SimpleNamespace(ok=True)
    return agent


@pytest.mark.asyncio
async def test_generation_stops_at_the_deadline_and_never_stacks():
    ctx = {"asset": "BTC", "price": 1.0, "formulas": {}, "timestamp": ""}
    # 20 ms per token, a 0.3 s budget: the loop must stop itself at ~0.3 s
    agent = _agent(_SlowGGUF(0.02), budget=0.3)
    assert agent.ready
    t0 = time.perf_counter()
    first = asyncio.create_task(agent.decide(ctx, 0.2))
    await asyncio.sleep(0.05)
    # a second window while the first vote is generating: skipped, not stacked
    second = await agent.decide(ctx, 0.2)
    assert second.status.value == "TIMEOUT" and "skipped" in second.error
    assert agent.skipped_busy == 1
    result = await first
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.5, elapsed                      # not 256 tokens x 20 ms = 5 s
    assert agent.overruns == 1 or result.decision == "BUY"
    assert agent._inflight is False
    # the JSON object closed -> the vote is usable even though it was cut short
    fast = _agent(_SlowGGUF(0.0005), budget=2.0)
    ok = await fast.decide(ctx, 0.2)
    assert ok.decision == "BUY" and ok.status.value == "ACTIVE"
    assert fast.tokens_per_second > 0 and 48 <= fast.vote_tokens <= 160
    assert fast.too_slow is False and fast.estimated_vote_s > 0

    # a CPU that needs far more than the budget: measured, then voted every 5th window
    slow = _agent(_SlowGGUF(0.02), budget=0.3)
    await slow.decide(ctx, 0.2)
    slow.prompt_tps = 5.0          # 5 prompt tokens per second -> dozens of seconds per vote
    slow._refresh_estimate(400)
    assert slow.too_slow and slow.estimated_vote_s > 40
    skipped = [await slow.decide(ctx, 0.2) for _ in range(4)]
    assert all("too slow" in r.error for r in skipped) and slow.skipped_slow == 4
    health = slow.health()
    assert health.extra["too_slow"] is True and "TOO SLOW" in health.detail


@pytest.mark.asyncio
async def test_the_stub_reads_the_window_formulas():
    from backend.agents.base import build_cycle_prompt
    from backend.agents.local_model_agent import _StubModel
    pinned = build_cycle_prompt({"asset": "BTC", "price": 1.0, "formulas": {"CCSv2": 0.4, "TAI": 0.3, "PIN": 0.7, "MPL": -0.5}})
    out = _StubModel().generate(pinned)
    assert '"SELL"' in out      # pinned tape: the microprice lean decides, not the bullish core
