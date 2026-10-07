"""Local model pool - up to three local models running at the same time.

Each slot is a full :class:`LocalModelAgent` (upload -> validate -> load ->
test inference -> monitor, Section 7.2).  Every slot is optional; whatever is
loaded answers every cycle **in parallel** and the answers are merged into the
single "local" opinion that fusion weights at 0.20:

* side  = the confidence-weighted majority of the models that answered
* confidence = mean confidence of the models on the winning side, scaled by
  the agreement (3 of 3 agreeing keeps it, a 2-1 split trims it)
* reasoning = one line per model, so the dashboard shows who said what

A slot that crashes or times out is reported as such and simply drops out of
the merge for that cycle - the others still count.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from backend.agents.base import AgentHealth, AgentResult, AgentStatus
from backend.agents.local_model_agent import (
    NAME,
    LocalModelAgent,
    ValidationResult,
    _human_bytes,
    process_memory,
)
from backend.core import config as cfg

MAX_MODELS = 3


class LocalModelPool:
    def __init__(self, settings=None, store_dir: Path | None = None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.store_dir = Path(store_dir or cfg.MODEL_STORE_DIR)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        # Slot 1 keeps the historical store directory so a model uploaded
        # before this round is still where the user left it.
        self.slots: list[LocalModelAgent] = [
            LocalModelAgent(self.settings, store_dir=self.store_dir if i == 0 else self.store_dir / f"slot{i + 1}")
            for i in range(MAX_MODELS)
        ]
        self.last_result: AgentResult | None = None
        self.last_results: dict[int, AgentResult] = {}

    # ------------------------------------------------------------------
    # Compatibility surface: the rest of the engine talks to "the local
    # agent"; slot 1 answers those questions unless a pool-wide view exists.
    # ------------------------------------------------------------------
    def slot(self, number: int | None) -> LocalModelAgent:
        index = min(max(int(number or 1), 1), MAX_MODELS) - 1
        return self.slots[index]

    @property
    def loaded(self) -> list[LocalModelAgent]:
        return [s for s in self.slots if s.ready]

    @property
    def ready(self) -> bool:
        return any(s.ready for s in self.slots)

    @property
    def crashed(self) -> bool:
        return any(s.crashed for s in self.slots) and not self.ready

    @property
    def stub(self) -> bool:
        return self.slots[0].stub

    @stub.setter
    def stub(self, value: bool) -> None:
        self.slots[0].stub = bool(value)

    @property
    def validation(self) -> ValidationResult | None:
        return self.slots[0].validation

    @property
    def model_name(self) -> str:
        names = [s.model_name for s in self.loaded]
        return " + ".join(names) if names else ""

    @property
    def last_error(self) -> str:
        return "; ".join(f"model {i + 1}: {s.last_error}" for i, s in enumerate(self.slots) if s.last_error)

    def status(self) -> AgentStatus:
        loaded = self.loaded
        if not loaded:
            return AgentStatus.ERROR if any(s.crashed for s in self.slots) else AgentStatus.DISABLED
        if all(s.stub for s in loaded):
            return AgentStatus.STUB
        return AgentStatus.ACTIVE

    async def ensure_stub(self) -> bool:
        return await self.slots[0].ensure_stub()

    # ------------------------------------------------------------------
    async def upload(self, filename: str, payload: bytes, slot: int = 1) -> dict:
        result = await self.slot(slot).upload(filename, payload)
        result["slot"] = min(max(int(slot), 1), MAX_MODELS)
        return result

    async def load(self, path: Path, slot: int = 1) -> dict:
        return await self.slot(slot).load(path)

    def unload(self, delete_file: bool = False, slot: int | None = None) -> dict:
        if slot is None:
            freed = 0
            for agent in self.slots:
                freed += int(agent.unload(delete_file=delete_file).get("freed_bytes", 0))
            return {"unloaded": True, "slots": MAX_MODELS, "freed_bytes": freed,
                    "freed_human": _human_bytes(freed)}
        result = self.slot(slot).unload(delete_file=delete_file)
        result["slot"] = min(max(int(slot), 1), MAX_MODELS)
        return result

    def check_memory(self) -> dict:
        payload = {"rss_bytes": process_memory(), "rss_human": _human_bytes(process_memory())}
        for index, agent in enumerate(self.slots):
            if agent.ready:
                info = agent.check_memory()
                if info.get("unloaded"):
                    payload[f"slot{index + 1}_unloaded"] = True
        return payload

    # ------------------------------------------------------------------
    async def decide(self, context: dict, weight: float) -> AgentResult:
        if self.slots[0].stub and not self.slots[0].ready:
            await self.slots[0].ensure_stub()
        active = [(i + 1, s) for i, s in enumerate(self.slots) if s.ready]
        if not active:
            crashed = any(s.crashed for s in self.slots)
            return AgentResult(
                agent=NAME,
                status=AgentStatus.ERROR if crashed else AgentStatus.DISABLED,
                model="",
                weight=weight,
                error=self.last_error or "No local model loaded",
            )
        if len(active) == 1:
            number, agent = active[0]
            result = await agent.decide(context, weight)
            self.last_results = {number: result}
            self.last_result = result
            return result

        started = time.perf_counter()
        outcomes = await asyncio.gather(
            *(agent.decide(context, weight) for _, agent in active), return_exceptions=True
        )
        results: dict[int, AgentResult] = {}
        for (number, agent), outcome in zip(active, outcomes):
            if isinstance(outcome, BaseException):
                outcome = AgentResult(agent=NAME, status=AgentStatus.ERROR, model=agent.model_name,
                                      weight=weight, error=str(outcome))
            results[number] = outcome
        self.last_results = results
        merged = merge_results(results, weight, latency_ms=(time.perf_counter() - started) * 1000.0)
        self.last_result = merged
        return merged

    # ------------------------------------------------------------------
    def health(self) -> AgentHealth:
        loaded = self.loaded
        status = self.status()
        if not loaded:
            detail = "No model loaded (up to 3 slots)"
            if any(s.crashed for s in self.slots):
                detail = "Crashed - reload required: " + self.last_error
        elif len(loaded) == 1:
            detail = loaded[0].health().detail
        else:
            detail = f"{len(loaded)} models in parallel · " + " · ".join(
                f"{s.model_name} {s.inference_ms:.0f}ms" for s in loaded
            )
        base = loaded[0] if loaded else self.slots[0]
        return AgentHealth(
            name=NAME,
            status=status,
            detail=detail,
            model=self.model_name,
            extra={
                "kind": base.model_kind,
                "parameter_count": base.parameter_count,
                "memory": process_memory(),
                "memory_human": _human_bytes(process_memory()),
                "stub": all(s.stub for s in loaded) if loaded else self.slots[0].stub,
                "last_error": self.last_error,
                "models_loaded": len(loaded),
                "slots": self.slots_payload(),
            },
        )

    def slots_payload(self) -> list[dict]:
        rows = []
        for index, agent in enumerate(self.slots):
            health = agent.health().to_dict()
            health["slot"] = index + 1
            health["ready"] = agent.ready
            health["file"] = str(agent.model_path.name) if agent.model_path else ""
            last = self.last_results.get(index + 1)
            if last is not None:
                health["last_decision"] = last.decision
                health["last_confidence"] = last.confidence
                health["last_latency_ms"] = round(last.latency_ms, 1)
            if agent.validation is not None:
                health["validation"] = agent.validation.to_dict()
            rows.append(health)
        return rows


def merge_results(results: dict[int, AgentResult], weight: float, latency_ms: float) -> AgentResult:
    """Confidence-weighted majority of the models that answered."""
    answered = {n: r for n, r in results.items() if r.decision}
    models = " + ".join(f"{n}:{r.model or 'model'}" for n, r in results.items())
    if not answered:
        errors = "; ".join(f"model {n}: {r.error or r.status.value}" for n, r in results.items())
        return AgentResult(agent=NAME, status=AgentStatus.ERROR, model=models, weight=weight,
                           error=errors or "no local model answered", latency_ms=latency_ms)
    tally: dict[str, float] = {}
    for r in answered.values():
        tally[r.decision] = tally.get(r.decision, 0.0) + max(0.05, float(r.confidence or 0.5))
    decision = max(tally, key=tally.get)
    winners = [r for r in answered.values() if r.decision == decision]
    mean_conf = sum(float(r.confidence or 0.5) for r in winners) / len(winners)
    agreement = len(winners) / len(answered)
    confidence = round(mean_conf * (0.5 + 0.5 * agreement), 4)
    reasoning = " | ".join(
        f"model {n} ({r.model or 'local'}): {r.decision} {float(r.confidence or 0):.0%}"
        + (f" - {r.reasoning}" if r.reasoning else "")
        for n, r in answered.items()
    )
    missing = [f"model {n}: {r.error or r.status.value}" for n, r in results.items() if not r.decision]
    stub = all(r.status is AgentStatus.STUB for r in answered.values())
    return AgentResult(
        agent=NAME,
        decision=decision,
        confidence=confidence,
        reasoning=f"{len(winners)}/{len(answered)} models agree on {decision}. {reasoning}",
        status=AgentStatus.STUB if stub else AgentStatus.ACTIVE,
        model=models,
        weight=weight,
        latency_ms=latency_ms,
        raw="\n".join(r.raw for r in answered.values() if r.raw),
        error="; ".join(missing),
    )
