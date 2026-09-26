"""Agent orchestrator - runs the four agents inside the 8-second lock budget.

Gemini, the local model and the GitHub agent are all called concurrently.  Each
has its own timeout (7 s) so the *slowest* agent still cannot push the lock past
t=8 s: the fusion step simply proceeds with whoever answered.
"""

from __future__ import annotations

import asyncio
import logging
import time

from backend.agents.base import AgentResult, AgentStatus
from backend.agents.gemini_agent import GeminiAgent
from backend.agents.github_agent import GitHubAgent
from backend.agents.local_model_agent import LocalModelAgent
from backend.core import config as cfg

log = logging.getLogger("drosophila.agents")


class AgentOrchestrator:
    def __init__(self, settings=None, local_stub: bool | None = None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.gemini = GeminiAgent(self.settings)
        self.local = LocalModelAgent(self.settings)
        self.github = GitHubAgent(self.settings)
        if local_stub is not None:
            self.local.stub = local_stub
        self.last_run_ms = 0.0
        self.last_results: dict[str, AgentResult] = {}

    # ------------------------------------------------------------------
    @property
    def agents(self) -> dict:
        return {"gemini": self.gemini, "local": self.local, "github": self.github}

    def weights(self) -> dict[str, float]:
        return {
            "drosophila": self.settings.weight_drosophila,
            "gemini": self.settings.weight_gemini,
            "local": self.settings.weight_local,
            "github": self.settings.weight_github,
        }

    # ------------------------------------------------------------------
    async def run_all(self, context: dict) -> dict[str, AgentResult]:
        weights = self.weights()
        started = time.perf_counter()

        gemini_task = asyncio.create_task(self.gemini.decide(context, weights["gemini"]))
        local_task = asyncio.create_task(self.local.decide(context, weights["local"]))
        github_task = asyncio.create_task(self.github.decide(context, weights["github"]))

        gathered = await asyncio.gather(
            gemini_task, local_task, github_task, return_exceptions=True
        )
        names = ("gemini", "local", "github")
        results: dict[str, AgentResult] = {}
        for name, outcome in zip(names, gathered):
            if isinstance(outcome, BaseException):
                results[name] = AgentResult(
                    agent=name,
                    status=AgentStatus.ERROR,
                    error=str(outcome),
                    weight=weights[name],
                )
            else:
                results[name] = outcome

        self.last_results = results
        self.last_run_ms = (time.perf_counter() - started) * 1000.0
        return results

    # ------------------------------------------------------------------
    def context_block(self, results: dict[str, AgentResult]) -> dict:
        """The per-agent summary embedded in every SIGNAL message."""
        block: dict[str, dict] = {}
        for name, result in results.items():
            block[name] = {
                "decision": result.decision,
                "confidence": round(result.confidence, 4) if result.confidence is not None else None,
                "status": result.status.value,
                "model": result.model,
                "latency_ms": round(result.latency_ms, 1),
                "reasoning": result.reasoning,
                "error": result.error,
            }
        return block

    def status_payload(self) -> dict:
        payload = {
            name: agent.health().to_dict() for name, agent in self.agents.items()
        }
        payload["weights"] = self.weights()
        payload["last_run_ms"] = round(self.last_run_ms, 1)
        return payload
