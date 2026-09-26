"""Gemini agent (Section 7.1).

Three fixes over v1.0:

1. **Structured output** - ``responseMimeType: application/json`` plus a
   ``responseSchema`` means the model cannot return unparseable prose.
2. **Timeout** - 7 seconds; if Gemini is slow the cycle proceeds without it
   rather than holding the Signal Lock Controller hostage.
3. **v2 prompt** - all 22 formula values, the news summary and brain status.

Rate limits back off exponentially: 10 -> 20 -> 40 -> 80 -> 160 -> 300 seconds.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from backend.agents.base import (
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    AgentHealth,
    AgentResult,
    AgentStatus,
    build_cycle_prompt,
    parse_agent_json,
)
from backend.core import config as cfg

log = logging.getLogger("drosophila.agents.gemini")

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
BACKOFF_SCHEDULE = (10.0, 20.0, 40.0, 80.0, 160.0, 300.0)
NAME = "gemini"


class GeminiAgent:
    def __init__(self, settings=None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.api_key = self.settings.gemini_api_key
        self.model = self.settings.gemini_model
        self.consecutive_failures = 0
        self.backoff_until = 0.0
        self.last_result: AgentResult | None = None
        self.last_error = ""
        self.calls = 0

    # ------------------------------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def set_key(self, key: str) -> None:
        self.api_key = (key or "").strip()
        self.settings.gemini_api_key = self.api_key
        self.consecutive_failures = 0
        self.backoff_until = 0.0

    def status(self) -> AgentStatus:
        if not self.configured:
            return AgentStatus.DISABLED
        if self.backoff_until > time.time():
            return AgentStatus.RATE_LIMITED
        if self.consecutive_failures >= 3:
            return AgentStatus.ERROR
        return AgentStatus.ACTIVE

    # ------------------------------------------------------------------
    async def decide(self, context: dict, weight: float) -> AgentResult:
        if not self.configured:
            return AgentResult(
                agent=NAME, status=AgentStatus.DISABLED, model=self.model,
                error="No Gemini API key configured", weight=weight,
            )
        now = time.time()
        if self.backoff_until > now:
            return AgentResult(
                agent=NAME,
                status=AgentStatus.RATE_LIMITED,
                model=self.model,
                weight=weight,
                error=f"Cooling down for {self.backoff_until - now:.0f}s",
            )

        prompt = build_cycle_prompt(context)
        started = time.perf_counter()
        try:
            decision, confidence, reasoning, raw = await asyncio.wait_for(
                self._call(prompt), timeout=self.settings.gemini_timeout_seconds
            )
        except asyncio.TimeoutError:
            self.consecutive_failures += 1
            return AgentResult(
                agent=NAME,
                status=AgentStatus.TIMEOUT,
                model=self.model,
                weight=weight,
                error=f"Timed out after {self.settings.gemini_timeout_seconds:.0f}s",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except PermissionError as exc:
            self.last_error = str(exc)
            return AgentResult(
                agent=NAME, status=AgentStatus.ERROR, model=self.model, weight=weight,
                error=f"Invalid key: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except RateLimited as exc:
            self._schedule_backoff()
            return AgentResult(
                agent=NAME,
                status=AgentStatus.RATE_LIMITED,
                model=self.model,
                weight=weight,
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            self.consecutive_failures += 1
            self.last_error = str(exc)
            return AgentResult(
                agent=NAME,
                status=AgentStatus.ERROR,
                model=self.model,
                weight=weight,
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        self.consecutive_failures = 0
        self.calls += 1
        result = AgentResult(
            agent=NAME,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            status=AgentStatus.ACTIVE if decision else AgentStatus.ERROR,
            model=self.model,
            weight=weight,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            raw=raw,
            error="" if decision else "model returned no usable decision",
        )
        self.last_result = result
        return result

    def _schedule_backoff(self) -> None:
        index = min(self.consecutive_failures, len(BACKOFF_SCHEDULE) - 1)
        delay = BACKOFF_SCHEDULE[index]
        self.consecutive_failures += 1
        self.backoff_until = time.time() + delay
        log.warning("Gemini rate limited; cooling down %.0fs", delay)

    async def _call(self, prompt: str) -> tuple[str | None, float | None, str, str]:
        url = f"{BASE_URL}/{self.model}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 512,
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA,
            },
        }
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        async with httpx.AsyncClient(timeout=self.settings.gemini_timeout_seconds) as client:
            response = await client.post(url, json=payload, headers=headers)
        if response.status_code in (401, 403):
            raise PermissionError("Gemini rejected the API key")
        if response.status_code == 429:
            raise RateLimited("Gemini rate limit (HTTP 429)")
        if response.status_code >= 400:
            raise RuntimeError(f"Gemini HTTP {response.status_code}: {response.text[:200]}")

        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return None, None, "", ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(str(p.get("text", "")) for p in parts)
        decision, confidence, reasoning = parse_agent_json(text)
        return decision, confidence, reasoning, text

    # ------------------------------------------------------------------
    async def test_key(self, key: str | None = None) -> dict:
        """'Test' button in Settings - validates without changing state."""
        candidate = (key or self.api_key or "").strip()
        if not candidate:
            return {"valid": False, "error": "No key entered"}
        url = f"{BASE_URL}/{self.model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "Reply with the single word: ok"}]}],
            "generationConfig": {"maxOutputTokens": 8, "temperature": 0.0},
        }
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.post(
                    url, json=payload, headers={"x-goog-api-key": candidate}
                )
            if response.status_code == 200:
                return {"valid": True, "detail": f"{self.model} reachable"}
            if response.status_code in (401, 403):
                return {"valid": False, "error": "Invalid Gemini API key"}
            if response.status_code == 429:
                return {"valid": True, "detail": "Key accepted (rate limited right now)"}
            return {"valid": False, "error": f"HTTP {response.status_code}"}
        except Exception as exc:  # noqa: BLE001
            return {"valid": False, "error": str(exc)}

    def health(self) -> AgentHealth:
        status = self.status()
        if status is AgentStatus.DISABLED:
            detail = "No API key"
        elif status is AgentStatus.RATE_LIMITED:
            detail = f"Cooling down {max(0, self.backoff_until - time.time()):.0f}s"
        elif self.last_result is not None:
            detail = f"{self.last_result.decision or 'no answer'} in {self.last_result.latency_ms:.0f}ms"
        else:
            detail = "Ready"
        return AgentHealth(
            name=NAME,
            status=status,
            detail=detail,
            model=self.model,
            extra={"calls": self.calls, "last_error": self.last_error},
        )


class RateLimited(RuntimeError):
    """Raised internally when the provider returns HTTP 429."""
