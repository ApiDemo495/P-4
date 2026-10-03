"""Gemini agent (Section 7.1).

Three fixes over v1.0:

1. **Structured output** - ``responseMimeType: application/json`` plus a
   ``responseSchema`` means the model cannot return unparseable prose.
2. **Timeout** - 7 seconds; if Gemini is slow the cycle proceeds without it
   rather than holding the Signal Lock Controller hostage.
3. **v2 prompt** - all 22 formula values, the news summary and brain status.

Up to three keys (primary + two temporary stand-ins) with automatic failover:
a rejected or rate-limited key is cooled down (10 -> 20 -> ... -> 300 s) and the
next healthy key answers the same call; the primary is preferred again as soon
as its cooldown ends.  See ``backend/agents/keyring.py``.
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
NAME = "gemini"


class GeminiAgent:
    def __init__(self, settings=None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.ring = self.settings.rings["gemini"]
        self.model = self.settings.gemini_model
        self.consecutive_failures = 0
        self.last_result: AgentResult | None = None
        self.last_error = ""
        self.last_failover = ""
        self.calls = 0

    # ------------------------------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self.ring)

    @property
    def api_key(self) -> str:
        """The key that would be used right now (primary unless cooling)."""
        return self.ring.current()

    def set_key(self, key: str, slot: int = 1) -> None:
        self.ring.set_slot(slot, key)
        self.settings.gemini_api_key = self.ring.primary
        self.consecutive_failures = 0

    def status(self) -> AgentStatus:
        if not self.configured:
            return AgentStatus.DISABLED
        if self.ring.all_cooling():
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
        prompt = build_cycle_prompt(context)
        started = time.perf_counter()
        deadline = started + self.settings.gemini_timeout_seconds
        # Failover: try the primary, then each healthy backup, all inside the
        # single 7 s budget.  A key that fails is cooled down in the ring so
        # the next cycle starts on the best healthy one.
        keys = self.ring.candidates()
        if not keys:
            wait = self.ring.soonest_recovery()
            return AgentResult(
                agent=NAME, status=AgentStatus.RATE_LIMITED, model=self.model, weight=weight,
                error=f"All {self.ring.status()['configured_slots']} Gemini keys cooling down; "
                      f"next retry in {wait:.0f}s",
            )
        failures: list[str] = []
        decision = confidence = None
        reasoning = raw = ""
        used_key = ""
        for key in keys:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.5:
                break
            slot = self.ring.slot_of(key)
            try:
                decision, confidence, reasoning, raw = await asyncio.wait_for(
                    self._call(prompt, key), timeout=remaining
                )
                used_key = key
                self.ring.report_success(key)
                break
            except asyncio.TimeoutError:
                self.ring.report_failure(key, "error", "timeout")
                failures.append(f"key {slot}: timed out")
            except PermissionError as exc:
                self.ring.report_failure(key, "rejected", str(exc))
                failures.append(f"key {slot}: rejected")
            except RateLimited as exc:
                cooled = self.ring.report_failure(key, "rate_limited", str(exc), exc.retry_after)
                failures.append(f"key {slot}: rate limited (cooling {cooled:.0f}s)")
            except Exception as exc:  # noqa: BLE001
                self.ring.report_failure(key, "error", str(exc))
                failures.append(f"key {slot}: {exc}")
            if failures and slot and len(keys) > 1:
                log.warning("Gemini %s -> switching to the next key", failures[-1])

        if not used_key:
            self.consecutive_failures += 1
            self.last_error = "; ".join(failures) or "no key attempted"
            if any("rate limited" in f for f in failures) and self.ring.all_cooling():
                status = AgentStatus.RATE_LIMITED
            elif any("timed out" in f for f in failures):
                status = AgentStatus.TIMEOUT
            else:
                status = AgentStatus.ERROR
            return AgentResult(
                agent=NAME, status=status, model=self.model, weight=weight,
                error=self.last_error, latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        self.consecutive_failures = 0
        self.calls += 1
        self.last_failover = "; ".join(failures)
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

    async def _call(self, prompt: str, key: str) -> tuple[str | None, float | None, str, str]:
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
        headers = {"Content-Type": "application/json", "x-goog-api-key": key}
        async with httpx.AsyncClient(timeout=self.settings.gemini_timeout_seconds) as client:
            response = await client.post(url, json=payload, headers=headers)
        if response.status_code in (401, 403):
            raise PermissionError("Gemini rejected the API key")
        if response.status_code == 429:
            raise RateLimited("Gemini rate limit (HTTP 429)", _retry_after(response))
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
            detail = f"All keys cooling down, next retry in {self.ring.soonest_recovery():.0f}s"
        elif self.last_result is not None:
            detail = f"{self.last_result.decision or 'no answer'} in {self.last_result.latency_ms:.0f}ms"
        else:
            detail = "Ready"
        return AgentHealth(
            name=NAME,
            status=status,
            detail=detail,
            model=self.model,
            extra={"calls": self.calls, "last_error": self.last_error,
                   "last_failover": self.last_failover, "keys": self.ring.status()},
        )


class RateLimited(RuntimeError):
    """Raised internally when the provider returns HTTP 429."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after(response) -> float | None:
    try:
        value = response.headers.get("retry-after")
        return float(value) if value else None
    except (TypeError, ValueError):
        return None
