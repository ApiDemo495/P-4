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
import re
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

#: Round AF/AH - Google retires model ids (``gemini-1.5-flash`` answers 404)
#: and ships new generations faster than any hard-coded list.  The agent asks
#: the API which models THIS key can use (ListModels) and ranks them by
#: generation: the NEWEST version wins (3.8 > 3.7 > ... > 3.5 > 3.0 > 2.5),
#: then the family (flash > flash-lite > pro - this engine needs a 7 s answer),
#: then GA over preview.  ``GEMINI_MODEL`` still pins one explicitly.  The
#: list below is only the tie-break order / the offline default.
PREFERRED_MODELS: tuple[str, ...] = (
    "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
    "gemini-3.5-flash-lite", "gemini-3.5-pro", "gemini-3-flash", "gemini-3-pro",
    "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest", "gemini-2.5-pro",
)
#: never pick these for a text decision
_EXCLUDE = ("embedding", "image", "tts", "live", "audio", "native-audio", "veo", "imagen", "robotics",
            "computer-use", "deep-research", "aqa", "learnlm", "gemma", "thinking-exp")
_VERSION = re.compile(r"gemini-(\d+)(?:\.(\d+))?")
_FAMILY_RANK = {"flash": 3, "flash-lite": 2, "pro": 1}


def model_rank(model_id: str) -> tuple:
    """Sort key: newest generation first, flash before pro, GA before preview.
    Returns ``None`` for ids that are not text-decision models."""
    m = model_id.lower()
    if not m.startswith("gemini") or any(x in m for x in _EXCLUDE):
        return None
    v = _VERSION.search(m)
    if v:
        major, minor = int(v.group(1)), int(v.group(2) or 0)
    elif "latest" in m:
        major, minor = 0, 0          # "gemini-flash-latest" - unknown generation, keep as fallback
    else:
        return None
    family = "flash-lite" if "flash-lite" in m else "flash" if "flash" in m else "pro" if "pro" in m else None
    if family is None:
        return None
    preview = any(x in m for x in ("preview", "exp", "-0", "latest"))
    return (major, minor, _FAMILY_RANK[family], 0 if preview else 1, m)


def choose_model(available: list[str], pinned: str = "") -> str:
    """Pick a usable model id from a ListModels answer (names come back as
    ``models/<id>``).  A pinned id wins when the key can use it; otherwise the
    newest flash-class model this key can call."""
    ids = [m.split("/", 1)[-1] for m in available]
    if pinned and pinned != "auto" and pinned in ids:
        return pinned
    ranked = sorted((r, i) for i in ids if (r := model_rank(i)) is not None)
    if ranked:
        return ranked[-1][1]
    text = [m for m in ids if m.startswith("gemini") and "embedding" not in m and "image" not in m]
    return text[0] if text else (pinned if pinned and pinned != "auto" else PREFERRED_MODELS[0])


async def list_models(key: str, timeout: float = 8.0) -> list[str]:
    """Model ids this key may call with generateContent (raises on HTTP errors)."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(f"{BASE_URL}?pageSize=200", headers={"x-goog-api-key": key})
    if response.status_code in (400, 401, 403):
        raise PermissionError(f"Gemini rejected the API key (HTTP {response.status_code})")
    if response.status_code == 429:
        raise RateLimited("Gemini rate limit (HTTP 429)", _retry_after(response))
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini HTTP {response.status_code}: {response.text[:200]}")
    out = []
    for m in response.json().get("models") or []:
        methods = m.get("supportedGenerationMethods") or []
        if "generateContent" in methods:
            out.append(str(m.get("name", "")))
    return out


class GeminiAgent:
    def __init__(self, settings=None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.ring = self.settings.rings["gemini"]
        self.model = self.settings.gemini_model
        self._model_resolved = False
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

    async def _resolve_model(self, key: str, force: bool = False) -> str:
        """Make sure ``self.model`` is an id this key can actually call."""
        if self._model_resolved and not force:
            return self.model
        available = await list_models(key)
        chosen = choose_model(available, self.settings.gemini_model)
        if chosen != self.model:
            log.info("Gemini model %s -> %s (what this key can use)", self.model, chosen)
        self.model = chosen
        self._model_resolved = True
        return chosen

    async def _call(self, prompt: str, key: str) -> tuple[str | None, float | None, str, str]:
        await self._resolve_model(key)
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
        if response.status_code == 404:
            # the model id was retired under us: re-discover once, retry once
            self._model_resolved = False
            await self._resolve_model(key, force=True)
            url = f"{BASE_URL}/{self.model}:generateContent"
            async with httpx.AsyncClient(timeout=self.settings.gemini_timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=headers)
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
        # Any key format Google issues is accepted (AIza… and the newer AQ.Ab…
        # keys alike): the API decides, not a prefix check.  Validation goes
        # through ListModels, which no model retirement can break, and the
        # answer names the model the engine will call with this key.
        try:
            available = await list_models(candidate)
        except PermissionError as exc:
            return {"valid": False, "error": f"Google rejected this key - {exc}"}
        except RateLimited:
            return {"valid": True, "detail": "Key accepted (rate limited right now)"}
        except Exception as exc:  # noqa: BLE001
            return {"valid": False, "error": f"could not reach generativelanguage.googleapis.com: {exc}"}
        if not available:
            return {"valid": False, "error": "key is valid but has no model with generateContent enabled"}
        chosen = choose_model(available, self.settings.gemini_model)
        if key is None or candidate == (self.api_key or "").strip():
            self.model, self._model_resolved = chosen, True
        return {"valid": True, "detail": f"key accepted - {len(available)} models available, using {chosen}",
                "model": chosen, "models": [m.split("/", 1)[-1] for m in available][:40]}

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
