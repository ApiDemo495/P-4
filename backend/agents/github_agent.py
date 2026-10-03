"""GitHub Models agent (Section 7.3) - optional fourth opinion.

Adds over v1.0: the PAT is tested the moment it is saved, in two steps, so a
user cannot end up with "saved but silently useless":

    GET https://api.github.com/user                                  -> PAT valid?
    GET https://models.inference.ai.azure.com/models                 -> Models reachable?

Transient 5xx responses retry three times and then rest *that PAT* for five
minutes; with a second or third PAT pasted in Settings the next one answers the
same call (``backend/agents/keyring.py``), and the primary is used again as
soon as it recovers.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from backend.agents.base import (
    SYSTEM_PROMPT,
    AgentHealth,
    AgentResult,
    AgentStatus,
    build_cycle_prompt,
    parse_agent_json,
)
from backend.core import config as cfg

log = logging.getLogger("drosophila.agents.github")

NAME = "github"
USER_URL = "https://api.github.com/user"
MODELS_URL = "https://models.inference.ai.azure.com"
CHAT_PATH = "/chat/completions"
COOLDOWN_SECONDS = 300.0
RETRIES = 3


class GitHubAgent:
    def __init__(self, settings=None) -> None:
        self.settings = settings or cfg.SETTINGS
        self.ring = self.settings.rings["github"]
        self.model = self.settings.github_models_model
        self.last_result: AgentResult | None = None
        self.last_error = ""
        self.last_failover = ""
        self.calls = 0

    # ------------------------------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self.ring)

    @property
    def token(self) -> str:
        return self.ring.current()

    def set_token(self, token: str, slot: int = 1) -> None:
        self.ring.set_slot(slot, token)
        self.settings.github_models_token = self.ring.primary

    def status(self) -> AgentStatus:
        if not self.configured:
            return AgentStatus.DISABLED
        if self.ring.all_cooling():
            return AgentStatus.INACTIVE
        return AgentStatus.ACTIVE

    # ------------------------------------------------------------------
    async def decide(self, context: dict, weight: float) -> AgentResult:
        if not self.configured:
            return AgentResult(
                agent=NAME, status=AgentStatus.DISABLED, model=self.model, weight=weight,
                error="No GitHub PAT configured",
            )
        keys = self.ring.candidates()
        if not keys:
            return AgentResult(
                agent=NAME, status=AgentStatus.INACTIVE, model=self.model, weight=weight,
                error=f"All {self.ring.status()['configured_slots']} GitHub PATs cooling down, "
                      f"retry in {self.ring.soonest_recovery():.0f}s",
            )

        prompt = build_cycle_prompt(context)
        started = time.perf_counter()
        deadline = started + self.settings.github_timeout_seconds
        failures: list[str] = []
        output = ""
        used = ""
        for key in keys:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.5:
                break
            slot = self.ring.slot_of(key)
            try:
                output = await asyncio.wait_for(self._call(prompt, key), timeout=remaining)
                used = key
                self.ring.report_success(key)
                break
            except asyncio.TimeoutError:
                self.ring.report_failure(key, "error", "timeout")
                failures.append(f"PAT {slot}: timed out")
            except PermissionError as exc:
                self.ring.report_failure(key, "rejected", str(exc))
                failures.append(f"PAT {slot}: rejected")
            except RateLimited as exc:
                cooled = self.ring.report_failure(key, "rate_limited", str(exc), exc.retry_after)
                failures.append(f"PAT {slot}: rate limited (cooling {cooled:.0f}s)")
            except Exception as exc:  # noqa: BLE001
                # Transient 5xx: this PAT rests for five minutes, the next one
                # answers now.
                self.ring.report_failure(key, "error", str(exc), COOLDOWN_SECONDS)
                failures.append(f"PAT {slot}: {exc}")
            if len(keys) > 1:
                log.warning("GitHub Models %s -> switching to the next PAT", failures[-1])
        if not used:
            self.last_error = "; ".join(failures) or "no PAT attempted"
            if any("timed out" in f for f in failures):
                status = AgentStatus.TIMEOUT
            elif any("rejected" in f for f in failures) and not self.ring.all_cooling():
                status = AgentStatus.ERROR
            else:
                status = AgentStatus.INACTIVE
            return AgentResult(
                agent=NAME, status=status, model=self.model, weight=weight,
                error=self.last_error, latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        self.last_failover = "; ".join(failures)

        decision, confidence, reasoning = parse_agent_json(output)
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
            raw=output,
            error="" if decision else "unparseable output",
        )
        self.last_result = result
        return result

    async def _call(self, prompt: str, token: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.settings.github_timeout_seconds) as client:
            for attempt in range(RETRIES):
                try:
                    response = await client.post(MODELS_URL + CHAT_PATH, json=payload, headers=headers)
                    if response.status_code in (401, 403):
                        raise PermissionError("GitHub PAT rejected by the Models API")
                    if response.status_code == 429:
                        retry = response.headers.get("retry-after")
                        raise RateLimited("GitHub Models rate limit (HTTP 429)",
                                          float(retry) if retry and retry.isdigit() else None)
                    if response.status_code >= 500:
                        raise RuntimeError(f"HTTP {response.status_code}")
                    if response.status_code >= 400:
                        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:160]}")
                    data = response.json()
                    return str(data["choices"][0]["message"]["content"])
                except (PermissionError, RateLimited):
                    raise
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if attempt < RETRIES - 1:
                        await asyncio.sleep(0.6 * (attempt + 1))
        raise RuntimeError(str(last_error))

    # ------------------------------------------------------------------
    async def test_token(self, token: str | None = None) -> dict:
        """Two-step validation, run when the user saves the PAT."""
        candidate = (token or self.token or "").strip()
        if not candidate:
            return {"valid": False, "error": "No token entered"}
        headers = {"Authorization": f"Bearer {candidate}", "Accept": "application/vnd.github+json"}
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                user = await client.get(USER_URL, headers=headers)
                if user.status_code != 200:
                    return {"valid": False, "error": "Invalid GitHub PAT"}
                login = str((user.json() or {}).get("login") or "")

                models = await client.get(MODELS_URL + "/models", headers=headers)
                if models.status_code != 200:
                    return {
                        "valid": False,
                        "error": "PAT valid but GitHub Models API not accessible",
                        "detail": f"signed in as {login}",
                    }
                return {"valid": True, "detail": f"Signed in as {login}; Models API accessible"}
        except Exception as exc:  # noqa: BLE001
            return {"valid": False, "error": str(exc)}

    def health(self) -> AgentHealth:
        status = self.status()
        if status is AgentStatus.DISABLED:
            detail = "Disabled (no PAT)"
        elif status is AgentStatus.INACTIVE:
            detail = f"All PATs temporarily unavailable, {self.ring.soonest_recovery():.0f}s left"
        else:
            detail = "Ready"
        return AgentHealth(
            name=NAME, status=status, detail=detail, model=self.model,
            extra={"calls": self.calls, "last_error": self.last_error,
                   "last_failover": self.last_failover, "keys": self.ring.status()},
        )


class RateLimited(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
