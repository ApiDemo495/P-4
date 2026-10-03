"""Key rings: up to three API keys per provider with automatic failover.

The user pastes one, two or three keys per provider.  Slot 1 is the
**primary** and is always preferred; slots 2 and 3 are temporary stand-ins.
When the key in use is rejected, rate limited or errors, it is put into a
cooldown and the next healthy slot takes over *for the very same call*.
The primary is re-tried the moment its cooldown expires, so the system
"comes back" to the primary automatically once it is fixed - the user never
has to press anything.

Cooldowns (per key, independent):

* rate limited (HTTP 429)  - 10 -> 20 -> 40 -> 80 -> 160 -> 300 s, doubling
  on every consecutive limit, reset on the first success
* rejected (HTTP 401/403)  - 10 minutes (the user may be fixing it right now)
* any other error / timeout - 60 s

Every ring is optional: an empty ring simply means "not configured".
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

MAX_SLOTS = 3
RATE_LIMIT_SCHEDULE = (10.0, 20.0, 40.0, 80.0, 160.0, 300.0)
REJECTED_COOLDOWN = 600.0
ERROR_COOLDOWN = 60.0

#: provider -> the .env / environment variable of slot 1 (slots 2 and 3 are
#: ``<NAME>_2`` and ``<NAME>_3``).
ENV_NAMES = {
    "gemini": "GEMINI_API_KEY",
    "github": "GITHUB_MODELS_TOKEN",
    "cryptopanic": "CRYPTOPANIC_API_KEY",
    "newsapi": "NEWSAPI_API_KEY",
}


def env_name(provider: str, slot: int) -> str:
    base = ENV_NAMES[provider]
    return base if slot <= 1 else f"{base}_{slot}"


def keys_from_env(provider: str) -> list[str]:
    """Slots 1..3 from the environment.  Slot 1 may also hold a
    comma-separated list (``KEY1,KEY2,KEY3``) for people who prefer it."""
    out: list[str] = []
    first = (os.environ.get(env_name(provider, 1)) or "").strip()
    if "," in first:
        out.extend(k.strip() for k in first.split(",") if k.strip())
    elif first:
        out.append(first)
    for slot in (2, 3):
        value = (os.environ.get(env_name(provider, slot)) or "").strip()
        if value:
            out.append(value)
    out = out[:MAX_SLOTS]
    return out + [""] * (MAX_SLOTS - len(out))


@dataclass
class KeySlot:
    key: str = ""
    cooldown_until: float = 0.0
    reason: str = ""
    consecutive_limits: int = 0
    successes: int = 0
    failures: int = 0
    last_used: float = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.key)

    def healthy(self, now: float | None = None) -> bool:
        return self.configured and self.cooldown_until <= (now or time.time())


@dataclass
class KeyRing:
    provider: str
    slots: list[KeySlot] = field(default_factory=lambda: [KeySlot() for _ in range(MAX_SLOTS)])
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, provider: str) -> "KeyRing":
        ring = cls(provider)
        for index, key in enumerate(keys_from_env(provider)):
            ring.slots[index].key = key
        return ring

    @classmethod
    def from_keys(cls, provider: str, keys: list[str] | tuple[str, ...]) -> "KeyRing":
        ring = cls(provider)
        for index, key in enumerate(list(keys)[:MAX_SLOTS]):
            ring.slots[index].key = (key or "").strip()
        return ring

    # ------------------------------------------------------------------
    @property
    def configured(self) -> bool:
        return any(s.configured for s in self.slots)

    def __bool__(self) -> bool:  # ``if ring:`` == "at least one key"
        return self.configured

    @property
    def primary(self) -> str:
        return self.slots[0].key

    def keys(self) -> list[str]:
        return [s.key for s in self.slots]

    def set_slot(self, slot: int, key: str) -> None:
        """Slot numbers are 1-based, like the boxes in Settings.  Setting a
        key clears its cooldown (the user just fixed it)."""
        index = min(max(int(slot), 1), MAX_SLOTS) - 1
        with self._lock:
            entry = self.slots[index]
            entry.key = (key or "").strip()
            entry.cooldown_until = 0.0
            entry.reason = ""
            entry.consecutive_limits = 0

    def set_keys(self, keys: list[str]) -> None:
        for index in range(MAX_SLOTS):
            self.set_slot(index + 1, keys[index] if index < len(keys) else "")

    # ------------------------------------------------------------------
    def candidates(self) -> list[str]:
        """Healthy keys in priority order (primary first).  If *every* key is
        cooling down, the one that recovers soonest is returned so a call is
        still attempted rather than the provider going silent."""
        now = time.time()
        with self._lock:
            healthy = [s.key for s in self.slots if s.healthy(now)]
            if healthy:
                return healthy
            cooling = [s for s in self.slots if s.configured]
            if not cooling:
                return []
            soonest = min(cooling, key=lambda s: s.cooldown_until)
            return [soonest.key] if soonest.cooldown_until - now < 5.0 else []

    def current(self) -> str:
        cands = self.candidates()
        return cands[0] if cands else ""

    def slot_of(self, key: str) -> int:
        for index, s in enumerate(self.slots):
            if s.key and s.key == key:
                return index + 1
        return 0

    def all_cooling(self) -> bool:
        now = time.time()
        return self.configured and not any(s.healthy(now) for s in self.slots)

    def soonest_recovery(self) -> float:
        """Seconds until some key becomes usable again (0 when one is)."""
        now = time.time()
        with self._lock:
            waits = [max(0.0, s.cooldown_until - now) for s in self.slots if s.configured]
        return min(waits) if waits else 0.0

    # ------------------------------------------------------------------
    def report_success(self, key: str) -> None:
        with self._lock:
            for s in self.slots:
                if s.key == key and key:
                    s.successes += 1
                    s.consecutive_limits = 0
                    s.cooldown_until = 0.0
                    s.reason = ""
                    s.last_used = time.time()

    def report_failure(self, key: str, kind: str, detail: str = "",
                       retry_after: float | None = None) -> float:
        """``kind`` is ``rate_limited`` | ``rejected`` | ``error``.  Returns
        the cooldown applied in seconds."""
        now = time.time()
        with self._lock:
            for s in self.slots:
                if s.key != key or not key:
                    continue
                s.failures += 1
                s.last_used = now
                if kind == "rate_limited":
                    index = min(s.consecutive_limits, len(RATE_LIMIT_SCHEDULE) - 1)
                    delay = RATE_LIMIT_SCHEDULE[index]
                    s.consecutive_limits += 1
                elif kind == "rejected":
                    delay = REJECTED_COOLDOWN
                else:
                    delay = ERROR_COOLDOWN
                if retry_after is not None and retry_after > 0:
                    delay = max(delay, float(retry_after))
                s.cooldown_until = now + delay
                s.reason = f"{kind}{': ' + detail if detail else ''}"
                return delay
        return 0.0

    # ------------------------------------------------------------------
    def status(self) -> dict:
        now = time.time()
        active = self.current()
        rows = []
        for index, s in enumerate(self.slots):
            rows.append({
                "slot": index + 1,
                "role": "primary" if index == 0 else "backup",
                "configured": s.configured,
                "masked": mask(s.key),
                "state": ("empty" if not s.configured else
                          "in use" if s.key == active else
                          "cooling" if s.cooldown_until > now else "standby"),
                "cooldown_seconds": round(max(0.0, s.cooldown_until - now), 1),
                "reason": s.reason,
                "successes": s.successes,
                "failures": s.failures,
            })
        return {
            "provider": self.provider,
            "configured": self.configured,
            "configured_slots": sum(1 for s in self.slots if s.configured),
            "active_slot": self.slot_of(active),
            "on_primary": bool(active) and active == self.primary,
            "all_cooling": self.all_cooling(),
            "slots": rows,
        }


def mask(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:4]}…{key[-4:]}"
