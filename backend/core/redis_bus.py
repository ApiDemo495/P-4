"""Redis access layer with a transparent in-memory fallback.

Section 13.1: "Redis -> Connection refused -> Fall back to in-memory dict.
Performance identical for our scale.  (No user notification - transparent
fallback.)"
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import time
from typing import Any

log = logging.getLogger("drosophila.redis")


class _MemoryStore:
    """Dict with per-key TTLs; the API mirrors the small subset of Redis we use."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[bytes, float | None]] = {}

    def _alive(self, key: str) -> bool:
        entry = self._data.get(key)
        if entry is None:
            return False
        _, expires = entry
        if expires is not None and expires < time.time():
            self._data.pop(key, None)
            return False
        return True

    def get(self, key: str) -> bytes | None:
        return self._data[key][0] if self._alive(key) else None

    def set(self, key: str, value: bytes, ex: int | None = None) -> None:
        expires = time.time() + ex if ex else None
        self._data[key] = (value, expires)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def ping(self) -> bool:
        return True

    def keys(self, pattern: str = "*") -> list[str]:
        prefix = pattern.rstrip("*")
        return [k for k in list(self._data) if k.startswith(prefix) and self._alive(k)]


class Store:
    """High-level async store - Redis when available, memory otherwise.

    The class never raises on connection problems; ``backend`` reports which
    mode is active so the health endpoint can surface it.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._redis: Any | None = None
        self._memory = _MemoryStore()
        self.backend = "memory"
        self._lock = asyncio.Lock()

    async def connect(self) -> str:
        async with self._lock:
            if self._redis is not None:
                return self.backend
            try:
                import redis.asyncio as aioredis

                client = aioredis.from_url(self.url, socket_connect_timeout=2, decode_responses=False)
                await asyncio.wait_for(client.ping(), timeout=2.0)
                self._redis = client
                self.backend = "redis"
                log.info("Redis connected: %s", self.url)
            except Exception as exc:  # noqa: BLE001
                self._redis = None
                self.backend = "memory"
                log.info("Redis unavailable (%s); using in-memory store", exc)
            return self.backend

    async def _ensure(self) -> None:
        if self._redis is None:
            await self.connect()

    async def get_bytes(self, key: str) -> bytes | None:
        await self._ensure()
        if self._redis is not None:
            try:
                return await self._redis.get(key)
            except Exception as exc:  # noqa: BLE001
                log.warning("Redis GET failed (%s); degrading to memory", exc)
                self._redis, self.backend = None, "memory"
        return self._memory.get(key)

    async def set_bytes(self, key: str, value: bytes, ttl: int | None = None) -> None:
        await self._ensure()
        if self._redis is not None:
            try:
                await self._redis.set(key, value, ex=ttl)
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("Redis SET failed (%s); degrading to memory", exc)
                self._redis, self.backend = None, "memory"
        self._memory.set(key, value, ex=ttl)

    async def get_pickle(self, key: str) -> Any | None:
        raw = await self.get_bytes(key)
        if raw is None:
            return None
        try:
            return pickle.loads(raw)
        except Exception:  # noqa: BLE001 - cache corruption is never fatal
            return None

    async def set_pickle(self, key: str, value: Any, ttl: int | None = None) -> None:
        await self.set_bytes(key, pickle.dumps(value, protocol=4), ttl)

    async def delete(self, key: str) -> None:
        await self._ensure()
        if self._redis is not None:
            try:
                await self._redis.delete(key)
                return
            except Exception:  # noqa: BLE001
                self._redis, self.backend = None, "memory"
        self._memory.delete(key)

    async def healthy(self) -> bool:
        if self._redis is None:
            return False
        try:
            await asyncio.wait_for(self._redis.ping(), timeout=1.0)
            return True
        except Exception:  # noqa: BLE001
            return False
