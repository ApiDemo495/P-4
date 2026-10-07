"""World-clock timing.

The 60-second cycle is phase-locked to the UTC minute (Section 9 of the
specification).  NTP is used to correct for system-clock drift; if NTP is
unreachable the module silently falls back to the system clock and re-attempts
every 60 seconds (Section 13.1).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

log = logging.getLogger("drosophila.clock")

NTP_HOSTS = ("pool.ntp.org", "time.google.com", "time.cloudflare.com")


class WorldClock:
    """UTC reference clock with optional NTP correction."""

    def __init__(self, ntp_enabled: bool = True) -> None:
        self._offset = 0.0
        self._ntp_enabled = ntp_enabled
        self._ntp_ok = False
        self._last_attempt = 0.0
        self._lock = asyncio.Lock()

    # -- queries ---------------------------------------------------------
    @property
    def ntp_synced(self) -> bool:
        return self._ntp_ok

    @property
    def offset(self) -> float:
        return self._offset

    def now(self) -> float:
        """Unix timestamp in seconds (UTC), NTP-corrected when available."""
        return time.time() + self._offset

    def now_dt(self) -> datetime:
        return datetime.fromtimestamp(self.now(), tz=timezone.utc)

    def iso(self) -> str:
        return self.now_dt().strftime("%Y-%m-%dT%H:%M:%SZ")

    def seconds_into_minute(self) -> float:
        return self.now() % 60.0

    def seconds_until_next_minute(self) -> float:
        return 60.0 - self.seconds_into_minute()

    # -- maintenance -----------------------------------------------------
    async def sync(self, force: bool = False) -> bool:
        """Attempt an NTP sync at most once per 60 seconds."""
        if not self._ntp_enabled:
            return False
        async with self._lock:
            now_mono = time.monotonic()
            if not force and (now_mono - self._last_attempt) < 60.0:
                return self._ntp_ok
            self._last_attempt = now_mono
            try:
                self._offset = await asyncio.wait_for(
                    asyncio.to_thread(self._query_ntp), timeout=4.0
                )
                self._ntp_ok = True
                log.info("NTP synced, offset %.3fs", self._offset)
            except Exception as exc:  # noqa: BLE001 - never fatal
                self._ntp_ok = False
                log.debug("NTP unavailable (%s); using system clock", exc)
            return self._ntp_ok

    @staticmethod
    def _query_ntp() -> float:
        import ntplib  # imported lazily so the app runs without the package

        client = ntplib.NTPClient()
        last_error: Exception | None = None
        for host in NTP_HOSTS:
            try:
                response = client.request(host, version=3, timeout=2)
                return float(response.offset)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise RuntimeError(f"all NTP hosts unreachable: {last_error}")

    async def run(self) -> None:
        """Background loop: keep the clock honest."""
        await self.sync(force=True)
        while True:
            await asyncio.sleep(60)
            await self.sync()


WORLD_CLOCK = WorldClock()
