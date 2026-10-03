"""Shared test configuration.

The environment is configured *before* ``backend.core.config`` is imported by
any test module, because :data:`backend.core.config.SETTINGS` is built at import
time.

Time is compressed by 20x for the whole suite (one signal cycle is 3 wall-clock
seconds instead of 60), which is exactly the mechanism the dashboard uses for
demos - every ratio in the protocol is preserved:

    lock deadline   8 s   -> 0.4 s
    formula refresh 15 s  -> 0.75 s
    outcome horizon 60 s  -> 3 s
    emergency      180 s  -> 9 s
"""

from __future__ import annotations

import os

os.environ.setdefault("TIME_SCALE", "20")
os.environ.setdefault("MARKET_DATA_MODE", "simulator")
os.environ.setdefault("MARKET_ALLOW_SIMULATOR", "1")
os.environ.setdefault("LOCAL_AGENT_STUB", "1")
os.environ.setdefault("LOG_LEVEL", "warning")
