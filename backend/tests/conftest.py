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
# Round AQ: never download the 130 MB connectome inside the test run (if the
# arrays are already on disk the real brain loads and is exercised).
os.environ.setdefault("CONNECTOME_AUTODOWNLOAD", "0")
os.environ.setdefault("MODEL_AUTOLOAD", "0")
os.environ.setdefault("MARKET_ALLOW_SIMULATOR", "1")
os.environ.setdefault("LOCAL_AGENT_STUB", "1")
os.environ.setdefault("LOG_LEVEL", "warning")
os.environ.setdefault("SOCIAL_ENABLED", "0")


# Round AW: the StateVault writes to .run/vault on the real disk.  Every test
# gets its own empty vault so one test's settled windows are never another
# test's "restored" history (and the repo's own vault is never touched).
import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _isolated_state_vault_per_module(tmp_path_factory):
    # module-scoped engines (test_prediction, test_e2e, ...) are built before
    # any function-scoped fixture runs
    previous = os.environ.get("STATE_VAULT_DIR")
    os.environ["STATE_VAULT_DIR"] = str(tmp_path_factory.mktemp("vault"))
    yield
    if previous is None:
        os.environ.pop("STATE_VAULT_DIR", None)
    else:
        os.environ["STATE_VAULT_DIR"] = previous


@pytest.fixture(autouse=True)
def _isolated_state_vault(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_VAULT_DIR", str(tmp_path / "vault"))
    yield
