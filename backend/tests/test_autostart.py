"""The zero-command contract: a fresh Codespace must need no commands typed.

The user's rule: "do everything as soon as I start new codespace, I don't want
to run any command."  That is a wiring contract, and wiring is exactly the kind
of thing that rots silently - a renamed script, a dropped hook, a port that
stops being forwarded - so it is tested here:

1.  ``devcontainer.json`` wires all three lifecycle hooks to the autostart
    script (create / start / attach), forwards port 8000 and asks for it to be
    public and opened in a browser;
2.  the autostart script is valid bash and executable;
3.  it self-heals: it re-provisions when the virtualenv is missing or
    ``requirements.txt`` changed (this is what makes a rebuilt container work);
4.  nothing in the path blocks on the optional Flutter download - it has to run
    in the background, or a user with a slow connection would wait minutes
    before the dashboard came up;
5.  the path is idempotent: every step is guarded, so running it twice is safe.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEVCONTAINER = ROOT / ".devcontainer" / "devcontainer.json"
AUTOSTART = ROOT / "tools" / "codespace_autostart.sh"


def _jsonc(path: Path) -> dict:
    """devcontainer.json allows // comments; json.loads does not."""
    raw = path.read_text()
    lines = []
    for line in raw.splitlines():
        if line.lstrip().startswith("//"):
            continue
        # drop trailing comments that are not inside a string
        stripped = re.sub(r'(?<!:)//(?![^"]*"[^"]*$).*$', "", line)
        lines.append(stripped)
    return json.loads("\n".join(lines))


@pytest.fixture(scope="module")
def config() -> dict:
    assert DEVCONTAINER.exists(), "the dev container definition is missing"
    return _jsonc(DEVCONTAINER)


def test_the_codespace_starts_everything_by_itself(config: dict) -> None:
    """All three lifecycle hooks must call the autostart script."""
    hooks = {
        "postCreateCommand": "--provision",
        "postStartCommand": "--start",
        "postAttachCommand": "--attach",
    }
    for key, mode in hooks.items():
        command = config.get(key, "")
        assert "codespace_autostart.sh" in command, f"{key} does not run the autostart script"
        assert mode in command, f"{key} should pass {mode}"
        assert command.startswith("bash "), f"{key} must call it through bash"


def test_the_one_port_story_survives(config: dict) -> None:
    """Port 8000 only, public, opened for the user without a click."""
    assert config.get("forwardPorts") == [8000]
    attributes = config.get("portsAttributes", {}).get("8000", {})
    assert attributes.get("visibility") == "public", "the forwarded URL must open without a login"
    assert attributes.get("onAutoForward") == "openBrowser", "the dashboard should open itself"
    # Any other port (a hand-started tool) must not pop a broken preview.
    assert config.get("otherPortsAttributes", {}).get("onAutoForward") == "silent"


def test_the_environment_switches_are_declared(config: dict) -> None:
    env = config.get("remoteEnv", {})
    assert env.get("PYTHONPATH") == "${containerWorkspaceFolder}"
    assert env.get("AUTO_FLUTTER") == "1", "the Flutter download is part of 'download everything'"


def test_the_autostart_script_is_valid_bash() -> None:
    assert AUTOSTART.exists(), "tools/codespace_autostart.sh is missing"
    bash = shutil.which("bash")
    assert bash, "bash is required"
    result = subprocess.run([bash, "-n", str(AUTOSTART)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert AUTOSTART.stat().st_mode & 0o111, "the autostart script should be executable"


def test_it_self_heals_a_missing_or_changed_environment() -> None:
    """A rebuilt container (no .venv) or an edited requirements.txt must re-install."""
    text = AUTOSTART.read_text()
    assert "needs_provision" in text
    assert "requirements.txt" in text, "the fingerprint must key on requirements.txt"
    assert ".provisioned" in text or "STAMP" in text, "provisioning needs a stamp to compare against"
    assert "setup.sh" in text, "self-healing must reuse the committed setup script"
    # The venv check has to look at the interpreter, not just the directory.
    assert ".venv/bin/python" in text or 'VENV/bin/python' in text


def test_the_flutter_download_never_blocks_the_dashboard() -> None:
    """The ~700 MB download runs in the background and is logged, not awaited."""
    text = AUTOSTART.read_text()
    flutter_block = text.split("start_flutter()", 1)[1].split("\n}", 1)[0]
    assert "nohup" in flutter_block, "the Flutter setup must be detached"
    assert "&" in flutter_block, "the Flutter setup must not be awaited"
    assert "flutter-setup.log" in text, "the download needs a log the user can read"
    # ... and the engine must start before/independently of it.
    attach = text.split('attach)', 1)[1]
    assert attach.index("start_engine") < attach.index("start_flutter")
    # postCreateCommand must not wait on it either: a bare `wait` with no
    # arguments would block container creation for the whole download.  Comments
    # are stripped first, so the note explaining this rule does not trip it.
    provision_block = text.split("  provision)", 1)[1].split(";;", 1)[0]
    commands = [
        line.strip()
        for line in provision_block.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "wait" not in commands, (
        "the provision step must never wait for the Flutter download"
    )


def test_the_engine_start_is_idempotent_and_supervised() -> None:
    text = AUTOSTART.read_text()
    assert "run.sh" in text and "--bg" in text, "the engine must be started through run.sh --bg"
    assert "engine_answers" in text, "a second run must notice the engine is already up"
    assert "wait_until_ready" in text, "the user should see the URL only once it answers"
    assert "wait_until_locked" in text, "and once a first prediction exists"


def test_it_publishes_the_url_it_prints() -> None:
    """The banner must use the Codespace URL when there is one."""
    text = AUTOSTART.read_text()
    assert "CODESPACE_NAME" in text
    assert "GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN" in text
    assert "codespace_url" in text


def test_the_restart_path_is_documented_for_the_user() -> None:
    """README: create the codespace, run nothing, and how to stop/restart."""
    readme = (ROOT / "README.md").read_text()
    assert "codespace_autostart" in readme or "zero commands" in readme.lower()
    assert "AUTO_FLUTTER" in readme, "the Flutter opt-out must be documented"
