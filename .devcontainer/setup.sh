#!/usr/bin/env bash
# =============================================================================
# DROSOPHILA TRADER v2.0 - dev container bootstrap.
#
# Idempotent: safe to re-run.  Every network step is optional and failure of an
# optional step (llama-cpp-python, redis) does not abort the setup
# of the core engine.
#
#   bash .devcontainer/setup.sh
#
# Environment switches:
#   INSTALL_LLAMA_CPP=1   build llama-cpp-python for real local GGUF inference
#   INSTALL_ONNX=1        install onnxruntime for local ONNX inference
#   INSTALL_NEUPRINT=1    install neuprint-python + caveclient (live connectome)
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log()  { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }

INSTALL_LLAMA_CPP="${INSTALL_LLAMA_CPP:-0}"
INSTALL_ONNX="${INSTALL_ONNX:-0}"
INSTALL_NEUPRINT="${INSTALL_NEUPRINT:-0}"

# -----------------------------------------------------------------------------
# 1. System packages
# -----------------------------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null 2>&1; then
  sudo_opt=""
  if [ "$(id -u)" -ne 0 ]; then
    # `sudo -n` never asks for a password: if it cannot run, we say so and go
    # on to the Python steps, which only need what is already installed.
    if sudo -n true 2>/dev/null; then sudo_opt="sudo -n"; else sudo_opt=""; fi
  fi
  # Only touch apt when something is actually missing: `apt-get update` alone
  # costs 30-90 s on a fresh Codespace and the python image already ships
  # python3, venv, git and curl.  redis is optional (the engine falls back to
  # its in-process cache), so it never justifies the wait on its own.
  missing=""
  python3 -c 'import venv, ensurepip' >/dev/null 2>&1 || missing="$missing python3 python3-venv python3-dev"
  command -v curl >/dev/null 2>&1 || missing="$missing curl"
  command -v git >/dev/null 2>&1 || missing="$missing git"
  command -v unzip >/dev/null 2>&1 || missing="$missing unzip"
  command -v xz >/dev/null 2>&1 || missing="$missing xz-utils"
  if [ -z "$missing" ]; then
    log "system packages already present - skipping apt (fast path)"
  elif [ "$(id -u)" -eq 0 ] || [ -n "$sudo_opt" ]; then
    log "installing system packages (non-interactive apt):$missing"
    $sudo_opt apt-get update -qq >/tmp/apt-update.log 2>&1 || warn "apt-get update failed - continuing (see /tmp/apt-update.log)"
    $sudo_opt apt-get install -y -qq --no-install-recommends \
        -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold \
        $missing >/tmp/apt-install.log 2>&1 || warn "some system packages failed to install (see /tmp/apt-install.log)"
  else
    warn "no password-less sudo here - skipping apt; python3 + venv must already exist"
  fi
fi

# -----------------------------------------------------------------------------
# 2. Python environment
# -----------------------------------------------------------------------------
# Pre-provisioned image (ghcr.io/apidemo495/p-4-dev): every requirement is
# already installed in /opt/venv.  Adopt it as .venv - nothing to download.
if { [ ! -d "$REPO_ROOT/.venv" ] || [ ! -x "$REPO_ROOT/.venv/bin/python" ]; } \
   && [ -x /opt/venv/bin/python ]; then
  rm -rf "$REPO_ROOT/.venv" 2>/dev/null || true
  ln -s /opt/venv "$REPO_ROOT/.venv" && ok "using the pre-installed environment /opt/venv as .venv"
fi
if [ ! -d "$REPO_ROOT/.venv" ] || [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
  log "creating virtualenv at .venv"
  rm -rf "$REPO_ROOT/.venv" 2>/dev/null || true
  if ! python3 -m venv "$REPO_ROOT/.venv" 2>/tmp/venv-error.log; then
    warn "python3 -m venv failed:"
    sed 's/^/      /' /tmp/venv-error.log | tail -5
    warn "install it and re-run:  sudo apt-get install -y python3-venv   (then: bash run.sh)"
    warn "without .venv, 'bash run.sh' still works by installing into the user site"
  fi
fi
PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

PIP_LOG=/tmp/pip-install.log
# Skip pip entirely when every required module already imports (pre-built
# image, or a second run): this is what makes a new Codespace start in seconds.
REQUIRED_MODULES="fastapi uvicorn numpy scipy httpx feedparser websockets pydantic dotenv redis ntplib msgpack multipart pytest markdown"
deps_ok() {
  "$PY" - "$REQUIRED_MODULES" <<'PYEOF' 2>/dev/null
import importlib, sys
for name in sys.argv[1].split():
    try:
        importlib.import_module(name)
    except Exception:
        sys.exit(1)
PYEOF
}
if deps_ok; then
  ok "every module in requirements.txt is already importable - skipping pip"
  SKIP_PIP=1
else
  SKIP_PIP=0
fi
if [ "$SKIP_PIP" = 0 ]; then
log "upgrading pip"
"$PY" -m pip install --quiet --retries 5 --timeout 60 --upgrade pip wheel setuptools >"$PIP_LOG" 2>&1 \
  || warn "pip self-upgrade had errors (continuing)"

log "installing requirements.txt (the output goes to $PIP_LOG)"
install_requirements() {
  "$PY" -m pip install --quiet --progress-bar off --prefer-binary --retries 5 --timeout 60 "$@" \
      -r "$REPO_ROOT/requirements.txt" >>"$PIP_LOG" 2>&1
}
if install_requirements; then
  ok "core dependencies installed"
elif install_requirements --index-url https://pypi.org/simple --no-cache-dir; then
  ok "core dependencies installed (second attempt, official index)"
else
  # PEP 668 ("externally managed environment") is the usual blocker when the
  # venv could not be created and pip is running against the system python.
  if install_requirements --user --break-system-packages --index-url https://pypi.org/simple; then
    ok "core dependencies installed into the user site (PEP 668 override)"
  else
    warn "dependency install FAILED - last 15 lines of $PIP_LOG:"
    tail -n 15 "$PIP_LOG" | sed 's/^/      /'
    warn "usual causes: no python3-venv, a proxy/VPN, or a company index that blocks pypi.org"
    warn "fix:  sudo apt-get update && sudo apt-get install -y python3-venv python3-pip && rm -rf .venv && bash run.sh"
  fi
fi
fi  # SKIP_PIP

if [ "$INSTALL_LLAMA_CPP" = "1" ]; then
  log "building llama-cpp-python (this takes several minutes)"
  CMAKE_ARGS="-DLLAMA_NATIVE=on" "$PY" -m pip install --quiet llama-cpp-python \
    && ok "llama-cpp-python installed" || warn "llama-cpp-python build failed; local agent will stay disabled"
fi

if [ "$INSTALL_ONNX" = "1" ]; then
  "$PY" -m pip install --quiet onnxruntime && ok "onnxruntime installed" || warn "onnxruntime install failed"
fi

if [ "$INSTALL_NEUPRINT" = "1" ]; then
  "$PY" -m pip install --quiet neuprint-python caveclient \
    && ok "neuPrint / CAVE clients installed" || warn "neuPrint clients failed to install"
fi

# -----------------------------------------------------------------------------
# 3. Configuration
# -----------------------------------------------------------------------------
if [ ! -f "$REPO_ROOT/.env" ]; then
  cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
  ok "created .env from .env.example - add your API keys there"
else
  log ".env already present, leaving it untouched"
fi

# -----------------------------------------------------------------------------
# 4. Redis (optional - the engine falls back to an in-process store)
# -----------------------------------------------------------------------------
if command -v redis-server >/dev/null 2>&1; then
  if ! (echo > /dev/tcp/127.0.0.1/6379) >/dev/null 2>&1; then
    log "starting redis-server in the background"
    redis-server --daemonize yes --save '' --appendonly no || warn "redis failed to start"
  fi
  ok "redis available on 127.0.0.1:6379" || true
else
  warn "redis-server not installed - using the in-memory store (fine for a single process)"
fi

# -----------------------------------------------------------------------------
# 5. Smoke check
# -----------------------------------------------------------------------------
log "verifying imports"
PYTHONPATH="$REPO_ROOT" "$PY" - <<'PYEOF' || warn "import check failed - inspect the traceback above"
import importlib, sys
mods = ["numpy", "scipy", "fastapi", "uvicorn", "httpx", "feedparser", "websockets",
        "pydantic", "dotenv", "redis", "ntplib", "msgpack", "multipart", "pytest"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{m} ({exc})")
if missing:
    print("MISSING:", "; ".join(missing)); sys.exit(1)
from backend.formulas.engine import ALL_FORMULAS
print(f"OK - {len(ALL_FORMULAS)} formulas registered")
PYEOF

echo
ok "setup complete"
cat <<EOF

  Start the engine + dashboard (one command, handles everything).
  EVERYTHING is served from ONE port - dashboard, /settings, /matrix,
  the WebSocket stream and the API:
      bash run.sh              # -> http://localhost:8000/
      bash run.sh --bg         # background
      bash run.sh --urls       # list every feature URL on that one port
      bash run.sh --clean      # stop the engine + any stray dev servers

  Diagnose the environment without starting anything:
      bash run.sh --check

  Run the end-to-end test suite (Appendix E, 13 scenarios):
      PYTHONPATH=$REPO_ROOT .venv/bin/python -m pytest -q

  Quick end-to-end cycle dump without the UI:
      PYTHONPATH=$REPO_ROOT .venv/bin/python -m backend.tests.smoke --seconds 90

  Add API keys without touching a terminal: open the dashboard and click
      "🔑 API keys"   (or the Settings page) - keys are tested before they are
      accepted and can be written to .env from there.

EOF
