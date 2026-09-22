#!/usr/bin/env bash
# =============================================================================
# DROSOPHILA TRADER v2.0 - dev container bootstrap.
#
# Idempotent: safe to re-run.  Every network step is optional and failure of an
# optional step (Flutter SDK, llama-cpp-python, redis) does not abort the setup
# of the core engine.
#
#   bash .devcontainer/setup.sh
#
# Environment switches:
#   INSTALL_LLAMA_CPP=1   build llama-cpp-python for real local GGUF inference
#   INSTALL_ONNX=1        install onnxruntime for local ONNX inference
#   INSTALL_FLUTTER=1     fetch the Flutter SDK for the mobile front end
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
INSTALL_FLUTTER="${INSTALL_FLUTTER:-0}"
INSTALL_NEUPRINT="${INSTALL_NEUPRINT:-0}"

# -----------------------------------------------------------------------------
# 1. System packages
# -----------------------------------------------------------------------------
if command -v apt-get >/dev/null 2>&1; then
  log "installing system packages (needs sudo if not root)"
  sudo_opt=""
  [ "$(id -u)" -ne 0 ] && sudo_opt="sudo"
  $sudo_opt apt-get update -qq || warn "apt-get update failed - continuing"
  $sudo_opt apt-get install -y -qq \
      python3 python3-venv python3-dev build-essential curl git unzip \
      redis-server || warn "some system packages failed to install"
fi

# -----------------------------------------------------------------------------
# 2. Python environment
# -----------------------------------------------------------------------------
if [ ! -d "$REPO_ROOT/.venv" ]; then
  log "creating virtualenv at .venv"
  python3 -m venv "$REPO_ROOT/.venv"
fi
PY="$REPO_ROOT/.venv/bin/python"
log "upgrading pip"
"$PY" -m pip install --quiet --upgrade pip wheel setuptools

log "installing requirements.txt"
"$PY" -m pip install --quiet -r "$REPO_ROOT/requirements.txt" \
  && ok "core dependencies installed" \
  || warn "dependency install reported errors - check the output above"

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
# 5. Flutter front end (optional)
# -----------------------------------------------------------------------------
if [ "$INSTALL_FLUTTER" = "1" ]; then
  if command -v flutter >/dev/null 2>&1; then
    ok "flutter already on PATH ($(flutter --version 2>/dev/null | head -1))"
  else
    log "cloning the Flutter SDK into ~/flutter (stable)"
    git clone --depth 1 -b stable https://github.com/flutter/flutter.git "$HOME/flutter" \
      && export PATH="$HOME/flutter/bin:$PATH" \
      && flutter --version \
      && (cd "$REPO_ROOT/frontend" && flutter pub get) \
      && ok "Flutter ready - 'cd frontend && flutter run'" \
      || warn "Flutter setup failed; the web dashboard is unaffected"
    cat <<'EOF'
    Add Flutter to your shell profile:
      export PATH="$HOME/flutter/bin:$PATH"
EOF
  fi
fi

# -----------------------------------------------------------------------------
# 6. Smoke check
# -----------------------------------------------------------------------------
log "verifying imports"
PYTHONPATH="$REPO_ROOT" "$PY" - <<'PYEOF' || warn "import check failed - inspect the traceback above"
import importlib, sys
mods = ["numpy", "scipy", "fastapi", "uvicorn", "httpx", "feedparser", "websockets"]
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

  Start the engine + dashboard (one command, handles everything):
      bash run.sh

  Diagnose the environment without starting anything:
      bash run.sh --check

  Build the Flutter client as a web app (served at /flutter):
      bash frontend/run_web.sh

  Run the end-to-end test suite (Appendix E, 13 scenarios):
      PYTHONPATH=$REPO_ROOT .venv/bin/python -m pytest -q

  Quick end-to-end cycle dump without the UI:
      PYTHONPATH=$REPO_ROOT .venv/bin/python -m backend.tests.smoke --seconds 90

  Add API keys without touching a terminal: open the dashboard and click
      "🔑 API keys"   (or the Settings page) - keys are tested before they are
      accepted and can be written to .env from there.

EOF
