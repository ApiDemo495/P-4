#!/usr/bin/env bash
# =============================================================================
# DROSOPHILA TRADER v2.0 - one-command launcher.
#
#   bash run.sh                 set up if needed, start the engine + dashboard
#   bash run.sh --check         diagnose the environment and exit (no server)
#   bash run.sh --bg            start in the background (survives the terminal)
#   bash run.sh --public        make the forwarded Codespaces port public
#   bash run.sh --setup-only    install everything, start nothing
#   bash run.sh --port 8080     use a different port
#
# Works in a GitHub Codespace, a devcontainer, WSL, macOS and Linux.  Every step
# that can fail is checked, and each failure prints the exact command to fix it.
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------- output ----
if [ -t 1 ]; then
  B=$'\033[1m'; DIM=$'\033[2m'; R=$'\033[0m'
  RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; CYN=$'\033[36m'
else
  B=""; DIM=""; R=""; RED=""; GRN=""; YEL=""; CYN=""
fi
say()  { printf '%s\n' "$*"; }
ok()   { printf '%s✔%s %s\n' "$GRN" "$R" "$*"; }
warn() { printf '%s!%s %s\n' "$YEL" "$R" "$*"; }
bad()  { printf '%s✘%s %s\n' "$RED" "$R" "$*"; }
step() { printf '\n%s==>%s %s%s%s\n' "$CYN" "$R" "$B" "$*" "$R"; }

MODE="run"
PORT="${PORT:-8000}"
PUBLIC=0
BG=0

while [ $# -gt 0 ]; do
  case "$1" in
    --check|--doctor)  MODE="check" ;;
    --setup-only)      MODE="setup" ;;
    --bg|--background) BG=1 ;;
    --public)          PUBLIC=1 ;;
    --port)            PORT="${2:-8000}"; shift ;;
    -h|--help)         sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) warn "ignoring unknown argument: $1" ;;
  esac
  shift
done

export PORT
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

VENV="$REPO_ROOT/.venv"
REQUIRED_MODULES="fastapi uvicorn numpy scipy httpx feedparser websockets"
PY=""

py_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; }
has_deps() {
  "$1" - "$REQUIRED_MODULES" <<'PYEOF' 2>/dev/null
import importlib, sys
for name in sys.argv[1].split():
    try:
        importlib.import_module(name)
    except Exception:
        sys.exit(1)
PYEOF
}

# --------------------------------------------------------------------------
# Pick an interpreter, in order of preference:
#   1. $PYTHON                        (explicit override)
#   2. an activated virtualenv
#   3. the repo-local .venv
#   4. any python 3.10+ that already has the dependencies (Codespaces images
#      often do - no venv needed)
#   5. create .venv from the first usable python 3.10+
# --------------------------------------------------------------------------
resolve_python() {
  if [ -n "${PYTHON:-}" ] && py_ok "$PYTHON"; then echo "$PYTHON"; return 0; fi
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    echo "$VIRTUAL_ENV/bin/python"; return 0
  fi
  if [ -x "$VENV/bin/python" ] && py_ok "$VENV/bin/python"; then
    echo "$VENV/bin/python"; return 0
  fi
  local candidate
  for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && py_ok "$candidate" && has_deps "$candidate"; then
      echo "$(command -v "$candidate")"; return 0
    fi
  done
  return 1
}

base_python() {
  local candidate
  for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && py_ok "$candidate"; then
      echo "$(command -v "$candidate")"; return 0
    fi
  done
  return 1
}

# ------------------------------------------------------------------ check ---
if [ "$MODE" = "check" ]; then
  step "Environment diagnosis"
  say "repo root        : $REPO_ROOT"
  SYS="$(base_python || true)"
  if [ -n "$SYS" ]; then
    say "python (base)    : $SYS ($("$SYS" -V 2>&1))"
  else
    say "python (base)    : not found"
  fi
  say "repo .venv       : $([ -x "$VENV/bin/python" ] && echo "$VENV" || echo 'not created yet')"
  PY="$(resolve_python || true)"
  say "interpreter used : ${PY:-<none - run: bash run.sh>}"
  if [ -n "$PY" ]; then
    say "python version   : $("$PY" -V 2>&1)"
    for module in $REQUIRED_MODULES; do
      if "$PY" -c "import $module" 2>/dev/null; then ok "import $module"; else bad "import $module -> run: bash run.sh"; fi
    done
  else
    bad "no usable Python 3.10+ with the dependencies found -> run: bash run.sh"
  fi
  say ""
  say ".env present     : $([ -f "$REPO_ROOT/.env" ] && echo yes || echo 'no - the web UI can create the keys instead')"
  for key in GEMINI_API_KEY CRYPTOPANIC_API_KEY NEWSAPI_API_KEY GITHUB_MODELS_TOKEN NEUPRINT_APPLICATION_CREDENTIALS; do
    if [ -f "$REPO_ROOT/.env" ]; then
      value="$(grep -E "^${key}=" "$REPO_ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2-)"
      if [ -n "${value:-}" ]; then ok "$key is set in .env"; else warn "$key is empty in .env (add it in the UI instead)"; fi
    fi
  done
  say ""
  say "codespace name   : ${CODESPACE_NAME:-<not a codespace>}"
  if command -v ss >/dev/null 2>&1; then
    say "port $PORT         : $(ss -ltn 2>/dev/null | grep -q ":${PORT} " && echo 'IN USE - pkill -f backend.api.main' || echo free)"
  fi
  say "flutter          : $(command -v flutter >/dev/null && flutter --version 2>/dev/null | head -1 || echo 'not on PATH - bash frontend/run_web.sh installs it')"
  say "flutter web build: $([ -d "$REPO_ROOT/frontend/build/web" ] && echo 'frontend/build/web exists -> served at /flutter' || echo 'not built -> bash frontend/run_web.sh')"
  say ""
  say "Start the app with:  bash run.sh"
  exit 0
fi

# ------------------------------------------------------------- interpreter ---
step "1/4  Python environment"
PY="$(resolve_python || true)"
if [ -z "$PY" ]; then
  BASE="$(base_python || true)"
  if [ -z "$BASE" ]; then
    bad "No Python 3.10+ found."
    say "    Codespace/devcontainer:  sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip"
    exit 1
  fi
  say "${DIM}creating $VENV from $BASE${R}"
  if "$BASE" -m venv "$VENV" 2>/dev/null; then
    PY="$VENV/bin/python"
  else
    warn "venv creation failed (python3-venv missing?) - installing into the system interpreter"
    PY="$BASE"
    SYSTEM_PIP=1
  fi
fi
ok "using $PY ($("$PY" -V 2>&1))"

# ------------------------------------------------------------------- deps ---
step "2/4  Dependencies"
if has_deps "$PY"; then
  ok "fastapi, uvicorn, numpy, scipy, httpx, feedparser, websockets already present"
else
  say "${DIM}installing requirements.txt (first run only, ~1 minute)${R}"
  "$PY" -m pip install --quiet --upgrade pip wheel >/dev/null 2>&1 || true
  PIP_FLAGS=""
  [ "${SYSTEM_PIP:-0}" = "1" ] && PIP_FLAGS="--user"
  if ! "$PY" -m pip install --quiet $PIP_FLAGS -r "$REPO_ROOT/requirements.txt"; then
    bad "pip install failed."
    say "    retry with an explicit index:"
    say "      $PY -m pip install --user --index-url https://pypi.org/simple -r requirements.txt"
    exit 1
  fi
  has_deps "$PY" || { bad "dependencies still missing after install"; exit 1; }
  ok "dependencies installed"
fi

# ----------------------------------------------------------------- config ---
step "3/4  Configuration"
if [ ! -f "$REPO_ROOT/.env" ]; then
  cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
  ok "created .env from .env.example"
  say "${DIM}   You do NOT have to edit it by hand: open the dashboard and click '🔑 API keys'.${R}"
  say "${DIM}   Keys are tested, applied immediately, and can be written to .env from the UI.${R}"
else
  ok ".env already present (left untouched)"
fi

if command -v redis-server >/dev/null 2>&1; then
  if ! (exec 3<>/dev/tcp/127.0.0.1/6379) 2>/dev/null; then
    redis-server --daemonize yes --save '' --appendonly no >/dev/null 2>&1 \
      && ok "redis started on 127.0.0.1:6379" \
      || warn "redis did not start - using the in-memory store (fine, single process)"
  else
    ok "redis already running on 127.0.0.1:6379"
  fi
else
  warn "redis-server not installed - using the in-memory store (fine, single process)"
fi

PORTS_URL=""
if [ -n "${CODESPACE_NAME:-}" ]; then
  PORTS_URL="https://${CODESPACE_NAME}-${PORT}.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}/"
fi

if [ "$MODE" = "setup" ]; then
  step "Setup complete"
  say "Start the app with:  bash run.sh"
  exit 0
fi

# ------------------------------------------------------------------ start ---
step "4/4  Starting the engine"

if [ "$PUBLIC" = 1 ] && [ -n "${CODESPACE_NAME:-}" ] && command -v gh >/dev/null 2>&1; then
  gh codespace ports visibility "${PORT}:public" -c "$CODESPACE_NAME" >/dev/null 2>&1 \
    && ok "port $PORT is now public" \
    || warn "could not change port visibility (org policy?) - it still works for you when signed in"
fi

if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  warn "port $PORT is already in use - the server may already be running."
  say "    stop it:   pkill -f 'backend.api.main'"
  say "    or move:   bash run.sh --port 8020"
  exit 1
fi

print_banner() {
  say ""
  say "${B}${GRN}  ================================================================${R}"
  say "${B}   DROSOPHILA TRADER v2.0${R}"
  say "   dashboard : ${B}${PORTS_URL:-http://localhost:${PORT}/}${R}"
  if [ -n "$PORTS_URL" ]; then
    say "   settings  : ${PORTS_URL}settings      ${DIM}<- put your API keys here${R}"
    say "   matrix    : ${PORTS_URL}matrix"
    if [ -d "$REPO_ROOT/frontend/build/web" ]; then
      say "   flutter   : ${PORTS_URL}flutter"
    else
      say "   flutter   : ${DIM}not built - run  bash frontend/run_web.sh${R}"
    fi
    say ""
    say "   ${DIM}A 'site can't be reached' page means the server is not up yet:${R}"
    say "   ${DIM}wait for the 'starting' lines below, then press reload.${R}"
  else
    say "   settings  : http://localhost:${PORT}/settings"
    say ""
    say "   ${DIM}Codespaces: PORTS tab -> globe icon next to port ${PORT}.${R}"
  fi
  say "${B}${GRN}  ================================================================${R}"
  say ""
}

if [ "$BG" = 1 ]; then
  LOG="$REPO_ROOT/server.log"
  : > "$LOG"
  nohup "$PY" -m backend.api.main >>"$LOG" 2>&1 &
  PID=$!
  printf '   waiting for startup'
  for _ in $(seq 1 30); do
    sleep 1; printf '.'
    if ! kill -0 "$PID" 2>/dev/null; then break; fi
    if grep -q "running on http" "$LOG" 2>/dev/null; then break; fi
  done
  printf '\n'
  if kill -0 "$PID" 2>/dev/null && grep -q "running on http" "$LOG"; then
    ok "running in the background (pid $PID)"
    print_banner
    say "   logs : tail -f $LOG"
    say "   stop : kill $PID    (or: pkill -f 'backend.api.main')"
    say ""
    grep -E "ready|brain |market data|news |cycle |dashboard" "$LOG" | tail -n 8
  else
    bad "the server exited during startup - full log:"
    say ""
    cat "$LOG"
    exit 1
  fi
else
  print_banner
  exec "$PY" -m backend.api.main
fi
