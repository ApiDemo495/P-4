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
#   bash run.sh --urls          print every feature URL (all on ONE port)
#   bash run.sh --status        is it running, and does the port really answer?
#   bash run.sh --stop          stop the engine (and its supervisor)
#   bash run.sh --clean         stop the engine AND any stray Flutter dev
#                               servers, then report what still holds the ports
#
# EVERYTHING the app offers - dashboard, API, WebSocket stream, matrix viewer,
# settings page and the Flutter build - is served from ONE port (8000).  There
# is no second port to forward, and any other listening port is somebody else's
# process (usually a `flutter run` dev server started by hand).
#
# ``--bg`` is idempotent (running it twice does not start a second server) and
# the engine runs under a supervisor that restarts it if it ever dies.  The
# start is verified with a real HTTP request, so "this page isn't working /
# HTTP 502" cannot happen silently again.
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
    --urls|--links)    MODE="urls" ;;
    --status|--ps)      MODE="status" ;;
    --supervise)        MODE="supervise" ;;
    --stop|--down)     MODE="stop" ;;
    --clean|--reset)   MODE="clean" ;;
    --port)            PORT="${2:-8000}"; shift ;;
    -h|--help)         sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) warn "ignoring unknown argument: $1" ;;
  esac
  shift
done

export PORT
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

VENV="$REPO_ROOT/.venv"
RUN_DIR="$REPO_ROOT/.run"
PIDFILE="$RUN_DIR/supervisor.pid"
SERVER_LOG="$REPO_ROOT/server.log"
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

# ------------------------------------------------------- ports / lifecycle ---
codespace_url() {
  if [ -n "${CODESPACE_NAME:-}" ]; then
    echo "https://${CODESPACE_NAME}-${PORT}.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}"
  else
    echo "http://localhost:${PORT}"
  fi
}

listeners() {  # print "<port> <process>" for our ports
  command -v ss >/dev/null 2>&1 || return 0
  ss -ltnp 2>/dev/null | grep -E ":(8000|8081|6379) " || true
}

port_is_open() {  # local TCP check - no HTTP involved
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | grep -q ":${PORT} "
  else
    (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null
  fi
}

http_ready() {  # 0 when the API answers - the only proof the port really works
  command -v curl >/dev/null 2>&1 || return 0
  curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1
}

supervisor_pid() {
  [ -f "$PIDFILE" ] || return 1
  local pid; pid="$(cat "$PIDFILE" 2>/dev/null)"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  # A pid file can outlive its process and be recycled by something unrelated,
  # so only trust it when the process really is our supervisor.  (Killing the
  # process group of a recycled pid would take down an innocent shell.)
  local args=""
  if [ -r "/proc/$pid/cmdline" ]; then
    args="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null)"
  else
    args="$(ps -p "$pid" -o args= 2>/dev/null)"
  fi
  case "$args" in
    *run.sh*supervise*) echo "$pid" ;;
    *) return 1 ;;
  esac
}

diagnose_502() {
  # Runs when the engine did not answer: say exactly why a browser got a 502.
  say ""
  say "${B}Why the browser says \"this page isn't working\" (HTTP 502):${R}"
  say "   Codespaces forwards port ${PORT} to a process *inside* this container."
  say "   If nothing is listening there - or the process died - the forwarder"
  say "   itself answers 502. It is not a DNS, firewall or CORS problem."
  say ""
  if pgrep -f "[b]ackend.api.main" >/dev/null 2>&1; then
    say "   engine process : alive"
  else
    say "   engine process : NOT RUNNING   <- this is the cause"
  fi
  say "   port ${PORT}      : $(port_is_open && echo listening || echo 'not listening')"
  say ""
  if [ -f "$SERVER_LOG" ]; then
    say "${B}Last 20 lines of server.log:${R}"
    tail -n 20 "$SERVER_LOG" | sed 's/^/   /'
    say ""
    grep -q "No module named" "$SERVER_LOG" 2>/dev/null \
      && say "   ${YEL}Missing dependency${R}   fix:  bash run.sh"
    grep -q "address already in use" "$SERVER_LOG" 2>/dev/null \
      && say "   ${YEL}Port busy${R}              fix:  bash run.sh --clean && bash run.sh --bg"
    grep -q "Traceback" "$SERVER_LOG" 2>/dev/null \
      && say "   ${YEL}Python traceback${R}      fix:  bash run.sh --check"
  fi
  say ""
  say "   ${B}Next steps, in order:${R}"
  say "     1)  bash run.sh            (foreground, so you see the error live)"
  say "     2)  bash run.sh --clean && bash run.sh --bg"
  say "     3)  bash run.sh --check"
}

print_urls() {
  local base; base="$(codespace_url)"
  say ""
  say "${B}Everything is on ONE port:${R} ${base}/"
  say ""
  say "   /            dashboard (widget panel, brain, history, news)"
  say "   /matrix      the 80x80 connectome heat-map"
  say "   /settings    keys, local model, brain reconnect, news poll"
  say "   /flutter     the Flutter client - $([ -d "$REPO_ROOT/frontend/build/web" ] && echo 'built and served' || echo 'not built yet: bash frontend/run_web.sh')"
  say "   /docs        the OpenAPI explorer"
  say "   /ws/signals  the WebSocket stream (used by both clients)"
  say ""
  if [ -n "${CODESPACE_NAME:-}" ]; then
    say "   Codespaces: PORTS tab -> port ${PORT} -> globe icon (or: bash run.sh --public)"
  fi
}

stop_server() {
  local pid stopped=0
  pid="$(supervisor_pid || true)"
  if [ -n "$pid" ]; then
    stopped=1
    # Verified pid: terminate the engine child first, then the supervisor.
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
    sleep 0.5
  fi
  # Backstop: a supervisor that lost its pid file would otherwise keep
  # restarting the engine behind our back.
  if pgrep -f "[r]un.sh --supervise" >/dev/null 2>&1; then
    stopped=1
    pkill -f "[r]un.sh --supervise" 2>/dev/null || true
  fi
  if pgrep -f "[b]ackend.api.main" >/dev/null 2>&1; then
    stopped=1
    pkill -f "[b]ackend.api.main" 2>/dev/null || true
    sleep 1
    if pgrep -f "[b]ackend.api.main" >/dev/null 2>&1; then
      pkill -9 -f "[b]ackend.api.main" 2>/dev/null || true
      sleep 0.5
    fi
  fi
  rm -f "$PIDFILE"
  if [ "$stopped" = 1 ]; then
    ok "engine stopped (the dashboard stays down until you start it again)"
    return 0
  fi
  say "nothing was running"
  return 1
}

stop_stray_dev_servers() {
  local found=0 pattern
  for pattern in "[f]lutter_tools" "[f]rontend_server" "[d]art.*web-server" "[d]art.*webserver"; do
    if pgrep -f "$pattern" >/dev/null 2>&1; then
      pkill -f "$pattern" 2>/dev/null || true
      found=1
    fi
  done
  [ "$found" = 1 ] && ok "stopped stray Flutter/Dart dev servers (they open random ports)" || true
}

report_open_ports() {
  say ""
  say "${B}Ports currently listening:${R}"
  local rows; rows="$(listeners)"
  if [ -z "$rows" ]; then
    say "   none of 8000 / 8081 / 6379 are in use"
  else
    printf '%s\n' "$rows" | sed 's/^/   /'
    say ""
    say "   8000 -> ours (everything).  Anything else is another tool you started."
  fi
  if [ -n "${CODESPACE_NAME:-}" ]; then
    say ""
    say "   ${DIM}Codespaces keeps a port in the PORTS tab until its process dies;${R}"
    say "   ${DIM}a stale entry showing 'not working' is a dead process - forward only ${PORT}.${R}"
  fi
}

if [ "$MODE" = "status" ]; then
  step "Status"
  pid="$(supervisor_pid || true)"
  [ -n "$pid" ] && ok "supervisor running (pid $pid)" || warn "supervisor not running"
  local count
  count="$(pgrep -f "[r]un.sh --supervise" 2>/dev/null | wc -l | tr -d ' ')"
  if [ "${count:-0}" -gt 1 ]; then
    warn "$count supervisors are running - bash run.sh --clean makes it one"
  fi
  pgrep -f "[b]ackend.api.main" >/dev/null 2>&1 \
    && ok "engine process alive" || warn "engine process not found"
  port_is_open && ok "port ${PORT} is listening" || bad "port ${PORT} is NOT listening"
  if http_ready; then
    ok "the port answers HTTP - the app is serving"
    command -v curl >/dev/null 2>&1 && \
      curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/api/health" \
      | tr ',' '\n' | grep -E '"ready"|"warming_up"|"start_error"|"healthy"' | sed 's/^/   /'
    say ""
    print_urls
  else
    bad "the port does not answer HTTP (this is what the browser shows as 502)"
    diagnose_502
  fi
  exit 0
fi

if [ "$MODE" = "stop" ]; then
  stop_server
  exit 0
fi

if [ "$MODE" = "supervise" ]; then
  # Internal: started by --bg, ended by --stop.  Keeps the engine alive so a
  # crash (or an OOM kill) turns into a 2-second restart instead of a 502.
  # --bg passes the resolved interpreter in DROSOPHILA_PY (this branch runs
  # before the environment steps, so $PY may be empty here).
  PY="${DROSOPHILA_PY:-}"
  [ -n "$PY" ] || PY="$(resolve_python || true)"
  [ -n "$PY" ] || PY="$(command -v python3 || true)"
  if [ -z "$PY" ]; then
    echo "[supervisor] no usable python interpreter - refusing to start"
    exit 1
  fi
  # The supervisor owns the pid file: no setsid/fork guesswork in the parent,
  # and --stop always knows exactly what to kill.
  mkdir -p "$RUN_DIR"
  echo $$ >"$PIDFILE"
  trap 'echo "[supervisor] stopping $(date -u +%FT%TZ)"; rm -f "$PIDFILE"; exit 0' TERM INT
  echo "[supervisor] started $(date -u +%FT%TZ) with $PY"
  while true; do
    echo "[supervisor] launching engine $(date -u +%FT%TZ)"
    "$PY" -m backend.api.main
    code=$?
    if [ "$code" = 0 ]; then
      echo "[supervisor] engine exited cleanly - not restarting"
      break
    fi
    echo "[supervisor] engine exited with code $code - restarting in 2s $(date -u +%FT%TZ)"
    sleep 2
  done
  exit 0
fi

if [ "$MODE" = "clean" ]; then
  step "Stopping everything this repo can start"
  stop_server || true
  stop_stray_dev_servers
  report_open_ports
  say ""
  ok "done - start again with: bash run.sh"
  exit 0
fi

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
    if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
      say "port $PORT         : IN USE (bash run.sh --stop, or --clean for everything)"
    else
      say "port $PORT         : free"
    fi
    say ""
    say "listening ports  :"
    local_rows="$(listeners)"
    if [ -z "$local_rows" ]; then
      say "                     none of 8000 / 8081 / 6379"
    else
      printf '%s\n' "$local_rows" | sed 's/^/                     /'
    fi
    say "                     ${DIM}(8000 is ours; other ports belong to flutter run or other tools)${R}"
  fi
  say "flutter          : $(command -v flutter >/dev/null && flutter --version 2>/dev/null | head -1 || echo 'not on PATH - bash frontend/run_web.sh installs it')"
  say "flutter web build: $([ -d "$REPO_ROOT/frontend/build/web" ] && echo 'frontend/build/web exists -> served at /flutter' || echo 'not built -> bash frontend/run_web.sh')"
  say ""
  say ""
  say "Start the app with:  bash run.sh"
  exit 0
fi

if [ "$MODE" = "urls" ]; then
  print_urls
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

is_externally_managed() {
  "$1" - <<'PYEOF' >/dev/null 2>&1
import os, sysconfig
std = sysconfig.get_path("stdlib") or ""
sys.exit(0 if std and os.path.exists(os.path.join(std, "EXTERNALLY-MANAGED")) else 1)
PYEOF
}

ensure_venv_module() {
  # Codespaces and slim images often ship python3 without venv/ensurepip.  That
  # is the usual reason "downloading the requirements" fails, so try to fix it
  # instead of just reporting it.
  if ! command -v apt-get >/dev/null 2>&1; then return 1; fi
  say "${DIM}   python3-venv is missing - trying to install it (needs sudo)${R}"
  if sudo -n true 2>/dev/null; then
    sudo -n apt-get update -qq >/dev/null 2>&1 || true
    sudo -n apt-get install -y -qq python3-venv python3-pip >/dev/null 2>&1 && return 0
  fi
  return 1
}

pip_install() {  # $1 = interpreter, rest = extra pip flags
  local py="$1"; shift
  say "${DIM}   pip install -r requirements.txt $*${R}"
  "$py" -m pip install --quiet --progress-bar off --retries 5 --timeout 60 "$@" \
      -r "$REPO_ROOT/requirements.txt"
}

if has_deps "$PY"; then
  ok "fastapi, uvicorn, numpy, scipy, httpx, feedparser, websockets already present"
else
  say "${DIM}installing requirements.txt (first run only, ~1 minute)${R}"
  "$PY" -m pip install --quiet --upgrade pip wheel >/dev/null 2>&1 || true

  PIP_FLAGS=""
  [ "${SYSTEM_PIP:-0}" = "1" ] && PIP_FLAGS="--user"

  if ! pip_install "$PY" $PIP_FLAGS; then
    warn "pip failed - retrying with an explicit index and no cache"
    pip_install "$PY" $PIP_FLAGS --index-url https://pypi.org/simple --no-cache-dir || true
  fi

  # PEP 668 ("externally managed environment") is the second classic failure.
  if ! has_deps "$PY" && is_externally_managed "$PY"; then
    warn "this interpreter is marked externally managed (PEP 668)"
    say "${DIM}   retrying with --user --break-system-packages (nothing outside your user dir is touched)${R}"
    pip_install "$PY" --user --break-system-packages --index-url https://pypi.org/simple || true
  fi

  # Last resort: make a venv now that the module may have been installed.
  if ! has_deps "$PY" && [ "${SYSTEM_PIP:-0}" = "1" ]; then
    if ensure_venv_module && "$BASE" -m venv "$VENV" >/dev/null 2>&1; then
      PY="$VENV/bin/python"
      ok "created $VENV after installing python3-venv"
      "$PY" -m pip install --quiet --upgrade pip wheel >/dev/null 2>&1 || true
      pip_install "$PY" --index-url https://pypi.org/simple || true
    fi
  fi

  if ! has_deps "$PY"; then
    bad "dependencies are still missing."
    say ""
    say "   Try these, in order:"
    say "     1)  sudo apt-get update && sudo apt-get install -y python3-venv python3-pip"
    say "         rm -rf .venv && bash run.sh"
    say "     2)  $PY -m pip install --user --index-url https://pypi.org/simple -r requirements.txt"
    say "     3)  corporate proxy/VPN? pip needs HTTPS to pypi.org:"
    say "         export HTTPS_PROXY=http://user:pass@host:port"
    say "     4)  full diagnosis:  bash run.sh --check"
    exit 1
  fi
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

if port_is_open; then
  # Already ours and answering?  `--bg` twice is a no-op, never an error.
  if [ "$BG" = 1 ] && http_ready && pgrep -f "[r]un.sh --supervise" >/dev/null 2>&1; then
    ok "already running and answering on port ${PORT} - nothing to start"
    print_banner
    print_urls
    exit 0
  fi
  warn "port $PORT is already in use - either this app or another program."
  say "    status:    bash run.sh --status"
  say "    stop ours: bash run.sh --stop"
  say "    free all:  bash run.sh --clean"
  say "    or move:   bash run.sh --port 8020"
  exit 1
fi

print_banner() {
  local base; base="$(codespace_url)"
  say ""
  say "${B}${GRN}  ================================================================${R}"
  say "${B}   DROSOPHILA TRADER v2.0 - everything on ONE port${R}"
  say "   ${B}${base}/${R}"
  say ""
  say "   /            dashboard   ${DIM}(widget panel, brain, news, history)${R}"
  say "   /settings    ${DIM}keys, local model, brain reconnect${R}"
  say "   /matrix      ${DIM}80x80 connectome${R}"
  if [ -d "$REPO_ROOT/frontend/build/web" ]; then
    say "   /flutter     Flutter client"
  else
    say "   /flutter     ${DIM}not built - bash frontend/run_web.sh${R}"
  fi
  say "   /docs        ${DIM}API explorer${R}"
  say ""
  say "   ${DIM}The URL answers once 'Uvicorn running on http://0.0.0.0:${PORT}' appears below -${R}"
  say "   ${DIM}if the tab says 'this page isn't working', wait for that line and reload.${R}"
  if [ -n "${CODESPACE_NAME:-}" ]; then
    say "   ${DIM}PORTS tab -> port ${PORT} -> globe icon (or: bash run.sh --public)${R}"
    say "   ${DIM}If another port shows 'not working', it is a dead process of some${R}"
    say "   ${DIM}other tool: bash run.sh --clean removes them.${R}"
  fi
  say "${B}${GRN}  ================================================================${R}"
  say ""
}

if [ "$BG" = 1 ]; then
  mkdir -p "$RUN_DIR"

  # Idempotent: never start a second copy of an app that is already answering.
  if http_ready && pgrep -f "[r]un.sh --supervise" >/dev/null 2>&1; then
    ok "already running - not starting a second copy (pid $(supervisor_pid || echo '?'))"
    print_banner
    print_urls
    exit 0
  fi
  if [ -n "$(supervisor_pid || true)" ] || pgrep -f "[b]ackend.api.main" >/dev/null 2>&1; then
    warn "a previous instance is still around - restarting it cleanly"
    stop_server || true
  fi
  if port_is_open; then
    bad "port ${PORT} is held by another program, not by this app."
    report_open_ports
    say ""
    say "   Free it:  bash run.sh --clean"
    say "   Or move:  bash run.sh --port 8020"
    exit 1
  fi

  LOG="$SERVER_LOG"
  : >"$LOG"
  export DROSOPHILA_PY="$PY"
  SUPERVISOR="$REPO_ROOT/$(basename "$0")"
  [ -f "$SUPERVISOR" ] || SUPERVISOR="$0"
  # Only ever one supervisor: a stale one would fight us for the port.
  if pgrep -f "[r]un.sh --supervise" >/dev/null 2>&1; then
    say "${DIM}   stopping a leftover supervisor first${R}"
    pkill -f "[r]un.sh --supervise" 2>/dev/null || true
    pkill -f "[b]ackend.api.main" 2>/dev/null || true
    sleep 1
  fi
  nohup bash "$SUPERVISOR" --supervise >>"$LOG" 2>&1 &
  printf '   starting'
  READY=0
  for _ in $(seq 1 60); do
    sleep 1; printf '.'
    if http_ready; then READY=1; break; fi
    if ! kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
      printf '\n'
      warn "the supervisor exited immediately - reading the log"
      break
    fi
  done
  printf '\n'

  if [ "$READY" = 1 ]; then
    ok "running in the background (supervisor pid $(supervisor_pid || echo '?'), log: server.log)"
    say "${DIM}   the supervisor restarts the engine automatically if it exits${R}"
    print_banner
    print_urls
    exit 0
  fi
  bad "the engine did not answer on port ${PORT} within 60 seconds."
  diagnose_502
  exit 1
else
  print_banner
  exec "$PY" -m backend.api.main
fi
