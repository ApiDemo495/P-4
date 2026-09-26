#!/usr/bin/env bash
# =============================================================================
# DROSOPHILA TRADER v2.0 - zero-command start for GitHub Codespaces.
#
# The user's rule: "I don't want to run any command."  So nothing here may ask
# a question, nothing may block on an optional download, and every step must be
# safe to run again.  `.devcontainer/devcontainer.json` wires the three hooks:
#
#   --provision   (postCreateCommand)  install everything, once per codespace
#   --start       (postStartCommand)   make sure the engine is running
#   --attach      (postAttachCommand)  same as --start, then print the status
#
# What it does, in order:
#   1. self-heal - if the virtualenv is missing or requirements.txt changed,
#      re-run `.devcontainer/setup.sh` (apt + venv + pip + redis + .env);
#   2. start the engine on the one port (`bash run.sh --bg`, idempotent, with a
#      supervisor that restarts it if it ever dies);
#   3. wait until /api/health answers and the first prediction is published;
#   4. make port 8000 public, so the forwarded URL opens in any browser;
#   5. kick off the Flutter SDK download + web build **in the background** and
#      log it to flutter-setup.log, so /flutter comes alive by itself.
#
# Environment switches (all optional):
#   AUTO_FLUTTER=0        skip the Flutter download and web build
#   PORT=8020             serve on a different port
#   AUTO_OPEN=0           do not try to make the port public
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE=""
for arg in "$@"; do
  case "$arg" in
    --provision|--start|--attach|--status) MODE="${arg#--}" ;;
    -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  esac
done
[ -n "$MODE" ] || MODE="attach"

VENV="$REPO_ROOT/.venv"
PY="$VENV/bin/python"
STATE_DIR="$REPO_ROOT/.devcontainer"
STAMP="$STATE_DIR/.provisioned"
FLUTTER_LOG="$REPO_ROOT/flutter-setup.log"
FLUTTER_PID="$STATE_DIR/flutter.pid"
PORT="${PORT:-8000}"
AUTO_FLUTTER="${AUTO_FLUTTER:-1}"
AUTO_OPEN="${AUTO_OPEN:-1}"

if [ -t 1 ]; then B=$'\033[1m'; DIM=$'\033[2m'; R=$'\033[0m'
  CYN=$'\033[36m'; GRN=$'\033[32m'; YEL=$'\033[33m'; RED=$'\033[31m'
else B=""; DIM=""; R=""; CYN=""; GRN=""; YEL=""; RED=""; fi
log()  { printf '%s[auto]%s %s\n' "$CYN" "$R" "$*"; }
ok()   { printf '%s[auto]%s %s\n' "$GRN" "$R" "$*"; }
warn() { printf '%s[auto]%s %s\n' "$YEL" "$R" "$*"; }
bad()  { printf '%s[auto]%s %s\n' "$RED" "$R" "$*"; }

mkdir -p "$STATE_DIR"

# -----------------------------------------------------------------------------
# URLs - the Codespace URL when there is one, localhost otherwise
# -----------------------------------------------------------------------------
codespace_url() {
  local domain="${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}"
  if [ -n "${CODESPACE_NAME:-}" ]; then
    printf 'https://%s-%s.%s' "$CODESPACE_NAME" "$PORT" "$domain"
  else
    printf 'http://localhost:%s' "$PORT"
  fi
}

# -----------------------------------------------------------------------------
# 1. Self-heal: is this environment provisioned *now*?
# -----------------------------------------------------------------------------
requirements_fingerprint() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$REPO_ROOT/requirements.txt" | cut -d' ' -f1
  else
    shasum -a 256 "$REPO_ROOT/requirements.txt" | cut -d' ' -f1
  fi
}

imports_ok() {
  [ -x "$PY" ] || return 1
  PYTHONPATH="$REPO_ROOT" "$PY" - <<'PYEOF' >/dev/null 2>&1
import importlib, sys
for name in ("numpy", "scipy", "fastapi", "uvicorn", "httpx", "feedparser", "websockets"):
    try:
        importlib.import_module(name)
    except Exception:
        sys.exit(1)
from backend.formulas.engine import ALL_FORMULAS
sys.exit(0 if len(ALL_FORMULAS) >= 22 else 1)
PYEOF
}

needs_provision() {
  [ -x "$PY" ] || return 0
  [ -f "$REPO_ROOT/.env" ] || return 0
  local want have
  want="$(requirements_fingerprint)"
  have="$(cat "$STAMP" 2>/dev/null || true)"
  [ "$want" = "$have" ] || return 0
  imports_ok || return 0
  return 1
}

provision() {
  if ! needs_provision; then
    log "environment already provisioned and up to date - nothing to download"
    return 0
  fi
  log "provisioning (system packages, virtualenv, requirements, .env, redis)"
  log "the download log for pip is /tmp/pip-install.log; this is the slow step, once"
  bash "$STATE_DIR/setup.sh" || warn "setup.sh reported problems - carrying on"
  if imports_ok; then
    requirements_fingerprint > "$STAMP"
    ok "provisioned: $(requirements_fingerprint | cut -c1-12)"
  else
    warn "the engine is still missing dependencies; run:  bash .devcontainer/setup.sh"
  fi
}

# -----------------------------------------------------------------------------
# 2. Flutter: download the SDK and build the web client, in the background
# -----------------------------------------------------------------------------
flutter_running() {
  [ -f "$FLUTTER_PID" ] || return 1
  local pid
  pid="$(cat "$FLUTTER_PID" 2>/dev/null || true)"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null
}

flutter_ready() {
  [ -f "$REPO_ROOT/frontend/build/web/index.html" ]
}

start_flutter() {
  if [ "$AUTO_FLUTTER" != "1" ]; then
    log "AUTO_FLUTTER=0 - skipping the Flutter download (set AUTO_FLUTTER=1 to enable)"
    return 0
  fi
  if flutter_ready; then
    ok "Flutter web bundle already built - served at /flutter"
    return 0
  fi
  if flutter_running; then
    log "Flutter setup is already running (pid $(cat "$FLUTTER_PID")) - log: flutter-setup.log"
    return 0
  fi
  log "starting the Flutter SDK download + web build in the background"
  log "it needs no input and does not block the dashboard; watch: tail -f flutter-setup.log"
  nohup env INSTALL_FLUTTER=1 bash "$REPO_ROOT/frontend/run_web.sh" \
    >"$FLUTTER_LOG" 2>&1 &
  echo $! > "$FLUTTER_PID"
}

# -----------------------------------------------------------------------------
# 3. The engine
# -----------------------------------------------------------------------------
engine_answers() {
  curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1
}

port_occupied() {
  # Not "is something listening" - curl already told us nothing *answers*.  A
  # socket that accepts a connection but serves no HTTP is a dead process still
  # holding the port, which is exactly the state that produces a blank page and
  # that no amount of restarting fixes.
  (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") >/dev/null 2>&1
}

start_engine() {
  if engine_answers; then
    log "the engine is already answering on port ${PORT}"
    return 0
  fi
  if port_occupied; then
    warn "port ${PORT} is held by something that does not answer - clearing it"
    bash "$REPO_ROOT/run.sh" --clean >/dev/null 2>&1 || true
  fi
  log "starting the engine on port ${PORT} (background, supervised)"
  bash "$REPO_ROOT/run.sh" --bg --port "$PORT" || warn "run.sh returned non-zero - see server.log"
}

wait_until_ready() {
  local waited=0 limit="${1:-150}"
  while [ "$waited" -lt "$limit" ]; do
    if engine_answers; then
      ok "engine up after ${waited}s"
      return 0
    fi
    sleep 3
    waited=$((waited + 3))
    if [ $((waited % 15)) -eq 0 ]; then
      printf '%s[auto]%s waiting for the engine (%ss) - it warms up the 80-node brain on first start\n' \
        "$DIM" "$R" "$waited"
    fi
  done
  bad "the engine did not answer within ${limit}s - diagnostics:"
  bash "$REPO_ROOT/run.sh" --status 2>&1 | sed 's/^/      /' || true
  return 1
}

prediction_age() {
  curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/api/signal/current" 2>/dev/null \
    | sed -n 's/.*"age_seconds"[[:space:]]*:[[:space:]]*\([0-9.]*\).*/\1s old/p' | head -1
}

wait_until_locked() {
  # The first prediction needs one window; do not make the user stare at a
  # "computing" panel wondering whether it worked.
  local waited=0
  while [ "$waited" -lt 40 ]; do
    local side
    side="$(curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/api/signal/current" 2>/dev/null \
      | sed -n 's/.*"signal"[[:space:]]*:[[:space:]]*"\(BUY\|SELL\)".*/\1/p' | head -1)"
    if [ -n "$side" ]; then
      LOCKED_SIDE="$side"
      ok "first prediction locked: ${side}"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  log "no prediction locked yet - the panel will fill on the next window"
  return 0
}

# -----------------------------------------------------------------------------
# 4. Make the forwarded port public, so the URL opens without a login dance
# -----------------------------------------------------------------------------
open_port() {
  [ "$AUTO_OPEN" = "1" ] || return 0
  [ -n "${CODESPACE_NAME:-}" ] || return 0
  command -v gh >/dev/null 2>&1 || return 0
  # Best effort: devcontainer.json already asks for a public port; this is the
  # belt to that pair of braces when the metadata has not applied yet.
  nohup gh codespace ports visibility "${PORT}:public" -c "$CODESPACE_NAME" \
    >/dev/null 2>&1 &
}

# -----------------------------------------------------------------------------
# 5. The banner the user actually reads
# -----------------------------------------------------------------------------
banner() {
  local url; url="$(codespace_url)"
  printf '\n'
  printf '%s================================================================%s\n' "$B" "$R"
  printf '%s  DROSOPHILA TRADER v2.0 - running, nothing to do%s\n' "$B" "$R"
  printf '%s================================================================%s\n' "$B" "$R"
  printf '  dashboard    %s/\n' "$url"
  printf '  api keys     %s/settings   (click, paste, test - no terminal)\n' "$url"
  printf '  brain matrix %s/matrix\n' "$url"
  printf '  api docs     %s/docs\n' "$url"
  if flutter_ready; then
    printf '  flutter app  %s/flutter\n' "$url"
  elif flutter_running; then
    printf '  flutter app  %s/flutter   (still building, log: flutter-setup.log)\n' "$url"
  else
    printf '  flutter app  %s/flutter   (not built; AUTO_FLUTTER=0 was set)\n' "$url"
  fi
  if [ -n "${LOCKED_SIDE:-}" ]; then
    printf '  right now    %s locked · %s\n' "$LOCKED_SIDE" "$(prediction_age)"
  fi
  # The *effective* timing: PREDICTION_MAX_AGE_SECONDS and OUTCOME_HORIZON_SECONDS
  # default to "0 = as long as its own window", so the raw config would print 0.
  local timing cadence maxage horizon
  timing="$(PYTHONPATH="$REPO_ROOT" "$PY" -c \
    'from backend.core import config as c; s = c.SETTINGS; print(int(s.cycle_period_seconds), int(s.prediction_expired), int(s.outcome_horizon))' \
    2>/dev/null || true)"
  read -r cadence maxage horizon <<EOF
$timing
EOF
  [ -n "${cadence:-}" ] || cadence=60
  printf '\n  %ss countdown · every panel refreshes together at t+15 / t+30 / t+45\n' "$cadence"
  printf '  the signal on screen is always this window'"'"'s own · scored %ss later\n' "${horizon:-60}"
  printf '  it restarts itself if it ever stops; this container brings it back on wake\n\n'
}

# -----------------------------------------------------------------------------
# Modes
# -----------------------------------------------------------------------------
case "$MODE" in
  provision)
    # NOTE: start_flutter detaches the download itself (nohup ... &).  Never
    # `wait` on it here: postCreateCommand would then hang for the minutes the
    # Flutter SDK takes, and the user would stare at a frozen "creating
    # container" screen for a step that is entirely optional.
    provision
    start_flutter
    ok "provisioning finished - the engine starts on the next attach"
    ;;
  start)
    provision
    start_engine
    open_port
    start_flutter
    wait_until_ready 150 || true
    ;;
  attach)
    provision
    start_engine
    open_port
    start_flutter
    wait_until_ready 150 || true
    wait_until_locked
    banner
    ;;
  status)
    printf 'engine: %s\n' "$(engine_answers && echo "answering on ${PORT}" || echo 'not answering')"
    printf 'flutter: %s\n' "$(flutter_ready && echo built || (flutter_running && echo building || echo 'not started'))"
    printf 'url: %s\n' "$(codespace_url)"
    ;;
esac

exit 0
