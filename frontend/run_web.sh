#!/usr/bin/env bash
# =============================================================================
# Build the Flutter client as a web app and hand it to the running backend.
#
#   bash frontend/run_web.sh
#
# The build is served by the FastAPI server at /flutter, so there is ONE port,
# ONE forwarded URL and no CORS. Open:  <dashboard-url>/flutter
#
# Why "flutter run -d chrome" does nothing in a Codespace:
#   * there is no Chrome/GPU to open a window in,
#   * `flutter` is only on PATH in the shell that installed it,
#   * the dev server binds a port nobody forwarded.
# A release build + the existing web server avoids all three.
#
# Options:
#   --dev      also start `flutter run -d web-server` on port 8081
#   --api URL  override the API base baked into the build
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$REPO_ROOT/frontend"
cd "$FRONTEND_DIR"

if [ -t 1 ]; then B=$'\033[1m'; DIM=$'\033[2m'; R=$'\033[0m'
  RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; CYN=$'\033[36m'
else B=""; DIM=""; R=""; RED=""; GRN=""; YEL=""; CYN=""; fi
ok()   { printf '%s✔%s %s\n' "$GRN" "$R" "$*"; }
warn() { printf '%s!%s %s\n' "$YEL" "$R" "$*"; }
bad()  { printf '%s✘%s %s\n' "$RED" "$R" "$*"; }
step() { printf '\n%s==>%s %s%s%s\n' "$CYN" "$R" "$B" "$*" "$R"; }

MODE="build"
API_BASE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dev) MODE="dev" ;;
    --api) API_BASE="${2:-}"; shift ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) warn "ignoring unknown argument: $1" ;;
  esac
  shift
done

# ------------------------------------------------------------- find flutter ---
find_flutter() {
  if command -v flutter >/dev/null 2>&1; then command -v flutter; return 0; fi
  for candidate in "$HOME/flutter/bin/flutter" /usr/local/flutter/bin/flutter \
                   /opt/flutter/bin/flutter /usr/lib/flutter/bin/flutter; do
    [ -x "$candidate" ] && { echo "$candidate"; return 0; }
  done
  return 1
}

step "1/4  Locating the Flutter SDK"
FLUTTER="$(find_flutter || true)"
if [ -z "$FLUTTER" ]; then
  warn "flutter is not installed (this is why 'nothing happened' before)"
  answer=""
  if [ -t 0 ]; then
    printf '    Clone the stable Flutter SDK into ~/flutter now? [y/N] '
    read -r answer || answer=""
  fi
  if [ "${answer:-}" != "y" ] && [ "${answer:-}" != "Y" ]; then
    cat <<'EOF'

    Install it yourself, then re-run this script:

      git clone --depth 1 -b stable https://github.com/flutter/flutter.git ~/flutter
      export PATH="$HOME/flutter/bin:$PATH"
      echo 'export PATH="$HOME/flutter/bin:$PATH"' >> ~/.bashrc

    (Or, from the repo root:  INSTALL_FLUTTER=1 bash .devcontainer/setup.sh)
EOF
    exit 1
  fi
  command -v git >/dev/null 2>&1 || { bad "git is required to clone Flutter"; exit 1; }
  git clone --depth 1 -b stable https://github.com/flutter/flutter.git "$HOME/flutter" \
    || { bad "clone failed (network?)"; exit 1; }
  FLUTTER="$HOME/flutter/bin/flutter"
fi
export PATH="$(dirname "$FLUTTER"):$PATH"
ok "flutter at $FLUTTER"

step "2/4  Preparing the project"
"$FLUTTER" config --enable-web >/dev/null 2>&1 || warn "could not set --enable-web (continuing)"
"$FLUTTER" --version 2>/dev/null | head -1 | sed "s/^/${DIM}    /;s/$/${R}/"
if ! "$FLUTTER" pub get >/tmp/flutter_pub_get.log 2>&1; then
  bad "flutter pub get failed:"
  tail -n 15 /tmp/flutter_pub_get.log
  exit 1
fi
ok "dependencies resolved"

step "3/4  Working out the API base URL"
if [ -z "$API_BASE" ]; then
  if [ -n "${CODESPACE_NAME:-}" ]; then
    DOMAIN="${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}"
    API_BASE="https://${CODESPACE_NAME}-${PORT:-8000}.${DOMAIN}"
  else
    API_BASE="http://localhost:${PORT:-8000}"
  fi
fi
ok "API_BASE=$API_BASE"

step "4/4  Building (first build takes 2-5 minutes)"
if [ "$MODE" = "dev" ]; then
  warn "dev mode: watch for the http://localhost:8081 line, then forward port 8081"
  exec "$FLUTTER" run -d web-server --web-hostname 0.0.0.0 --web-port 8081 \
    --dart-define=API_BASE="$API_BASE"
fi

if ! "$FLUTTER" build web --release --base-href /flutter/ \
      --dart-define=API_BASE="$API_BASE" >/tmp/flutter_build.log 2>&1; then
  bad "flutter build web failed - last lines of the log:"
  tail -n 25 /tmp/flutter_build.log
  exit 1
fi

BUILD_DIR="$FRONTEND_DIR/build/web"
[ -f "$BUILD_DIR/index.html" ] || { bad "no index.html in $BUILD_DIR"; exit 1; }
SIZE="$(du -sh "$BUILD_DIR" | cut -f1)"
ok "built into frontend/build/web ($SIZE)"

if [ -n "${CODESPACE_NAME:-}" ]; then
  URL="https://${CODESPACE_NAME}-${PORT:-8000}.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}/flutter"
else
  URL="http://localhost:${PORT:-8000}/flutter"
fi

cat <<EOF

${B}${GRN}  ================================================================${R}
   Flutter web build ready — served by the engine (restart it if it was
   already running, so the new mount is picked up):

     ${B}${URL}${R}

   API base baked in: ${API_BASE}
   Dashboard:         ${URL%/flutter}
${B}${GRN}  ================================================================${R}

   If that URL 404s: the backend was started before the build existed.
   Stop it and start it again:   bash run.sh --bg
EOF
