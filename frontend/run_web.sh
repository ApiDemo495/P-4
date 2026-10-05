#!/usr/bin/env bash
# =============================================================================
# Build the Flutter client as a web app and hand it to the running backend.
#
#   bash frontend/run_web.sh
#
# The build is served by the FastAPI server at /flutter, so there is ONE port,
# ONE forwarded URL and no CORS. Open:  <dashboard-url>/flutter
#
# Options:
#   (none)     build the release bundle and serve it at /flutter  <- ONE port
#   --check    report whether the SDK is installed and whether a build exists
#   --dev      optional hot-reload dev server, FIXED at port 8081 (it is the
#              only case in which a second port appears; never let flutter
#              pick a random one)
#
# Why "flutter run -d chrome" does nothing in a Codespace:
#   * there is no Chrome/GPU to open a window in,
#   * `flutter` is only on PATH in the shell that installed it,
#   * the dev server binds a port nobody forwarded.
# A release build + the existing web server avoids all three.
#
# Options:
#   --api URL  override the API base baked into the build
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$REPO_ROOT/frontend"
cd "$FRONTEND_DIR"

if [ -t 1 ]; then B=$'\033[1m'; DIM=$'\033[2m'; R=$'\033[0m'
  RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; CYN=$'\033[36m'
else B=""; DIM=""; R=""; RED=""; GRN=""; YEL=""; CYN=""; fi
say()  { printf '%s\n' "$*"; }
ok()   { printf '%s✔%s %s\n' "$GRN" "$R" "$*"; }
warn() { printf '%s!%s %s\n' "$YEL" "$R" "$*"; }
bad()  { printf '%s✘%s %s\n' "$RED" "$R" "$*"; }
step() { printf '\n%s==>%s %s%s%s\n' "$CYN" "$R" "$B" "$*" "$R"; }

MODE="build"
API_BASE=""
PORT="${PORT:-8000}"
while [ $# -gt 0 ]; do
  case "$1" in
    --dev) MODE="dev" ;;
    --check|--doctor) MODE="check" ;;
    --api) API_BASE="${2:-}"; shift ;;
    --port) PORT="${2:-8000}"; shift ;;
    -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) warn "ignoring unknown argument: $1" ;;
  esac
  shift
done

# ------------------------------------------------------------- find flutter ---
find_flutter() {
  if command -v flutter >/dev/null 2>&1; then command -v flutter; return 0; fi
  local remembered=""
  [ -f "$REPO_ROOT/.run/flutter_home" ] && remembered="$(cat "$REPO_ROOT/.run/flutter_home")/bin/flutter"
  for candidate in "${FLUTTER_HOME:-/nonexistent}/bin/flutter" "$remembered" "$HOME/flutter/bin/flutter" \
                   "$HOME/flutter-sdk/bin/flutter" /usr/local/flutter/bin/flutter \
                   /opt/flutter/bin/flutter /usr/lib/flutter/bin/flutter; do
    [ -n "$candidate" ] || continue
    [ -x "$candidate" ] && { echo "$candidate"; return 0; }
  done
  return 1
}

if [ "$MODE" = "check" ]; then
  step "Flutter web client - diagnosis"
  FLUTTER="$(find_flutter || true)"
  if [ -n "$FLUTTER" ]; then
    ok "flutter found: $FLUTTER"
    "$FLUTTER" --version 2>/dev/null | head -1 | sed "s/^/    /"
  else
    warn "flutter is not installed (~700 MB download, optional)"
    say "    install:  bash frontend/run_web.sh        (it clones the stable SDK)"
    say "    ${DIM}the web dashboard at http://localhost:${PORT}/ already has every${R}"
    say "    ${DIM}feature; the Flutter client is the same app in Dart.${R}"
  fi
  if [ -d "$FRONTEND_DIR/build/web" ]; then
    ok "web build present: frontend/build/web ($(du -sh "$FRONTEND_DIR/build/web" | cut -f1))"
    say "    served by the engine at:  http://localhost:${PORT}/flutter"
  else
    warn "no web build yet - run:  bash frontend/run_web.sh"
  fi
  say ""
  say "ONE port rule: the release build is served by the engine, so /flutter needs"
  say "no extra port.  Only --dev opens a second one, and it is fixed at 8081."
  exit 0
fi

step "1/4  Locating the Flutter SDK"
FLUTTER="$(find_flutter || true)"
if [ -z "$FLUTTER" ]; then
  warn "flutter is not installed (this is why 'nothing happened' before)"
  answer=""
  # Only ask when a person is actually there.  The automatic start runs this
  # script detached (INSTALL_FLUTTER=1, stdin from /dev/null); a prompt in that
  # situation would stop the download for ever without anyone seeing it.
  if [ "${INSTALL_FLUTTER:-0}" != "1" ] && [ -t 0 ] && [ -t 1 ]; then
    printf '    Clone the stable Flutter SDK into ~/flutter now? [y/N] '
    read -r -t 60 answer || answer=""
  fi
  if [ "${answer:-}" != "y" ] && [ "${answer:-}" != "Y" ] && [ "${INSTALL_FLUTTER:-0}" != "1" ]; then
    cat <<'EOF'

    Install it yourself, then re-run this script:

      git clone --depth 1 --single-branch -b stable \
        https://github.com/flutter/flutter.git ~/flutter
      export PATH="$HOME/flutter/bin:$PATH"

    You do NOT need Flutter to use the app: the web dashboard on port 8000 has
    every feature, and the Flutter client is the same app written in Dart.

    (Or, from the repo root:  INSTALL_FLUTTER=1 bash frontend/run_web.sh)
EOF
    exit 1
  fi
  FLUTTER_DIR="${FLUTTER_HOME:-$HOME/flutter}"
  if [ -z "${FLUTTER_HOME:-}" ] && [ -f "$REPO_ROOT/.run/flutter_home" ]; then
    FLUTTER_DIR="$(cat "$REPO_ROOT/.run/flutter_home")"
  fi

  # -- pre-flight 1: anything already at $FLUTTER_DIR that is not a working SDK
  #    is the reason for "fatal: destination path already exists" - a partial
  #    clone (with or without .git), an empty folder, a wrongly-cased FLUTTER.
  #    Move it aside (never fail on rm), and if the path still cannot be freed,
  #    install next to it and remember the new home for every later run.
  for stale in "$FLUTTER_DIR" "$HOME/FLUTTER" "$HOME/Flutter"; do
    [ -e "$stale" ] || continue
    [ -x "$stale/bin/flutter" ] && continue
    warn "removing an incomplete Flutter SDK left at $stale by an earlier attempt"
    rm -rf "$stale" 2>/dev/null || mv "$stale" "${stale}.broken.$$" 2>/dev/null || true
  done
  if [ -e "$FLUTTER_DIR" ] && [ ! -x "$FLUTTER_DIR/bin/flutter" ]; then
    FLUTTER_DIR="$HOME/flutter-sdk"
    warn "could not free the old path - installing into $FLUTTER_DIR instead"
    rm -rf "$FLUTTER_DIR" 2>/dev/null || true
  fi
  mkdir -p "$REPO_ROOT/.run" && printf '%s\n' "$FLUTTER_DIR" >"$REPO_ROOT/.run/flutter_home"
  # -- pre-flight 2: disk - the SDK + Dart + web engine need ~3 GB free.
  free_mb="$(df -Pm "$HOME" 2>/dev/null | awk 'NR==2 {print $4}')"
  if [ -n "${free_mb:-}" ] && [ "$free_mb" -lt 3000 ]; then
    bad "only ${free_mb} MB free under $HOME - the Flutter SDK needs about 3 GB."
    say "    free some space (docker system prune, old venvs) or use a larger Codespace machine,"
    say "    then click Restart on /flutter.  The web dashboard on port ${PORT} is unaffected."
    exit 1
  fi

  ok_attempt=0
  parent="$(dirname "$FLUTTER_DIR")"
  staging="$parent/.flutter.extracting.$$"

  # -- route A (preferred): the official release archive - ONE resumable file,
  #    exactly what flutter.dev's download button serves.  Much faster than a
  #    git clone of the whole flutter repository and immune to "already exists".
  step_archive() {
    local url="$1" tmp_tar
    tmp_tar="$parent/.flutter_sdk_download.tar.xz"     # fixed name so -C - resumes
    say "${DIM}   downloading $url${R}"
    if ! curl -fL --retry 5 --retry-delay 3 -C - -sS -o "$tmp_tar" "$url"; then
      # a resume against a changed file can fail: start that one over
      rm -f "$tmp_tar"
      curl -fL --retry 5 --retry-delay 3 -sS -o "$tmp_tar" "$url" || return 1
    fi
    rm -rf "$staging" && mkdir -p "$staging"
    if tar -xJf "$tmp_tar" -C "$staging" && [ -x "$staging/flutter/bin/flutter" ]; then
      rm -rf "$FLUTTER_DIR"
      mv "$staging/flutter" "$FLUTTER_DIR" && rm -rf "$staging" && rm -f "$tmp_tar" && return 0
    fi
    rm -rf "$staging"; rm -f "$tmp_tar"
    return 1
  }
  releases_json="https://storage.googleapis.com/flutter_infra_release/releases/releases_linux.json"
  archive_path="$(curl -fsSL --retry 3 --max-time 30 "$releases_json" 2>/dev/null | python3 -c '
import json, sys
data = json.load(sys.stdin)
stable = data["current_release"]["stable"]
for rel in data["releases"]:
    if rel["hash"] == stable and rel.get("dart_sdk_arch", "x64") == "x64":
        print(rel["archive"]); break
' 2>/dev/null || true)"
  candidates=""
  [ -n "$archive_path" ] && candidates="https://storage.googleapis.com/flutter_infra_release/releases/${archive_path}"
  # Known-good stable archives, newest first, when the index is unreachable.
  for v in 3.35.5 3.35.4 3.35.3 3.32.8 3.32.5 3.29.3; do
    candidates="$candidates https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_${v}-stable.tar.xz"
  done
  for url in $candidates; do
    step_archive "$url" && { ok_attempt=1; ok "Flutter SDK installed from the release archive"; break; }
    warn "archive route failed for $(basename "$url")"
  done

  # -- route B: shallow git clone into a *fresh* directory, then move it into
  #    place (so git can never complain that the destination exists).
  if [ "$ok_attempt" != 1 ] && command -v git >/dev/null 2>&1; then
    for attempt in 1 2 3; do
      rm -rf "$staging"
      say "${DIM}   cloning the stable Flutter SDK (~700 MB, attempt $attempt)${R}"
      if git clone --depth 1 --single-branch -b stable \
           https://github.com/flutter/flutter.git "$staging" 2>&1 | tail -n 4 | sed 's/^/      /' \
         && [ -x "$staging/bin/flutter" ]; then
        rm -rf "$FLUTTER_DIR"
        mv "$staging" "$FLUTTER_DIR" && { ok_attempt=1; ok "Flutter SDK cloned"; break; }
      fi
      rm -rf "$staging"
      warn "git attempt $attempt failed"
      sleep $((attempt * 5))
    done
  elif [ "$ok_attempt" != 1 ]; then
    warn "git is not installed - skipping the clone route"
  fi

  if [ "$ok_attempt" != 1 ]; then
    bad "could not download the Flutter SDK (both github.com and storage.googleapis.com routes failed)."
    cat <<EOF

    Check from this terminal what is blocked:
      curl -sI https://github.com | head -1
      curl -sI https://storage.googleapis.com | head -1

    Then click Restart on /flutter, or run by hand:
      INSTALL_FLUTTER=1 bash frontend/run_web.sh

    Manual install (any machine, no git needed):
      curl -fL -o /tmp/flutter.tar.xz https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_3.35.5-stable.tar.xz
      rm -rf ~/flutter && tar -xJf /tmp/flutter.tar.xz -C ~
      bash frontend/run_web.sh
    (any flutter_linux_<version>-stable.tar.xz from https://docs.flutter.dev/install/archive works;
     the script finds ~/flutter/bin/flutter and only builds.)

    The web dashboard on port ${PORT} has every feature meanwhile.
EOF
    exit 1
  fi
  # Flutter refuses to run from a directory owned by another user (a tarball
  # extracted as root, a Codespace whose HOME changed) - allow ours explicitly.
  git config --global --add safe.directory "$FLUTTER_DIR" >/dev/null 2>&1 || true
  FLUTTER="$FLUTTER_DIR/bin/flutter"
fi
export PATH="$(dirname "$FLUTTER"):$PATH"
ok "flutter at $FLUTTER"
# No analytics prompt, no "welcome" banner, no first-run questions: the
# automatic start must never wait on a keyboard.
export FLUTTER_SUPPRESS_ANALYTICS=true CI=true
"$FLUTTER" --disable-analytics >/dev/null 2>&1 || true
# The Dart SDK and the web engine are fetched on first use; do it explicitly so
# a failure is reported here (with a retry) instead of half-way through a build.
say "${DIM}   downloading the Dart SDK + web engine (flutter precache --web, 1-3 min)${R}"
precache_ok=0
for attempt in 1 2 3 4 5; do
  if "$FLUTTER" precache --web >/tmp/flutter_precache.log 2>&1; then precache_ok=1; break; fi
  warn "flutter precache --web failed (attempt $attempt) - retrying in $((attempt * 5)) s"
  tail -n 3 /tmp/flutter_precache.log 2>/dev/null | sed 's/^/      /'
  sleep $((attempt * 5))
done
[ "$precache_ok" = 1 ] || warn "precache kept failing - continuing; the build step will download what it needs"

step "2/4  Preparing the project"
"$FLUTTER" config --enable-web >/dev/null 2>&1 || warn "could not set --enable-web (continuing)"
if [ ! -f web/index.html ]; then
  # the web scaffold (web/index.html, manifest, icons) is tracked in git; if a
  # checkout lost it, regenerate it so "flutter build web" never refuses with
  # "This project is not configured for the web".
  warn "web/ scaffold missing - regenerating it"
  "$FLUTTER" create . --platforms web --project-name drosophila_trader >/dev/null 2>&1 \
    || bad "could not regenerate the web scaffold"
fi
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
  cat <<EOF

   ${YEL}Dev mode uses a SECOND port on purpose${R} (hot reload), and it is fixed
   at 8081 - never a random one:

     http://localhost:8081/          (inside the Codespace)
     PORTS tab -> 8081 -> globe icon (from your browser)

   For the single-port setup, press Ctrl-C and run:  bash frontend/run_web.sh
EOF
  exec "$FLUTTER" run -d web-server \
    --web-hostname 0.0.0.0 --web-port 8081 \
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
   Flutter web build ready - served by the engine on the SAME port, so there
   is nothing else to forward:

     ${B}${URL}${R}          <- the app
     ${URL%/flutter}          <- the dashboard (restart it if it was already
                                 running, so the new mount is picked up)

   API base baked in: ${API_BASE}
${B}${GRN}  ================================================================${R}

   If /flutter 404s, the backend predates the build:  bash run.sh --stop && bash run.sh --bg
   Ports looking broken?  bash run.sh --clean   (kills stale servers)
EOF
