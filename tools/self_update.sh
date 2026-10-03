#!/usr/bin/env bash
# =============================================================================
# Self-update: bring THIS checkout up to date with its own branch, in place.
#
#   bash tools/self_update.sh --check     JSON: is the branch behind origin?
#   bash tools/self_update.sh --apply     fetch + fast-forward + re-provision if
#                                         requirements changed + restart engine
#
# Rules (this is what makes it safe to run unattended):
#   * never `checkout`, never switch or create a branch, never touch `main`;
#     the current branch is the only thing fetched and the only thing moved;
#   * fast-forward ONLY - if the branch has diverged nothing is touched and the
#     JSON says so;
#   * local edits are kept: git refuses a fast-forward that would overwrite an
#     edited file, and then this script leaves everything exactly as it was;
#   * a detached HEAD (no branch) is left alone.
#
# It is called by tools/codespace_autostart.sh on every start/attach, by the
# engine (GET /api/update/status, POST /api/update/apply) and, in a Codespace,
# automatically every few minutes (AUTO_UPDATE=0 turns that off).
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PORT="${PORT:-8000}"
MODE="check"
RESTART=1
for arg in "$@"; do
  case "$arg" in
    --check) MODE="check" ;;
    --apply) MODE="apply" ;;
    --no-restart) RESTART=0 ;;
    --port) : ;;
    *) [[ "$arg" =~ ^[0-9]+$ ]] && PORT="$arg" ;;
  esac
done

json() {  # json key=value ... (values are strings unless they look like numbers/bools)
  local out="{" first=1 kv k v
  for kv in "$@"; do
    k="${kv%%=*}"; v="${kv#*=}"
    [ "$first" = 1 ] || out+=","
    first=0
    if [[ "$v" =~ ^(-?[0-9]+|true|false|null)$ ]]; then out+="\"$k\":$v"
    else v="${v//\\/\\\\}"; v="${v//\"/\\\"}"; out+="\"$k\":\"$v\""; fi
  done
  printf '%s}\n' "$out"
}

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  json ok=false reason="not a git working tree" behind=0; exit 0
fi
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo HEAD)"
if [ "$BRANCH" = "HEAD" ]; then
  json ok=false reason="detached HEAD - not touching it" behind=0 branch=HEAD; exit 0
fi
LOCAL="$(git rev-parse --short HEAD 2>/dev/null)"

# Fetch only our own branch, by explicit refspec (never the whole remote).
FETCH_ERR=""
if ! timeout 90 git fetch -q origin "+refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}" 2>/tmp/self_update_fetch.err; then
  FETCH_ERR="$(tr '\n' ' ' </tmp/self_update_fetch.err | cut -c1-200)"
fi
REMOTE="$(git rev-parse --short "origin/${BRANCH}" 2>/dev/null || echo "")"
if [ -z "$REMOTE" ]; then
  json ok=false reason="origin/${BRANCH} unknown: ${FETCH_ERR:-no such remote branch}" branch="$BRANCH" local="$LOCAL" behind=0; exit 0
fi
BEHIND="$(git rev-list --count "HEAD..origin/${BRANCH}" 2>/dev/null || echo 0)"
AHEAD="$(git rev-list --count "origin/${BRANCH}..HEAD" 2>/dev/null || echo 0)"
DIRTY="$(git status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')"
CAN_FF=false; [ "$AHEAD" = 0 ] && CAN_FF=true
LAST_SUBJECT="$(git log -1 --format=%s "origin/${BRANCH}" 2>/dev/null | cut -c1-120)"

if [ "$MODE" = "check" ]; then
  json ok=true branch="$BRANCH" local="$LOCAL" remote="$REMOTE" behind="$BEHIND" ahead="$AHEAD" \
       dirty_files="$DIRTY" can_fast_forward="$CAN_FF" fetch_error="$FETCH_ERR" latest="$LAST_SUBJECT"
  exit 0
fi

# ------------------------------------------------------------------ apply ---
LOG="$REPO_ROOT/.run/update.log"
mkdir -p "$REPO_ROOT/.run"
exec >>"$LOG" 2>&1
echo "==> self-update $(date -u +%FT%TZ) on ${BRANCH}: local ${LOCAL}, origin ${REMOTE}, behind ${BEHIND}, ahead ${AHEAD}"
if [ "$BEHIND" = 0 ]; then echo "    already up to date"; exit 0; fi
if [ "$CAN_FF" != true ]; then
  echo "    the branch has diverged (ahead ${AHEAD}) - refusing to move it; nothing changed"; exit 2
fi
REQ_BEFORE="$(sha256sum requirements.txt 2>/dev/null | cut -d' ' -f1)"
# An untracked file that the incoming commits add (typically a copy of this
# very script dropped in by hand to bootstrap) blocks the fast-forward.  If it
# is byte-identical to what is coming, letting git write it is a no-op: remove
# the copy.  Anything that differs is a real local file and stays.
while IFS= read -r path; do
  [ -n "$path" ] || continue
  if git cat-file -e "origin/${BRANCH}:${path}" 2>/dev/null \
     && git cat-file -p "origin/${BRANCH}:${path}" | cmp -s - "$path"; then
    echo "    untracked ${path} is identical to the incoming version - letting git own it"
    rm -f "$path"
  fi
done < <(git ls-files --others --exclude-standard)
# Same for a *tracked* file edited to exactly the incoming content: restore
# the HEAD version so the fast-forward can write the identical bytes back.
while IFS= read -r path; do
  [ -n "$path" ] || continue
  if git cat-file -e "origin/${BRANCH}:${path}" 2>/dev/null \
     && git cat-file -p "origin/${BRANCH}:${path}" | cmp -s - "$path"; then
    echo "    modified ${path} already equals the incoming version - letting git own it"
    git restore --worktree --source=HEAD -- "$path" 2>/dev/null || true
  fi
done < <(git diff --name-only)
if ! git merge --ff-only -q "origin/${BRANCH}"; then
  echo "    fast-forward refused (a local edit overlaps an incoming change) - nothing changed"
  git status --porcelain --untracked-files=no | head -20
  exit 3
fi
echo "    now at $(git rev-parse --short HEAD): $(git log -1 --format=%s | cut -c1-100)"
REQ_AFTER="$(sha256sum requirements.txt 2>/dev/null | cut -d' ' -f1)"
if [ "$REQ_BEFORE" != "$REQ_AFTER" ]; then
  echo "    requirements.txt changed - installing"
  bash run.sh --setup-only 9>&- || echo "    (setup reported problems - see above)"
  rm -f .run/.provisioned
fi
if [ "$RESTART" = 1 ]; then
  echo "    restarting the engine on port ${PORT}"
  bash run.sh --stop 9>&- >/dev/null 2>&1 || true
  bash run.sh --bg --port "$PORT" 9>&- || echo "    (run.sh --bg returned non-zero - see server.log)"
fi
echo "    done $(date -u +%FT%TZ)"
