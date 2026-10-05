#!/usr/bin/env bash
# Commit frontend/build/web to the branch the workflow ran on.  Never touches
# main.  On failure the log tail is posted on the branch's open PR so it can
# be read without the Actions UI.
set -uo pipefail
LOG=/tmp/commit.log
: >"$LOG"
run() { echo "+ $*" >>"$LOG"; "$@" >>"$LOG" 2>&1; }

report() {
  local rc=$1
  cat "$LOG"
  if [ "$rc" -ne 0 ]; then
    {
      echo "### commit step failed (run ${RUN_ID:-?}, exit $rc)"
      echo '```'
      tail -n 60 "$LOG"
      echo '```'
    } >/tmp/c.md
    PR=$(gh pr list --head "${GITHUB_REF_NAME}" --state open --json number --jq '.[0].number' 2>/dev/null || true)
    [ -n "$PR" ] && gh pr comment "$PR" --body-file /tmp/c.md >/dev/null 2>&1 || true
  fi
  exit "$rc"
}

run git config user.name "flutter-web bot"
run git config user.email "actions@users.noreply.github.com"
run git add -f frontend/build/web || report $?
if git diff --cached --quiet; then
  echo "bundle unchanged - nothing to commit" >>"$LOG"
  report 0
fi
run git commit -m "Flutter web bundle (built by GitHub Actions) [skip ci]" || report $?
# "flutter pub get" rewrites analysis_options.yaml / pubspec.lock; drop those
# so the rebase below starts from a clean tree (only the bundle is wanted)
run git checkout -- .
run git status --short
# the branch may have moved while we built: rebase our one commit onto it
run git fetch origin "${GITHUB_REF_NAME}" || report $?
if ! run git rebase "origin/${GITHUB_REF_NAME}"; then
  run git rebase --abort
  report 1
fi
run git push origin "HEAD:${GITHUB_REF_NAME}" || report $?
report 0
