#!/usr/bin/env bash
# Stop: fires only when the content of modified Python actually changed since
# the last time this hook spoke, in this session.
#
# The research analogue of "delete the failing test" is claiming a number that
# no run produced. Nothing here can execute the pipeline, so the reminder names
# what an honest report looks like when you cannot run anything.
#
# It must stay rare to stay read. `git status` alone is not a change signal:
# an untracked file (MOSE.py) is listed on every turn forever, which would fire
# this on every turn and train the reader to skip it.
set -u
payload=$(cat 2>/dev/null || echo '{}')
cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0

changed=$(git status --porcelain -- '*.py' 2>/dev/null | head -20)
[ -n "$changed" ] || exit 0

# Fingerprint the content, not just the status lines.
fp=$(
  printf '%s\n' "$changed"
  printf '%s\n' "$changed" | awk '{print $NF}' | while IFS= read -r f; do
    [ -f "$f" ] && sha1sum "$f" 2>/dev/null
  done
)
fp=$(printf '%s' "$fp" | sha1sum | cut -d' ' -f1)

sid=$(printf '%s' "$payload" | jq -r '.session_id // "nosession"' 2>/dev/null || echo nosession)
state="${TMPDIR:-/tmp}/claude-evidence-gate-${sid}"

if [ -f "$state" ] && [ "$(cat "$state" 2>/dev/null)" = "$fp" ]; then
  exit 0   # nothing moved since we last said this
fi
printf '%s' "$fp" > "$state" 2>/dev/null

jq -nc --arg c "$changed" '{
  hookSpecificOutput: {
    hookEventName: "Stop",
    additionalContext: ("Python changed in the working tree:\n" + $c +
      "\nBefore reporting: this checkout has no torch, no GPU and no data/, so no result here was measured. Say \"unrun\" for anything not executed on the server, and never quote a metric without the command and log path that produced it. Editing an eval, a metric, a trie or an edit/test split is a change to the experiment, not a bug fix: flag it explicitly.")
  }
}'
