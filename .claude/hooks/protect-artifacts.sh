#!/usr/bin/env bash
# PreToolUse (Bash): block commands that destroy irreplaceable run artifacts.
#
# data/ is ~85 GB of prepared NQ320k; the GR checkpoints are ~4.5 days of 3090
# time. None of it is in git (see .gitignore) and none of it is on this laptop.
# A stray removal or redirect costs a week, so it is a hook, not a CLAUDE.md line.
#
# Two scoping rules learned the hard way, each from a false positive that
# blocked legitimate work in this repo:
#
#   1. The destructive verb must be in COMMAND POSITION (start, or after a
#      separator). Without it the pattern fires on a path merely quoted inside
#      an argument, which once blocked the edit to this very file.
#   2. For copies, DOME.py must be the DESTINATION (final token). `cp DOME.py
#      elsewhere` reads the baseline and is fine; `cp x DOME.py` overwrites it.
#
# This is a regex over raw command text and cannot tell quoted from unquoted.
# A command that embeds a separator and a destructive verb inside a string is
# still denied. That direction is fail-safe and is left as is.
set -u
payload=$(cat)
cmd=$(printf '%s' "$payload" | jq -r '.tool_input.command // empty')
[ -n "$cmd" ] || exit 0

deny() {
  jq -nc --arg r "$1" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: $r
    }
  }'
  exit 0
}

VERB='(^|[;&|]|\$\()[[:space:]]*(sudo[[:space:]]+)?'
TARGET='(\bdata/|\bGR/|\bEdit/|\bdsets/|\.pt\b|\bcorpus_vectors|\bcorpus_docids)'

# Destructive verbs aimed at the expensive, gitignored trees or at checkpoints.
if printf '%s' "$cmd" | grep -qE "${VERB}(rm|rmdir|shred|mv|truncate)\b[^|;&]*${TARGET}"; then
  deny "Blocked: this touches run artifacts that are not in git and cost days to rebuild (the prepared corpus, the GR trees, the checkpoints). Ask the user before removing or moving any of them."
fi

# Writes that land ON the frozen baseline: redirects, tee, in-place sed.
if printf '%s' "$cmd" | grep -qE "(>|>>|\btee\b|${VERB}sed\b[^|;&]*-i)[^|;&]*\bDOME\.py\b"; then
  deny "Blocked: DOME.py is the frozen upstream baseline and the MOSE diff is measured against it. Write to MOSE.py instead."
fi

# A copy whose DESTINATION is the baseline. Reading it out is fine.
if printf '%s' "$cmd" | grep -qE "${VERB}(cp|install)\b[^|;&]*[[:space:]]DOME\.py[[:space:]]*(\$|[;&|])"; then
  deny "Blocked: that would overwrite DOME.py, the frozen upstream baseline. Copying DOME.py somewhere else is fine; writing onto it is not."
fi
