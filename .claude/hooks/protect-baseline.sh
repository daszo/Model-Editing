#!/usr/bin/env bash
# PreToolUse (Write|Edit|NotebookEdit): keep reference files immutable.
#
# DOME.py is the unmodified upstream baseline. Every MOSE claim is measured as
# a diff against it, so an edit to DOME.py destroys the comparison silently.
# notes/paper.md is a transcript of the source papers, not our writing.
set -u
payload=$(cat)
f=$(printf '%s' "$payload" | jq -r '.tool_input.file_path // empty')
[ -n "$f" ] || exit 0

base=$(basename "$f")
case "$base" in
  DOME.py)
    reason="DOME.py is the frozen upstream baseline. MOSE work goes in MOSE.py; the DOME-vs-MOSE diff is the experiment. If the baseline genuinely needs a fix, the user must say so explicitly and it must land as its own commit."
    ;;
  paper.md)
    case "$f" in
      */notes/paper.md)
        reason="notes/paper.md is a transcript of the source papers, not project writing. Put your own analysis in notes/research-directions.md or a new note."
        ;;
      *) exit 0 ;;
    esac
    ;;
  *) exit 0 ;;
esac

jq -nc --arg r "$reason" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: $r
  }
}'
