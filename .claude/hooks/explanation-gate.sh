#!/usr/bin/env bash
# PostToolUse (Write|Edit): the Explanation Gate.
#
# After Sankaranarayanan, "Mitigating 'Epistemic Debt' in GenAI-Scaffolded
# Novice Programming using Metacognitive Scripts" (L@S '26, arXiv:2602.20206).
# The paper's VibeCheck plugin intercepts any atomic AI insertion of >= 2 lines
# or >= 50 characters and blocks the merge until the human explains the code's
# causal logic well enough to score >= 3 on the SOLO taxonomy. Group B
# (unrestricted AI) failed a later maintenance task 77% of the time; Group C
# (gated) failed 39%, on identical code with identical Phase 1 utility.
#
# Here the risk is not a maintenance task, it is a reviewer. If Claude writes
# MOSE and the author cannot say why a term is in the update, the debt comes
# due at submission, not at compile time.
#
# Scope is deliberately narrow: the method file only. Gating every edit is how
# a gate gets ignored (72% of the paper's own subjects called it annoying).
# Set MOSE_GATE=off to disable for a session.
set -u
[ "${MOSE_GATE:-on}" = "off" ] && exit 0

payload=$(cat)
f=$(printf '%s' "$payload" | jq -r '.tool_response.filePath // .tool_input.file_path // empty')
[ -n "$f" ] || exit 0

case "$(basename "$f")" in
  MOSE.py) ;;                       # the method under construction
  *) exit 0 ;;
esac

# Size of the atomic insertion, per the paper's Rule Enforcement Engine.
read -r added_chars added_lines <<EOF
$(printf '%s' "$payload" | jq -r '
  (.tool_input.content // "") as $w
  | (.tool_input.new_string // "") as $n
  | (.tool_input.old_string // "") as $o
  | (if $w != "" then ($w|length) else (($n|length) - ($o|length)) end) as $c
  | (if $w != "" then ($w|split("\n")|length) else (($n|split("\n")|length) - ($o|split("\n")|length)) end) as $l
  | "\($c) \($l)"')
EOF
added_chars=${added_chars:-0}; added_lines=${added_lines:-0}
[ "$added_chars" -ge 50 ] || [ "$added_lines" -ge 2 ] || exit 0

jq -nc --arg f "$f" '{
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    additionalContext: ("EXPLANATION GATE armed for " + $f + ".\n" +
      "A substantive change just landed in the method file. Before you report this work as done, run the teach-back protocol in the explanation-gate skill: put the causal question to the user with AskUserQuestion, have the solo-judge subagent score the answer, and loop on sub-threshold answers with guiding questions only.\n" +
      "Do not answer the question for them, do not embed the answer in the options, and do not skip the gate because the change looks small or the user seems to know it already. If the user declines, record that they declined and move on without arguing.")
  }
}'
