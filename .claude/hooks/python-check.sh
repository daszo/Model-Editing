#!/usr/bin/env bash
# PostToolUse (Write|Edit): the only checks that actually run on this machine.
#
# torch and transformers are not installed here and there is no GPU, so nothing
# in this repo can be executed or imported locally. Parsing the file and
# inspecting its AST is the one real pass/fail signal available; without it a
# syntax error rides to the server and dies four minutes into a job.
set -u
payload=$(cat)
f=$(printf '%s' "$payload" | jq -r '.tool_response.filePath // .tool_input.file_path // empty')
case "$f" in *.py) ;; *) exit 0 ;; esac
[ -f "$f" ] || exit 0

root="${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"
msgs=""
add() { msgs="${msgs:+$msgs

}$1"; }

# 1. Syntax. compile() rather than py_compile so no __pycache__ is written.
out=$(python3 - "$f" <<'PYCHK' 2>&1
import sys, traceback
p = sys.argv[1]
try:
    compile(open(p, "rb").read(), p, "exec")
except SyntaxError as e:
    print("".join(traceback.format_exception_only(e)).rstrip())
    sys.exit(1)
PYCHK
) || add "SyntaxError in $f, this file will not run:
$out"

# 2. Type annotations. Project rule: every variable carries a type hint.
#    Scoped to lines that differ from the DOME.py baseline for MOSE.py, so the
#    2,828 inherited lines are not reported as 412 findings.
if [ -x "$root/.claude/scripts/typecheck.py" ]; then
  tout=$("$root/.claude/scripts/typecheck.py" "$f" 2>&1) || add "Missing type annotations (project rule: every variable is annotated):
$tout"
fi

# 3. Line endings. DOME.py and MOSE.py are CRLF upstream. An editor that
#    silently converts them turns a three-line change into a 2,828-line diff.
case "$(basename "$f")" in
  MOSE.py|DOME.py)
    head -c 20000 "$f" | grep -qU $'\r' || add "$(basename "$f") lost its CRLF line endings. It is CRLF upstream; converting it makes the whole file show as changed. Restore CRLF before continuing."
    ;;
esac

[ -n "$msgs" ] || exit 0
jq -nc --arg m "$msgs" '{
  hookSpecificOutput: { hookEventName: "PostToolUse", additionalContext: $m },
  systemMessage: $m
}'
