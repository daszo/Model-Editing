#!/usr/bin/env bash
# The whole local verification loop. Run it before reporting any code change.
#
# This checkout has no torch, no GPU and no data/, so nothing here proves the
# pipeline works. It proves the code parses, the baseline is still pristine and
# the diff is readable. That is the entire honest local signal.
set -u
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 1
fail=0
note() { printf '  %s\n' "$1"; }

echo "1. Python syntax"
while IFS= read -r f; do
  [ -f "$f" ] || continue
  if ! err=$(python3 - "$f" <<'PYCHK' 2>&1
import sys, traceback
p = sys.argv[1]
try:
    compile(open(p, "rb").read(), p, "exec")
except SyntaxError as e:
    print("".join(traceback.format_exception_only(e)).rstrip())
    sys.exit(1)
PYCHK
  ); then
    note "FAIL $f"; note "$err"; fail=1
  fi
done < <(git ls-files '*.py'; git ls-files --others --exclude-standard '*.py')
[ $fail -eq 0 ] && note "ok, all tracked .py parse"

echo "2. Baseline untouched"
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  if git diff --quiet HEAD -- DOME.py 2>/dev/null; then
    note "ok, DOME.py matches HEAD"
  else
    note "FAIL DOME.py differs from HEAD. It is the frozen upstream baseline that MOSE is measured against."
    fail=1
  fi
fi

echo "3. Line endings"
for f in DOME.py MOSE.py; do
  [ -f "$f" ] || continue
  if head -c 20000 "$f" | grep -qU $'\r'; then
    note "ok, $f still CRLF"
  else
    note "FAIL $f lost CRLF; the diff will show every line as changed."
    fail=1
  fi
done

echo "4. Type annotations"
if [ -x .claude/scripts/typecheck.py ]; then
  tfiles=$( { git ls-files '*.py'; git ls-files --others --exclude-standard '*.py'; } | grep -v '^DOME\.py$' )
  if [ -n "$tfiles" ]; then
    # shellcheck disable=SC2086
    if tout=$(.claude/scripts/typecheck.py $tfiles 2>&1); then
      note "ok, every annotatable binding is annotated"
    else
      printf '%s\n' "$tout" | grep ERROR | sed 's/^/  /'
      note "project rule: every variable carries a type hint"
      fail=1
    fi
  fi
else
  note "skipped, typecheck.py not executable"
fi

echo
if [ $fail -eq 0 ]; then
  echo "PASS (local checks only, nothing was executed)"
else
  echo "FAIL"
fi
exit $fail
