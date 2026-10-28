#!/usr/bin/env bash
# Install this Claude Code harness into another checkout, on this machine or
# another one.
#
#   ./.claude/install.sh /path/to/repo     install into that repo
#   ./.claude/install.sh                   install into the current repo
#   ./.claude/install.sh --pack            build a tarball to carry elsewhere
#   ./.claude/install.sh --check /path     verify an install without changing it
#
# Idempotent. An existing .claude/settings.json is MERGED, never overwritten:
# its permission arrays are unioned and its hooks are kept alongside these.
# An existing CLAUDE.md is never touched; the new one is written beside it as
# CLAUDE.md.incoming for you to reconcile by hand, because that file is the one
# piece of the harness that is genuinely project-specific.
#
# Everything the harness needs is git, jq, python3 and bash. No network.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="install"
TARGET=""

while [ $# -gt 0 ]; do
  case "$1" in
    --pack)  MODE="pack" ;;
    --check) MODE="check" ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
    *)  TARGET="$1" ;;
  esac
  shift
done

say()  { printf '  %s\n' "$1"; }
fail() { printf 'install: %s\n' "$1" >&2; exit 1; }

# --- dependencies ------------------------------------------------------------
echo "Dependencies"
missing=""
for dep in git jq python3 bash; do
  if command -v "$dep" >/dev/null 2>&1; then
    say "ok, $dep $("$dep" --version 2>/dev/null | head -1 | cut -c1-40)"
  else
    missing="$missing $dep"
    say "MISSING $dep"
  fi
done
[ -z "$missing" ] || fail "install these first:$missing"

pyver=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$pyver" in
  3.[0-8]) fail "python3 $pyver is too old; typecheck.py needs 3.9+ for builtin generics" ;;
  *) say "ok, python3 $pyver supports the annotations typecheck.py emits" ;;
esac

# --- pack --------------------------------------------------------------------
if [ "$MODE" = "pack" ]; then
  out="${TARGET:-$PWD/claude-harness.tar.gz}"
  tar -C "$SRC" -czf "$out" \
    --exclude='.claude/.env' \
    --exclude='.claude/worktrees' \
    .claude CLAUDE.md
  echo
  echo "Packed $out"
  echo "On the other machine:"
  echo "  tar -xzf $(basename "$out") -C /path/to/repo"
  echo "  /path/to/repo/.claude/install.sh /path/to/repo"
  exit 0
fi

# --- resolve target ----------------------------------------------------------
[ -n "$TARGET" ] || TARGET="$PWD"
TARGET="$(cd "$TARGET" 2>/dev/null && pwd)" || fail "no such directory: $TARGET"
echo
echo "Target: $TARGET"
git -C "$TARGET" rev-parse --git-dir >/dev/null 2>&1 \
  && say "ok, git repository" \
  || say "note, not a git repository; the baseline and ratchet checks will skip"

# --- check mode --------------------------------------------------------------
if [ "$MODE" = "check" ]; then
  echo
  echo "Installed pieces"
  rc=0
  for f in hooks/protect-baseline.sh hooks/protect-artifacts.sh hooks/python-check.sh \
           hooks/explanation-gate.sh hooks/evidence-gate.sh \
           scripts/check.sh scripts/typecheck.py scripts/solo-judge.sh \
           settings.json README.md; do
    if [ -e "$TARGET/.claude/$f" ]; then say "ok, $f"; else say "MISSING $f"; rc=1; fi
  done
  [ -f "$TARGET/CLAUDE.md" ] && say "ok, CLAUDE.md" || { say "MISSING CLAUDE.md"; rc=1; }
  jq -e . "$TARGET/.claude/settings.json" >/dev/null 2>&1 \
    && say "ok, settings.json is valid JSON" || { say "BROKEN settings.json"; rc=1; }
  exit $rc
fi

# --- copy --------------------------------------------------------------------
echo
echo "Installing"
mkdir -p "$TARGET/.claude"
for d in hooks agents skills scripts; do
  [ -d "$SRC/.claude/$d" ] || continue
  mkdir -p "$TARGET/.claude/$d"
  cp -R "$SRC/.claude/$d/." "$TARGET/.claude/$d/"
  say "ok, $d"
done
cp "$SRC/.claude/README.md" "$TARGET/.claude/README.md" 2>/dev/null && say "ok, README.md" || true
chmod +x "$TARGET"/.claude/hooks/*.sh "$TARGET"/.claude/scripts/*.sh "$TARGET"/.claude/scripts/*.py 2>/dev/null || true

# CLAUDE.md is project-specific: never clobber one that already exists.
if [ -f "$TARGET/CLAUDE.md" ] && ! cmp -s "$SRC/CLAUDE.md" "$TARGET/CLAUDE.md"; then
  cp "$SRC/CLAUDE.md" "$TARGET/CLAUDE.md.incoming"
  say "kept your CLAUDE.md; new one written as CLAUDE.md.incoming, reconcile by hand"
else
  cp "$SRC/CLAUDE.md" "$TARGET/CLAUDE.md"
  say "ok, CLAUDE.md"
fi

# settings.json: merge, do not replace.
NEW="$SRC/.claude/settings.json"
CUR="$TARGET/.claude/settings.json"
if [ -f "$CUR" ] && ! cmp -s "$NEW" "$CUR"; then
  jq -e . "$CUR" >/dev/null 2>&1 || fail "existing $CUR is not valid JSON; fix or move it first"
  cp "$CUR" "$CUR.bak.$(date +%Y%m%d%H%M%S)"
  jq -s '
    .[0] as $cur | .[1] as $new
    | $cur * $new
    | .permissions.allow = (($cur.permissions.allow // []) + ($new.permissions.allow // []) | unique)
    | .permissions.ask   = (($cur.permissions.ask   // []) + ($new.permissions.ask   // []) | unique)
    | .permissions.deny  = (($cur.permissions.deny  // []) + ($new.permissions.deny  // []) | unique)
    | .hooks = (
        (($cur.hooks // {}) | keys) + (($new.hooks // {}) | keys) | unique
        | map({ (.): ((($cur.hooks[.]) // []) + (($new.hooks[.]) // []) | unique) })
        | add
      )
  ' "$CUR" "$NEW" > "$CUR.merged" || fail "merge failed; your settings.json is untouched"
  mv "$CUR.merged" "$CUR"
  say "ok, settings.json merged (backup alongside it)"
else
  cp "$NEW" "$CUR"
  say "ok, settings.json"
fi

# Keep the OpenAI key out of git.
if [ -f "$TARGET/.gitignore" ] && ! grep -qx '.claude/.env' "$TARGET/.gitignore" 2>/dev/null; then
  printf '\n# Explanation Gate judge credentials, never commit\n.claude/.env\n' >> "$TARGET/.gitignore"
  say "ok, added .claude/.env to .gitignore"
fi
if [ ! -f "$TARGET/.claude/.env" ]; then
  cat > "$TARGET/.claude/.env.example" <<'ENV'
# Copy to .claude/.env and fill in. Never commit the real file.
# Used by .claude/scripts/solo-judge.sh, the Explanation Gate judge.
export OPENAI_API_KEY=sk-...
# Defaults to gpt-4o, the model the paper used. Run
#   .claude/scripts/solo-judge.sh --models
# to see what your key can reach.
export OPENAI_JUDGE_MODEL=gpt-4o
ENV
  say "ok, wrote .claude/.env.example"
fi

# --- self-test ---------------------------------------------------------------
echo
echo "Self-test"
t() {
  got=$(jq -nc --arg c "$2" '{tool_name:"Bash",tool_input:{command:$c}}' \
        | "$TARGET/.claude/hooks/protect-artifacts.sh" 2>/dev/null \
        | jq -r '.hookSpecificOutput.permissionDecision // empty')
  [ -z "$got" ] && got=allow
  if [ "$got" = "$1" ]; then say "ok, $2 -> $got"; else say "FAIL $2 -> $got (wanted $1)"; return 1; fi
}
rc=0
t deny  'rm -rf data/nq320k'          || rc=1
t allow 'ls data/'                    || rc=1
t allow 'diff DOME.py MOSE.py'        || rc=1

got=$(jq -nc '{tool_name:"Edit",tool_input:{file_path:"'"$TARGET"'/DOME.py"}}' \
      | "$TARGET/.claude/hooks/protect-baseline.sh" 2>/dev/null \
      | jq -r '.hookSpecificOutput.permissionDecision // "allow"')
[ "$got" = "deny" ] && say "ok, DOME.py edit -> deny" || { say "FAIL DOME.py edit -> $got"; rc=1; }

tmp="$(mktemp -d)"; printf 'x = 1\n' > "$tmp/probe.py"
out=$("$TARGET/.claude/scripts/typecheck.py" "$tmp/probe.py" 2>&1 || true)
case "$out" in *ERROR*) say "ok, typecheck flags an unannotated binding" ;;
               *) say "FAIL typecheck did not flag $tmp/probe.py"; rc=1 ;; esac
rm -rf "$tmp"

if [ -n "${OPENAI_API_KEY:-}" ]; then
  if "$TARGET/.claude/scripts/solo-judge.sh" --selftest >/dev/null 2>&1; then
    say "ok, OpenAI judge reachable and returns a valid verdict"
  else
    say "note, OPENAI_API_KEY is set but the judge call failed; run solo-judge.sh --selftest"
  fi
else
  say "note, no OPENAI_API_KEY; the gate will fall back to the Claude subagent judge"
fi

echo
if [ $rc -eq 0 ]; then
  echo "Installed. Open Claude Code in $TARGET and run /hooks once so it picks up settings.json."
else
  echo "Installed, but the self-test found problems above."
fi
exit $rc
