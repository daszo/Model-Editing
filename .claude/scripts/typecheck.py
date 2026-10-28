#!/usr/bin/env python3
"""Flag Python bindings that carry no type annotation.

Project rule: every variable gets a type hint.

The rule is enforced as a RATCHET, not retroactively. Only lines that are new
or changed are checked; the pre-existing backlog is baselined. Turning the rule
on repo-wide today would report ~300 findings across utility and benchmark code
that is not part of the research contribution, and a gate that reports 300
things on its first run is a gate that gets switched off.

What each file is compared against:

  MOSE.py        the DOME.py baseline. It is a 2,828-line copy of it and the
                 DOME-to-MOSE diff is the contribution, so annotating inherited
                 lines would bury the method in noise.
  tracked files  their committed version at HEAD.
  new files      nothing; they are checked whole, because all of it is new.

Pass --all to check entire files instead, when you want to see the backlog.

Two severities, because not every binding can carry an inline annotation:

  ERROR  assignments, function parameters and return types. PEP 526 and PEP 484
         cover these; there is no excuse for a bare one.
  NOTE   for-targets, with-targets, except-targets and tuple unpacking. Python
         has no inline annotation syntax for these. To satisfy the rule
         strictly, pre-declare the name (`i: int` above the loop). Reported so
         the rule is not silently narrowed, not failed on.

Exit 1 if any ERROR is found in scope.
"""

from __future__ import annotations

import ast
import difflib
import subprocess
import sys
from pathlib import Path
from typing import Final, Iterator, NamedTuple

BASELINE_FILE: Final[str] = "DOME.py"
SCOPED_TO_BASELINE: Final[frozenset[str]] = frozenset({"MOSE.py"})


class Finding(NamedTuple):
    line: int
    severity: str
    message: str


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        done: subprocess.CompletedProcess[str] = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return done.stdout if done.returncode == 0 else None


def _changed(new: list[str], old: list[str]) -> set[int]:
    """1-indexed line numbers of `new` that differ from `old`."""
    out: set[int] = set()
    matcher: difflib.SequenceMatcher[str] = difflib.SequenceMatcher(
        None, old, new, autojunk=False
    )
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert"):
            out.update(range(j1 + 1, j2 + 1))
    return out


def scope_for(path: Path, source: str) -> set[int] | None:
    """Lines to check. None means the whole file."""
    lines: list[str] = source.splitlines()
    root: Path = path.parent if path.parent.exists() else Path(".")

    if path.name in SCOPED_TO_BASELINE:
        baseline: Path = path.parent / BASELINE_FILE
        if baseline.is_file():
            return _changed(lines, baseline.read_text(errors="replace").splitlines())
        return None

    top: str | None = _git(["rev-parse", "--show-toplevel"], root)
    if top is None:
        return None
    repo: Path = Path(top.strip())
    try:
        rel: str = str(path.resolve().relative_to(repo))
    except ValueError:
        return None
    committed: str | None = _git(["show", f"HEAD:{rel}"], repo)
    if committed is None:
        return None  # untracked or new: all of it is new
    return _changed(lines, committed.splitlines())


def _name_of(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"...{node.attr}"
    if isinstance(node, ast.Subscript):
        return "subscript target"
    return type(node).__name__


def _targets(node: ast.expr) -> Iterator[ast.expr]:
    if isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            yield from _targets(elt)
    else:
        yield node


def annotated_names(tree: ast.AST) -> set[str]:
    """Names annotated anywhere in the file.

    PEP 526 annotates a name once at its first binding; a later `x = f()` is a
    reassignment, not a fresh unannotated variable. Without this, every loop
    accumulator and every rebind would be reported, which is both wrong and
    noisy enough to get the check switched off.
    """
    names: set[str] = set()
    node: ast.AST
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a: ast.arguments = node.args
            every: list[ast.arg] = [*a.posonlyargs, *a.args, *a.kwonlyargs]
            if a.vararg:
                every.append(a.vararg)
            if a.kwarg:
                every.append(a.kwarg)
            arg: ast.arg
            for arg in every:
                if arg.annotation is not None:
                    names.add(arg.arg)
    return names


def walk(tree: ast.AST) -> Iterator[Finding]:
    known: set[str] = annotated_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                parts: list[ast.expr] = list(_targets(target))
                if len(parts) > 1:
                    yield Finding(
                        node.lineno,
                        "NOTE",
                        f"tuple unpacking ({', '.join(_name_of(p) for p in parts)}) "
                        "cannot be annotated inline; pre-declare each name",
                    )
                    continue
                only: ast.expr = parts[0]
                if isinstance(only, (ast.Attribute, ast.Subscript)):
                    continue  # self.x = ... is annotated at the class level
                if isinstance(only, ast.Name) and only.id in known:
                    continue  # already annotated at its first binding
                yield Finding(
                    node.lineno,
                    "ERROR",
                    f"`{_name_of(only)}` has no type annotation; "
                    f"write `{_name_of(only)}: T = ...`",
                )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args: ast.arguments = node.args
            every: list[ast.arg] = [*args.posonlyargs, *args.args, *args.kwonlyargs]
            if args.vararg:
                every.append(args.vararg)
            if args.kwarg:
                every.append(args.kwarg)
            for arg in every:
                if arg.annotation is None and arg.arg not in ("self", "cls"):
                    yield Finding(
                        arg.lineno,
                        "ERROR",
                        f"parameter `{arg.arg}` of `{node.name}` has no type annotation",
                    )
            if node.returns is None:
                yield Finding(
                    node.lineno,
                    "ERROR",
                    f"`{node.name}` has no return type annotation",
                )

        elif isinstance(node, ast.For):
            for part in _targets(node.target):
                yield Finding(
                    node.lineno,
                    "NOTE",
                    f"loop variable `{_name_of(part)}` cannot be annotated inline; "
                    "pre-declare it above the loop",
                )

        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars is not None:
                    for part in _targets(item.optional_vars):
                        yield Finding(
                            node.lineno,
                            "NOTE",
                            f"context variable `{_name_of(part)}` cannot be annotated inline",
                        )

        elif isinstance(node, ast.ExceptHandler) and node.name:
            yield Finding(
                node.lineno,
                "NOTE",
                f"exception name `{node.name}` cannot be annotated inline",
            )


def check(path: Path, check_all: bool) -> int:
    try:
        source: str = path.read_text(errors="replace")
        tree: ast.AST = ast.parse(source, filename=str(path))
    except SyntaxError:
        return 0  # python-check.sh owns syntax; do not double-report

    scope: set[int] | None = None if check_all else scope_for(path, source)
    findings: list[Finding] = sorted(
        f for f in walk(tree) if scope is None or f.line in scope
    )
    errors: list[Finding] = [f for f in findings if f.severity == "ERROR"]
    notes: list[Finding] = [f for f in findings if f.severity == "NOTE"]

    for f in errors:
        print(f"{path}:{f.line}: ERROR {f.message}")
    for f in notes:
        print(f"{path}:{f.line}: note  {f.message}")
    if errors:
        where: str = "" if scope is None else " (new or changed lines only)"
        print(f"{len(errors)} unannotated binding(s) in {path}{where}")
    return 1 if errors else 0


def main(argv: list[str]) -> int:
    check_all: bool = "--all" in argv
    paths: list[Path] = [Path(a) for a in argv[1:] if a != "--all"]
    if not paths:
        print("usage: typecheck.py [--all] FILE.py [FILE.py ...]", file=sys.stderr)
        return 2
    worst: int = 0
    p: Path
    for p in paths:
        if p.suffix == ".py" and p.is_file():
            worst = max(worst, check(p, check_all))
    return worst


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
