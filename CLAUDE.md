# Model-Editing (DOME to MOSE)

Research repo. DOME edits T5 decoder FFN weights to add new documents to a
generative-retrieval index. The project extends it with MOSE so the edit
survives a stream of rounds instead of one shot. Read `notes/research-directions.md`
for the claim under test and `notes/diagnostic-plan.md` for the build plan
before proposing MOSE work.

## This checkout cannot run anything

No torch, no transformers, no GPU, no `data/`, no `GR/`. Those live on the home
lab server (`~/projects/DOME`, its own `.venv`, one 3090). Here you can read,
edit and parse Python. You cannot import it, train, edit a model or evaluate.

IMPORTANT: never report a metric, a retrieval score or "it works" from this
machine. Nothing here has been executed. Say "unrun" and name the command the
user would run on the server.

## Verify with

```bash
.claude/scripts/check.sh      # syntax, baseline intact, line endings
```

That is the whole local loop. It proves the code parses, not that it is correct.

## File roles

| Path | Role |
| ---------------------------- | -------------------------------------------- |
| `DOME.py` | Frozen upstream baseline. Never edit. Hook-blocked. |
| `MOSE.py` | The working copy. All MOSE changes go here. |
| `notes/paper.md` | Transcript of the source papers. Read-only. |
| `notes/repro-status.md` | Runbook and current state. Update when state changes. |
| `notes/research-directions.md` | Lit check and the argument. |
| `notes/diagnostic-plan.md` | Spec for the streaming diagnostic. |

`MOSE.py` is currently a byte-identical copy of `DOME.py`. Both are CRLF
upstream; keep them CRLF or the diff shows all 2,828 lines as changed.

The DOME-to-MOSE diff is the contribution. Keep it readable: no reformatting,
no renaming, no drive-by cleanups in `MOSE.py`. If a change is not MOSE, it
does not belong in that diff.

## Python style

Every variable carries a type hint, along with every function parameter and
return type. Where Python has no inline annotation syntax (loop targets,
`with` targets, tuple unpacking), pre-declare the name above the statement.

Enforced as a ratchet by `.claude/scripts/typecheck.py`: only new and changed
lines are checked, so the pre-existing backlog stays baselined. For `MOSE.py`
that means lines differing from the `DOME.py` baseline. Never retrofit
annotations onto inherited DOME lines; it would bury the method in a
2,828-line diff. Run `.claude/scripts/typecheck.py --all <file>` to see the
backlog deliberately.

## Experiment integrity

- Changing an eval, a metric, a docID trie or an edit/test query split changes
  the experiment. Never do it to make a number look better, and flag it
  explicitly when it is the actual task.
- Test queries are never fed into an edit. That leak invalidates the result and
  is invisible in the output.
- Every reported number needs the command, the git SHA and the log path that
  produced it. No provenance, no number.
- Record runs with `/experiment-log` so results stay attributable.

## Long runs

Full training is roughly 4.5 days on the 3090 and the pipeline needs about
85 GB free. Never start a training or editing run without asking first.
`GR/training.py` has no resume: a crash restarts from epoch 0.

## Known gotcha: the edit entry point does not match

`main()` calls `editor.apply_docid_edit(mt=..., edit_requests=..., copy=...,
return_orig_weights=..., cov_data_file=..., accumulated_kk=...,
use_multiprocessing=...)` and unpacks two values (`MOSE.py:2788`). No such
method exists. `T5DocIDDOME` defines `apply_dome_edit(self, mt, edit_targets,
keep_original_weights=True)` returning one object (`MOSE.py:1892`). Name,
signature and arity all differ, so stage 3 raises `AttributeError` as shipped.
`notes/diagnostic-plan.md` repeats the wrong name when it says round chaining
is just repeated calls. Reconcile this before building streaming on top of it.

## Uncommitted upstream fixes

`notes/repro-status.md` lists four fixes to `GR/*` and `DOME.py` made on
2026-08-31 that are still uncommitted and live only on the server. Do not
assume this checkout reflects them.
