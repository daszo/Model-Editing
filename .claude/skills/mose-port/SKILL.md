---
name: mose-port
description: Port one MOSE component into MOSE.py against the DOME baseline, keeping the diff reviewable and the change verifiable.
---

Port the component named in $ARGUMENTS. One component per pass. If the request
covers several, do the first and say what remains.

**1. Read the spec before the code.** Find the component in `notes/paper.md`.
Quote the equation or algorithm step you are implementing, and its section, in
your response. If the transcript does not cover it, fetch the paper and say so.
Do not start from an idea of what MOSE does.

**2. Find the seam in DOME.** The edit core is `T5DocIDDOME` in `MOSE.py`
(class at line 790), whose real entry point is `apply_dome_edit` (line 1892),
with the covariance machinery in `CovarianceManager` (line 111) and the worker
functions above it. Name the functions you will touch and what each currently
does, then say where the new behaviour attaches. Chaining rounds means calling
the edit entry again with the previously edited model; prefer that to rewriting
the edit core.

Note `main()` calls `apply_docid_edit` with a different signature, and no such
method exists. See the entry-point mismatch in CLAUDE.md before you build on
either. `notes/diagnostic-plan.md` repeats the wrong name; do not copy it.

**3. Check the plan against the diagnostic.** `notes/diagnostic-plan.md` sets
the streaming protocol this work is measured under. If your approach conflicts
with it, raise that before writing code.

**4. Implement in `MOSE.py` only.** `DOME.py` is frozen and hook-blocked. Match
the surrounding style, keep CRLF, and change nothing the component does not
need. The DOME-to-MOSE diff is the contribution, so every hunk must be
defensible as part of the method.

**5. Verify what can be verified here.** Run `.claude/scripts/check.sh`. Then
run `diff DOME.py MOSE.py` and read the whole diff yourself, confirming each
hunk belongs to this component.

**6. Report honestly.** Nothing in this checkout runs: no torch, no GPU, no
`data/`. State that the change is unrun, give the exact command the user would
run on the server to exercise it, and say what result would confirm or refute
the component working.

**7. Get a second opinion.** For anything touching the edit arithmetic, the
preservation constraint or the eval, use the `paper-faithfulness` subagent to
check the implementation against the paper in a fresh context, and act on what
it finds before calling the port done.
