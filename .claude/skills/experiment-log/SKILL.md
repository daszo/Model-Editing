---
name: experiment-log
description: Record a run in notes/experiments.md with its command, git SHA, config and log path, so the number stays attributable.
disable-model-invocation: true
---

Append one entry to `notes/experiments.md` for the run described in
$ARGUMENTS. Create the file with a `# Experiment log` heading if it does not
exist. Newest entry at the top, under the heading.

Gather each field yourself where you can, from the run's log, the config it
printed and `git rev-parse HEAD`. Ask the user only for what you genuinely
cannot recover. Do not invent a value to fill a row: write `unknown` and say
what would recover it.

If the run has not actually happened, do not write an entry. Say what needs to
run first.

```markdown
## <date> <short name>

| field | value |
| ------------- | -------------------------------------------------- |
| command | the exact command, as run |
| host | server or laptop |
| git SHA | `<sha>`, clean or dirty |
| config | the hyperparameters that differ from the defaults |
| rounds | R, and the docid slice per round |
| log | path to the log or stdout capture |
| duration | wall clock |

**Result.** The metrics, each with the eval set it came from.

**Read.** What this does or does not settle about the claim under test.

**Caveats.** Anything that makes this incomparable to earlier runs: a changed
seed, a changed metric, a changed split, a crash and restart, a different
checkpoint.
```

After writing it, show the user the entry and state plainly whether the result
supports, weakens or does not address the claim in
`notes/research-directions.md`. A run that fails to show the predicted effect
is a result. Record it the same way.
