---
name: explanation-gate
description: Teach-back gate before a change to the method is accepted. Poses one causal question about the change, has an independent judge score the answer on the SOLO taxonomy, and loops with guiding questions until it passes.
---

Run the Explanation Gate for the change in $ARGUMENTS, or for the most recent
substantive change to `MOSE.py` if nothing is named.

After Sankaranarayanan, L@S '26 (arXiv:2602.20206). In that study two groups
got the same model and produced statistically indistinguishable working code.
Then the code broke. The ungated group repaired it 23% of the time, the gated
group 62%. The gate is not about code quality; it is about whether the human
holds the causal model when the code stops behaving.

The stake here is a reviewer asking why a term is in the update, or a number
coming out wrong at round 7 with no intuition for which of five layers to look
at first.

## Protocol

**1. Pick the one question.** Look at the change and find the single causal
link that the change turns on, the one whose absence would leave the user
unable to debug or defend it. Ask about that. Good questions here are of the
form: what does this term protect and at whose expense, what does the baseline
do instead at round 3, which matrix does this accumulate into and why that one,
what would you expect to see if this is wrong.

Bad questions ask what the code does, ask for a definition, or can be answered
by reading the diff aloud.

**2. Put it to the user with AskUserQuestion.** One question. Offer answer
options only when they are genuinely competing explanations, never when one
option is the answer and the others are filler; the user can always write their
own. Do not answer it yourself, in the question text, in the options, or in the
preamble.

**3. Judge it.** Never score it yourself. You wrote the code, you know the
answer, and you will read understanding into a vague answer that is not there.
That is the self-preference bias the paper's cross-vendor design exists to
exclude.

Preferred, matching the paper: write the change and the user's answer to two
files under the scratchpad and run

```bash
.claude/scripts/solo-judge.sh <code-file> <explanation-file>
```

which asks OpenAI at temperature 0.1 and returns `{"score": 1-5, "feedback":
"..."}`. If it reports no `OPENAI_API_KEY`, say so once and fall back to the
`solo-judge` subagent, noting in your report that the judge was same-vendor on
this run.

**4. Loop or pass.** Below 3, put the judge's guiding questions to the user and
judge again. The paper's median is 2 attempts; if you reach 4 without
convergence, stop and say plainly which link is not landing, then let the user
decide whether to keep going. Above or at 3, say so, relay the one thing at the
next level up that they did not cover, and continue with the work.

**5. Record it.** Note in your report that the gate ran and what it scored.
If the user declined the gate, record that instead. Either is fine; a silent
skip is not.

## Rules

- The gate interrogates the user, never the other way round. Do not turn it
  into a quiz you then answer.
- Never withhold the code or refuse to continue working. The paper's gate
  blocked a merge; you are blocking a claim of completion, not the user's
  access to their own repository.
- If the user says skip, drop it immediately and without comment. Friction that
  is argued about stops being friction and becomes nagging. 72% of the paper's
  subjects called the gate annoying even as it halved their failure rate.
- Do not run the gate on your own harness edits, on notes, or on anything
  outside the method. Scope is `MOSE.py`.
