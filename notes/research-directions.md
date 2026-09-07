# Research directions off the DOME paper — working notes

Scratch notes for a small, publishable project building on **DOME** ("Model
Editing for New Document Integration in Generative Information Retrieval",
WWW'26, arXiv:2603.02773). Not a proposal — just the reasoning and the lit-check
findings so I can turn it into a real proposal myself.

---

## 1. The seam I'm circling

DOME adds new documents to a generative-retrieval (GR) model by **editing the
existing decoder FFN weights** (ROME/MEMIT-style: locate critical layers 14–18,
optimize edit vectors with a hybrid soft/hard-label strategy, apply a constrained
closed-form update). It's fast, rehearsal-free, keeps the model the same size,
and barely forgets the initial corpus.

**But DOME is only ever evaluated as a *single* editing operation.** The setup is
90% initial / 10% new, edited in once (§6.1). RQ4 (Appendix A.2) sounds like the
streaming question but just makes that single batch bigger (100→1000 docs in
**one** shot). There is no round 1, then round 2 on top of it, then round 3.

Real "dynamic corpora" — DOME's own motivation — are **streams**: documents keep
arriving, so you'd apply DOME again and again. In-place model editing is *known*
to erode under repeated application (that's why AlphaEdit and WISE exist). And
DOME's preservation constraint uses `K0 K0^T` from the **initial** corpus only —
in round *t* it has no term protecting the docs added in rounds `1..t-1`, and
every round keeps writing into the same 5 layers. So the testable prediction is:
**earlier-added docs get clobbered as the stream goes on, and edit interference
accumulates in layers 14–18.**

---

## 2. Lit check — and a correction to myself

I first said "nobody measures retention of previously-added documents." **That's
wrong at the field level** — the continual-GR line already owns the streaming
benchmark and the retention metric. What nobody has done is put *in-place model
editing* on it. Corrected map:

**Continual GR (handle the stream, but by ADDING capacity or replay):**
- **Continual Learning for GR over Dynamic Corpora** (CIKM'23, arXiv:2308.14968) —
  the foundational dynamic-corpus GR benchmark: incremental indexing, forgetting.
  *This likely defines the streaming protocol to adopt.*
- **DSI++** (updating transformer memory w/ new docs) — replay + flatter-minima
  regularization to fight forgetting. Retraining-flavored.
- **MixLoRA-DSI** (EMNLP'25, arXiv:2507.09924, code: LouisDo2108/MixLoRA-DSI) —
  replaces selected T5-decoder FFN layers with a dynamically **expandable
  mixture of LoRA experts**; OOD-driven expert expansion, top-k router, RQ
  docIDs, NQ320k + MS MARCO. **Adds parameters.**
- **PromptDSI** — prompt-based rehearsal-free continual GR. **Adds prompts.**
- **PAMT — Parametric Memory Head for Continual GR** (2026, arXiv:2604.23388) —
  ⚠️ *closest competitor.* Freezes the backbone, **attaches an external
  product-key memory head** that emits residual hidden-space corrections;
  sparse memory-value updates chosen by decoding-time access stats. Evaluates on
  **5 sequential disjoint slices** of MS MARCO / NQ with **backward/forward
  transfer** metrics. Positions vs DSI++, CLEVER, MixLoRA-DSI — **not vs DOME or
  model editing.** So the streaming eval + retention metric already exist; PAMT
  just doesn't do it with in-place editing.
- **IncDSI** (ICML'23) — real-time insertion via constrained optimization, but
  **atomic docIDs only** + codebook expansion; doesn't fit structured RQ docIDs.

**Lifelong model editing (for LLM facts, not GR — the toolbox to borrow from):**
- **DOME** — the only in-place editor for GR, single-shot only.
- **AlphaEdit** (ICLR'25 Outstanding, arXiv:2410.02355, code: jianghoucheng/
  AlphaEdit) — projects each update onto the **null space** of preserved-knowledge
  keys so sequential edits don't disturb old knowledge. Extends MEMIT to lifelong.
  *Caveat:* a repro study (arXiv:2606.26783) shows the protection is **bounded** —
  it degrades once edit counts get large. So not a guaranteed win.
- **WISE** (NeurIPS'24, arXiv:2405.14768) — dual memory: frozen main + editable
  **side memory** + a trained **router**; "knowledge sharding" puts different edit
  batches in orthogonal subspaces merged without conflict. The "impossible
  triangle" (reliability / generalization / locality) in lifelong editing.
- **HoReN** (arXiv:2605.08143) — Hopfield-retrieval for large-scale sequential
  model editing. Another lifelong-editing mechanism.
- **GenRecEdit** — model editing for generative *recommendation* with cold-start
  items. Nearest transfer of editing to a retrieval-like setting; different domain.

**The sharpened gap:** in-place, capacity-preserving, rehearsal-free model editing
(DOME family) has never been made to work over a document *stream* for GR. The
methods that handle the stream all grow the model (LoRA/prompts/memory head) or
replay. That's the wedge — and the *distinct selling point* of an editing-based
method is that it keeps the architecture and size fixed and needs no router over
a growing expert set.

---

## 3. How you go from limitation → solution (the method, not magic)

The repeatable recipe I keep coming back to:

1. **Sharpen the limitation into a falsifiable prediction with a named
   mechanism.** Not "DOME may not scale" but "DOME forgets round-1 docs by round
   T because `K0` is stale and layers 14–18 saturate." A mechanism is what a fix
   grabs onto.
2. **Confirm it empirically BEFORE inventing anything.** Run the smallest
   experiment that would expose the failure. Everyone skips this; it's the most
   important. If DOME survives streams fine, there's no paper — better to learn
   that in week 1. This step *is* your first contribution (editing on the
   continual benchmark).
3. **Read the failure to locate the fix.** The diagnostic says *how* it breaks
   (which rounds sag, whether it correlates with edit-vector interference), and
   the shape points at the knob.
4. **Borrow the fix from an adjacent solved problem.** The core move in applied
   ML. Sequential LLM editing is solved-ish (AlphaEdit null-space, WISE
   side-memory, HoReN). Continual GR is solved-ish (LoRA experts, replay). The
   contribution = transport one mechanism into DOME's setting **and** adapt it to
   what makes GR special: the edit-vector *indistinguishability* from docID
   overlap that DOME itself identified (§4.2).
5. **Ship the smallest version that beats the diagnostic baseline**, compared
   against the continual-GR methods on *their* streaming protocol.

---

## 4. Recommended direction + fix menu

**"Lifelong / streaming document integration for GR via model editing."**
Diagnostic first, then a capacity-preserving fix. Fixes, cheap → ambitious, each
tied to a paper:

- **Growing preservation set** — refresh `(K0,V0)` after each round to include
  previously-added docs so the constraint protects them. Minimal, principled;
  maybe half the battle.
- **Null-space / orthogonal edits (AlphaEdit → GR)** — project each round's update
  into the null space of prior rounds' keys so edits stop interfering. DOME
  *cites* AlphaEdit but never adopts its lifelong mechanism. Watch the "bounded
  protection" caveat — may need a GR-specific twist.
- **Side memory + routing (WISE → GR)** — keep edits in a separate memory, route
  queries base-vs-edited. Most novel, most work. Tension: a *separate* memory is
  arguably "adding capacity," so to keep the capacity-preserving selling point,
  lean on the sharding-into-orthogonal-subspaces-of-the-existing-weights idea
  rather than a big side network.

**Why this fits my situation (1× RTX 3090):** train the base GR model **once**
(~7.5 d, or a t5-base / smaller-corpus pilot in hours), then every streaming
experiment is on the *editing* side — seconds-to-minutes per doc, so a 10-round
stream is hours, not days. Expensive part paid once; research part cheap. See
[[dome-3090-benchmarks]].

### Alternatives (both editing-side, also cheap)
- **Cold-start / content-scarce integration** (à la GenRecEdit): DOME needs the
  new doc's text + pseudo-queries. Edit a doc in from just a title, or no query
  signal. Narrower, clean.
- **Edit-friendly docID spaces**: DOME's whole problem is edit-vector
  indistinguishability from docID overlap (and there's a real duplicate-docID
  collision bug in the RQ scheme in this repo, see [[dome-pipeline-fixes]]). How
  does docID scheme (RQ / PQ / atomic / semantic) govern editability at scale?
  Interesting but needs a base retraining per scheme → less HW-friendly.

---

## 5. Where to begin — the diagnostic (first ~week)

**Full build spec: [[diagnostic-plan]]** (data splitting, files to add, functions
to reuse/extract, exact metric formulas, sanity checks, 3090 sizing).

Don't design the fix yet. Build the diagnostic:

1. Base GR model — start with a **t5-base pilot** to iterate in hours.
2. Split the "new" documents into ~5–10 sequential rounds (mirror the PAMT /
   Continual-GR-over-Dynamic-Corpora disjoint-slice protocol so it's comparable).
3. Apply DOME round by round.
4. After **every** round, measure three curves: retrieval on (a) the original
   corpus, (b) **each earlier round's docs** (backward transfer), (c) the newest
   round (forward transfer). Reuse PAMT's backward/forward-transfer definitions so
   numbers are directly comparable.
5. If (b) sags with rounds → gap confirmed, and its shape tells you which fix to
   reach for. If (b) holds → surprising, valid, redirect fast (saved months).

Also run AlphaEdit and MEMIT round-by-round as editing baselines (DOME already
compares to them single-shot), and MixLoRA-DSI / PAMT as the continual-GR
baselines on the same slices.

---

## 6. Open risks / to verify before committing
- **PAMT overlap is the main threat.** Read it (and MixLoRA-DSI) in full;
  differentiate crisply on "in-place editing of existing weights, no added
  memory/experts/router-over-growing-set." If a reviewer sees editing as just
  "another sparse-memory update like PAMT," the wedge is gone.
- **Confirm DOME actually degrades sequentially.** The whole project rests on the
  diagnostic in §5 showing sag. Do that experiment first.
- **Check for any 2026 preprint already doing sequential editing for GR** —
  searches so far show none, but re-check near submission.
- Keep `--epochs`/`--batch_size` fixed across resumes when training the base model
  (see [[dome-pipeline-fixes]]).

### Sources
DOME arXiv:2603.02773 · Continual GR over Dynamic Corpora arXiv:2308.14968 (CIKM'23) ·
MixLoRA-DSI arXiv:2507.09924 (EMNLP'25) · PAMT arXiv:2604.23388 · IncDSI (ICML'23) ·
DSI++ · PromptDSI · AlphaEdit arXiv:2410.02355 (ICLR'25) + repro arXiv:2606.26783 ·
WISE arXiv:2405.14768 (NeurIPS'24) · HoReN arXiv:2605.08143 · GenRecEdit.
