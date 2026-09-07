# Build plan — the streaming DOME diagnostic

Goal: turn DOME's single-shot edit into a **stream of R rounds** and measure
whether previously-added documents get forgotten as the stream goes on. I build
this myself; this is the spec, not the code. Context: [[research-directions]].
The claim under test: *DOME degrades on earlier rounds because its preservation
constraint `K0K0^T` is fixed to the initial corpus and edits pile into layers
14–18.* Confirm that BEFORE designing any fix.

Grounding facts (verified in this repo):
- New-doc universe = `corpus2.json` = 10,000 docs, docids 99,650–109,649.
  Their RQ docIDs are in `corpus_docids_rq_2.json`; real queries in `train2.json`
  (8,730 pairs, not every doc has one); pseudo-queries in `train2.json.qg.json`
  (10/doc).
- `get_knowns.py` builds edit-request files under `Edit/data/` as lists of
  `{known_id, subject, prompt, attribute}` where `attribute` = the docID string
  `"$a$$b$$c$$d$"`. It **dedupes to one query/doc and random-samples 1000 new
  docs** — so DOME's single-shot "new set" is ~1000 docs. The stream slices this.
- `DOME.py` already exposes the edit core: `editor = T5DocIDDOME(hparams)` then
  `edited_model, orig_weights = editor.apply_docid_edit(model=..., edit_requests
  =..., cov_data_file=..., ...)`. **Chaining rounds = calling this R times, feeding
  the previous `edited_model` back in as `model`.** No need to rewrite DOME.
- Eval/constrained decoding pattern lives in `GR/training.py::eval_model`
  (builds a `MarisaTrie` of docIDs, `generate(..., prefix_allowed_tokens_fn=...)`,
  R@k/MRR). Reuse it, but the trie must **grow** each round.

---

## A. Data prep — split new docs into rounds  (new script `GR/split_rounds.py`)

1. Load `corpus2.json`, `train2.json`, `train2.json.qg.json`,
   `corpus_docids_rq_2.json`. Work in **global docid space** (99,650–109,649).
2. Pick the new-doc set to stream. For a long stream use more of corpus2 than
   DOME's 1000 sample — e.g. all docs that have ≥1 real query (so they're
   evaluable). Deterministically shuffle with a fixed seed (record it).
3. Chunk into **R disjoint rounds** (start R=10). Each round r gets a disjoint set
   of docids `S_r`. Slices must be disjoint and cover no docid twice.
4. For each doc in `S_r`, split its **real queries** into `edit` (~80%) and
   `test` (~20%). Test queries are **never** fed to the edit. Docs with only one
   real query → put it in `test` and rely on pseudo-queries for editing (note
   which docs are edit-only-via-pseudo).
5. Emit, per round, in the `Edit/data/`-style dict format:
   - `data/nq320k/rounds/round_{r}/query2docid_edit.json`  (edit real queries)
   - `data/nq320k/rounds/round_{r}/query2docid_pseudo.json` (pseudo for `S_r`)
   - `data/nq320k/rounds/round_{r}/mem2docid.json`          (doc text for `S_r`)
   - `data/nq320k/rounds/round_{r}/query2docid_test.json`   (held-out eval)
   - `data/nq320k/rounds/round_{r}/docids.json`             (`S_r` → docID list)
   Reuse the exact attribute construction from `get_knowns.py`
   (`'$' + '$$'.join(corpusids[docid][1]) + '$'`).
6. Also emit the **initial-corpus eval set**: held-out initial queries. Use
   `dev1.json` (the dev set `get_knowns.py` already maps) filtered to docids in
   corpus1, saved as `rounds/initial_test.json`. This is slice s=0.

Keep the preservation/covariance file (`cov_data_file`) = the **initial** corpus
data (`Edit/data/query2docid_full_rq.json`, the DOME default) and **do not change
it between rounds** — a stale `K0` is exactly the thing being tested.

---

## B. Trie must grow  (extract `GR/util/trie.py::build_trie(docid_lists, tokenizer)`)

Factor the trie-build out of `training.py`. Maintain an **active docID set**:
- start = all corpus1 docIDs (initial retrievable set),
- after applying round r, add `S_r`'s docIDs.
Rebuild the trie from the active set after every round. If you don't, constrained
decoding can never emit a new doc's docID → you'd measure fake "forgetting". This
is the single easiest bug to introduce.

---

## C. Evaluator  (new `eval_gr.py::evaluate(model, trie, test_set, tokenizer)`)

Reuse `eval_model`'s decode loop. Input: a model, the current trie, and a
test-query set with gold docIDs. For each query, constrained-beam-decode top-k,
strip `$`, compare to gold docID string. Return R@1 / R@10 / R@100 / MRR@100.
Keep test sets small per slice (≈300–500 queries) — constrained beam search is
the slow part, and you call this O(R^2) times (every slice after every round).

---

## D. Orchestrator  (new `run_stream_diagnostic.py`)

```
mt = ModelAndTokenizer(gr_checkpoint=34.pt)          # base GR model
editor = T5DocIDDOME(hparams)                          # DOME defaults, layers 14-18
active = docids(corpus1); trie = build_trie(active)
A = {}                                                 # A[r][s] = metric matrix

# row r=0: before any edit
A[0][0] = evaluate(mt.model, trie, initial_test)

for r in 1..R:
    reqs   = load(round_r/query2docid_edit.json)       # real edit queries
    pseudo = load(round_r/query2docid_pseudo.json)
    docs   = load(round_r/mem2docid.json)
    t0 = time()
    mt.model, _ = editor.apply_docid_edit(              # <-- chain: feed model back
        model=mt.model, edit_requests=reqs, pseudo=pseudo, doc_edit_requests=docs,
        cov_data_file=INITIAL_COV_FILE)                 # preservation FIXED to corpus1
    tpd_r = (time()-t0)/len(S_r)
    active += docids(S_r); trie = build_trie(active)    # grow trie
    A[r][0] = evaluate(mt.model, trie, initial_test)    # initial retention
    for s in 1..r:                                      # every earlier + current round
        A[r][s] = evaluate(mt.model, trie, round_s/query2docid_test.json)
    dump(A, tpd_r)
```

Notes:
- `apply_docid_edit`'s exact kwarg names: check `DOME.py::main()` (it passes
  `edit_requests`, `doc_edit_requests`, `cov_data_file`, `num_cases`,
  `pseudo_num`). Match them; don't guess.
- Covariance is cached by `CovarianceManager` on first call — one-time cost.
- Save `A` and TPD after every round so a crash doesn't lose the run.

---

## E. Metrics to add  (compute from the `A[r][s]` matrix; use R@10 as the base)

Let slice 0 = initial corpus, slices 1..R = the rounds. `A[r][s]` = R@10 on
slice s's test set measured **after** round r (defined for s ≤ r).

- **Per-slice retention curves** — for each earlier slice s, plot `A[r][s]` for
  r = s..R. Downward slope = forgetting of docs added in round s. **This is the
  headline figure.**
- **Backward Transfer (BWT)** — mean drop of earlier slices by the end:
  `BWT = mean_{s=1..R-1} ( A[R][s] − A[s][s] )`.  Negative ⇒ forgetting.
  (`A[s][s]` = a slice's score right after it was added; `A[R][s]` = at the end.)
- **Forgetting on initial corpus (Fn)** — DOME's metric:
  `Fn = A[0][0] − A[R][0]` (R@10). Track `A[r][0]` across r too.
- **Forward / plasticity** — `mean_{r=1..R} A[r][r]` (how well the newest round is
  learned each time). Watch for decline as layers 14–18 saturate.
- **Final Average Accuracy** — `ACC = mean_{s=0..R} A[R][s]`.
- **Efficiency** — TPD per round (should stay ~flat ⇒ editing stays cheap over the
  stream; the practical selling point vs retraining/expansion methods).

Report R@1 / MRR@100 the same way if you want them; R@10 is the anchor so numbers
compare to DOME Table 2–3 and PAMT's backward/forward transfer.

---

## F. Baselines / controls (same slices, same seed)

- **DOME single-shot** on the *union* of all R slices at once → the "editing
  upper bound"; the streamed run's gap below this = the degradation you're
  showing.
- **Round-by-round MEMIT and AlphaEdit** (DOME already wraps them single-shot) →
  shows whether the streaming failure is DOME-specific or general to editing.
- **MixLoRA-DSI / PAMT** on the same slices → the continual-GR reference (their
  home turf). Optional for the diagnostic; needed for the eventual paper.
- **From-Scratch retrain** on corpus1+corpus2 → true upper bound (expensive; cite
  DOME's number instead of re-running unless cooling/time allows).

---

## G. Sanity checks (do these before trusting any curve)

1. **Round-1 identity:** streamed round 1 == DOME single-shot on slice 1
   (same edit, same eval) → confirms the loop wiring is faithful.
2. **Trie contains new docIDs:** after round r, assert every docid in `S_1..S_r`
   is decodable (pick a few, force-decode) — else "forgetting" is a trie bug.
3. **No docID collisions** across the active set (corpus1 ∪ S_1..S_r): reuse the
   uniqueness check / overflow warning from the fixed `get_docid.py`
   ([[dome-pipeline-fixes]]). Two docs sharing a docID makes the metric lie.
4. **Test isolation:** assert `query2docid_test` ∩ `query2docid_edit` = ∅ per round.
5. **Determinism:** fixed seed for the shuffle + query split; log it.

---

## H. Sizing on the RTX 3090 (see [[dome-3090-benchmarks]])

- Base GR model: train **once**. Use a **t5-base pilot** (~2.5 d, or reuse a
  checkpoint) to get the whole loop working before spending on t5-large.
- Editing: DOME TPD ≈ 2 s/doc → 1000 docs/round ≈ ~30–40 min/round; covariance is
  cached once (~1–3 h first time). 10 rounds ≈ a few hours of editing.
- Eval dominates wall-clock (beam search, O(R^2) slice-evals). Keep per-slice test
  sets ≈300–500 queries; that's the main knob. Whole diagnostic ≈ 1 day on t5-base.

## I. Keep FIXED so it's a fair diagnostic
Same hyperparams as DOME single-shot; edit layers 14–18; preservation
`cov_data_file` = initial corpus, never refreshed; same docID scheme (RQ). The
**only** variable is single-shot vs streamed. (Refreshing the preservation set,
null-space projection, side-memory = the *fixes* you try later, one at a time,
each re-run through this same harness so the curves are directly comparable.)
```
