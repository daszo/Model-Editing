# DOME reproduction — status & runbook

Last updated: 2026-08-31. Nothing is running; no stage has been started.

## Where things stand

**Done and verified**

- `data/nq320k/` fully prepared by `prepare_nq320k.py` (source: `dqmis/NQ320K-NCI`).
  - `corpus1.json` 99,650 docs · `corpus2.json` 10,000 docs (the held-out "new" docs)
  - `train1.json` 298,643 pairs · `train2.json` 8,730 · `dev.json`/`dev1.json` 6,036
  - `train1.json.qg.json` 996,500 pseudo-query pairs (99,650 docs x 10, docids 0..99,649)
  - `train2.json.qg.json` 100,000 pairs (10,000 docs x 10, docids 99,650..109,649)
  - Pseudo-queries were re-verified after the forced shutdown on 2026-08-31: valid JSON,
    exactly 10 per doc, contiguous docids, no blanks, no leftover `.tmp`. Complete.
- Models in `../huggingface/`: `bert-base-uncased`, `t5-large`, `doc2query-t5-base-msmarco`.
- Seven modules that the repo imports but never shipped were reconstructed in an earlier
  session: `GR/util/io.py`, `GR/Dataset/NewNQDataset.py`, `GR/genre/trie.py`,
  `GR/eval/eval.py`, `util/nethook.py`, `dsets/knowns.py`, `layer_stats.py`.

**Not started** — no `corpus_vectors_*`, no `corpus_docids_rq_*`, no checkpoints, no `Edit/`.

## Fixes applied 2026-08-31 (uncommitted; see `git diff`)

1. **`GR/get_docid.py`** — used to emit only `corpus_docids_rq_1`, but `get_knowns.py`
   reads `_1` **and** `_2`. Now does both in one run: fits the RQ quantizer on corpus1,
   then *encodes* corpus2 with those same codebooks (refitting would give new documents
   an inconsistent codebook). Corpus1's random collision-digits are passed forward as
   `reserved` so corpus2 cannot be handed a docID corpus1 already owns.
   Also: overflow (an RQ code with >128 docs, which upstream silently maps to digit 0 and
   thereby duplicates a docID) now prints an explicit warning, and the uniqueness check
   runs across the concatenated corpus. Smoke-tested on 300+20 synthetic docs:
   320 docIDs, 0% duplication, `corpusids[global_docid]` invariant holds.
2. **`DOME.py`** — `main()` never passed `epoch_path`, so a missing GR checkpoint only
   *warned* and DOME would have edited a raw pretrained t5-large that has never seen a
   docID, producing meaningless numbers. Added `--gr_checkpoint`
   (default `data/output/t5-large-rq-ckpts/34.pt`), passed it through, and made a missing
   checkpoint raise. Repointed `--model_name` to `../huggingface/t5-large` and the four
   data files to `Edit/data/`.
3. **`GR/get_knowns.py`** — `Edit/data/` does not exist in the repo, so every write failed
   after all the work was done; `write_json` now creates it. Replaced two list rebuilds
   inside 100k-iteration loops with a set. Smoke-tested: emits all 7 files DOME reads.
4. **`GR/get_vectors.py`** — pad to longest-in-batch instead of always 500 tokens (same
   embeddings, much less wasted compute); dropped a `.squeeze()` that breaks when the
   final batch holds exactly one document.

Line endings: `DOME.py`, `GR/get_knowns.py`, `GR/get_vectors.py` are CRLF upstream and
were kept CRLF. `GR/get_docid.py` is LF upstream.

## Before starting: grow the disk

Currently 65 GB with ~18 GB free; a full run needs ~85 GB on top of what is used.
Target **150 GB**. `sda1` already fills the disk, so: grow the virtual disk, then
`growpart /dev/sda 1` and `resize2fs /dev/sda1`.

| item | size |
| --- | --- |
| `corpus_vectors_1/2.json` | ~5 GB |
| 10 x t5-large checkpoints | 28 GB |
| DOME edited model + covariance cache | ~4 GB |

## Run order

```bash
cd ~/projects/DOME
./.venv/bin/python GR/get_vectors.py          # 1a: BERT-embed 109,650 docs
./.venv/bin/python GR/get_docid.py            # 1b: emits corpus_docids_rq_1 AND _2
./.venv/bin/accelerate launch --mixed_precision bf16 GR/training.py \
    --epochs 35 --eval_interval 5             # 1c: -> 34.pt, 7 checkpoints
./.venv/bin/python GR/get_knowns.py           # 2:  writes Edit/data/*.json
./.venv/bin/python DOME.py                    # 3:  the editing
```

`--epochs 35` lands the final checkpoint exactly on the `34.pt` DOME expects.

## Watch for

- **GPU memory.** 24 GB total; AdamW fp32 on 770M params is ~12.4 GB before activations.
  Use `accelerate launch --mixed_precision bf16` (plain `Accelerator()` defaults to fp32).
  If batch 128 OOMs, drop to `--batch_size 64`. Confirm in the first minutes of Stage 1c.
- **Step rate.** 1.39M samples/epoch at batch 128 = ~10.9k steps/epoch, ~381k steps for
  35 epochs. At ~1 s/step that is ~4.5 days. Unmeasured on this 3090 — watch the first
  100 steps and extrapolate before committing the week.
- **`training.py` has no resume.** A crash at epoch 40 restarts from epoch 0 with the
  checkpoint sitting unused. Open item: add resume (load newest `.pt`, restore epoch and
  scheduler position) before kicking off the long run.

## Open decisions

- [ ] Grow disk to 150 GB
- [ ] Add resume-from-checkpoint to `GR/training.py` (recommended, not yet done)
- [ ] Commit the four fixes above (currently uncommitted working-tree changes)
