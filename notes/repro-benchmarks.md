# DOME reproduction — hardware feasibility benchmark (RTX 3090)

**Question asked:** run the DOME experiment, cut it short to estimate the full
wall-clock, do the same for smaller/quantized variants, and decide whether this
home-lab box (single RTX 3090, 24 GB) can realistically reproduce the paper.

**Date:** 2026-09-05 · **GPU:** RTX 3090 24 GB, driver 595.84, 350 W limit ·
torch 2.13+cu130, transformers 5.16.1, accelerate 1.14.

---

## TL;DR verdict

**Yes, reproducible — budget ~8–11 days wall-clock for the faithful t5-large run,**
almost all of it in GR training (`GR/training.py`). Everything else (embedding,
docID, DOME editing) is hours, not days. But two things dominate the number, and
one of them is a surprise:

1. **bf16 is mandatory.** bf16 = ~7.5 d (35 ep); fp32 = ~15.5 d. Always launch
   with `accelerate launch --mixed_precision bf16`. The repo's `Accelerator()`
   defaults to **fp32** — if you just `python GR/training.py`, you silently pay
   2×.
2. **This particular 3090 thermal-throttles hard.** It boosts to 1950 MHz cold,
   then collapses to **~250–765 MHz within seconds** and stays there for the rest
   of a run — a ~4× clock loss. It sits at 76–82 °C drawing only ~160 W of its
   350 W budget, i.e. **cooling-limited, not power-limited.** Fixing airflow /
   fan curve / paste is the single biggest lever available — potentially bigger
   than switching models. (See "The throttling story" below — it's why my first
   measurement was 3× too optimistic.)

Disk is no longer a constraint: **259 G total / 210 G free** after the +200 G
grow. The full run needs ~4.4 G (vectors) + ~30 G (t5-large checkpoints, one
every 5 epochs at 2.95 G each) + small — all fits comfortably now.

---

## How these numbers were produced (and the one big gotcha)

I built a standalone harness (`scratchpad/bench_train.py`) that reproduces
`GR/training.py`'s inner step **exactly** — same model, +128 docID tokens,
resized embeddings, same AdamW/Adafactor, same short `query→docid` batch shape
from the real `train1.json` — but with synthetic docID targets, so I could sweep
model / precision / optimizer / batch without waiting on the hour-long Stage-0
embedding or the 2.65 GB corpus load. Per-step FLOPs and memory are identical to
the real loop.

I then **validated it against the real `GR/training.py`** launched under
accelerate+bf16: it reported `Training data size: 1394793` (exactly our count),
used 19.5 GB, and stepped at ~1.7 s/step — matching the harness's *sustained*
number. Loss fell 20.6 → 13.6 over the first ~140 steps, so it trains.

**The gotcha:** my very first harness reading was 0.485 s/step (would imply ~2 d).
That was a **cold-GPU artifact** — the card was boosting to 1950 MHz because the
heatsink was still cold. Every warm / heat-soaked measurement, and the real
script, land at ~1.7 s/step. **All numbers below are the sustained (heat-soaked)
values**, which is what a multi-day run actually experiences.

---

## Results

### Epoch size
Dataset = 298,643 real queries + 996,500 pseudo-queries + 99,650 documents =
**1,394,793 samples/epoch** → **10,897 steps/epoch** at batch 128. Default script
= 50 epochs (paper-style convergence ~35 is enough; both shown).

### Model comparison — bf16 + AdamW, batch 128 (sustained)

| model | params | s/step | VRAM | h/epoch | 35 epochs | 50 epochs |
|---|---|---|---|---|---|---|
| **t5-large** (paper) | 738 M | 1.695 | 19.3 G | 5.13 | **7.5 d** | **10.7 d** |
| t5-base | 223 M | 0.564 | 6.7 G | 1.71 | **2.5 d** | 3.6 d |
| t5-small | 61 M | 0.117 | 2.2 G | 0.35 | **0.5 d** | 0.75 d |

Throttling hits t5-large *hardest* (most power → most heat), so the smaller
models' relative advantage grows under sustained load: t5-base is 3.0× faster
per step than t5-large, t5-small 14.5×.

### Precision / optimizer — t5-large, batch 128 (sustained)

| config | s/step | VRAM | 35 epochs | note |
|---|---|---|---|---|
| **bf16 + AdamW** | 1.695 | 19.3 G | **7.5 d** | recommended |
| fp32 + AdamW | 3.515 | 20.0 G | 15.5 d | 2× slower — avoid |
| bf16 + Adafactor | 1.924 | 13.4 G | 8.5 d | saves 6 G, 13 % slower |

Adafactor is the memory lever, not a speed lever: it frees ~6 G of optimizer
state but is slightly slower. On the 3090, AdamW+bf16 already fits at 19.3 G, so
**use AdamW** unless you need the headroom for a bigger batch.

### Batch size — t5-large, bf16 (directional; measured pre-throttle-discovery)
- **batch 256 → OOM.** 128 is the practical ceiling for t5-large on 24 G.
- batch 64 gave *worse* throughput per sample than 128 (small-batch inefficiency).
- **Batch 128 is the sweet spot.** t5-base/small have headroom for 256–384 but
  gain little; 128 keeps things comparable to the paper.

### Stage 0 — corpus embedding (`GR/get_vectors.py`, bert-base, batch 64)
27.9 docs/s → **~65 min** for all 109,650 docs. Output vectors JSON ≈ 4.4 G
(re-stores full doc text + 768-d embeddings). One-time.

### DOME editing (`DOME.py`) — estimated, not directly timed
Requires a trained GR checkpoint (which we don't have yet), so I did not time it
end-to-end. From the code: DOME precomputes a **400,000-sample** activation
covariance (ROME-style, Eq. 8), which is a one-time cost **cached to disk**
(`cache/covariance/`), then applies fast closed-form per-document FFN edits.
Expect the covariance pass to be the heavy part (order ~1–3 h at BERT/T5 forward
throughput) and the actual edits to be minutes. Small next to GR training.

---

## The throttling story (why the hardware matters more than the model)

`nvidia-smi dmon` during a sustained t5-large run, SM clock (`pclk`):

```
idle           : 210 MHz
first ~3 s     : 1860–1950 MHz   ← cold boost (the misleading 0.485 s/step)
thereafter     : 255–765 MHz     ← collapses and stays, temp 76–82 °C, ~160 W
```

160 W out of a 350 W limit with clocks on the floor = the cooler can't keep the
die (or hotspot/VRM) in the boost band, so the card parks at low clocks. This is
**the** finding for "is this hardware realistic": the 3090 *can* do the work, but
this box's cooling is throwing away ~4× of its compute. Concretely, recovering
even half of that clock would move the t5-large run from ~7.5 d toward ~4 d.

**Cheap things to try before the long run:** raise the fan curve / set a static
high fan (`nvidia-settings` or a fan-control script; needs the X server or
`nvidia-smi -pm 1` + a coolbits fan tool), improve case airflow, re-paste if the
card is old, or *cap the power limit* (`sudo nvidia-smi -pl 250`) — counter-
intuitively, a lower, stable power target sometimes holds higher average clocks
than a thermally-oscillating 350 W. Measure a 2-minute `bench_train.py` run after
each change; the s/step number tells you immediately if it helped.

---

## Recommended path forward

1. **Fix cooling first.** Biggest lever, costs no accuracy. Re-run
   `scratchpad/bench_train.py --model_name t5-large` (2 min) to confirm s/step
   drops before committing a week.
2. **Always launch bf16:**
   `accelerate launch --mixed_precision bf16 --num_processes 1 GR/training.py --batch_size 128`.
3. **Add resume-to-checkpoint to `training.py` before starting.** A 7–11 day run
   *will* be interrupted (you already force-rebooted once). The script currently
   has **no resume** — an interruption loses everything since the last 5-epoch
   save. This is a ~20-line change (load newest `.pt`, restore epoch + scheduler
   step, optionally prune old checkpoints to save disk). Strongly recommended.
4. **Run a t5-base pilot (2.5 d) end-to-end first.** It won't match the paper's
   Hits@k (smaller model = lower retrieval), but it exercises the *entire*
   pipeline — training → `get_knowns` → DOME editing → eval — and surfaces any
   remaining breakage for the price of 2.5 days instead of finding it on day 7 of
   the t5-large run. t5-small (0.5 d) is an even faster correctness smoke test.
5. **Then commit the faithful t5-large run** (~7.5 d at 35 epochs with current
   cooling; less if cooling improves).

### End-to-end budget, faithful t5-large (current cooling)
| stage | time |
|---|---|
| Stage 0 embedding | ~1.1 h |
| get_docid + get_knowns | ~0.5 h |
| **GR training (35 ep, bf16)** | **~7.5 d** |
| DOME covariance precompute (cached) | ~1–3 h |
| DOME editing + eval | ~1 h |
| **total** | **~8 days** (fp32 path: ~16 d) |

---

## What I did *not* do
- No full training to convergence (only step-time; that's the ask).
- DOME editing not timed end-to-end (needs a trained checkpoint) — estimated
  from code.
- The synthetic `corpus_docids_rq_1.json` I used to launch the real script for
  validation was **deleted** afterward — the real pipeline must regenerate it via
  `get_vectors` → `get_docid`. No stray checkpoints or artifacts were left; only
  the 4 previously-modified source files remain in `git status`.

Raw numbers + re-runnable harnesses are in `notes/benchmarks/`:
`results.json`, `bench_train.py` (training step), `bench_stage0.py` (embedding),
`bench_dataloader.py` (data-load throughput), `bench_accel.py` (accelerate-
overhead isolation). Run any from the repo root with `./.venv/bin/python
notes/benchmarks/bench_train.py --model_name t5-large`.
