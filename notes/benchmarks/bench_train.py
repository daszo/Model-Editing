"""Standalone benchmark that reproduces GR/training.py's inner training step
EXACTLY (same model, +128 docID tokens, resized embeddings, same AdamW/Adafactor,
same short query->docid batch shape) but with synthetic docID targets so we can
sweep model / batch / precision / optimizer WITHOUT waiting on Stage-0 embedding
or loading the 2.65 GB corpus. Per-step compute is identical to the real loop;
only the docID *values* are synthetic, which does not change FLOPs or memory.

Outputs one JSON line with steady-state step time, throughput, and peak VRAM,
plus projected epoch / full-run wall time for the real 1,394,793-sample epoch.
"""
import argparse, json, os, random, time, sys
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Adafactor
from torch.optim import AdamW

REAL_EPOCH_SAMPLES = 1_394_793  # 298,643 real + 996,500 pseudo + 99,650 docs (corpus1)


def _docid_to_target(t):
    return "$" + "$$".join(t) + "$"


class BenchDataset(Dataset):
    """Real query texts (realistic token lengths) + synthetic 4-digit docIDs."""
    def __init__(self, texts, id_num, max_len, max_target_len, tokenizer):
        self.texts = texts
        self.id_num = id_num
        self.max_len = max_len
        self.max_target_len = max_target_len
        self.tok = tokenizer
        rng = random.Random(0)
        self.docids = [[str(rng.randint(0, id_num - 1)) for _ in range(4)]
                       for _ in range(len(texts))]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return str(self.texts[i]), _docid_to_target(self.docids[i])

    def collate_fn(self, batch):
        texts = [b[0] for b in batch]
        targets = [b[1] for b in batch]
        enc = self.tok(texts, padding=True, truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        labels = self.tok(targets, padding=True, truncation=True,
                          max_length=self.max_target_len, return_tensors="pt").input_ids
        labels[labels == self.tok.pad_token_id] = -100
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": labels}


def resolve_model(name):
    local = f"../huggingface/{name}"
    return local if os.path.isdir(local) else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="t5-large")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--precision", choices=["fp32", "bf16", "bf16-full"], default="bf16")
    ap.add_argument("--optimizer", choices=["adamw", "adafactor"], default="adamw")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--id_num", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--warmup_steps", type=int, default=8)
    ap.add_argument("--measure_steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train_json", default="data/nq320k/train1.json")
    args = ap.parse_args()

    result = {"model": args.model_name, "batch": args.batch_size,
              "precision": args.precision, "optimizer": args.optimizer,
              "grad_ckpt": args.grad_ckpt}

    try:
        import json as _j
        texts = [x[0] for x in _j.load(open(args.train_json))]
        # only need enough rows to fill warmup+measure batches
        need = args.batch_size * (args.warmup_steps + args.measure_steps + 2)
        texts = (texts * (need // len(texts) + 1))[:need]

        path = resolve_model(args.model_name)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
        for i in range(args.id_num):
            tok.add_tokens([f"${i}$"])
        model.resize_token_embeddings(len(tok))
        result["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)

        if args.grad_ckpt:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        model_dtype = torch.bfloat16 if args.precision == "bf16-full" else torch.float32
        model = model.to("cuda", dtype=model_dtype)
        model.train()

        if args.optimizer == "adamw":
            opt = AdamW(model.parameters(), lr=args.lr)
        else:
            opt = Adafactor(model.parameters(), lr=args.lr, scale_parameter=False,
                            relative_step=False, warmup_init=False)

        ds = BenchDataset(texts, args.id_num, args.max_len, 32, tok)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=ds.collate_fn, drop_last=True)

        autocast = args.precision == "bf16"  # fp32 params, bf16 compute (== accelerate bf16)
        torch.cuda.reset_peak_memory_stats()
        step_times = []
        it = iter(dl)
        step = 0
        total = args.warmup_steps + args.measure_steps
        while step < total:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl); batch = next(it)
            batch = {k: v.cuda() for k, v in batch.items()}
            torch.cuda.synchronize(); t0 = time.time()
            if autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
            else:
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
            torch.cuda.synchronize()
            dt = time.time() - t0
            if step >= args.warmup_steps:
                step_times.append(dt)
            step += 1

        step_times.sort()
        median = step_times[len(step_times) // 2]
        mean = sum(step_times) / len(step_times)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        steps_per_epoch = -(-REAL_EPOCH_SAMPLES // args.batch_size)  # ceil
        result.update({
            "ok": True,
            "median_step_s": round(median, 4),
            "mean_step_s": round(mean, 4),
            "samples_per_s": round(args.batch_size / median, 1),
            "peak_vram_gb": round(peak_gb, 2),
            "steps_per_epoch": steps_per_epoch,
            "epoch_hours": round(steps_per_epoch * median / 3600, 2),
            "run35_days": round(steps_per_epoch * median * 35 / 86400, 2),
            "run50_days": round(steps_per_epoch * median * 50 / 86400, 2),
        })
    except RuntimeError as e:
        msg = str(e)
        result["ok"] = False
        result["error"] = "OOM" if "out of memory" in msg.lower() else msg[:200]
        torch.cuda.empty_cache()
    except Exception as e:
        result["ok"] = False
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    print("BENCH_RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
