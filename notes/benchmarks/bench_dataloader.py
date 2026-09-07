"""Diagnose whether GR/training.py is data-loading bound. Times pure dataloader
throughput (no GPU) using the SAME on-the-fly tokenization the training collate
uses, across num_workers settings, and reports tokenizer fast/slow."""
import json, time, sys, random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

path = "../huggingface/t5-large"
tok = AutoTokenizer.from_pretrained(path)
for i in range(128):
    tok.add_tokens([f"${i}$"])
print("tokenizer.is_fast =", tok.is_fast, "| type =", type(tok).__name__)

texts = [x[0] for x in json.load(open("data/nq320k/train1.json"))]
rng = random.Random(0)
docids = [["$%d$" % rng.randint(0, 127) for _ in range(4)] for _ in range(len(texts))]


class DS(Dataset):
    def __len__(self): return len(texts)
    def __getitem__(self, i):
        return str(texts[i]), "$" + "$$".join(docids[i]) + "$"
    def collate(self, batch):
        t = [b[0] for b in batch]; y = [b[1] for b in batch]
        enc = tok(t, padding=True, truncation=True, max_length=32, return_tensors="pt")
        lab = tok(y, padding=True, truncation=True, max_length=32, return_tensors="pt").input_ids
        return enc, lab

ds = DS()
# raw single-batch tokenize cost (main thread)
b = [ds[i] for i in range(128)]
t0 = time.time()
for _ in range(20):
    ds.collate(b)
print("single-thread tokenize: %.1f ms/batch (128)" % ((time.time()-t0)/20*1000))

for nw in [0, 4, 8, 16]:
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=nw,
                    collate_fn=ds.collate, drop_last=True,
                    persistent_workers=(nw > 0), prefetch_factor=(4 if nw > 0 else None))
    it = iter(dl)
    for _ in range(5): next(it)          # warm workers
    t0 = time.time(); n = 40
    for _ in range(n): next(it)
    dt = time.time() - t0
    print("num_workers=%2d : %.3f s/batch  -> %.0f samples/s (data only)" % (nw, dt/n, 128*n/dt))
