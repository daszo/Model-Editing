"""Replicate GR/training.py's EXACT step (Accelerator(mixed_precision=bf16),
accelerator.prepare, accelerator.backward, accelerator.clip_grad_norm_) and time
it with CUDA sync, to explain why the real script runs ~1.7 s/step vs the plain
harness's 0.485. Toggle each accelerate feature to find the culprit."""
import json, time, random, sys
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.optim import AdamW
from accelerate import Accelerator

MODE = sys.argv[1] if len(sys.argv) > 1 else "full"  # full|noclip|plainautocast

acc = Accelerator(mixed_precision="bf16")
path = "../huggingface/t5-large"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForSeq2SeqLM.from_pretrained(path)
for i in range(128):
    tok.add_tokens([f"${i}$"])
model.resize_token_embeddings(len(tok))
opt = AdamW(model.parameters(), lr=1e-3)

texts = [x[0] for x in json.load(open("data/nq320k/train1.json"))][:128*60]
rng = random.Random(0)


class DS(Dataset):
    def __len__(self): return len(texts)
    def __getitem__(self, i):
        return str(texts[i]), "$" + "$$".join("$%d$" % rng.randint(0,127) for _ in range(4)) + "$"
    def collate(self, b):
        t=[x[0] for x in b]; y=[x[1] for x in b]
        enc=tok(t,padding=True,truncation=True,max_length=32,return_tensors="pt")
        lab=tok(y,padding=True,truncation=True,max_length=32,return_tensors="pt").input_ids
        lab[lab==tok.pad_token_id]=-100
        return {"input_ids":enc["input_ids"],"attention_mask":enc["attention_mask"],"labels":lab}

ds=DS()
dl=DataLoader(ds,batch_size=128,shuffle=False,num_workers=4,collate_fn=ds.collate,drop_last=True)

if MODE=="plainautocast":
    model=model.cuda(); model.train()
else:
    model,opt,dl=acc.prepare(model,opt,dl)
    model.train()

times=[]
it=iter(dl); step=0
while step<48:
    try: batch=next(it)
    except StopIteration: it=iter(dl); batch=next(it)
    if MODE=="plainautocast":
        batch={k:v.cuda() for k,v in batch.items()}
    torch.cuda.synchronize(); t0=time.time()
    if MODE=="plainautocast":
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss=model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    else:
        loss=model(**batch).loss
        acc.backward(loss)
        if MODE=="full":
            acc.clip_grad_norm_(model.parameters(),1.)
    opt.step(); opt.zero_grad()
    torch.cuda.synchronize(); dt=time.time()-t0
    if step>=8: times.append(dt)
    step+=1

times.sort()
print("MODE=%s  median=%.3f s/step  mean=%.3f  peak_gb=%.2f" % (
    MODE, times[len(times)//2], sum(times)/len(times),
    torch.cuda.max_memory_allocated()/1e9))
