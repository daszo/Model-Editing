"""Measure Stage-0 embedding throughput (GR/get_vectors.py) on a sample of the
real corpus, then project to the full 109,650 docs and estimate output size.
Mirrors get_vectors.py: bert-base-uncased, [CLS] embedding, batch 64, max_len 500.
"""
import json, time, sys
import torch
from transformers import BertTokenizer, BertModel

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
path = "../huggingface/bert-base-uncased"
tok = BertTokenizer.from_pretrained(path)
model = BertModel.from_pretrained(path).to("cuda").eval()

# real docs from corpus1 (load full, slice first N; RAM is 54 GB so fine)
docs = json.load(open("data/nq320k/corpus1.json"))[:N]

bs = 64
torch.cuda.synchronize(); t0 = time.time()
emb_bytes = 0
with torch.no_grad():
    for i in range(0, len(docs), bs):
        batch = docs[i:i + bs]
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True,
                  max_length=500, return_attention_mask=True)
        enc = {k: v.cuda() for k, v in enc.items()}
        out = model(**enc)
        e = out.last_hidden_state[:, 0, :].cpu().tolist()
        emb_bytes += sum(len(json.dumps(x)) for x in e)
torch.cuda.synchronize()
dt = time.time() - t0
dps = len(docs) / dt
TOTAL = 109_650
proj_min = TOTAL / dps / 60
# output JSON ~ [doc_text, index, [768 floats]] ; embedding text dominates size
avg_emb_chars = emb_bytes / len(docs)
proj_out_gb = (avg_emb_chars * TOTAL) / 1e9
print("STAGE0 " + json.dumps({
    "sample_docs": len(docs), "docs_per_s": round(dps, 1),
    "proj_full_minutes": round(proj_min, 1),
    "proj_vectors_json_gb_est": round(proj_out_gb, 2),
}))
