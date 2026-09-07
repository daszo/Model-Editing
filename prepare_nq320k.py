"""
Prepare the NQ320k data files that the DOME / GR pipeline expects.

Source: https://huggingface.co/datasets/dqmis/NQ320K-NCI  (NCI preprocessing recipe)
  data/corpus.jsonl : {"docid": "<int>", "document": "<text>"}
  data/train.jsonl  : {"query": "...", "docid": "<int>", "title": ..., ...}
  data/valid.jsonl  : same shape as train.jsonl

The scripts in this repo read a flat layout under data/nq320k/. In every file the
document is referenced by its POSITIONAL index into the concatenated corpus
(corpus1 + corpus2). See GR/get_knowns.py:45 -> corpusids[query[1]] and
GR/get_vectors.py:35 -> corpus = [doc for doc, index in ...].

Produced files (all consumed by GR/*.py and DOME.py):
  data/nq320k/corpus1.json            list[str]            docs[:split]
  data/nq320k/corpus2.json            list[str]            docs[split:]  (held-out "new" docs)
  data/nq320k/train1.json             list[[query, docid]] train pairs into corpus1
  data/nq320k/train2.json             list[[query, docid]] train pairs into corpus2
  data/nq320k/dev.json                list[[query, docid]] valid pairs into corpus1
  data/nq320k/dev1.json               list[[query, docid]] copy of dev.json
  data/nq320k/train1.json.qg.json     []  placeholder pseudo-queries (see note below)
  data/nq320k/train2.json.qg.json     []  placeholder pseudo-queries

NOTE on .qg.json: the pipeline expects docT5query-generated pseudo queries
(query augmentation). The public dataset does not ship them, so we write empty
lists, which the code accepts. Replace them with real generated queries in the
same [[query, docid], ...] format to fully reproduce the paper.

Usage:
    python prepare_nq320k.py                # default 10000 held-out docs -> corpus2
    python prepare_nq320k.py --holdout 10000
"""
import argparse
import json
import os

from huggingface_hub import hf_hub_download

REPO = "dqmis/NQ320K-NCI"
OUT_DIR = "data/nq320k"


def read_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  wrote {path}  ({len(data)} items)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=10000,
                    help="number of last documents put in corpus2 (the 'new' docs to edit)")
    args = ap.parse_args()

    print(f"Downloading {REPO} (corpus/train/valid)...")
    corpus_path = hf_hub_download(REPO, "data/corpus.jsonl", repo_type="dataset")
    train_path = hf_hub_download(REPO, "data/train.jsonl", repo_type="dataset")
    valid_path = hf_hub_download(REPO, "data/valid.jsonl", repo_type="dataset")

    # --- Corpus: order by integer docid so list position == docid ---
    print("Loading corpus...")
    corpus_rows = read_jsonl(corpus_path)
    corpus_rows.sort(key=lambda r: int(r["docid"]))
    docids = [int(r["docid"]) for r in corpus_rows]
    n = len(docids)
    # Sanity: docids must be contiguous 0..n-1 for positional indexing to be valid.
    assert docids == list(range(n)), (
        f"docids are not a contiguous 0..{n-1} range; positional indexing would be wrong")
    documents = [r["document"] for r in corpus_rows]
    print(f"  {n} documents")

    split = n - args.holdout
    assert 0 < split < n, f"holdout {args.holdout} invalid for corpus of size {n}"

    write_json(f"{OUT_DIR}/corpus1.json", documents[:split])
    write_json(f"{OUT_DIR}/corpus2.json", documents[split:])
    print(f"  corpus1 = docs[0:{split}], corpus2 = docs[{split}:{n}]")

    # --- Query pairs -> [[query, docid], ...] split by which corpus the doc is in ---
    def to_pairs(rows):
        return [[r["query"], int(r["docid"])] for r in rows]

    train_pairs = to_pairs(read_jsonl(train_path))
    valid_pairs = to_pairs(read_jsonl(valid_path))
    print(f"Loaded {len(train_pairs)} train / {len(valid_pairs)} valid pairs")

    train1 = [p for p in train_pairs if p[1] < split]
    train2 = [p for p in train_pairs if p[1] >= split]
    dev = [p for p in valid_pairs if p[1] < split]

    write_json(f"{OUT_DIR}/train1.json", train1)
    write_json(f"{OUT_DIR}/train2.json", train2)
    write_json(f"{OUT_DIR}/dev.json", dev)
    write_json(f"{OUT_DIR}/dev1.json", dev)  # code references both dev.json and dev1.json

    # --- Pseudo-query placeholders (see module docstring) ---
    write_json(f"{OUT_DIR}/train1.json.qg.json", [])
    write_json(f"{OUT_DIR}/train2.json.qg.json", [])

    print("\nDone. Files are in data/nq320k/")
    print("Reminder: train1/train2.json.qg.json are EMPTY placeholders; supply real "
          "docT5query pseudo queries for full reproduction.")


if __name__ == "__main__":
    main()
