"""
Generate docT5query pseudo-queries for the NQ320k corpus.

For each document we sample N queries from castorini/doc2query-t5-base-msmarco
(the docTTTTTquery model used by the DSI/NCI generative-retrieval recipe) and
emit pairs in the exact format the pipeline consumes:

    [[pseudo_query_text, global_docid], ...]

global_docid MUST be the position of the document in the CONCATENATED corpus
(corpus1 + corpus2), because GR/get_knowns.py and GR/training.py resolve a pair
via corpusids[query[1]]. corpus1 docs start at 0; corpus2 docs start at
--docid-offset (= len(corpus1)).

Outputs are written incrementally so the run is crash-resilient / resumable.

Examples:
    # corpus1 -> data/nq320k/train1.json.qg.json  (offset 0)
    python generate_pseudo_queries.py \
        --corpus data/nq320k/corpus1.json \
        --out    data/nq320k/train1.json.qg.json \
        --docid-offset 0

    # corpus2 -> data/nq320k/train2.json.qg.json  (offset = len(corpus1))
    python generate_pseudo_queries.py \
        --corpus data/nq320k/corpus2.json \
        --out    data/nq320k/train2.json.qg.json \
        --docid-offset 99650
"""
import argparse
import json
import os

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_DIR = "../huggingface/doc2query-t5-base-msmarco"


def load_corpus(path):
    with open(path) as f:
        return json.load(f)


def load_done_docids(out_path):
    """Return set of global docids already generated, for resume."""
    if not os.path.exists(out_path):
        return set()
    try:
        with open(out_path) as f:
            data = json.load(f)
        return {pair[1] for pair in data}, data
    except (json.JSONDecodeError, IndexError, KeyError):
        return set(), []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--docid-offset", type=int, default=0,
                    help="global docid of corpus[0] (= len(corpus1) for corpus2)")
    ap.add_argument("--num-queries", type=int, default=10, help="pseudo queries per document")
    ap.add_argument("--batch-size", type=int, default=32, help="documents per forward pass")
    ap.add_argument("--max-input-len", type=int, default=256)
    ap.add_argument("--max-output-len", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="only process first N docs (debug)")
    ap.add_argument("--flush-every", type=int, default=200, help="write output every N docs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_DIR} on {device} ...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    if device == "cuda":
        model = model.half()
    model.to(device).eval()

    corpus = load_corpus(args.corpus)
    if args.limit:
        corpus = corpus[:args.limit]
    total = len(corpus)

    # Resume support: skip docs already present in the output file.
    done = load_done_docids(args.out)
    if isinstance(done, tuple):
        done_ids, results = done
    else:
        done_ids, results = done, []
    if done_ids:
        print(f"Resuming: {len(done_ids)} docids already in {args.out}")

    def flush():
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f)
        os.replace(tmp, args.out)

    from tqdm import tqdm
    pending = [(i, doc) for i, doc in enumerate(corpus)
               if (args.docid_offset + i) not in done_ids]
    print(f"{total} docs total, {len(pending)} to generate "
          f"({args.num_queries} queries each), batch={args.batch_size}")

    processed_since_flush = 0
    for start in tqdm(range(0, len(pending), args.batch_size)):
        batch = pending[start:start + args.batch_size]
        docs = [d for _, d in batch]
        enc = tokenizer(docs, max_length=args.max_input_len, truncation=True,
                        padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_length=args.max_output_len,
                do_sample=True,
                top_k=10,
                num_return_sequences=args.num_queries,
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        # decoded is flat: [doc0_q0..doc0_qN-1, doc1_q0..]; regroup per doc.
        for bi, (local_i, _) in enumerate(batch):
            gid = args.docid_offset + local_i
            qs = decoded[bi * args.num_queries:(bi + 1) * args.num_queries]
            for q in qs:
                q = q.strip()
                if q:
                    results.append([q, gid])
        processed_since_flush += len(batch)
        if processed_since_flush >= args.flush_every:
            flush()
            processed_since_flush = 0

    flush()
    print(f"Done. Wrote {len(results)} pseudo-query pairs to {args.out}")


if __name__ == "__main__":
    main()
