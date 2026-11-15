import json

import numpy as np

from app.config import settings
from app.evaluation.metrics import hit_at_k, is_relevant, percentile, precision_at_k, reciprocal_rank
from app.ingestion.loaders import load_documents
from app.retrieval.retrievers import Retriever
from app.retrieval.store import ensure_index

CHUNKERS = ["fixed", "recursive"]
RETRIEVERS = ["dense", "bm25", "hybrid", "hybrid_rerank"]
KS = [3, 5, 10]


def main():
    docs = load_documents(settings.data_dir / "raw")
    with open(settings.data_dir / "eval" / "eval.jsonl", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]
    print(f"{len(docs)} docs, {len(dataset)} questions\n")

    results = []
    print(f"{'chunker':<10}{'retriever':<15}{'k':<4}{'Hit@k':<8}{'MRR':<8}{'P@k':<8}{'p50ms':<8}{'p95ms':<8}")
    for chunker in CHUNKERS:
        col, chunks = ensure_index("corpus", docs, chunker)
        for mode in RETRIEVERS:
            retriever = Retriever(col, chunks, mode)
            for k in KS:
                hits, rrs, precs, times = [], [], [], []
                for q in dataset:
                    got, ms = retriever.retrieve(q["question"], k)
                    rel = [is_relevant(c.text, q["evidence"]) for c in got]
                    hits.append(hit_at_k(rel, k))
                    rrs.append(reciprocal_rank(rel))
                    precs.append(precision_at_k(rel, k))
                    times.append(ms)
                row = {
                    "chunker": chunker, "retriever": mode, "k": k, "n": len(dataset),
                    "hit": float(np.mean(hits)), "mrr": float(np.mean(rrs)),
                    "precision": float(np.mean(precs)),
                    "p50_ms": percentile(times, 50), "p95_ms": percentile(times, 95),
                }
                results.append(row)
                print(f"{chunker:<10}{mode:<15}{k:<4}{row['hit']:<8.3f}{row['mrr']:<8.3f}"
                      f"{row['precision']:<8.3f}{row['p50_ms']:<8.1f}{row['p95_ms']:<8.1f}")

    out = settings.data_dir / "results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()