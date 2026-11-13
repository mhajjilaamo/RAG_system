"""
Evaluate retrieval of a saved index against a LegalBench-RAG benchmark.

    python -m scripts.evaluate indexes/privacy_qa/20261003_003723 --k 8
    python -m scripts.evaluate <index_dir> --k 3 --max-questions 50 --full   # full benchmark file

Writes results/<corpus>/run_<corpus>_<timestamp>.json
"""
import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone

from src.eval_metrics import precision_k, recall_k, reciprocal_rank
from src.pipeline import Index

CHUNK_SIZE = 500     # chunkers' default max size
CHUNK_OVERLAP = 0


def evaluate(index_dir, n_results, max_questions=None, mini=True, verbose=False):
    index = Index(index_dir)
    cfg = index.config
    corpus = cfg.corpus

    timestamp = datetime.now(timezone.utc).isoformat()
    benchmark_path = f"data/legalBench-RAG/benchmarks/{corpus}{'_mini' if mini else ''}.json"
    with open(benchmark_path, "r", encoding="utf-8") as f:
        tests = json.load(f)["tests"][:max_questions]

    eval_results = {
        "run": {
            "timestamp": timestamp,
            "dataset": "LEGAL BENCH RAG",
            "benchmark file": benchmark_path,
            "index dir": index_dir,
            "corpus": corpus,
            "num questions": len(tests),
            "chunk strategy": cfg.chunking_strategy,
            "chunk size": CHUNK_SIZE,
            "chunk overlap": CHUNK_OVERLAP,
            "embedding model": cfg.embedding_model,
            "k": n_results,
        },
        "aggregates": {},
        "per_question": [],
    }

    recall_sum = precision_sum = rr_sum = 0
    for i, test in enumerate(tests, start=1):
        question, snippets = test["query"], test["snippets"]
        results = index.query(question, n_results)

        recall = recall_k(results, snippets)
        precision = precision_k(results, snippets)
        rr = reciprocal_rank(results, snippets)
        recall_sum += recall
        precision_sum += precision
        rr_sum += rr

        if verbose:
            print("question :", question)
            print("expected :", *[s["answer"] for s in snippets], sep="\n")
            print(f"top {n_results} :", *[c.text for c in results], sep="\n")
        print(f"[{i}/{len(tests)}] recall={recall:.3f} precision={precision:.3f} rr={rr:.3f}")

        eval_results["per_question"].append({
            "question": question,
            "gold snippets": snippets,
            "retrieved results": [asdict(r) for r in results],
            "recall": recall,
            "precision": precision,
            "reciprocal rank": rr,
        })

        # Gemini free-tier rate limit: one query embedding per question
        if cfg.embedding_model == "gemini" and i % 100 == 0:
            time.sleep(60)

    n = len(tests)
    eval_results["aggregates"] = {
        "mean recall": recall_sum / n,
        "mean precision": precision_sum / n,
        "mean reciprocal rank": rr_sum / n,
    }
    print(json.dumps(eval_results["aggregates"], indent=4))

    out_dir = os.path.join("results", corpus)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"run_{corpus}_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
    print("saved :", path)
    return eval_results["aggregates"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("index_dir", help="e.g. indexes/privacy_qa/20261003_003723")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--full", action="store_true", help="use the full benchmark instead of *_mini.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    evaluate(args.index_dir, args.k, args.max_questions, mini=not args.full, verbose=args.verbose)


if __name__ == "__main__":
    main()
