"""Answer every question with each condition: C0, C1, C2 (GraphRAG global search on one community level),
TS (the same map-reduce over the source chunks) and SS (vector RAG). Same context budget for all.

Answers are appended to results/graphrag/answers.jsonl; rerun to resume.

usage: uv run scripts/answer.py --conditions C0 C1 C2 SS        (needs Ollama with nomic-embed-text)
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from aml_graphrag.adapters.ollama import OllamaEmbedder, OllamaLLM
from aml_graphrag.application.answering import VectorIndex, global_search, vector_rag
from aml_graphrag.application.reports import report_text


def chunk_index(index_dir, embedder):
    chunks = [json.loads(line) for line in (index_dir / "chunks.jsonl").open()]
    texts = [f"({c['title']}) {c['text']}" for c in chunks]
    cache = index_dir / "chunk_vectors.npy"
    if cache.exists():
        return VectorIndex(embedder, texts, np.load(cache)), texts
    index = VectorIndex(embedder, texts)
    np.save(cache, index.vectors)
    return index, texts


def main(args):
    questions = json.loads(args.questions.read_text())
    reports = [json.loads(line) for line in (args.index / "reports.jsonl").open()]
    by_level = {f"C{level}": [report_text(r) for r in reports if r["level"] == level] for level in range(3)}
    llm = OllamaLLM(args.model, num_ctx=args.num_ctx)
    ss_index, chunk_texts = chunk_index(args.index, OllamaEmbedder())

    def run(condition, q):
        start = time.time()
        if condition == "SS":
            out = vector_rag(llm, q["question"], ss_index, args.budget)
        else:
            units = chunk_texts if condition == "TS" else by_level[condition]
            out = global_search(llm, q["question"], units, args.budget, seed=q["id"])
        return {"question": q["id"], "condition": condition, "seconds": round(time.time() - start, 1), **out}

    log = args.out / "answers.jsonl"
    done = {(a["question"], a["condition"]) for a in map(json.loads, log.open())} if log.exists() else set()
    todo = [(c, q) for c in args.conditions for q in questions[:args.limit] if (q["id"], c) not in done]
    print(f"{len(todo)} answers to write", flush=True)
    with log.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        for n, row in enumerate(pool.map(lambda cq: run(*cq), todo), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"  {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=["C0", "C1", "C2", "SS"])
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--budget", type=int, default=3000, help="context tokens, the same for every condition")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--num-ctx", type=int, default=15360)
    parser.add_argument("--questions", type=Path, default=Path("results/graphrag/questions.json"))
    parser.add_argument("--index", type=Path, default=Path("data/index"))
    parser.add_argument("--out", type=Path, default=Path("results/graphrag"))
    main(parser.parse_args())
