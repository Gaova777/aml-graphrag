"""GraphRAG step 1-2: chunk the corpus and extract entities and relationships from every chunk.

Writes data/index/chunks.jsonl and data/index/extractions.jsonl as it goes; rerun to resume.

usage: uv run scripts/extract.py            (needs Ollama: OLLAMA_NUM_PARALLEL=3 for --workers 3)
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from aml_graphrag.adapters.ollama import OllamaLLM
from aml_graphrag.application.indexing import chunk, clean, extract


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    chunks = []
    for doc in json.loads(args.manifest.read_text()):
        text = clean((args.raw / f"{doc['revision']}.txt").read_text(), args.max_tokens)
        for i, piece in enumerate(chunk(text)):
            chunks.append({"id": f"{doc['revision']}-{i}", "title": doc["title"], "text": piece})
    with open(args.out / "chunks.jsonl", "w") as f:
        f.writelines(json.dumps(c) + "\n" for c in chunks)

    log = args.out / "extractions.jsonl"
    done = {json.loads(line)["chunk"] for line in log.open()} if log.exists() else set()
    todo = [c for c in chunks if c["id"] not in done]
    print(f"{len(chunks)} chunks, {len(todo)} to go", flush=True)
    # 12288 context cells shared by 3 parallel slots: ~4k tokens each, enough for one chunk and its graph
    llm = OllamaLLM(args.model, num_ctx=4096 * args.workers)

    def run(c):
        for attempt in range(2):
            try:
                return c, extract(llm, c["text"])
            except json.JSONDecodeError:  # the reply hit the context limit and the JSON was cut
                pass
        return c, None

    start = time.time()
    with open(log, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        for n, fut in enumerate(as_completed(pool.submit(run, c) for c in todo), 1):
            c, ex = fut.result()
            if ex is None:
                print(f"  gave up on {c['id']} ({c['title']})", flush=True)
                continue
            f.write(json.dumps({"chunk": c["id"], **ex}) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"  {n}/{len(todo)}  {(time.time() - start) / n:.1f} s per chunk", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus/manifest.json"))
    parser.add_argument("--raw", type=Path, default=Path("data/corpus/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/index"))
    main(parser.parse_args())
