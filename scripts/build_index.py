"""GraphRAG steps 3-5: merge the extractions into one graph, find hierarchical communities, write their reports.

Reads data/index/extractions.jsonl; writes graph.json, communities.json and reports.jsonl (resumable).

usage: uv run scripts/build_index.py --levels 3
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aml_graphrag.adapters.ollama import OllamaLLM
from aml_graphrag.application.communities import build_graph, hierarchical_leiden, largest_component
from aml_graphrag.application.indexing import merge
from aml_graphrag.application.reports import community_context, write_report


def main(args):
    extractions = [json.loads(line) for line in (args.index / "extractions.jsonl").open()]
    nodes, edges = merge([(e["chunk"], e) for e in extractions])
    g = build_graph(nodes, edges)
    giant = largest_component(g)
    print(f"{len(nodes)} entities, {len(edges)} relationships; largest component {giant.vcount()} entities "
          f"({giant.vcount() / g.vcount():.0%}), {giant.ecount()} relationships")
    keep = set(giant.vs["name"])
    nodes = {n: v for n, v in nodes.items() if n in keep}
    edges = {p: e for p, e in edges.items() if p[0] in keep and p[1] in keep}
    degree = dict(zip(giant.vs["name"], giant.degree()))

    communities = [c for c in hierarchical_leiden(giant, args.max_size, seed=args.seed) if c["level"] < args.levels]
    for level in range(args.levels):
        sizes = [len(c["nodes"]) for c in communities if c["level"] == level]
        print(f"  C{level}: {len(sizes)} communities, median size {sorted(sizes)[len(sizes) // 2]}")
    (args.index / "graph.json").write_text(json.dumps({
        "nodes": nodes, "edges": [{"source": s, "target": t, **e} for (s, t), e in edges.items()]}))
    (args.index / "communities.json").write_text(json.dumps(communities))

    log = args.index / "reports.jsonl"
    reports = {json.loads(line)["community"]: json.loads(line) for line in log.open()} if log.exists() else {}
    llm = OllamaLLM(args.model, num_ctx=args.num_ctx)
    start, new = time.time(), 0
    with log.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        for level in reversed(range(args.levels)):  # bottom-up: a parent can use its children's reports
            todo = [c for c in communities if c["level"] == level and c["id"] not in reports]

            def run(c):
                children = [reports[k["id"]] for k in communities if k["parent"] == c["id"] and k["id"] in reports]
                children.sort(key=lambda r: -r["size"])
                return c, write_report(llm, community_context(c, nodes, edges, degree, children, args.budget))

            for c, report in pool.map(run, todo):
                reports[c["id"]] = row = {"community": c["id"], "level": level, "size": len(c["nodes"]), **report}
                f.write(json.dumps(row) + "\n")
                f.flush()
                new += 1
                if new % 25 == 0:
                    print(f"  {new} reports, {(time.time() - start) / new:.1f} s each", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--max-size", type=int, default=10)
    parser.add_argument("--budget", type=int, default=3000, help="context tokens per report")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--num-ctx", type=int, default=15360)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--index", type=Path, default=Path("data/index"))
    main(parser.parse_args())
