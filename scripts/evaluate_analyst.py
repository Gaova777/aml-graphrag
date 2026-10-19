"""Does grounding stop the assistant from making things up? The same question on every alert, three conditions.

For each claim: does it cite something it was given (code), do its numbers match what it cites (code), and does
the cited evidence actually support it (a different model, qwen3:8b, as judge). Also: when the explanation is
unstable (stability < 0.5), does the answer warn about it?

usage: uv run scripts/evaluate_analyst.py answer      (qwen3:4b-instruct writes the answers)
       uv run scripts/evaluate_analyst.py judge       (qwen3:8b checks the cited claims)
       uv run scripts/evaluate_analyst.py report
"""
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aml_graphrag.adapters.ollama import OllamaEmbedder, OllamaLLM
from aml_graphrag.application.analyst import answer, typology_query
from aml_graphrag.application.answering import VectorIndex, count
from aml_graphrag.application.reports import report_text
from aml_graphrag.domain.claims import check
from aml_graphrag.domain.context import pack

SUPPORT_SCHEMA = {"type": "object", "properties": {
    "verdict": {"type": "string", "enum": ["supported", "partly", "not_supported"]}}, "required": ["verdict"]}

SUPPORT_PROMPT = """Does the evidence support the claim? Answer "supported" if everything the claim states is in the
evidence, "partly" if some of it is, "not_supported" otherwise.

Claim: {claim}

Evidence:
{evidence}"""


def sources_for(alert, condition, chunk_index, report_index, budget):
    query = typology_query(alert["shapes"])
    if condition == "text_rag":
        found = pack(chunk_index.search(query), budget, count)
        return {f"S{n}": t for n, t in enumerate(found, 1)}
    if condition == "graph":
        found = pack(report_index.search(query), budget, count)
        return {f"C{n}": t for n, t in enumerate(found, 1)}
    return None


def run_answers(args):
    alerts = json.loads(args.alerts.read_text())
    embedder = OllamaEmbedder()
    chunks = [json.loads(line) for line in (args.index / "chunks.jsonl").open()]
    import numpy as np
    chunk_index = VectorIndex(embedder, [f"({c['title']}) {c['text']}" for c in chunks],
                              np.load(args.index / "chunk_vectors.npy"))
    reports = [json.loads(line) for line in (args.index / "reports.jsonl").open()]
    report_index = VectorIndex(embedder, [report_text(r) for r in reports if r["level"] == 1])
    llm = OllamaLLM(args.model, num_ctx=args.num_ctx)

    def run(job):
        alert, condition = job
        facts = {"rf_score": alert["facts"]["rf_score"]} if condition != "graph" else alert["facts"]
        sources = sources_for(alert, condition, chunk_index, report_index, args.budget)
        reply = answer(llm, alert["tx"], facts, sources)
        return {"tx": alert["tx"], "condition": condition, **reply}

    log = args.out / "answers.jsonl"
    done = {(a["tx"], a["condition"]) for a in map(json.loads, log.open())} if log.exists() else set()
    todo = [(a, c) for a in alerts for c in ("no_graph", "text_rag", "graph") if (a["tx"], c) not in done]
    print(f"{len(todo)} answers to write", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        for n, row in enumerate(pool.map(run, todo), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 30 == 0:
                print(f"  {n}/{len(todo)}", flush=True)


def run_judge(args):
    answers = [json.loads(line) for line in (args.out / "answers.jsonl").open()]
    llm = OllamaLLM(args.judge, num_ctx=args.num_ctx)
    log = args.out / "support.jsonl"
    done = {(s["tx"], s["condition"], s["claim"]) for s in map(json.loads, log.open())} if log.exists() else set()
    todo = []
    for a in answers:
        for n, claim in enumerate(a["claims"]):
            if check(claim, a["evidence"]) in ("ok", "number_mismatch") and (a["tx"], a["condition"], n) not in done:
                todo.append((a, n, claim))

    def run(job):
        a, n, claim = job
        evidence = "\n".join(f"[{c}] {a['evidence'][c]}" for c in claim["cites"])
        verdict = llm.json(SUPPORT_PROMPT.format(claim=claim["text"], evidence=evidence), SUPPORT_SCHEMA)["verdict"]
        return {"tx": a["tx"], "condition": a["condition"], "claim": n, "verdict": verdict}

    print(f"{len(todo)} cited claims to check", flush=True)
    with log.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        for row in pool.map(run, todo):
            f.write(json.dumps(row) + "\n")
            f.flush()


def run_report(args):
    alerts = {a["tx"]: a for a in json.loads(args.alerts.read_text())}
    answers = [json.loads(line) for line in (args.out / "answers.jsonl").open()]
    support = {(s["tx"], s["condition"], s["claim"]): s["verdict"] for s in map(json.loads, (args.out / "support.jsonl").open())}
    rows = []
    for condition in ("no_graph", "text_rag", "graph"):
        subset = [a for a in answers if a["condition"] == condition]
        claims = [(a, n, c) for a in subset for n, c in enumerate(a["claims"])]
        status = [check(c, a["evidence"]) for a, n, c in claims]
        verdict = [support.get((a["tx"], condition, n)) for a, n, c in claims]
        grounded = sum(s == "ok" and v == "supported" for s, v in zip(status, verdict))
        unstable = [a for a in subset if alerts[a["tx"]]["explanation"]["stability"] < 0.5]
        warned = sum("stab" in json.dumps(a["caveats"] + [c["text"] for c in a["claims"]]).lower() for a in unstable)
        n = len(claims)
        rows.append({
            "condition": condition, "answers": len(subset), "claims": n, "claims_per_answer": round(n / len(subset), 2),
            "no_citation": round(status.count("no_citation") / n, 3),
            "bad_citation": round(status.count("bad_citation") / n, 3),
            "number_mismatch": round(status.count("number_mismatch") / n, 3),
            "judged_not_supported": round(sum(v == "not_supported" for v in verdict) / n, 3),
            "grounded": round(grounded / n, 3),
            "warns_when_unstable": f"{warned}/{len(unstable)}" if condition == "graph" else "",
        })
    with open(args.out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["answer", "judge", "report"])
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--judge", default="qwen3:8b")
    parser.add_argument("--budget", type=int, default=1500, help="tokens of background sources")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--num-ctx", type=int, default=15360)
    parser.add_argument("--alerts", type=Path, default=Path("data/alerts.json"))
    parser.add_argument("--index", type=Path, default=Path("data/index"))
    parser.add_argument("--out", type=Path, default=Path("results/analyst"))
    args = parser.parse_args()
    {"answer": run_answers, "judge": run_judge, "report": run_report}[args.step](args)
