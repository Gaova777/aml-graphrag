"""Does grounding stop the assistant from making things up? The same question on every alert, three conditions.

For each claim: does it cite something it was given (code), do its numbers match what it cites (code), and does
the cited evidence actually support it (a different model, qwen3:8b, as judge). Also: when the explanation is
unstable (stability < 0.5), does the answer warn about it?

usage: uv run scripts/evaluate_analyst.py answer      (qwen3:4b-instruct writes the answers)
       uv run scripts/evaluate_analyst.py judge       (qwen3:8b checks the cited claims)
       uv run scripts/evaluate_analyst.py report
       uv run scripts/evaluate_analyst.py uncited     (control: free answers, no citation rule)
"""
import argparse
import os
import re
import shutil
import subprocess
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


FREE_PROMPT = """You assist an anti-money laundering analyst reviewing an alert on a Bitcoin transaction.

{facts}

Question: {question}"""

INVENTED_SCHEMA = {"type": "object", "properties": {
    "invents": {"type": "boolean"}, "example": {"type": "string"}}, "required": ["invents", "example"]}

INVENTED_PROMPT = """Below is the only information an assistant had about a Bitcoin transaction, and its answer.
Note that the model's features are anonymized: nobody knows what they measure.

Does the answer present as fact, or as the likely reason for the alert, anything specific about this transaction or
about what the model's features measure (amounts, counterparties, addresses, mixers, darknet links, transaction size or
frequency, patterns) that the information does not give? Generic caution ("a high score is not proof") doesn't count.
Answer invents=true or false, and quote one example (empty if false).

Information:
{facts}

Answer:
{answer}"""


def run_uncited(args):
    """Control: the no-graph condition without the rule to cite. How often does a free answer invent specifics?"""
    from aml_graphrag.application.analyst import QUESTION, fact_lines

    alerts = json.loads(args.alerts.read_text())[: args.uncited]
    log = args.out / "uncited.jsonl"
    if log.exists():  # answers already written: only judge again
        replies = [json.loads(line)["answer"] for line in log.open()]
    else:
        llm = OllamaLLM(args.model, num_ctx=args.num_ctx)
        with ThreadPoolExecutor(args.workers) as pool:
            replies = list(pool.map(lambda a: llm.text(FREE_PROMPT.format(
                facts=fact_lines({"rf_score": a["facts"]["rf_score"]}), question=QUESTION.format(tx=a["tx"]))), alerts))
        subprocess.run([shutil.which("ollama") or os.path.expanduser("~/.local/opt/ollama/bin/ollama"),
                        "stop", args.model], check=False)  # free the GPU for the judge
    judge = OllamaLLM(args.judge, num_ctx=args.num_ctx)
    with ThreadPoolExecutor(args.workers) as pool:
        verdicts = list(pool.map(lambda ar: judge.json(INVENTED_PROMPT.format(
            facts=fact_lines({"rf_score": ar[0]["facts"]["rf_score"]}), answer=ar[1]), INVENTED_SCHEMA),
            zip(alerts, replies)))
    rows = [{"tx": a["tx"], "answer": r, **v} for a, r, v in zip(alerts, replies, verdicts)]
    with open(log, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{sum(r['invents'] for r in rows)} of {len(rows)} free answers state specifics they weren't given")


# a warning about the explanation, in the answer's own words
UNSTABLE = re.compile(r"unstable|instab|not stable|low stability|inconsistent|different edges", re.I)
DISAGREES = re.compile(r"(graph model|gnn|graphsage)[^.]{0,80}(does not|doesn't|did not|not) (flag|agree|consider)", re.I)


def run_report(args):
    alerts = {a["tx"]: a for a in json.loads(args.alerts.read_text())}
    gnn_threshold = json.loads(args.metrics.read_text())["graphsage_both"][0]["threshold"]
    answers = [json.loads(line) for line in (args.out / "answers.jsonl").open()]
    support = {(s["tx"], s["condition"], s["claim"]): s["verdict"] for s in map(json.loads, (args.out / "support.jsonl").open())}
    rows = []
    for condition in ("no_graph", "text_rag", "graph"):
        subset = [a for a in answers if a["condition"] == condition]
        claims = [(a, n, c) for a in subset for n, c in enumerate(a["claims"])]
        status = [check(c, a["evidence"]) for a, n, c in claims]
        verdict = [support.get((a["tx"], condition, n)) for a, n, c in claims]
        grounded = sum(s == "ok" and v == "supported" for s, v in zip(status, verdict))

        def says(pattern, a):
            return bool(pattern.search(" ".join([a["summary"]] + a["caveats"] + [c["text"] for c in a["claims"]])))

        unstable = [a for a in subset if alerts[a["tx"]]["explanation"]["stability"] < 0.5]
        stable = [a for a in subset if alerts[a["tx"]]["explanation"]["stability"] >= 0.5]
        disagree = [a for a in subset if alerts[a["tx"]]["gnn"] < gnn_threshold]
        n = len(claims)
        rows.append({
            "condition": condition, "answers": len(subset), "claims": n, "claims_per_answer": round(n / len(subset), 2),
            "items_cited_per_answer": round(sum(len({i for c in a["claims"] for i in c["cites"]}) for a in subset) / len(subset), 2),
            "no_citation": round(status.count("no_citation") / n, 3),
            "bad_citation": round(status.count("bad_citation") / n, 3),
            "number_mismatch": round(status.count("number_mismatch") / n, 3),
            "judged_partly": round(sum(v == "partly" for v in verdict) / n, 3),
            "judged_not_supported": round(sum(v == "not_supported" for v in verdict) / n, 3),
            "grounded": round(grounded / n, 3),
            "warns_when_unstable": f"{sum(says(UNSTABLE, a) for a in unstable)}/{len(unstable)}",
            "warns_when_stable": f"{sum(says(UNSTABLE, a) for a in stable)}/{len(stable)}",
            "says_graph_model_disagrees": f"{sum(says(DISAGREES, a) for a in disagree)}/{len(disagree)}",
        })
    with open(args.out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["answer", "judge", "report", "uncited"])
    parser.add_argument("--uncited", type=int, default=30, help="alerts for the no-citation control")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--judge", default="qwen3:8b")
    parser.add_argument("--budget", type=int, default=1500, help="tokens of background sources")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--num-ctx", type=int, default=15360)
    parser.add_argument("--alerts", type=Path, default=Path("data/alerts.json"))
    parser.add_argument("--index", type=Path, default=Path("data/index"))
    parser.add_argument("--metrics", type=Path, default=Path("results/gnn/metrics.json"))
    parser.add_argument("--out", type=Path, default=Path("results/analyst"))
    args = parser.parse_args()
    {"answer": run_answers, "judge": run_judge, "report": run_report, "uncited": run_uncited}[args.step](args)
