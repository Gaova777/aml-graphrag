"""Head-to-head judging (section 4.1.4): for each question and pair of conditions, a different model judges the two
answers on comprehensiveness, diversity, empowerment and directness, once in each order.

usage: uv run scripts/judge.py --pairs C0:SS C1:SS C1:C0
"""
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aml_graphrag.adapters.ollama import OllamaLLM
from aml_graphrag.application.evaluation import CRITERIA, judge
from aml_graphrag.domain.judging import position_bias, score_pair, win_rates


def main(args):
    questions = {q["id"]: q["question"] for q in json.loads(args.questions.read_text())}
    answers = {(a["question"], a["condition"]): a["answer"] for a in map(json.loads, (args.out / "answers.jsonl").open())}
    llm = OllamaLLM(args.model, num_ctx=args.num_ctx)
    log = args.out / f"judgements_{args.model.replace(':', '_')}.jsonl"
    done = {(j["question"], j["a"], j["b"], j["order"]) for j in map(json.loads, log.open())} if log.exists() else set()

    pairs = [tuple(p.split(":")) for p in args.pairs]
    todo = []
    for a, b in pairs:
        for qid in questions:
            if (qid, a) in answers and (qid, b) in answers:
                todo += [(qid, a, b, order) for order in ("ab", "ba") if (qid, a, b, order) not in done]

    def run(job):
        qid, a, b, order = job
        first, second = (answers[qid, a], answers[qid, b]) if order == "ab" else (answers[qid, b], answers[qid, a])
        return {"question": qid, "a": a, "b": b, "order": order, "verdict": judge(llm, questions[qid], first, second)}

    print(f"{len(todo)} judgements to make", flush=True)
    with log.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        for n, row in enumerate(pool.map(run, todo), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"  {n}/{len(todo)}", flush=True)

    judgements = {(j["question"], j["a"], j["b"], j["order"]): j["verdict"] for j in map(json.loads, log.open())}
    rows = []
    for a, b in pairs:
        qids = [q for q in questions if (q, a, b, "ab") in judgements and (q, a, b, "ba") in judgements]
        first = [judgements[q, a, b, "ab"] for q in qids]
        swapped = [judgements[q, a, b, "ba"] for q in qids]
        rates = win_rates([score_pair(f, s) for f, s in zip(first, swapped)])
        rows.append({"a": a, "b": b, "questions": len(qids), **{c: round(rates[c], 3) for c in CRITERIA},
                     "position_flips": round(position_bias(first, swapped), 3)})
        print(f"  {a} vs {b} ({len(qids)} questions): " + ", ".join(f"{c} {rates[c]:.0%}" for c in CRITERIA)
              + f"; verdict flipped with the order {rows[-1]['position_flips']:.0%} of the time")
    with open(args.out / f"win_rates_{args.model.replace(':', '_')}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=["C0:SS", "C1:SS", "C1:C0"])
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--num-ctx", type=int, default=15360)
    parser.add_argument("--questions", type=Path, default=Path("results/graphrag/questions.json"))
    parser.add_argument("--out", type=Path, default=Path("results/graphrag"))
    main(parser.parse_args())
