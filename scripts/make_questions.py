"""The paper's question generation (section 4.1.2): users -> tasks -> global questions, from a description of the
corpus only, so no question is written with one document in mind.

usage: uv run scripts/make_questions.py --users 5 --tasks 5 --per-task 2
"""
import argparse
import json
from pathlib import Path

from aml_graphrag.adapters.ollama import OllamaLLM
from aml_graphrag.application.evaluation import questions_for, users_and_tasks

DESCRIPTION = ("106 English Wikipedia articles about money laundering and financial crime: laundering techniques, "
               "anti-money laundering laws and agencies, cryptocurrency crime (mixers, darknet markets, ransomware, "
               "exchanges), and notable cases involving banks, crypto exchanges, drug cartels and leaked documents.")


def main(args):
    llm = OllamaLLM(args.model, temperature=0.7)
    users = users_and_tasks(llm, DESCRIPTION, args.users, args.tasks)
    questions = []
    for u in users[:args.users]:
        for task in u["tasks"][:args.tasks]:
            for q in questions_for(llm, DESCRIPTION, u["user"], task, args.per_task):
                questions.append({"id": len(questions), "user": u["user"], "task": task, "question": q})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(questions, indent=1) + "\n")
    print(f"{len(questions)} questions")
    for q in questions[:: max(1, len(questions) // 8)]:
        print(" -", q["question"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--per-task", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("results/graphrag/questions.json"))
    main(parser.parse_args())
