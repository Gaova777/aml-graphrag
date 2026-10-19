"""Score Elliptic transactions with GraphSAGE and with Weber et al.'s random forest, on the same temporal split.

Also reproduces the paper's own setting (train on steps 1-34, test on 35-49, random forest on all features) to check
our numbers against their Table 3, and the per-timestep illicit F1 that shows the dark market shutdown at step 43.

usage: uv run --group gnn scripts/train_gnn.py
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from aml_graphrag.adapters.elliptic import load
from aml_graphrag.application.gnn import best_f1_threshold, illicit_probability, masks, standardize, train
from aml_graphrag.domain.metrics import f1_at, precision_at_k


def report(y, p, threshold):
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "precision_at_100": precision_at_k(list(y), list(p), 100), **f1_at(list(y), list(p), threshold)}


def main(args):
    d = load(args.raw)
    x = standardize(d["x"], d["timestep"])
    split = masks(d["timestep"], d["y"])
    y = d["y"]
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    xt, et = torch.tensor(x, device=device), torch.tensor(d["edge_index"], device=device)
    yt = torch.tensor(y, device=device)
    st = {k: torch.tensor(v, device=device) for k, v in split.items()}

    results, per_step = {}, []
    sage_runs = []
    for seed in range(args.seeds):
        model, val_pr = train(xt, et, yt, st, seed=seed)
        p = illicit_probability(model, xt, et).cpu().numpy()
        threshold = best_f1_threshold(y[split["val"]], p[split["val"]])
        sage_runs.append({"seed": seed, "threshold": threshold, "val": report(y[split["val"]], p[split["val"]], threshold),
                          "test": report(y[split["test"]], p[split["test"]], threshold)})
        print(f"  GraphSAGE seed {seed}: val PR-AUC {val_pr:.3f}, test PR-AUC {sage_runs[-1]['test']['pr_auc']:.3f}", flush=True)
        if seed == 0:
            args.models.mkdir(parents=True, exist_ok=True)
            torch.save({"state": model.state_dict(), "threshold": threshold}, args.models / "graphsage.pt")
            np.save(args.models / "graphsage_scores.npy", p)
            sage_p, sage_threshold = p, threshold
    results["graphsage"] = sage_runs

    train_idx = split["train"]
    rf = RandomForestClassifier(n_estimators=50, max_features=50, n_jobs=-1, random_state=0)  # Weber et al.'s settings
    rf.fit(x[train_idx], y[train_idx])
    rf_p = rf.predict_proba(x)[:, 1]
    rf_threshold = best_f1_threshold(y[split["val"]], rf_p[split["val"]])
    results["random_forest"] = {"threshold": rf_threshold, "val": report(y[split["val"]], rf_p[split["val"]], rf_threshold),
                                "test": report(y[split["test"]], rf_p[split["test"]], rf_threshold)}

    # the paper's exact setting: train 1-34, test 35-49, decision at 0.5
    paper_train = (d["timestep"] <= 34) & (y >= 0)
    rf34 = RandomForestClassifier(n_estimators=50, max_features=50, n_jobs=-1, random_state=0).fit(x[paper_train], y[paper_train])
    p34 = rf34.predict_proba(x[split["test"]])[:, 1]
    results["weber_reproduction"] = {"ours": f1_at(list(y[split["test"]]), list(p34), 0.5),
                                     "paper": {"precision": 0.971, "recall": 0.675, "f1": 0.788}}

    for step in range(35, 50):
        m = (d["timestep"] == step) & (y >= 0)
        per_step.append({"timestep": step, "illicit": int((y[m] == 1).sum()), "licit": int((y[m] == 0).sum()),
                         "graphsage_f1": f1_at(list(y[m]), list(sage_p[m]), sage_threshold)["f1"],
                         "random_forest_f1": f1_at(list(y[m]), list(rf_p[m]), rf_threshold)["f1"]})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=1) + "\n")
    with open(args.out / "per_timestep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_step[0]))
        writer.writeheader()
        writer.writerows(per_step)
    print(json.dumps({k: v["test"] for k, v in [("random_forest", results["random_forest"])]}, indent=1))
    print(json.dumps(results["weber_reproduction"], indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--raw", type=Path, default=Path("data/elliptic/raw"))
    parser.add_argument("--models", type=Path, default=Path("data/models"))
    parser.add_argument("--out", type=Path, default=Path("results/gnn"))
    main(parser.parse_args())
