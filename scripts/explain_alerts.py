"""Build the alert queue: test-period transactions above the alert threshold, with their graph facts and a
GNNExplainer explanation (5 seeds, stability, fidelity).

Writes data/alerts.json (derived from Elliptic, so not committed) and results/gnn/explanations.csv.

usage: uv run --group gnn scripts/explain_alerts.py --n 100
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.utils import to_undirected

from aml_graphrag.adapters.elliptic import load
from aml_graphrag.application.analyst import transaction_facts
from aml_graphrag.application.explain import explain
from aml_graphrag.application.gnn import GraphSAGE, SPLIT, standardize
from aml_graphrag.domain.patterns import shape


def neighborhood(i, top_edges, succ, pred, d, rf, max_nodes):
    """Nodes and money-flow edges within two steps of the alert, for the console to draw. The explanation's edges
    are always kept; past max_nodes, the rest of the two-step ring is dropped."""
    keep = [i]
    one = list(dict.fromkeys(succ[i] + pred[i]))
    keep += one
    index = {int(d["tx_id"][k]): k for k in keep}
    for e in top_edges:
        for t in (e["source"], e["target"]):
            if t not in index:
                index[t] = int(np.flatnonzero(d["tx_id"] == t)[0])
    for k in one:
        for m in succ[k] + pred[k]:
            if len(index) >= max_nodes:
                break
            index.setdefault(int(d["tx_id"][m]), m)
    nodes = set(index.values())
    highlighted = {(e["source"], e["target"]) for e in top_edges}
    edges = [{"source": int(d["tx_id"][a]), "target": int(d["tx_id"][b]),
              "explanation": (int(d["tx_id"][a]), int(d["tx_id"][b])) in highlighted}
             for a in nodes for b in succ[a] if b in nodes]
    return {"nodes": [{"tx": int(d["tx_id"][k]), "rf": round(float(rf[k]), 3), "alert": k == i} for k in nodes],
            "edges": edges}


def main(args):
    d = load(args.raw)
    rf, gnn = np.load(args.models / "scores.npy")
    metrics = json.loads(args.metrics.read_text())
    thresholds = {"random_forest": metrics["random_forest"]["threshold"],
                  "graphsage": metrics["graphsage_both"][0]["threshold"]}
    checkpoint = torch.load(args.models / "graphsage.pt")
    x = torch.tensor(standardize(d["x"], d["timestep"]))
    edges = to_undirected(torch.tensor(d["edge_index"]))
    model = GraphSAGE(x.shape[1])
    model.load_state_dict(checkpoint["state"])

    succ, pred = defaultdict(list), defaultdict(list)
    for a, b in d["edge_index"].T.tolist():
        succ[a].append(b)
        pred[b].append(a)

    test = (d["timestep"] >= SPLIT["test"][0]) & (d["timestep"] <= SPLIT["test"][1])
    flagged = np.flatnonzero(test & (rf >= thresholds["random_forest"]))
    chosen = np.random.default_rng(args.seed).choice(flagged, size=min(args.n, len(flagged)), replace=False)
    print(f"{len(flagged)} test transactions above the alert threshold; explaining {len(chosen)}", flush=True)

    alerts, rows, graphs = [], [], {}
    for n, i in enumerate(sorted(chosen, key=lambda i: -rf[i]), 1):
        i = int(i)
        one = set(succ[i]) | set(pred[i])
        two = {m for k in one for m in succ[k] + pred[k]} - one - {i}
        neighbors = {"inputs": len(pred[i]), "outputs": len(succ[i]), "n_1hop": len(one), "n_2hop": len(two),
                     "risky_1hop": int(sum(rf[k] >= thresholds["random_forest"] for k in one)),
                     "risky_2hop": int(sum(rf[k] >= thresholds["random_forest"] for k in two))}
        exp = explain(model, x, edges, i, seeds=range(args.runs), k=args.k)
        for e in exp["edges"]:
            a, b = e["pair"]
            a, b = (a, b) if b in succ[a] else (b, a)  # show the edge the way the money flowed
            touches = "touches the alert directly" if i in (a, b) else "elsewhere in its neighborhood"
            e["source"], e["target"] = int(d["tx_id"][a]), int(d["tx_id"][b])
            e["text"] = (f"Edge {e['source']} -> {e['target']} ({touches}); alert scores {rf[a]:.2f} -> {rf[b]:.2f}; "
                         f"importance {e['weight']:.2f}.")
            del e["pair"]
        shapes = shape(i, succ, pred)
        facts = transaction_facts(i, d, {"random_forest": rf, "graphsage": gnn}, thresholds, neighbors, shapes, exp)
        alerts.append({"tx": int(d["tx_id"][i]), "index": i, "timestep": int(d["timestep"][i]),
                       "label": int(d["y"][i]), "rf": float(rf[i]), "gnn": float(gnn[i]), "shapes": shapes,
                       "neighbors": neighbors, "explanation": exp, "facts": facts})
        rows.append({"tx": alerts[-1]["tx"], "timestep": alerts[-1]["timestep"], "label": alerts[-1]["label"],
                     "rf": round(float(rf[i]), 4), "gnn": round(float(gnn[i]), 4), "shapes": "|".join(shapes),
                     "neighborhood": exp["neighborhood"], "stability": round(exp["stability"], 4),
                     "p_full": round(exp["p_full"], 4), "p_without_top": round(exp["p_without_top"], 4),
                     "p_only_top": round(exp["p_only_top"], 4), "p_no_edges": round(exp["p_no_edges"], 4)})
        graphs[alerts[-1]["tx"]] = neighborhood(i, exp["edges"], succ, pred, d, rf, args.max_nodes)
        if n % 10 == 0:
            print(f"  {n}/{len(chosen)}", flush=True)

    args.alerts.write_text(json.dumps(alerts, indent=1))
    (args.alerts.parent / "neighborhoods.json").write_text(json.dumps(graphs))
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "explanations.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    s = np.array([r["stability"] for r in rows])
    flagged = np.array([r["gnn"] >= thresholds["graphsage"] for r in rows])
    # fidelity for the class the graph model predicted: how much its probability falls without the top edges
    sign = np.where(flagged, 1, -1)
    fall = sign * np.array([r["p_full"] - r["p_without_top"] for r in rows])
    print(f"stability: median {np.median(s):.2f}, below 0.5 for {np.mean(s < 0.5):.0%}")
    for name, m in (("graph model agrees", flagged), ("graph model disagrees", ~flagged)):
        print(f"  {name}: {m.sum()} alerts, stability median {np.median(s[m]):.2f}, removing the top edges lowers the "
              f"predicted class by 0.1+ for {(fall[m] >= 0.1).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=120)
    parser.add_argument("--raw", type=Path, default=Path("data/elliptic/raw"))
    parser.add_argument("--models", type=Path, default=Path("data/models"))
    parser.add_argument("--metrics", type=Path, default=Path("results/gnn/metrics.json"))
    parser.add_argument("--alerts", type=Path, default=Path("data/alerts.json"))
    parser.add_argument("--out", type=Path, default=Path("results/gnn"))
    main(parser.parse_args())
