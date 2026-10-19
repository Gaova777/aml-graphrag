"""Which edges made the graph model flag a transaction, how stable that answer is, and whether it's faithful."""
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

from aml_graphrag.domain.stability import consensus, stability


def explain(model, x, edge_index, node: int, seeds=range(5), k: int = 5, hops: int = 3, epochs: int = 200) -> dict:
    """GNNExplainer once per seed on the node's computation graph.

    stability: mean pairwise Jaccard of the top-k edges across seeds (the thesis' measure).
    fidelity: the model's illicit probability with the full neighborhood, without the k consensus edges, and with
    only them. If removing them doesn't move the score, they weren't the reason, however stable they are.
    """
    subset, sub_edges, center, _ = k_hop_subgraph(node, hops, edge_index, relabel_nodes=True)
    sub_x, center = x[subset], int(center)
    explainer = Explainer(
        model=model, algorithm=GNNExplainer(epochs=epochs, lr=0.01), explanation_type="model",
        edge_mask_type="object", node_mask_type=None,
        model_config={"mode": "multiclass_classification", "task_level": "node", "return_type": "raw"},
    )
    pairs = [tuple(sorted((int(subset[a]), int(subset[b])))) for a, b in sub_edges.t().tolist()]
    runs = []
    for seed in seeds:
        torch.manual_seed(seed)
        mask = explainer(sub_x, sub_edges, index=center).edge_mask.tolist()
        weights = {}
        for pair, w in zip(pairs, mask):  # an undirected edge is two directed copies: keep the stronger
            weights[pair] = max(weights.get(pair, 0.0), w)
        runs.append(weights)
    top = consensus(runs, k)
    mean = {p: sum(r[p] for r in runs) / len(runs) for p in top}

    def prob(keep):
        model.eval()
        with torch.no_grad():
            return model(sub_x, sub_edges[:, keep]).softmax(-1)[center, 1].item()

    in_top = torch.tensor([p in set(top) for p in pairs])
    return {
        "edges": [{"pair": p, "weight": mean[p]} for p in top],
        "stability": stability(runs, k),
        "neighborhood": len(subset),
        "p_full": prob(torch.ones(len(pairs), dtype=torch.bool)),
        "p_without_top": prob(~in_top),
        "p_only_top": prob(in_top),
    }
