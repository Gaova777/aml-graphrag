"""GraphSAGE scoring of Elliptic transactions, set up as in the thesis (class weighting, early stopping on PR-AUC)."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import nn
from torch_geometric.nn import SAGEConv

# Weber et al.'s split; the last train steps are held out to pick the epoch and the threshold
SPLIT = {"train": (1, 30), "val": (31, 34), "test": (35, 49)}


def masks(timestep, y):
    """Labelled nodes of each split. Unknown nodes stay in the graph and pass messages, but are never scored."""
    return {name: (timestep >= lo) & (timestep <= hi) & (y >= 0) for name, (lo, hi) in SPLIT.items()}


def standardize(x, timestep):
    train = (timestep >= SPLIT["train"][0]) & (timestep <= SPLIT["train"][1])
    mean, std = x[train].mean(0), x[train].std(0) + 1e-6
    return (x - mean) / std


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden=64, layers=3, dropout=0.455):
        super().__init__()
        sizes = [in_channels] + [hidden] * (layers - 1) + [2]
        self.convs = nn.ModuleList(SAGEConv(a, b) for a, b in zip(sizes, sizes[1:]))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.dropout(F.relu(conv(x, edge_index)), self.dropout, self.training)
        return self.convs[-1](x, edge_index)


def illicit_probability(model, x, edge_index):
    model.eval()
    with torch.no_grad():
        return model(x, edge_index).softmax(-1)[:, 1]


def train(x, edge_index, y, split, seed=0, epochs=300, patience=30, lr=0.0036, weight_decay=7.3e-5):
    """Hyperparameters are the thesis' Optuna best for GraphSAGE; returns the model at its best validation PR-AUC."""
    torch.manual_seed(seed)
    model = GraphSAGE(x.shape[1]).to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_y = y[split["train"]]
    weight = torch.tensor([1.0, (train_y == 0).sum().item() / (train_y == 1).sum().item()], device=x.device)
    best, best_state, waited = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, edge_index)[split["train"]], train_y, weight=weight)
        loss.backward()
        opt.step()
        p = illicit_probability(model, x, edge_index)[split["val"]].cpu().numpy()
        score = average_precision_score(y[split["val"]].cpu().numpy(), p)
        if score > best:
            best, best_state, waited = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


def best_f1_threshold(y_true, p):
    """The threshold with the highest illicit F1 (picked on validation, then frozen)."""
    order = np.argsort(-p)
    tp = np.cumsum(y_true[order] == 1)
    precision, recall = tp / np.arange(1, len(p) + 1), tp / max(tp[-1], 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(p[order][f1.argmax()])
