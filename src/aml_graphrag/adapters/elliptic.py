"""The Elliptic Bitcoin transaction graph (Weber et al., 2019) from its three CSV files.

Download them from https://www.kaggle.com/datasets/ellipticco/elliptic-data-set (or let PyTorch Geometric's
EllipticBitcoinDataset fetch them) into data/elliptic/raw/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

LICIT, ILLICIT, UNKNOWN = 0, 1, -1
EXPECTED = {"illicit": 4545, "licit": 42019, "unknown": 157205}


def load(raw: Path = Path("data/elliptic/raw")) -> dict:
    features = pd.read_csv(raw / "elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(raw / "elliptic_txs_classes.csv")
    edges = pd.read_csv(raw / "elliptic_txs_edgelist.csv")

    tx_id = features[0].to_numpy()
    index = pd.Series(np.arange(len(tx_id)), index=tx_id)
    # raw classes: '1' illicit, '2' licit, 'unknown'
    label = classes.set_index("txId")["class"].map({"1": ILLICIT, "2": LICIT, "unknown": UNKNOWN})
    y = label.loc[tx_id].to_numpy()
    counts = {"illicit": int((y == ILLICIT).sum()), "licit": int((y == LICIT).sum()), "unknown": int((y == UNKNOWN).sum())}
    if counts != EXPECTED:
        raise ValueError(f"unexpected label counts {counts}, expected {EXPECTED}")
    return {
        "tx_id": tx_id,
        "timestep": features[1].to_numpy(),
        # column 1 is the timestep: as a feature it would only let the model memorize periods
        "x": features.iloc[:, 2:].to_numpy(dtype=np.float32),
        "y": y,
        "edge_index": np.stack([index.loc[edges["txId1"]].to_numpy(), index.loc[edges["txId2"]].to_numpy()]),
    }
