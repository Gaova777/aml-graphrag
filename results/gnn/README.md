# The transaction side: who scores the alerts, and can the graph model explain them?

Elliptic: 203,769 Bitcoin transactions in 49 timesteps (about two weeks each), 234,355 money-flow edges,
4,545 labelled illicit, 42,019 licit, the rest unknown. Split as in Weber et al. (2019): train on steps 1–34
(1–30 to fit, 31–34 to pick the epoch and the threshold), test on 35–49. Only labelled transactions are scored;
unknown ones stay in the graph and pass messages.

```
uv run --group gnn scripts/train_gnn.py         # metrics.json, per_timestep.csv
uv run --group gnn scripts/explain_alerts.py    # explanations.csv
```

## First, does our setup match the paper?

Weber et al.'s random forest on all features, trained on steps 1–34 and tested on 35–49:

| | precision | recall | F1 (illicit) |
|---|---:|---:|---:|
| Weber et al. 2019, Table 3 | 0.971 | 0.675 | 0.788 |
| this repo | 0.936 | 0.724 | 0.816 |

Close enough to trust the rest.

## Random forest vs. GraphSAGE

Test set (steps 35–49, 16,670 labelled transactions, 1,083 illicit). Three seeds for GraphSAGE.

| | PR-AUC | precision@100 | F1 at the validation threshold | ROC-AUC |
|---|---:|---:|---:|---:|
| GraphSAGE, the thesis setup (hears only from the transactions that funded it) | 0.40–0.43 | 0.53–0.65 | 0.48–0.49 | 0.86–0.87 |
| GraphSAGE, both directions | 0.48–0.53 | 0.63–0.68 | 0.56–0.59 | 0.86–0.87 |
| Random forest (Weber et al.'s settings) | **0.77** | **1.00** | **0.81** | 0.90 |

([`metrics.json`](metrics.json))

- **The graph model loses**, as it did in the Elliptic paper (their GCN lost to their random forest). The 165 features
  already include aggregates of each transaction's neighbors, so message passing has little left to add.
- **Letting messages go both ways helps** (+0.08 PR-AUC): a transaction also learns from where its outputs went.
- **ROC-AUC hides it.** All three models score 0.86–0.90 on ROC-AUC, while the top of their rankings is very different
  (precision@100 from 0.53 to 1.00). With 6.5% illicit, the thesis' lesson holds: look at PR-AUC and precision@k.
- So in the product **the random forest scores the alerts**, and the graph model is kept for what only it can do:
  pointing at the neighbors that pushed its own score.

## Every model breaks at step 43

Illicit F1 per test timestep ([`per_timestep.csv`](per_timestep.csv)):

| step | 35 | 36 | 37 | 38 | 39 | 40 | 41 | 42 | **43** | 44 | 45 | 46 | 47 | 48 | 49 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random forest | .97 | .94 | .73 | .94 | .92 | .78 | .94 | .88 | **.00** | .06 | .00 | .40 | .00 | .00 | .00 |
| GraphSAGE (both) | .81 | .34 | .54 | .63 | .59 | .55 | .56 | .76 | **.00** | .04 | .00 | .00 | .00 | .00 | .03 |

Weber et al. saw the same thing and tie it to the shutdown of a large dark market around step 43: the illicit
transactions after it don't look like the ones before. A model trained on the past can't see a new pattern;
in production this is the moment to retrain and to watch drift, not to trust the score.

## The alert queue and its explanations

100 transactions picked at random among the 4,645 test-period transactions above the alert threshold. 80 of them
were never labelled by Elliptic and 20 are labelled illicit, which is what an analyst's queue looks like. For each,
GNNExplainer runs 5 times with different seeds and we keep the 5 edges the runs agree on most
([`explanations.csv`](explanations.csv)).

- **Stability.** The mean Jaccard of the top-5 edges between runs has a median of 0.37. **56 of 100 explanations
  are unstable** (below 0.5): run it again and it points at other edges.
- **The graph model agrees with the alert on 40 of the 100** (17 of the 20 labelled illicit are among them).
  - On those 40, the explanation of "why illicit" is the least stable (median 0.29), and removing its 5 edges lowers
    the illicit score by 0.1 or more for only 7. The model calls them illicit because of their own features; the
    highlighted edges are mostly decoration.
  - On the 60 it doesn't flag, removing the highlighted edges *raises* the illicit score by 0.1 or more for 36. There
    the explanation is real: these neighbors are what make the transaction look normal to the graph model.
- **Shapes.** 56 alerts are the end of a chain (one input, outputs not spent within the timestep), 18 are sources
  (no inputs in the graph), and only one each is a fan-in or part of a peeling chain.

This is the thesis' finding seen in a product: **a stable explanation isn't necessarily a faithful one, and a
plausible one may be neither.** So the assistant is given the stability and the fidelity numbers and is asked to
say so, instead of presenting the highlighted edges as "the reason".
