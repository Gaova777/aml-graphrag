# Does the assistant make things up?

The GraphRAG paper judges how answers *look*; it says it didn't measure fabrication. This is that measurement, on the
question an analyst actually asks: *"Why was transaction X flagged as suspicious, and how much should I trust that?"*

Same 100 alerts (see [results/gnn](../gnn)), same model (qwen3:4b-instruct), three amounts of context:

- **no_graph:** only the alert score and its rank in the timestep.
- **text_rag:** the score, plus the Wikipedia passages closest to the transaction's shape (plain vector RAG).
- **graph:** the score, plus facts from the transaction graph (neighbors, shape, the graph model's score and whether
  it agrees), the GNNExplainer edges with their stability and fidelity, and the GraphRAG community reports that match
  the shape.

In all three the model has to answer with a list of claims, each citing the ids of the items that support it.

```
uv run scripts/evaluate_analyst.py answer     # answers.jsonl
uv run scripts/evaluate_analyst.py judge      # support.jsonl (qwen3:8b)
uv run scripts/evaluate_analyst.py report     # summary.csv
uv run scripts/evaluate_analyst.py uncited    # uncited.jsonl, the control without the citation rule
```

## How a claim is checked

1. **Code:** does it cite something, and only ids it was given? Does every number in it appear in what it cites
   (0.912, 0.91 and 91% count as the same)? See [`domain/claims.py`](../../src/aml_graphrag/domain/claims.py).
2. **A different model** (qwen3:8b) reads the claim and only the evidence it cites, and says supported, partly
   or not supported.

A claim is **grounded** if it passes the code checks and the judge says supported.

## Results

([`summary.csv`](summary.csv))

| | no_graph | text_rag | graph |
|---|---:|---:|---:|
| claims per answer | 2.4 | 3.0 | 3.0 |
| distinct items cited per answer | 1.0 | 3.0 | **4.6** |
| claims without a citation | 0% | 0% | 0% |
| citing something it wasn't given | 0% | 1.3% | 0% |
| numbers that don't match the source | 0.4% | 0% | 1.0% |
| judged only partly supported | 3.4% | 1.7% | 6.6% |
| judged not supported | 0.8% | 0.3% | 2.3% |
| **grounded** | **95.8%** | **96.7%** | **90.4%** |
| warns when the explanation is unstable (stability < 0.5) | – | – | **56 of 56** |
| warns although the explanation is stable | – | – | 17 of 44 |
| says the graph model doesn't flag it, when it doesn't | – | – | **57 of 60** |

And the control: the no_graph condition **without** the rule to cite, as a free answer, on the first 30 alerts
([`uncited.jsonl`](uncited.jsonl)):

| | |
|---|---:|
| free answers that state specifics they weren't given | **30 of 30** |

Almost all of them say what the model's features are ("transaction size, frequency, recipient patterns",
"darknet links"). Nobody knows that: Elliptic's features are anonymized. The judge flags all 30, and a plain keyword
search for those words finds them in all 30 too. With the same information and the citation rule, they appear in 2 of
the 100 no_graph answers.

## What I take from it

- **The citation rule is what stops the invention, not the graph.** Asked to cite, a 4B model with nothing but a
  score stays inside what it has: it repeats the score, and that's it. Without the rule it fills the gap with plausible,
  specific, invented reasons in every single answer. That's the failure an analyst can't see coming.
- **The graph is what makes the answer worth reading.** Only the graph condition tells the analyst something they
  don't already see on the screen: that the graph model disagrees (57 of 60 times it does), and that the explanation
  is unstable (all 56 times it is). It cites 4.6 different facts per answer, against one.
- **Richer evidence, subtler mistakes.** Grounded claims drop from ~96% to 90% in the graph condition. The errors are
  readings, not inventions: an edge described as "alert scores 0.24 → 1.00" (the scores of its two ends) became
  "the edge makes the score jump from 0.24 to 1.00"; "the score barely changes without these edges" when it rose.
  Their numbers match the source, so only the judge can catch them. That's also a lesson about wording the facts:
  "0.24 → 1.00" invites a causal reading.
- **It errs on the side of caution.** It also warns about instability in 17 of the 44 stable explanations. For an
  analyst that's the better mistake, but it's a mistake.

## Caveats

- One model, one prompt, 100 alerts. The judge is itself a small model and some of its "not supported" verdicts are
  wrong (it rejected "low stability (0.22)" when the fact said 0.22). Read the claims in
  [`answers.jsonl`](answers.jsonl) and the verdicts in [`support.jsonl`](support.jsonl).
- The warning and disagreement rates come from keyword patterns over the answer text
  ([`scripts/evaluate_analyst.py`](../../scripts/evaluate_analyst.py)); they can miss a paraphrase.
- The control has 30 alerts, not 100, and only the no_graph context.
