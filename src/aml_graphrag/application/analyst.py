"""The analyst assistant: why was this transaction flagged, and how much should I trust that?

Three ways to answer the same question, compared in scripts/evaluate_analyst.py:
  no_graph:  the model only gets the alert score
  text_rag:  the score plus encyclopedia passages about laundering (plain vector RAG)
  graph:     the score, facts computed from the transaction graph, the GNN explanation with its stability and
             fidelity, and the GraphRAG community reports that match the transaction's shape
Every claim must cite the id of the fact or source it comes from.
"""
from aml_graphrag.ports.llm import LLM

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"},
            "cites": {"type": "array", "items": {"type": "string"}},
        }, "required": ["text", "cites"]}},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "claims", "caveats"],
}

ANSWER_PROMPT = """You assist an anti-money laundering analyst reviewing an alert on a Bitcoin transaction from the
Elliptic dataset. The transaction's 165 features are anonymized, so never guess what they mean.

Question: {question}

Answer with:
- summary: two or three sentences
- claims: the specific statements behind your answer; each one cites, in "cites", the ids of the items below
  that support it (for example ["rf_score"] or ["edge_2", "S1"]). Only state what those items say.
- caveats: what the analyst should be careful about

{context}"""

QUESTION = "Why was transaction {tx} flagged as suspicious, and how much should I trust that?"


def fact_lines(facts: dict) -> str:
    return "Facts about the transaction:\n" + "\n".join(f"[{k}] {v}" for k, v in facts.items())


def source_lines(sources: dict) -> str:
    return "Background sources:\n" + "\n\n".join(f"[{k}] {v}" for k, v in sources.items())


def answer(llm: LLM, tx, facts: dict, sources: dict | None = None) -> dict:
    context = fact_lines(facts) + ("\n\n" + source_lines(sources) if sources else "")
    reply = llm.json(ANSWER_PROMPT.format(question=QUESTION.format(tx=tx), context=context), ANSWER_SCHEMA)
    return reply | {"evidence": facts | (sources or {})}


def transaction_facts(i, data, scores, thresholds, neighbors, shapes, explanation) -> dict:
    """Plain-language facts, each with an id the answer can cite. Numbers are rounded the way they'll be shown."""
    rf, gnn = scores["random_forest"], scores["graphsage"]
    step = data["timestep"][i]
    same_step = rf[data["timestep"] == step]
    rank = int((same_step > rf[i]).sum()) + 1
    facts = {
        "rf_score": f"Alert score (random forest on the transaction's own features): {rf[i]:.2f}; the alert threshold is "
                    f"{thresholds['random_forest']:.2f}. Rank {rank} of {len(same_step)} transactions in timestep {step}.",
        "gnn_score": f"Graph model score (GraphSAGE, which also looks at neighboring transactions): {gnn[i]:.2f}; its "
                     f"threshold is {thresholds['graphsage']:.2f}.",
        "inputs": f"It received bitcoins from {neighbors['inputs']} earlier transactions and sent to "
                  f"{neighbors['outputs']} later ones.",
        "risky_neighbors": f"{neighbors['risky_1hop']} of its {neighbors['n_1hop']} direct neighbors and "
                           f"{neighbors['risky_2hop']} of its {neighbors['n_2hop']} neighbors two steps away are above "
                           f"the alert threshold themselves.",
        "shape": "Graph shape: " + (", ".join(shapes) if shapes else "none of fan-in, fan-out, peeling chain, "
                                    "source or end of chain") + ".",
    }
    if explanation:
        facts["explanation"] = (
            f"GNNExplainer, run 5 times with different seeds on its {explanation['neighborhood']}-transaction "
            f"neighborhood, agrees on the edges below. Stability (mean Jaccard of the top-5 edges between runs): "
            f"{explanation['stability']:.2f} (1 = same edges every run; below 0.5 the explanation is unstable).")
        facts["fidelity"] = (
            f"Graph model score with the whole neighborhood {explanation['p_full']:.2f}, without the 5 highlighted "
            f"edges {explanation['p_without_top']:.2f}, with only those edges {explanation['p_only_top']:.2f}, with no "
            f"edges at all (its own features only) {explanation['p_no_edges']:.2f}.")
        for n, e in enumerate(explanation["edges"], 1):
            facts[f"edge_{n}"] = e["text"]
    return facts


def typology_query(shapes: list[str]) -> str:
    """What to look up in the encyclopedia side for a transaction with these shapes."""
    words = {"fan-in": "many sources pooled into one, structuring, smurfing, mixing",
             "fan-out": "funds split and sent to many destinations, layering, distribution",
             "peeling chain": "bitcoin peeling chain, layering through successive transactions, tumbler",
             "end of chain": "where laundered funds end up, cashing out, exchanges, final destination of illicit funds",
             "source": "where illicit bitcoins come from, ransomware payments, darknet market proceeds, hacks"}
    return "Bitcoin money laundering: " + ("; ".join(words[s] for s in shapes) if shapes else
                                          "illicit transactions, darknet markets, ransomware, mixers, exchanges")
