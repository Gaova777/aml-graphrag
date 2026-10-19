import numpy as np

from aml_graphrag.application.analyst import answer, transaction_facts, typology_query
from aml_graphrag.domain.claims import check
from tests.fakes import FakeLLM

DATA = {"timestep": np.array([35, 35, 35, 36])}
SCORES = {"random_forest": np.array([0.2, 0.93, 0.5, 0.9]), "graphsage": np.array([0.1, 0.81, 0.3, 0.2])}
THRESHOLDS = {"random_forest": 0.41, "graphsage": 0.6}
NEIGHBORS = {"inputs": 1, "outputs": 2, "n_1hop": 3, "n_2hop": 7, "risky_1hop": 1, "risky_2hop": 4}
EXPLANATION = {"neighborhood": 11, "stability": 0.42, "p_full": 0.81, "p_without_top": 0.35, "p_only_top": 0.77,
               "edges": [{"text": "Edge 5 -> 6 (touches the alert directly); alert scores 0.88 -> 0.93; importance 0.71."}]}


def test_facts_rank_within_the_timestep_and_carry_stability_and_fidelity():
    facts = transaction_facts(1, DATA, SCORES, THRESHOLDS, NEIGHBORS, ["peeling chain"], EXPLANATION)
    assert "Rank 1 of 3 transactions in timestep 35" in facts["rf_score"]
    assert "0.42" in facts["explanation"] and "0.35" in facts["fidelity"]
    assert facts["shape"] == "Graph shape: peeling chain."
    assert set(facts) >= {"rf_score", "gnn_score", "inputs", "risky_neighbors", "edge_1"}


def test_answers_keep_the_evidence_they_were_given_so_claims_can_be_checked():
    facts = transaction_facts(1, DATA, SCORES, THRESHOLDS, NEIGHBORS, [], None)
    llm = FakeLLM(json_replies=[{"summary": "s", "caveats": [], "claims": [
        {"text": "The alert score is 0.93.", "cites": ["rf_score"]},
        {"text": "It was paid by a known mixer.", "cites": []}]}])
    reply = answer(llm, 123, facts, {"S1": "Mixers pool coins."})
    assert [check(c, reply["evidence"]) for c in reply["claims"]] == ["ok", "no_citation"]
    assert "[S1] Mixers pool coins." in llm.prompts[0]


def test_typology_query_follows_the_shape():
    assert "peeling chain" in typology_query(["peeling chain"])
    assert "darknet" in typology_query([])
