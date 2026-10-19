from aml_graphrag.domain.claims import check, numbers

EVIDENCE = {"rf_score": "Random forest illicit score: 0.912 (rank 3 of 1,341 in timestep 35)",
            "stability": "Explanation stability (Jaccard of top-5 edges over 5 runs): 0.47"}


def test_numbers_and_percentages():
    assert numbers("score 0.91, 91% and 1,341 transactions") == [0.91, 0.91, 1341.0]


def test_a_claim_is_ok_only_if_it_cites_real_evidence_with_the_same_numbers():
    assert check({"text": "The model scored it 0.91.", "cites": ["rf_score"]}, EVIDENCE) == "ok"
    assert check({"text": "It is in the top 91% ... rank 3.", "cites": ["rf_score"]}, EVIDENCE) == "ok"
    assert check({"text": "The model scored it 0.95.", "cites": ["rf_score"]}, EVIDENCE) == "number_mismatch"
    assert check({"text": "It received funds from a mixer.", "cites": []}, EVIDENCE) == "no_citation"
    assert check({"text": "Stable explanation.", "cites": ["edge_9"]}, EVIDENCE) == "bad_citation"
