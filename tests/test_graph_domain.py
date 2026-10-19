from aml_graphrag.domain.patterns import peeling_chain, shape
from aml_graphrag.domain.stability import consensus, jaccard, stability


def test_identical_runs_are_perfectly_stable():
    run = {"a": 0.9, "b": 0.8, "c": 0.1}
    assert stability([run, run, run], k=2) == 1.0


def test_disagreeing_runs():
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    runs = [{"a": 0.9, "b": 0.8, "c": 0.1}, {"a": 0.1, "b": 0.8, "c": 0.9}]
    assert stability(runs, k=2) == 1 / 3
    assert consensus(runs, k=2)[0] == "b"


def test_peeling_chain_follows_one_input_two_outputs():
    # 1 -> 2 -> 3 -> 4, each also paying out to a side transaction
    succ = {1: [2, 11], 2: [3, 12], 3: [4, 13], 0: [1]}
    pred = {1: [0], 2: [1], 3: [2], 4: [3], 11: [1], 12: [2], 13: [3]}
    assert peeling_chain(2, succ, pred) == 3
    assert "peeling chain" in shape(2, succ, pred)
    assert peeling_chain(4, succ, pred) == 0


def test_fan_in_and_fan_out():
    pred = {"x": list(range(3))}
    succ = {"x": ["y"]}
    assert shape("x", succ, pred) == ["fan-in"]
    assert shape("y", succ, {"y": ["x"]}) == ["end of chain"]
    assert shape("z", {"z": ["a", "b", "c"]}, {}) == ["source", "fan-out"]
