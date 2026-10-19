from aml_graphrag.application.answering import VectorIndex, count, global_search, vector_rag
from tests.fakes import FakeEmbedder, FakeLLM


def test_global_search_drops_useless_points_and_reduces_the_rest_by_score():
    llm = FakeLLM(
        json_replies=[{"points": [{"description": "mixers hide the trail", "score": 40}]},
                      {"points": [{"description": "nothing here", "score": 0},
                                  {"description": "shell companies hide owners", "score": 90}]}],
        text_replies=["final answer"],
    )
    units = ["report one " * 50, "report two " * 50]
    out = global_search(llm, "how is money hidden?", units, budget=120)
    assert out == {"answer": "final answer", "map_calls": 2, "points": 2}
    reduce_prompt = llm.prompts[-1]
    assert reduce_prompt.index("shell companies") < reduce_prompt.index("mixers")
    assert "nothing here" not in reduce_prompt


def test_global_search_without_relevant_data_says_so():
    llm = FakeLLM(json_replies=[{"points": [{"description": "no data", "score": 0}]}])
    out = global_search(llm, "q", ["a report"], budget=100)
    assert out["points"] == 0 and "doesn't contain" in out["answer"]


def test_vector_rag_puts_the_closest_chunks_first():
    index = VectorIndex(FakeEmbedder(), ["zzzz yyyy", "bitcoin mixer", "xxxx"])
    assert index.search("bitcoin mixers")[0] == "bitcoin mixer"
    llm = FakeLLM(text_replies=["answer"])
    assert vector_rag(llm, "bitcoin mixers", index, budget=count("bitcoin mixer"))["chunks"] == 1
