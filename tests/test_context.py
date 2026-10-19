from aml_graphrag.domain.context import batches, pack


def words(text):
    return len(text.split())


def test_pack_stops_at_the_first_item_that_does_not_fit():
    assert pack(["a b", "c d e", "f"], budget=5, count=words) == ["a b", "c d e"]
    assert pack(["a b c d e f"], budget=5, count=words) == []


def test_batches_keep_order_and_budget():
    assert batches(["a b", "c d e", "f", "g h i j k l"], budget=5, count=words) == [["a b", "c d e"], ["f"], ["g h i j k l"]]
