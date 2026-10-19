import igraph as ig

from aml_graphrag.application.communities import build_graph, hierarchical_leiden, largest_component


def test_big_communities_are_split_again_and_small_ones_are_not():
    g = ig.Graph.Famous("Zachary")
    g.vs["name"] = [str(i) for i in range(g.vcount())]
    g.es["weight"] = [1] * g.ecount()
    communities = hierarchical_leiden(g, max_size=8)
    top = [c for c in communities if c["level"] == 0]
    assert sorted(n for c in top for n in c["nodes"]) == sorted(g.vs["name"])
    for c in communities:
        children = [k for k in communities if k["parent"] == c["id"]]
        if children:
            assert len(c["nodes"]) > 8
            assert sorted(n for k in children for n in k["nodes"]) == sorted(c["nodes"])


def test_graph_keeps_names_and_weights():
    nodes = {"A": {}, "B": {}, "C": {}, "D": {}}
    edges = {("A", "B"): {"weight": 3}, ("C", "D"): {"weight": 1}, ("B", "C"): {"weight": 2}, }
    g = build_graph(nodes, {("A", "B"): edges[("A", "B")]})
    assert g.vcount() == 4 and largest_component(g).vs["name"] == ["A", "B"]
    assert build_graph(nodes, edges).es["weight"] == [3, 1, 2]
