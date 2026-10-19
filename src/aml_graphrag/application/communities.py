"""Graph and hierarchical Leiden communities (Edge et al., 2024, section 3.1.4)."""
import igraph as ig


def build_graph(nodes: dict, edges: dict) -> ig.Graph:
    names = sorted(nodes)
    index = {n: i for i, n in enumerate(names)}
    g = ig.Graph(n=len(names), edges=[(index[s], index[t]) for s, t in edges])
    g.vs["name"] = names
    g.es["weight"] = [e["weight"] for e in edges.values()]
    return g


def largest_component(g: ig.Graph) -> ig.Graph:
    return g.connected_components().giant()


def hierarchical_leiden(g: ig.Graph, max_size: int = 10, max_levels: int = 4, seed: int = 0) -> list[dict]:
    """Leiden on the whole graph, then again inside every community bigger than max_size, and so on.

    Each community: {"id", "level", "parent", "nodes"} with node names. Level 0 is the root level.
    """
    import random

    random.seed(seed)
    ig.set_random_number_generator(random)
    communities = []

    def split(sub: ig.Graph, level: int, parent):
        parts = sub.community_leiden(objective_function="modularity", weights="weight", n_iterations=-1)
        if len(parts) < 2:
            return
        for members in parts:
            names = [sub.vs[i]["name"] for i in members]
            cid = len(communities)
            communities.append({"id": cid, "level": level, "parent": parent, "nodes": names})
            if len(members) > max_size and level + 1 < max_levels:
                split(sub.induced_subgraph(members), level + 1, cid)

    split(g, 0, None)
    return communities
