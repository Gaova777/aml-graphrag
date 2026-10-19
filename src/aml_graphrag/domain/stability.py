"""Is an explanation stable? The same explainer, run again with another seed, should point at the same edges."""
from itertools import combinations


def top_k(weights: dict, k: int) -> set:
    return set(sorted(weights, key=lambda e: -weights[e])[:k])


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def stability(runs: list[dict], k: int) -> float:
    """Mean pairwise Jaccard of the top-k edges across runs (1 = always the same edges)."""
    tops = [top_k(r, k) for r in runs]
    pairs = list(combinations(tops, 2))
    return sum(jaccard(a, b) for a, b in pairs) / len(pairs)


def consensus(runs: list[dict], k: int) -> list:
    """Edges by how many runs put them in their top-k, then by mean weight: what the runs agree on."""
    votes, total = {}, {}
    for r in runs:
        for e in top_k(r, k):
            votes[e] = votes.get(e, 0) + 1
        for e, w in r.items():
            total[e] = total.get(e, 0.0) + w
    return sorted(votes, key=lambda e: (-votes[e], -total[e]))[:k]
