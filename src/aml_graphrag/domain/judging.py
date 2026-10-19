"""Turning an LLM judge's votes into win rates, the paper's way, plus the order swap it didn't do."""


def score_pair(first: dict, swapped: dict) -> dict:
    """Score of answer A against B for each criterion, from two judgements: A shown first, then B shown first.

    A win is 1, a tie 0.5, a loss 0 (the paper's scoring), averaged over the two orders, so a judge that always
    prefers whatever comes first scores a tie.
    """
    points = {1: 1.0, 0: 0.5, 2: 0.0}
    return {c: (points[first[c]] + 1.0 - points[swapped[c]]) / 2 for c in first}


def win_rates(scores: list[dict]) -> dict:
    """Average score of A over all questions, per criterion."""
    return {c: sum(s[c] for s in scores) / len(scores) for c in scores[0]}


def position_bias(first: list[dict], swapped: list[dict]) -> float:
    """How often the verdict flips when only the order changes (ties excluded): 0 is a consistent judge."""
    flips = total = 0
    for a, b in zip(first, swapped):
        for c in a:
            if a[c] and b[c]:
                total += 1
                flips += a[c] == b[c]  # same slot wins both times means the winner changed
    return flips / total if total else 0.0
