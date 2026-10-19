"""Fitting text into a model's context: the paper's 'add until the token budget is full'."""
from collections.abc import Callable, Iterable


def pack(items: Iterable[str], budget: int, count: Callable[[str], int]) -> list[str]:
    """Items in the given order until the next one doesn't fit."""
    kept, used = [], 0
    for item in items:
        n = count(item)
        if used + n > budget:
            break
        kept.append(item)
        used += n
    return kept


def batches(items: Iterable[str], budget: int, count: Callable[[str], int]) -> list[list[str]]:
    """Split items into consecutive groups of at most budget tokens each (an item bigger than budget goes alone)."""
    groups, current, used = [], [], 0
    for item in items:
        n = count(item)
        if current and used + n > budget:
            groups.append(current)
            current, used = [], 0
        current.append(item)
        used += n
    if current:
        groups.append(current)
    return groups
