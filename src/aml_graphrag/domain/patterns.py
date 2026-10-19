"""Shapes in the transaction graph that analysts know (Elliptic's features are anonymous, so shape is what we can name).

In Elliptic an edge a -> b means bitcoins flowed from transaction a into transaction b.
"""


def peeling_chain(node, succ: dict, pred: dict, max_len: int = 50) -> int:
    """Length of the peeling chain through node: consecutive transactions with one input and two outputs
    (a payment peeled off, the change moving on). 0 if node isn't one."""

    def peel(n):
        return len(pred.get(n, ())) == 1 and len(succ.get(n, ())) == 2

    if not peel(node):
        return 0
    length, seen = 1, {node}
    back = node
    while len(seen) < max_len:
        [back] = pred[back]
        if back in seen or not peel(back):
            break
        seen.add(back)
        length += 1
    forward = node
    while len(seen) < max_len:
        nxt = [n for n in succ.get(forward, ()) if peel(n) and n not in seen]
        if not nxt:
            break
        forward = nxt[0]
        seen.add(forward)
        length += 1
    return length


def shape(node, succ: dict, pred: dict, fan: int = 5) -> list[str]:
    names = []
    if len(pred.get(node, ())) >= fan:
        names.append("fan-in")
    if len(succ.get(node, ())) >= fan:
        names.append("fan-out")
    if peeling_chain(node, succ, pred) >= 3:
        names.append("peeling chain")
    return names
