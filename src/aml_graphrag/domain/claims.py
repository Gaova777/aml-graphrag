"""Checking an answer's claims against what it cites, with code rather than with another model where possible."""
import re

NUMBER = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(%?)")


def numbers(text: str) -> list[float]:
    found = []
    for digits, percent in NUMBER.findall(text):
        value = float(digits.replace(",", ""))
        found.append(value / 100 if percent else value)
    return found


def same_number(a: float, b: float) -> bool:
    """Equal up to the rounding an answer would do (0.912 ~ 0.91 ~ 91%)."""
    return abs(a - b) <= max(0.006, 0.006 * abs(b))


def check(claim: dict, evidence: dict) -> str:
    """ok | no_citation | bad_citation | number_mismatch

    evidence maps the ids the model was allowed to cite to their text. Every number in the claim must appear in
    something it cites; small counts written as words aren't checked.
    """
    cites = claim.get("cites") or []
    if not cites:
        return "no_citation"
    if any(c not in evidence for c in cites):
        return "bad_citation"
    cited = [n for c in cites for n in numbers(evidence[c])]
    for n in numbers(claim["text"]):
        if not any(same_number(n, m) for m in cited):
            return "number_mismatch"
    return "ok"
