"""Community summaries (Edge et al., 2024, section 3.1.5), written bottom-up."""
from aml_graphrag.application.indexing import ENC
from aml_graphrag.domain.context import pack
from aml_graphrag.ports.llm import LLM

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "rating": {"type": "number", "minimum": 0, "maximum": 10},
        "rating_explanation": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "object", "properties": {
            "summary": {"type": "string"}, "explanation": {"type": "string"},
        }, "required": ["summary", "explanation"]}},
    },
    "required": ["title", "summary", "rating", "rating_explanation", "findings"],
}

REPORT_PROMPT = """You are helping a financial crime analyst. Below is a community of related entities from a knowledge
graph built from encyclopedia articles about money laundering and financial crime, with the relationships between them.

Write a report about this community:
- title: a short, specific name for the community (use its key entities)
- summary: an executive summary of how the entities relate and why they matter
- rating: from 0 to 10, how important this community is for understanding money laundering and financial crime
- rating_explanation: one sentence
- findings: 3 to 7 key insights, each with a short summary and a paragraph of explanation

Use only the information below.

{context}"""


def count(text: str) -> int:
    return len(ENC.encode(text))


def entity_line(name, node, max_descriptions=3):
    unique = list(dict.fromkeys(node["descriptions"]))[:max_descriptions]
    return f"- {name} ({node['type']}): {' '.join(unique)}"


def edge_line(pair, edge, max_descriptions=2):
    unique = list(dict.fromkeys(edge["descriptions"]))[:max_descriptions]
    return f"- {pair[0]} -> {pair[1]}: {' '.join(unique)}"


def report_text(report: dict) -> str:
    findings = "\n".join(f"- {f['summary']}: {f['explanation']}" for f in report["findings"])
    return f"# {report['title']}\n{report['summary']}\n{findings}"


def community_context(community, nodes, edges, degree, sub_reports, budget):
    """Relationships by prominence (the combined degree of their ends), then the entities, until the budget is full.

    If the community's own elements don't fit and its sub-communities already have reports, those reports
    replace the raw elements, biggest sub-communities first (the paper's substitution for higher levels).
    """
    members = set(community["nodes"])
    inside = sorted(((p, e) for p, e in edges.items() if p[0] in members and p[1] in members),
                    key=lambda pe: -(degree[pe[0][0]] + degree[pe[0][1]]))
    lines = [edge_line(p, e) for p, e in inside]
    entity_lines = [entity_line(n, nodes[n]) for n in sorted(members, key=lambda n: -degree[n])]
    full = ["Relationships:"] + lines + ["Entities:"] + entity_lines
    if sum(count(x) for x in full) <= budget or not sub_reports:
        rels = pack(lines, budget * 2 // 3, count)
        ents = pack(entity_lines, budget - sum(count(x) for x in rels), count)
        return "\n".join(["Relationships:"] + rels + ["Entities:"] + ents)
    return "\n\n".join(pack([report_text(r) for r in sub_reports], budget, count))


def write_report(llm: LLM, context: str) -> dict:
    return llm.json(REPORT_PROMPT.format(context=context), REPORT_SCHEMA)
