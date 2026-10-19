"""GraphRAG indexing (Edge et al., 2024, section 3.1): chunks -> entities and relationships -> one graph."""
from collections import defaultdict

import tiktoken

from aml_graphrag.ports.llm import LLM

ENC = tiktoken.get_encoding("cl100k_base")
# the paper's default types plus two that matter in this corpus
ENTITY_TYPES = ["person", "organization", "geo", "event", "technique", "law"]

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"},
            "type": {"type": "string", "enum": ENTITY_TYPES},
            "description": {"type": "string"},
        }, "required": ["name", "type", "description"]}},
        "relationships": {"type": "array", "items": {"type": "object", "properties": {
            "source": {"type": "string"},
            "target": {"type": "string"},
            "description": {"type": "string"},
            "strength": {"type": "integer", "minimum": 1, "maximum": 10},
        }, "required": ["source", "target", "description", "strength"]}},
    },
    "required": ["entities", "relationships"],
}

EXTRACT_PROMPT = """Read the text and extract a knowledge graph from it.

1. Entities: the named people, organizations, places (geo) and events (cases, scandals, hacks, trials) in the text,
plus the techniques (ways to launder, hide or move money, types of crime) and laws (laws, regulations, sanctions lists,
standards) it talks about. Skip generic things like dates, amounts, sentences or job titles.
For each one: its full name (written the same way every time), its type, and one short sentence with what the text
says about it.

2. Relationships: pairs of those entities that the text clearly relates. For each one: source, target, one short
sentence saying how they are related according to the text, and a strength from 1 to 10.

Only use what the text says. Do not add outside knowledge.

Text:
{text}"""


TAIL_SECTIONS = ("== See also ==", "== References ==", "== Notes ==", "== Further reading ==", "== External links ==")


def clean(text: str, max_tokens: int | None = None) -> str:
    """Drop the link lists at the end of an article and keep its first max_tokens (the lead and early sections)."""
    cut = min([i for i in (text.find(h) for h in TAIL_SECTIONS) if i >= 0], default=len(text))
    text = text[:cut].strip()
    return ENC.decode(ENC.encode(text)[:max_tokens]) if max_tokens else text


def chunk(text: str, size: int = 600, overlap: int = 100) -> list[str]:
    tokens = ENC.encode(text)
    step = size - overlap
    return [ENC.decode(tokens[i:i + size]) for i in range(0, max(len(tokens) - overlap, 1), step)]


def extract(llm: LLM, text: str) -> dict:
    return llm.json(EXTRACT_PROMPT.format(text=text), EXTRACT_SCHEMA)


def key(name: str) -> str:
    return " ".join(name.upper().split())


def merge(extractions: list[tuple[str, dict]]) -> tuple[dict, dict]:
    """Same name -> same node; repeated relationships add up, as in the paper (exact string matching)."""
    entities = defaultdict(lambda: {"types": defaultdict(int), "descriptions": [], "chunks": set()})
    relations = defaultdict(lambda: {"descriptions": [], "weight": 0, "chunks": set()})
    for chunk_id, ex in extractions:
        for e in ex["entities"]:
            node = entities[key(e["name"])]
            node["types"][e["type"]] += 1
            node["descriptions"].append(e["description"])
            node["chunks"].add(chunk_id)
        for r in ex["relationships"]:
            s, t = key(r["source"]), key(r["target"])
            if s == t:
                continue
            for name in (s, t):  # an endpoint the model forgot to list as an entity still becomes a node
                entities[name]["chunks"].add(chunk_id)
            edge = relations[tuple(sorted((s, t)))]
            edge["descriptions"].append(r["description"])
            edge["weight"] += r["strength"]
            edge["chunks"].add(chunk_id)
    nodes = {name: {"type": max(n["types"], key=n["types"].get) if n["types"] else "unknown",
                    "descriptions": n["descriptions"], "chunks": sorted(n["chunks"])}
             for name, n in entities.items()}
    edges = {pair: {"descriptions": r["descriptions"], "weight": r["weight"], "chunks": sorted(r["chunks"])}
             for pair, r in relations.items()}
    return nodes, edges
