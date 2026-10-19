"""The ways of answering a question compared in the paper (section 3.2 and 4.1.3).

global_search: map-reduce over community reports of one level (C0, C1, C2) or over raw chunks (TS).
vector_rag:    the usual RAG, the chunks most similar to the question (SS).
All of them get the same context budget.
"""
import random

import numpy as np

from aml_graphrag.application.indexing import ENC
from aml_graphrag.domain.context import batches, pack
from aml_graphrag.ports.llm import LLM, Embedder

MAP_SCHEMA = {
    "type": "object",
    "properties": {"points": {"type": "array", "items": {"type": "object", "properties": {
        "description": {"type": "string"}, "score": {"type": "integer", "minimum": 0, "maximum": 100},
    }, "required": ["description", "score"]}}},
    "required": ["points"],
}

MAP_PROMPT = """Answer the question using only the data below. Give a list of key points; each one a paragraph with
a helpfulness score from 0 to 100 (how much it helps answer the question). If the data has nothing relevant, return
one point saying so with score 0. Do not make anything up.

Question: {question}

Data:
{context}"""

REDUCE_PROMPT = """You are a financial crime analyst. Answer the question with a well-organized response
using only the key points below, which come from different parts of a body of encyclopedia articles and are sorted by
importance. Combine and organize them; drop what doesn't help. Do not make anything up.

Question: {question}

Key points:
{context}"""

DIRECT_PROMPT = """You are a financial crime analyst. Answer the question with a well-organized response using only
the text passages below. Do not make anything up.

Question: {question}

Passages:
{context}"""


def count(text: str) -> int:
    return len(ENC.encode(text))


def global_search(llm: LLM, question: str, units: list[str], budget: int, seed: int = 0) -> dict:
    units = units[:]
    random.Random(seed).shuffle(units)  # the paper shuffles so that no report is always first
    points = []
    groups = batches(units, budget, count)
    for group in groups:
        reply = llm.json(MAP_PROMPT.format(question=question, context="\n\n".join(group)), MAP_SCHEMA)
        points += [p for p in reply["points"] if p["score"] > 0]
    points.sort(key=lambda p: -p["score"])
    context = pack([f"- (score {p['score']}) {p['description']}" for p in points], budget, count)
    if not context:
        return {"answer": "The data doesn't contain an answer to this question.", "map_calls": len(groups), "points": 0}
    return {"answer": llm.text(REDUCE_PROMPT.format(question=question, context="\n".join(context))),
            "map_calls": len(groups), "points": len(points)}


class VectorIndex:
    def __init__(self, embedder: Embedder, texts: list[str], vectors=None):
        self.embedder = embedder
        self.texts = texts
        self.vectors = np.asarray(vectors if vectors is not None else self._embed(texts), dtype=np.float32)
        self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)

    def _embed(self, texts, batch=32):
        return [v for i in range(0, len(texts), batch) for v in self.embedder.embed(texts[i:i + batch])]

    def search(self, query: str) -> list[str]:
        q = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        order = np.argsort(-(self.vectors @ (q / np.linalg.norm(q))))
        return [self.texts[i] for i in order]


def vector_rag(llm: LLM, question: str, index: VectorIndex, budget: int) -> dict:
    context = pack(index.search(question), budget, count)
    return {"answer": llm.text(DIRECT_PROMPT.format(question=question, context="\n\n".join(context))),
            "chunks": len(context)}
