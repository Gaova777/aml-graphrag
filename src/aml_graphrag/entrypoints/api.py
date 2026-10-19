"""HTTP API for the analyst console.

GET  /alerts               the alert queue
GET  /alerts/{tx}          facts, explanation and the transaction's neighborhood (nodes and edges to draw)
POST /alerts/{tx}/ask      the grounded answer, with every claim checked against what it cites
GET  /communities          GraphRAG community reports by level
POST /global               a question about the whole corpus, answered by global search over one level
GET  /health
"""
import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aml_graphrag.adapters.langfuse_tracer import tracer
from aml_graphrag.application.analyst import answer, typology_query
from aml_graphrag.application.answering import VectorIndex, count, global_search
from aml_graphrag.application.reports import report_text
from aml_graphrag.domain.claims import check
from aml_graphrag.domain.context import pack

DATA = Path(os.environ.get("AML_DATA", "data"))
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")

app = FastAPI(title="aml-graphrag")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:4200"], allow_methods=["*"], allow_headers=["*"])


@lru_cache
def alerts() -> dict:
    return {a["tx"]: a for a in json.loads((DATA / "alerts.json").read_text())}


@lru_cache
def neighborhoods() -> dict:
    return json.loads((DATA / "neighborhoods.json").read_text())


@lru_cache
def reports() -> list[dict]:
    return [json.loads(line) for line in (DATA / "index" / "reports.jsonl").open()]


@lru_cache
def tracing():
    return tracer()


@lru_cache
def llm():
    from aml_graphrag.adapters.ollama import OllamaLLM
    return OllamaLLM("qwen3:4b-instruct", url=OLLAMA, num_ctx=15360, tracer=tracing())


@lru_cache
def report_index():
    from aml_graphrag.adapters.ollama import OllamaEmbedder
    return VectorIndex(OllamaEmbedder(url=OLLAMA), [report_text(r) for r in reports() if r["level"] == 1])


def summary(a: dict) -> dict:
    return {k: a[k] for k in ("tx", "timestep", "rf", "gnn", "shapes")} | {
        "stability": a["explanation"]["stability"], "neighbors": a["neighbors"]}


@app.get("/health")
def health():
    return {"alerts": len(alerts()), "communities": len(reports())}


@app.get("/alerts")
def list_alerts():
    return [summary(a) for a in sorted(alerts().values(), key=lambda a: -a["rf"])]


@app.get("/alerts/{tx}")
def get_alert(tx: int):
    a = alerts().get(tx) or _missing(tx)
    return summary(a) | {"facts": a["facts"], "explanation": a["explanation"], "graph": neighborhoods()[str(tx)]}


@app.post("/alerts/{tx}/ask")
def ask(tx: int):
    a = alerts().get(tx) or _missing(tx)
    with tracing().span("analyst.ask", tx=tx) as span:
        found = pack(report_index().search(typology_query(a["shapes"])), 1500, count)
        sources = {f"C{n}": t for n, t in enumerate(found, 1)}
        reply = answer(llm(), tx, a["facts"], sources)
        for claim in reply["claims"]:
            claim["check"] = check(claim, reply["evidence"])
        span.update(output={k: reply[k] for k in ("summary", "claims", "caveats")})
        if reply["claims"]:
            ok = sum(c["check"] == "ok" for c in reply["claims"]) / len(reply["claims"])
            tracing().score("cited_claims", ok, comment="share of claims that cite given evidence with matching numbers")
    return reply


@app.get("/communities")
def communities(level: int = 0):
    return [{k: r[k] for k in ("community", "level", "size", "title", "summary", "rating", "findings")}
            for r in sorted(reports(), key=lambda r: -r["rating"]) if r["level"] == level]


class GlobalQuestion(BaseModel):
    question: str
    level: int = 0


@app.post("/global")
def ask_global(q: GlobalQuestion):
    units = [report_text(r) for r in reports() if r["level"] == q.level]
    if not units:
        raise HTTPException(404, f"no communities at level {q.level}")
    with tracing().span("global.search", question=q.question, level=q.level) as span:
        out = global_search(llm(), q.question, units, budget=3000)
        span.update(output=out)
    return out


def _missing(tx):
    raise HTTPException(404, f"transaction {tx} is not in the alert queue")
