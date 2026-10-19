import json

from fastapi.testclient import TestClient

from aml_graphrag.entrypoints import api
from tests.fakes import FakeLLM

ALERT = {"tx": 7, "index": 0, "timestep": 40, "label": 1, "rf": 0.9, "gnn": 0.7, "shapes": ["fan-in"],
         "neighbors": {"inputs": 6, "outputs": 1, "n_1hop": 7, "n_2hop": 20, "risky_1hop": 2, "risky_2hop": 5},
         "explanation": {"stability": 0.4, "neighborhood": 28, "p_full": 0.7, "p_without_top": 0.2, "p_only_top": 0.6,
                         "edges": []},
         "facts": {"rf_score": "Alert score 0.90."}}


def client(tmp_path, monkeypatch, llm):
    (tmp_path / "index").mkdir()
    (tmp_path / "alerts.json").write_text(json.dumps([ALERT]))
    (tmp_path / "neighborhoods.json").write_text(json.dumps({"7": {"nodes": [], "edges": []}}))
    (tmp_path / "index" / "reports.jsonl").write_text(json.dumps(
        {"community": 0, "level": 0, "size": 4, "title": "Mixers", "summary": "s", "rating": 7, "findings": []}) + "\n")
    monkeypatch.setattr(api, "DATA", tmp_path)
    for f in (api.alerts, api.neighborhoods, api.reports, api.llm, api.report_index):
        f.cache_clear()
    monkeypatch.setattr(api, "llm", lambda: llm)
    monkeypatch.setattr(api, "report_index", lambda: type("I", (), {"search": lambda self, q: ["# Mixers\ns"]})())
    return TestClient(api.app)


def test_queue_detail_and_checked_answer(tmp_path, monkeypatch):
    llm = FakeLLM(json_replies=[{"summary": "s", "caveats": ["unstable"], "claims": [
        {"text": "Score 0.90.", "cites": ["rf_score"]}, {"text": "Linked to a mixer.", "cites": ["C9"]}]}])
    c = client(tmp_path, monkeypatch, llm)
    assert c.get("/alerts").json()[0]["stability"] == 0.4
    assert c.get("/alerts/7").json()["graph"] == {"nodes": [], "edges": []}
    assert c.get("/alerts/8").status_code == 404
    claims = c.post("/alerts/7/ask").json()["claims"]
    assert [cl["check"] for cl in claims] == ["ok", "bad_citation"]
    assert c.get("/communities", params={"level": 0}).json()[0]["title"] == "Mixers"
