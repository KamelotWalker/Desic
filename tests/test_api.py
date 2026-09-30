import json
import random

import pytest
from fastapi.testclient import TestClient

from desic.api import create_app
from desic.demo import loan_row


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(str(tmp_path))) as c:
        yield c


MODEL = {
    "name": "loan",
    "target": "decision",
    "classes": ["approve", "reject"],
    "features": [{"name": "income", "type": "numeric"}, {"name": "city", "type": "categorical", "values": ["a", "b"]}],
}


def test_create_decide_feedback_cycle(client):
    r = client.post("/api/models", json=MODEL)
    assert r.status_code == 201, r.text
    assert client.post("/api/models", json=MODEL).status_code == 409

    d = client.post("/api/models/loan/decide", json={"features": {"income": 10, "city": "a"}}).json()
    assert d["prediction"] is None and d["source"] == "none"
    fb = client.post("/api/models/loan/feedback", json={"decision_id": d["id"], "label": "reject"})
    assert fb.status_code == 200 and fb.json()["correct"] is False
    assert client.post("/api/models/loan/feedback", json={"decision_id": d["id"], "label": "reject"}).status_code == 409

    rng = random.Random(0)
    rows = []
    for _ in range(600):
        inc = rng.uniform(0, 100)
        rows.append({"income": inc, "city": rng.choice("ab"), "decision": "approve" if inc > 40 else "reject"})
    r = client.post("/api/models/loan/learn", json={"rows": rows}).json()
    assert r["learned"] == 600
    d = client.post("/api/models/loan/decide", json={"features": {"income": 90, "city": "b"}}).json()
    assert d["prediction"] == "approve" and d["source"] == "model"
    assert d["explanation"]["path"][0]["feature"] == "income"

    detail = client.get("/api/models/loan").json()
    assert detail["learned"] == 601
    assert detail["counters"]["pending"] == 1
    assert detail["history"]
    assert client.get("/api/models/loan/tree").json()["type"] == "split"
    assert client.get("/api/models/loan/learned-rules").json()
    pending = client.get("/api/models/loan/decisions?pending=true&uncertain_first=true").json()
    assert [p["id"] for p in pending] == [d["id"]]


def test_hard_rules_override_model(client):
    client.post("/api/models", json=MODEL)
    rules = [{"name": "blocklist", "decision": "reject", "priority": 10,
              "conditions": [{"feature": "city", "op": "==", "value": "b"}]}]
    saved = client.put("/api/models/loan/rules", json={"rules": rules}).json()
    d = client.post("/api/models/loan/decide", json={"features": {"income": 99, "city": "b"}}).json()
    assert d["source"] == "rule" and d["prediction"] == "reject" and d["rule"]["id"] == saved[0]["id"]
    client.post("/api/models/loan/feedback", json={"decision_id": d["id"], "label": "approve"})
    rule = client.get("/api/models/loan").json()["rules"][0]
    assert rule["hits"] == 1 and rule["overridden"] == 1
    bad = client.put("/api/models/loan/rules", json={"rules": [{"decision": "x", "conditions": []}]})
    assert bad.status_code == 400


def test_upload_and_train_dataset(client):
    rng = random.Random(3)
    rows = [loan_row(rng) for _ in range(800)]
    header = list(rows[0])
    csv_text = ";".join(header) + "\n" + "\n".join(";".join(str(r[h]) for h in header) for r in rows)
    r = client.post("/api/datasets", files={"file": ("loans.csv", csv_text.encode())}, data={"name": "loans"})
    assert r.status_code == 201, r.text
    ds = r.json()
    assert ds["n_rows"] == 800 and ds["profile"]["employment"]["type"] == "categorical"
    assert ds["profile"]["credit_score"]["type"] == "numeric"

    job = client.post(f"/api/datasets/{ds['id']}/train", json={"model": "loans", "target": "decision", "holdout": 0.25}).json()
    for _ in range(200):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] != "running":
            break
        import time; time.sleep(0.05)
    assert job["status"] == "done", job
    assert job["result"]["holdout"]["accuracy"] > 0.7
    assert client.get("/api/models/loans").json()["schema"]["target"] == "decision"


def test_bad_upload(client):
    r = client.post("/api/datasets", files={"file": ("x.json", b'{"nope": 1}')})
    assert r.status_code == 400


def test_websocket_streams_events(client):
    client.post("/api/models", json=MODEL)
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"
        d = client.post("/api/models/loan/decide", json={"features": {"income": 1}}).json()
        ev = ws.receive_json()
        assert ev["type"] == "decision" and ev["decision"]["id"] == d["id"]
        client.post("/api/models/loan/feedback", json={"decision_id": d["id"], "label": "reject"})
        types = [ws.receive_json()["type"] for _ in range(2)]
        assert types == ["metrics", "feedback"]


def test_models_survive_restart(tmp_path):
    with TestClient(create_app(str(tmp_path))) as c:
        c.post("/api/models", json=MODEL)
        c.post("/api/models/loan/learn", json={"rows": [{"income": i, "decision": "approve" if i > 50 else "reject"} for i in range(300)]})
    with TestClient(create_app(str(tmp_path))) as c:
        m = c.get("/api/models/loan").json()
        assert m["learned"] == 300
        assert c.post("/api/models/loan/decide", json={"features": {"income": 90}}).json()["prediction"] == "approve"


def test_dashboard_served(client):
    assert "Desic" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_generate_design_and_dataset_with_fake_provider(client, monkeypatch):
    import desic.api as api_mod
    from tests.test_generation import FakeProvider

    fake = FakeProvider()
    monkeypatch.setattr(api_mod, "make_provider", lambda cfg: fake)
    body = {"provider": "anthropic", "api_key": "sk-test", "description": "approve loans"}
    design = client.post("/api/generate/design", json=body)
    assert design.status_code == 200, design.text
    job = client.post("/api/generate/dataset", json={**body, "design": design.json(), "n_rows": 80,
                                                     "train_model": "gen_model"}).json()
    import time
    for _ in range(200):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    assert job["result"]["rows"] == 80
    ds = client.get(f"/api/datasets/{job['result']['dataset']}").json()
    assert ds["source"] == "generated" and ds["meta"]["design"]["target"] == "decision"
    assert "sk-test" not in str(client.get("/api/jobs").json())  # key never kept
    for _ in range(200):
        tj = client.get(f"/api/jobs/{job['result']['train_job']}").json()
        if tj["status"] != "running":
            break
        time.sleep(0.05)
    assert tj["status"] == "done", tj
    assert client.get("/api/models/gen_model").json()["learned"] > 0
