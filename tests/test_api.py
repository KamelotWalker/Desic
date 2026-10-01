import random
import time

import pytest
from fastapi.testclient import TestClient

from desic.api import create_app
from desic.demo import loan, ticket

DEPT = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payments", "technical": "bugs", "sales": "pricing"}}


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(str(tmp_path))) as c:
        yield c


def wait_job(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def teach_department(client, n=300, seed=0):
    client.post("/v1/questions", json={"name": "department", **DEPT})  # 409 if it already exists: fine
    rng = random.Random(seed)
    examples = [{"state": t, "answer": d} for t, d, _ in (ticket(rng) for _ in range(n))]
    r = client.post("/v1/questions/department/learn", json={"examples": examples})
    assert r.status_code == 200, r.text
    return r.json()


def test_decide_registers_questions_and_abstains_when_cold(client):
    r = client.post("/v1/decide", json={"state": "I was charged twice", "questions": {"department": DEPT}})
    assert r.status_code == 200, r.text
    a = r.json()["answers"]["department"]
    assert a["type"] == "choice" and a["abstain"] is True and a["source"] == "student"
    assert set(a["probabilities"]) == {"billing", "technical", "sales"}
    q = client.get("/v1/questions/department").json()
    assert q["options"]["billing"] == "payments"


def test_feedback_cycle_and_prequential_metrics(client):
    client.post("/v1/decide", json={"state": "x", "questions": {"department": DEPT}})
    teach_department(client)
    r = client.post("/v1/decide", json={"state": "please refund my payment, charged twice", "questions": {"department": {}},
                                        "explain": True}).json()
    a = r["answers"]["department"]
    assert a["choice"] == "billing" and not a["abstain"]
    assert a["explanation"]["experts"]["linear"]["awake"]
    fb = client.post("/v1/feedback", json={"decision_id": r["id"], "answers": {"department": "billing"}})
    assert fb.status_code == 200 and fb.json()["answers"]["department"]["correct"] is True
    assert client.post("/v1/feedback", json={"decision_id": r["id"], "answers": {"department": "sales"}}).status_code == 409
    detail = client.get("/v1/questions/department").json()
    assert detail["labels"] == 301 and detail["metrics"]["ece"] is not None and detail["history"]
    assert detail["log"]["human"] == 1 and detail["log"]["dataset"] == 300


def test_multiple_question_types_in_one_call(client):
    r = client.post("/v1/decide", json={"state": {"ticket": "app crashes", "plan": "pro"}, "questions": {
        "urgent": {"type": "noul", "instructions": "needs help today"},
        "severity": {"type": "score", "criteria": {"low": "cosmetic", "medium": "workaround", "high": "blocking"}},
    }}).json()
    assert r["answers"]["urgent"]["type"] == "noul" and 0 <= r["answers"]["urgent"]["probability"] <= 1
    assert r["answers"]["severity"]["type"] == "score" and 0 <= r["answers"]["severity"]["score"] <= 2
    fb = client.post("/v1/feedback", json={"decision_id": r["id"], "answers": {"urgent": True, "severity": 2}}).json()
    assert fb["answers"]["urgent"]["label"] == "true" and fb["answers"]["severity"]["label"] == "high"
    bad = client.post("/v1/decide", json={"state": "x", "questions": {"urgent": {"type": "choice", "criteria": ["a", "b"]}}})
    assert bad.status_code == 422


def test_hot_learning_from_a_single_correction(client):
    teach_department(client)
    state = "The courier left my package at the wrong address"
    r = client.post("/v1/decide", json={"state": state, "questions": {"department": {}}}).json()
    wrong = r["answers"]["department"]["choice"]
    right = "sales" if wrong != "sales" else "billing"
    client.post("/v1/feedback", json={"decision_id": r["id"], "answers": {"department": right}})
    again = client.post("/v1/decide", json={"state": state, "questions": {"department": {}}}).json()
    assert again["answers"]["department"]["choice"] == right


def test_rules_override_the_student(client):
    teach_department(client)
    rules = [{"name": "legal", "decision": "sales", "priority": 5, "conditions": [{"feature": "$text", "op": "contains", "value": "lawyer"}]}]
    assert client.put("/v1/questions/department/rules", json={"rules": rules}).status_code == 200
    r = client.post("/v1/decide", json={"state": "my lawyer says the refund is late", "questions": {"department": {}}}).json()
    a = r["answers"]["department"]
    assert a["source"] == "rule" and a["choice"] == "sales" and a["rule"]["name"] == "legal"
    bad = client.put("/v1/questions/department/rules", json={"rules": [{"decision": "nope", "conditions": [{"feature": "x"}]}]})
    assert bad.status_code == 400


def test_retract_rebuild_snapshot_rollback(client):
    teach_department(client, n=200)
    r = client.post("/v1/decide", json={"state": "invoice is wrong", "questions": {"department": {}}}).json()
    client.post("/v1/feedback", json={"decision_id": r["id"], "answers": {"department": "technical"}})  # a bad label
    events = client.get("/v1/questions/department/feedback?limit=5").json()
    bad = next(e for e in events if e["source"] == "human")
    assert client.post(f"/v1/feedback/{bad['id']}/retract").json()["retracted"] is True
    snap = client.post("/v1/questions/department/snapshots", json={"note": "before"}).json()
    job = client.post("/v1/questions/department/rebuild", json={}).json()
    job = wait_job(client, job["id"])
    assert job["status"] == "done" and job["result"]["replayed"] == 200
    assert client.get("/v1/questions/department").json()["labels"] == 200
    versions = [s["version"] for s in client.get("/v1/questions/department/snapshots").json()]
    assert snap["version"] in versions
    back = client.post("/v1/questions/department/rollback", json={"version": snap["version"]}).json()
    assert back["labels"] == 200  # the snapshot was taken after the retraction, which applied at once


def test_settings_patch_and_reset(client):
    client.post("/v1/questions", json={"name": "department", **DEPT})
    r = client.patch("/v1/questions/department", json={"settings": {"abstain_threshold": 0.9, "teacher_mode": "off"},
                                                       "add_options": ["legal"], "descriptions": {"legal": "contracts"}})
    assert r.status_code == 200
    d = r.json()
    assert d["settings"]["abstain_threshold"] == 0.9 and d["options"]["legal"] == "contracts"
    assert client.patch("/v1/questions/department", json={"settings": {"teacher_mode": "sometimes"}}).status_code == 400
    teach_department(client, n=50)
    assert client.post("/v1/questions/department/reset").json()["labels"] == 0


def test_dataset_upload_and_training_json_state(client):
    rng = random.Random(3)
    rows = []
    for _ in range(900):
        s, y = loan(rng)
        rows.append({**s["applicant"], "amount": s["loan"]["amount"], "decision": y})
    header = list(rows[0])
    csv_text = ";".join(header) + "\n" + "\n".join(";".join(str(r[h]) for h in header) for r in rows)
    ds = client.post("/v1/datasets", files={"file": ("loans.csv", csv_text.encode())}, data={"name": "loans"}).json()
    assert ds["n_rows"] == 900 and ds["profile"]["credit_score"]["type"] == "number"
    job = client.post(f"/v1/datasets/{ds['id']}/train", json={"task": "loan", "answer_column": "decision"}).json()
    job = wait_job(client, job["id"])
    assert job["status"] == "done", job
    assert job["result"]["holdout"]["accuracy"] > 0.75
    q = client.get("/v1/questions/loan").json()
    assert set(q["options"]) == {"approve", "review", "reject"}


def test_dataset_training_text_state(client):
    rng = random.Random(4)
    lines = ["[" + ",".join(f'{{"message": "{t}", "team": "{d}"}}' for t, d, _ in (ticket(rng) for _ in range(300))) + "]"]
    ds = client.post("/v1/datasets", files={"file": ("t.json", lines[0].encode())}).json()
    assert ds["profile"]["message"]["type"] == "text"
    job = wait_job(client, client.post(f"/v1/datasets/{ds['id']}/train",
                                       json={"task": "team", "answer_column": "team", "state_columns": ["message"]}).json()["id"])
    assert job["status"] == "done" and job["result"]["holdout"]["accuracy"] > 0.8


def test_websocket_streams_decisions_and_feedback(client):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"
        r = client.post("/v1/decide", json={"state": "hi", "questions": {"department": DEPT}}).json()
        types = []
        while "decision" not in types:
            types.append(ws.receive_json()["type"])
        client.post("/v1/feedback", json={"decision_id": r["id"], "answers": {"department": "sales"}})
        ev = ws.receive_json()
        assert ev["type"] == "feedback" and ev["task"] == "department"


def test_persistence_across_restarts(tmp_path):
    with TestClient(create_app(str(tmp_path))) as c:
        c.post("/v1/decide", json={"state": "x", "questions": {"department": DEPT}})
        teach_department(c, n=100)
    with TestClient(create_app(str(tmp_path))) as c:
        q = c.get("/v1/questions/department").json()
        assert q["labels"] == 100
        r = c.post("/v1/decide", json={"state": "refund my invoice please", "questions": {"department": {}}}).json()
        assert r["answers"]["department"]["choice"] == "billing"


def test_teacher_endpoints_without_teacher(client):
    t = client.get("/v1/teacher").json()
    assert t["configured"] is False
    assert client.post("/v1/teacher/test").status_code == 502
    r = client.put("/v1/teacher", json={"provider": "anthropic", "api_key": "sk-ant-secret-1234"}).json()
    assert r["configured"] and r["api_key_hint"] == "…1234" and "secret" not in str(r)
    assert client.put("/v1/teacher", json={"provider": "nope"}).status_code == 400


def test_dashboard_served(client):
    assert "Desic" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_short_text_rows_train_as_plain_text(client):
    """Regression (first user trial): short messages became {"state": ...} objects in
    training while the same message arrives as plain text at decision time."""
    rows = [{"state": "Kargom nerede?", "answer": "kargo"}, {"state": "Faturamı gönderir misiniz", "answer": "fatura"}] * 10
    import json as _json
    ds = client.post("/v1/datasets", files={"file": ("s.json", _json.dumps(rows).encode())}).json()
    job = wait_job(client, client.post(f"/v1/datasets/{ds['id']}/train",
                                       json={"task": "route", "answer_column": "answer", "holdout": 0}).json()["id"])
    assert job["status"] == "done", job
    r = client.post("/v1/decide", json={"state": "Kargom nerede?", "questions": {"route": {}}, "explain": True}).json()
    exp = r["answers"]["route"]["explanation"]
    # the very same state was learned: an exact match (still on probation, so in the patch layer)
    assert exp["patches"][0]["similarity"] == 1.0 and exp["patches"][0]["answer"] == "kargo"
    assert exp["experts"]["tree"]["awake"] is False


def test_retracting_a_label_undoes_it_at_once(client):
    teach_department(client)
    state = "The courier left my package at the wrong address"
    first = client.post("/v1/decide", json={"state": state, "questions": {"department": {}}}).json()
    before = first["answers"]["department"]["choice"]
    wrong = "sales" if before != "sales" else "billing"
    client.post("/v1/feedback", json={"decision_id": first["id"], "answers": {"department": wrong}})
    after = client.post("/v1/decide", json={"state": state, "questions": {"department": {}}}).json()
    assert after["answers"]["department"]["choice"] == wrong
    ev = next(e for e in client.get("/v1/questions/department/feedback?limit=5").json() if e["source"] == "human")
    r = client.post(f"/v1/feedback/{ev['id']}/retract").json()
    assert r["applied"] is True and r["undo"]["on_probation"] == 1 and r["undo"]["replayed"] == 0
    undone = client.post("/v1/decide", json={"state": state, "questions": {"department": {}}}).json()
    assert undone["answers"]["department"]["choice"] == before  # no rebuild needed
    assert client.post(f"/v1/feedback/{ev['id']}/restore").json()["applied"] is True
    again = client.post("/v1/decide", json={"state": state, "questions": {"department": {}}}).json()
    assert again["answers"]["department"]["choice"] == wrong


def test_undo_the_most_recent_labels(client):
    teach_department(client, n=200)
    rng = random.Random(5)
    bad = [{"state": t, "answer": "sales"} for t, d, _ in (ticket(rng) for _ in range(40)) if d == "billing"][:10]
    client.post("/v1/questions/department/learn", json={"examples": bad})
    r = client.post("/v1/questions/department/retract-recent", json={"n": len(bad)}).json()
    assert len(r["events"]) == len(bad) and r["undo"]["retracted"] == len(bad) and r["hint"] is None
    log = client.get(f"/v1/questions/department/feedback?limit={len(bad)}").json()
    assert all(e["retracted"] for e in log)
    q = client.get("/v1/questions/department").json()
    assert q["patches"]["entries"] == 200 and q["patches"]["on_probation"] <= q["patches"]["probation"]


def test_students_saved_before_the_patch_layer_are_wrapped(tmp_path):
    from desic.core import DecisionTask, QuestionSpec
    from desic.storage import Storage

    st = Storage(tmp_path / "desic.db")
    old = DecisionTask(QuestionSpec.parse("department", DEPT))
    rng = random.Random(0)
    for t, d, _ in (ticket(rng) for _ in range(150)):
        old.learn(t, d, source="dataset")
    st.save_task("department", old.spec.to_dict(), old)
    st.close()
    with TestClient(create_app(str(tmp_path))) as c:
        q = c.get("/v1/questions/department").json()
        assert q["labels"] == 150 and q["metrics"]["labels"] == 150 and q["patches"]["entries"] == 0
        r = c.post("/v1/decide", json={"state": "refund my invoice please", "questions": {"department": {}}}).json()
        assert r["answers"]["department"]["choice"] == "billing"


def test_decision_contract_settings_and_logging(client):
    teach_department(client, n=200)
    r = client.patch("/v1/questions/department", json={"settings": {"cost_wrong": 10, "cost_abstain": 1}})
    assert r.status_code == 200 and r.json()["settings"]["cost_wrong"] == 10.0
    d = client.post("/v1/decide", json={"state": "refund my invoice please", "questions": {"department": {}}}).json()
    dec = d["answers"]["department"]["decision"]
    assert dec["rule"] == "costs" and dec["threshold"] == 0.9 and dec["propensity"] == 1.0
    assert dec["policy_version"] == 2 and dec["model_version"] >= 200  # P1: logged separately
    bad = client.patch("/v1/questions/department", json={"settings": {"risk_budget": 2}})
    assert bad.status_code == 400
    r = client.patch("/v1/questions/department", json={"settings": {"cost_wrong": None, "cost_abstain": None, "risk_budget": 0.05}})
    assert r.json()["settings"]["cost_wrong"] is None and r.json()["settings"]["risk_budget"] == 0.05
    logged = client.get(f"/v1/decisions/{d['id']}").json()["answers"]["department"]
    assert logged["decision"]["rule"] == "costs" and logged["decision"]["policy_version"] == dec["policy_version"]
    assert logged["decision"]["model_version"] == dec["model_version"]


def test_replay_a_candidate_policy_on_past_decisions(client):
    teach_department(client, n=200)
    rng = random.Random(9)
    for t, d, _ in (ticket(rng) for _ in range(40)):
        r = client.post("/v1/decide", json={"state": t, "questions": {"department": {}}}).json()
        client.post("/v1/feedback", json={"decision_id": r["id"], "answers": {"department": d}})
    before = client.get("/v1/questions/department").json()
    rep = client.post("/v1/questions/department/policy/replay", json={"settings": {"abstain_threshold": 0.99}}).json()
    assert rep["decisions_with_feedback"] == 40
    assert rep["candidate"]["coverage"] <= rep["current"]["coverage"]
    after = client.get("/v1/questions/department").json()
    assert after["version"] == before["version"] and after["policy"] == before["policy"]  # nothing changed
    assert after["settings"]["abstain_threshold"] == 0.6
