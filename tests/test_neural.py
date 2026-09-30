"""Neural student: training, gating, shadow, promotion — skipped without PyTorch."""

import random
import time

import pytest

torch = pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from desic.api import create_app  # noqa: E402
from desic.demo import ticket  # noqa: E402
from desic.neural.model import DecisionNet, ScratchBackbone, collate  # noqa: E402
from desic.neural.runtime import NeuralManager  # noqa: E402
from desic.neural.text import MARK, ScratchTokenizer, question_text, state_to_text  # noqa: E402

DEPT = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payments", "technical": "bugs", "sales": "pricing"}}
FAST = {"epochs": 3, "batch_size": 16, "max_len": 96}


def wait_job(client, job_id, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def seed_questions(client, n=500):
    client.post("/v1/questions", json={"name": "department", **DEPT})
    client.post("/v1/questions", json={"name": "urgent", "type": "noul", "instructions": "needs a response today"})
    rng = random.Random(0)
    tickets = [ticket(rng) for _ in range(n)]
    client.post("/v1/questions/department/learn", json={"examples": [{"state": t, "answer": d} for t, d, _ in tickets]})
    client.post("/v1/questions/urgent/learn", json={"examples": [{"state": t, "answer": u} for t, _, u in tickets]})
    return tickets


def train(client, **cfg):
    r = client.patch("/v1/neural/config", json={**FAST, **cfg})
    assert r.status_code == 200, r.text
    job = wait_job(client, client.post("/v1/neural/train").json()["id"])
    assert job["status"] == "done", job
    return job["result"]


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(str(tmp_path))) as c:
        yield c


# ------------------------------------------------------------------ units
def test_text_and_collate():
    first = question_text("choice", "Which team?", {"billing": "payments", "sales": ""})
    assert first.count(MARK) == 2
    assert state_to_text({"a": {"b": 1}, "tags": ["x", "y"]}) == "a.b: 1 | tags[]: x, y"
    tok = ScratchTokenizer()
    ids = tok.encode(first, "refund please", 64)
    assert ids[0] == tok.CLS and ids.count(tok.MASK) == 2
    assert tok.encode(question_text("choice", "q", {f"o{i}": "long description here" for i in range(60)}), "x", 64) is None
    net = DecisionNet(ScratchBackbone(hidden=32, heads=2, max_len=64))
    batch = [tok.encode(first, "refund", 64), tok.encode(question_text("noul", "urgent?", {"true": "", "false": ""}), "asap", 64)]
    ids_t, attn, markers = collate(batch, tok.MASK, torch.device("cpu"))
    logits, act = net(ids_t, attn, markers)
    assert logits.shape == (2, 2) and act.shape == (2,)
    assert torch.isfinite(logits).all()


def test_offline_gate():
    ok = {"test": {"overall": {"n": 50, "nll": 0.3, "accuracy": 0.9}}, "prior": {"nll": 1.0}}
    assert NeuralManager._gate(ok)[0] == "shadow"
    useless = {"test": {"overall": {"n": 50, "nll": 1.1, "accuracy": 0.4}}, "prior": {"nll": 1.0}}
    assert NeuralManager._gate(useless)[0] == "rejected"
    worse = {**ok, "baseline": {"overall": {"n": 50, "nll": 0.2}}}
    assert NeuralManager._gate(worse)[0] == "rejected"


# -------------------------------------------------------------- end to end
def test_train_promote_and_decide_with_the_neural_expert(client, tmp_path):
    seed_questions(client)
    status = client.get("/v1/neural").json()
    assert status["available"] and status["active"] is None
    res = train(client, shadow_min=0)
    assert res["status"] == "active", res
    assert res["test"]["accuracy"] > 0.8

    r = client.post("/v1/decide", json={"state": "the app keeps crashing after login, please help today",
                                        "questions": {"department": {}, "urgent": {}}, "explain": True}).json()
    exp = r["answers"]["department"]["explanation"]
    assert exp["experts"]["neural"]["awake"] is True
    assert 0.0 <= exp["neural"]["act"] <= 1.0
    q = client.get("/v1/questions/department").json()
    assert "neural" in q["expert_weights"] and q["neural_attached"]

    ck = client.get("/v1/neural").json()["checkpoints"][0]
    assert ck["status"] == "active" and set(ck["report"]["temperatures"]) == {"department", "urgent"}
    assert ck["report"]["prior"]["nll"] > ck["report"]["test"]["overall"]["nll"]


def test_neural_survives_restart(tmp_path):
    with TestClient(create_app(str(tmp_path))) as c:
        seed_questions(c, n=300)
        ck = train(c, shadow_min=0)["checkpoint"]
    with TestClient(create_app(str(tmp_path))) as c:
        assert c.get("/v1/health").json()["neural"] == ck
        r = c.post("/v1/decide", json={"state": "refund my invoice", "questions": {"department": {}}, "explain": True}).json()
        assert r["answers"]["department"]["explanation"]["experts"]["neural"]["awake"]


def test_shadow_is_judged_on_live_feedback(client):
    tickets = seed_questions(client)
    res = train(client, shadow_min=5)
    assert res["status"] == "shadow"
    for t, d, _ in tickets[:5]:
        r = client.post("/v1/decide", json={"state": t + " (again)", "questions": {"department": {}}}).json()
        client.post("/v1/feedback", json={"decision_id": r["id"], "answers": {"department": d}})
    ck = client.get("/v1/neural").json()["checkpoints"][0]
    assert ck["shadow"]["n"] == 5
    assert ck["status"] in ("active", "rejected")


def test_manual_actions_and_second_candidate_is_compared(client):
    seed_questions(client, n=400)
    first = train(client, shadow_min=0)["checkpoint"]
    second = train(client, shadow_min=100)  # continues from the active checkpoint, must not be worse
    ck2 = next(c for c in client.get("/v1/neural").json()["checkpoints"] if c["id"] == second["checkpoint"])
    assert ck2["report"]["baseline"]["checkpoint"] == first
    assert ck2["report"]["init"].startswith("continued from")
    if ck2["status"] == "shadow":
        s = client.post(f"/v1/neural/checkpoints/{ck2['id']}/promote").json()
        assert s["active"] == ck2["id"]
    s = client.post(f"/v1/neural/checkpoints/{client.get('/v1/neural').json()['active']}/retire").json()
    assert s["active"] is None
    r = client.post("/v1/decide", json={"state": "refund", "questions": {"department": {}}, "explain": True}).json()
    assert "neural" not in r["answers"]["department"]["explanation"]["experts"] or \
        not r["answers"]["department"]["explanation"]["experts"]["neural"]["awake"]
    assert client.post("/v1/neural/checkpoints/nope/promote").status_code == 404


def test_rlcd_objective_trains(client):
    seed_questions(client, n=300)
    res = train(client, objective="rlcd", shadow_min=0, epochs=2)
    ck = client.get("/v1/neural").json()["checkpoints"][0]
    assert ck["report"]["train"]["objective"] == "rlcd" and res["test"]["n"] > 0


def test_config_validation(client):
    assert client.patch("/v1/neural/config", json={"objective": "magic"}).status_code == 400
    assert client.patch("/v1/neural/config", json={"nope": 1}).status_code == 400
    assert client.post("/v1/neural/train").status_code in (202, 503)


def test_hugging_face_backbone_from_a_local_directory(client, tmp_path):
    """Same code path as ModernBERT / mmBERT, with a tiny randomly initialised BERT (no download)."""
    transformers = pytest.importorskip("transformers")
    tickets = seed_questions(client, n=300)
    words = sorted({w for t, _, _ in tickets for w in t.lower().replace(".", " ").replace(",", " ").replace("!", " ").split()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ":", "choice", "noul", "which", "team", "?", "billing", "technical",
             "sales", "payments", "bugs", "pricing", "true", "false", "yes", "no", "the", "statement", "is", "needs", "a",
             "response", "today", *words]
    d = tmp_path / "tiny-bert"
    d.mkdir()
    (d / "vocab.txt").write_text("\n".join(dict.fromkeys(vocab)))
    tok = transformers.BertTokenizerFast(vocab_file=str(d / "vocab.txt"))
    tok.save_pretrained(d)
    cfg = transformers.BertConfig(vocab_size=len(tok), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                                  intermediate_size=128, max_position_embeddings=128)
    transformers.BertModel(cfg).save_pretrained(d)
    res = train(client, backbone=str(d), max_len=96, shadow_min=0, epochs=2, lr_backbone=5e-4)
    assert res["status"] in ("active", "rejected")
    ck = client.get("/v1/neural").json()["checkpoints"][0]
    assert ck["backbone"] == str(d)
    if res["status"] == "active":
        r = client.post("/v1/decide", json={"state": "refund please", "questions": {"department": {}}, "explain": True}).json()
        assert r["answers"]["department"]["explanation"]["experts"]["neural"]["awake"]
