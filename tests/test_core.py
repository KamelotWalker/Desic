import pickle
import random

import pytest

from desic.core import DecisionTask, HoeffdingTree, QuestionSpec, SpecError
from desic.core.calibration import TemperatureCalibrator, reliability, risk_coverage, temper
from desic.core.drift import DDM, DRIFT
from desic.core.experts import Mixture
from desic.core.features import Featurizer, flatten, fold, state_hash
from desic.core.rules import condition_matches, first_match, validate_rule
from desic.core.tree import simplify_conditions
from desic.demo import loan, ticket

DEPT = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payments", "technical": "bugs", "sales": "pricing"}}


def trained(name, raw, examples):
    t = DecisionTask(QuestionSpec.parse(name, raw))
    for s, y in examples:
        t.learn(s, y, source="dataset")
    return t


# ------------------------------------------------------------------ spec
def test_spec_parsing_and_labels():
    s = QuestionSpec.parse("dept", DEPT)
    assert s.option_names == ["billing", "technical", "sales"]
    assert s.label_of("Billing") == "billing"
    assert s.label_of("legal") == "legal"  # open world for choice
    n = QuestionSpec.parse("urgent", {"type": "noul", "instructions": "needs help today"})
    assert n.option_names == ["true", "false"]
    assert n.label_of(True) == "true" and n.label_of("evet") == "true" and n.label_of(0) == "false"
    sc = QuestionSpec.parse("sev", {"type": "score", "criteria": ["low", "mid", "high"]})
    assert sc.label_of(2) == "high" and sc.label_of("mid") == "mid"
    with pytest.raises(SpecError):
        sc.label_of("extreme")
    with pytest.raises(SpecError):
        QuestionSpec.parse("x", {"type": "choice", "criteria": ["only"]})
    with pytest.raises(SpecError):
        QuestionSpec.parse("bad name!", DEPT)
    with pytest.raises(SpecError):
        s.merge(QuestionSpec.parse("dept", {"type": "noul"}))


# -------------------------------------------------------------- features
def test_features_text_and_json():
    assert fold("ŞİKAYET ığdır") == "sikayet igdir"
    f = Featurizer()
    x = f.extract("Kartımdan iki kez çekildi")
    assert {"w:kartimdan", "p5:karti", "b:iki_kez"} <= set(x.sparse)
    j = f.extract({"customer": {"plan": "pro", "age": 31}, "tags": ["vip"], "body": "the app crashes when I log in"})
    assert "k:customer.plan=pro" in j.sparse and "k:tags[]=vip" in j.sparse and "w:crashes" in j.sparse
    assert j.scalars["customer.age"] == 31.0 and j.scalars["customer.plan"] == "pro"
    assert flatten("hi") == {"$text": "hi"}
    assert state_hash({"a": 1, "b": 2}) == state_hash({"b": 2, "a": 1})


# ------------------------------------------------------------------ task
def test_text_choice_learns_and_explains():
    rng = random.Random(0)
    t = trained("department", DEPT, [ticket(rng)[:2] for _ in range(400)])
    m = t.metrics.summary()
    assert m["accuracy"] > 0.9 and m["ece"] < 0.1
    a = t.answer("kartımdan fazla para çekildi, iade istiyorum")
    pub = t.public(a)
    assert pub["choice"] == "billing" and not pub["abstain"]
    assert abs(sum(pub["probabilities"].values()) - 1) < 1e-3
    exp = t.explain(a)
    assert exp["experts"]["linear"]["awake"] and exp["linear"]["for"]


def test_json_state_uses_tree():
    rng = random.Random(1)
    t = trained("loan", {"type": "choice", "criteria": ["approve", "review", "reject"]}, [loan(rng) for _ in range(2500)])
    test = [loan(rng) for _ in range(300)]
    acc = sum(t.public(t.answer(s))["choice"] == y for s, y in test) / len(test)
    assert acc > 0.85
    assert t.summary()["expert_weights"]["tree"] > 0.3
    assert "path" in t.explain(t.answer(test[0][0]))["tree"]


def test_unfamiliar_states_are_pulled_toward_dont_know():
    rng = random.Random(6)
    t = trained("department", DEPT, [ticket(rng)[:2] for _ in range(300)])
    familiar = t.answer("please refund my payment, the invoice is wrong")
    strange = t.answer("Zzyzx quorble flanging the vorpal gimbals")
    assert familiar["familiarity"] > 0.5 and not t.public(familiar)["abstain"]
    assert strange["familiarity"] < 0.2 and t.public(strange)["abstain"]


def test_cold_start_abstains():
    t = DecisionTask(QuestionSpec.parse("q", DEPT))
    pub = t.public(t.answer("anything"))
    assert pub["abstain"] and pub["confidence"] == pytest.approx(1 / 3, abs=1e-3)


def test_one_correction_changes_the_next_answer():
    rng = random.Random(2)
    t = trained("department", DEPT, [ticket(rng)[:2] for _ in range(300)])
    s = "My parcel arrived broken and nobody answers"
    before = t.public(t.answer(s))["choice"]
    target = next(o for o in ("sales", "billing", "technical") if o != before)
    t.learn(s, target)
    assert t.public(t.answer(s))["choice"] == target  # the memory expert reacts instantly


def test_noul_and_score():
    rng = random.Random(3)
    tickets = [ticket(rng) for _ in range(500)]
    u = trained("urgent", {"type": "noul", "instructions": "needs help today"}, [(t, str(urg)) for t, _, urg in tickets])
    pub = u.public(u.answer("the app crashes, this is urgent, we are losing sales"))
    assert pub["type"] == "noul" and pub["answer"] is True and 0 <= pub["probability"] <= 1
    sev = trained("sev", {"type": "score", "criteria": ["low", "medium", "high"]},
                  [(t, "high" if d == "technical" and urg else ("medium" if d == "technical" else "low")) for t, d, urg in tickets])
    pub = sev.public(sev.answer("the app crashes as soon as I open it. Please help!"))
    assert pub["level"] == "high" and 1.0 < pub["score"] <= 2.0
    assert sev.metrics.summary()["rps"] is not None


def test_soft_teacher_labels_do_not_count_as_ground_truth():
    t = DecisionTask(QuestionSpec.parse("q", DEPT))
    t.learn("refund please", {"billing": 0.8, "sales": 0.2}, source="teacher")
    assert t.metrics.n == 0 and t.labels_by_source == {"teacher": 1}
    assert t.public(t.answer("refund please"))["choice"] == "billing"


def test_adapts_after_labels_flip():
    rng = random.Random(4)
    words = {"a": ["refund", "charge", "invoice"], "b": ["crash", "error", "bug"]}

    def ex(flip):
        y = rng.choice("ab")
        s = " ".join(rng.sample(words[y], 2) + [rng.choice(["hi", "please", "today"])])
        return s, ({"a": "b", "b": "a"}[y] if flip else y)

    t = trained("q", {"type": "choice", "criteria": ["a", "b"]}, [ex(False) for _ in range(500)])
    for _ in range(300):
        s, y = ex(True)
        t.learn(s, y)
    recent = [r[1] for r in list(t.metrics.records)[-100:]]
    assert sum(recent) / len(recent) > 0.9
    assert any(e["type"] == "drift" for e in t.events)


def test_task_pickles():
    rng = random.Random(5)
    t = trained("department", DEPT, [ticket(rng)[:2] for _ in range(100)])
    t2 = pickle.loads(pickle.dumps(t))
    assert t2.public(t2.answer("refund me")) == t.public(t.answer("refund me"))


# ------------------------------------------------------------ calibration
def test_temperature_calibration_fixes_underconfidence():
    rng = random.Random(0)
    cal = TemperatureCalibrator(refit_every=50, min_samples=50)
    for _ in range(400):  # right 95% of the time but only says 0.6
        correct = rng.random() < 0.95
        raw = {"a": 0.6, "b": 0.4} if correct else {"a": 0.4, "b": 0.6}
        cal.add(raw, "a")
    assert cal.t == pytest.approx(0.5, abs=0.01)  # sharpened, but never beyond the 2x safety bound
    assert 0.65 < cal.apply({"a": 0.6, "b": 0.4})["a"] < 0.75
    over = TemperatureCalibrator(refit_every=50, min_samples=50)
    for _ in range(400):  # says 0.95 but is right only 60% of the time
        over.add({"a": 0.95, "b": 0.05}, "a" if rng.random() < 0.6 else "b")
    assert over.t > 1.5 and over.apply({"a": 0.95, "b": 0.05})["a"] < 0.8
    assert temper({"a": 0.6, "b": 0.4}, 1.0) == {"a": 0.6, "b": 0.4}


def test_reliability_and_risk_coverage():
    recs = [(0.9, True)] * 9 + [(0.9, False)] + [(0.6, True)] * 3 + [(0.6, False)] * 2
    ece, bins = reliability(recs)
    assert ece == pytest.approx(0.0, abs=1e-6) and len(bins) == 2
    rc = risk_coverage(recs, points=3)
    assert rc[0]["accuracy"] >= rc[-1]["accuracy"] and rc[-1]["coverage"] == 1.0


def test_mixture_prefers_the_better_expert_and_recovers():
    m = Mixture(["good", "bad"])
    for _ in range(30):
        m.update({"good": {"x": 0.9, "y": 0.1}, "bad": {"x": 0.2, "y": 0.8}}, {"x": 1.0})
    assert m.weights()["good"] > 0.95
    for _ in range(60):  # the world changes
        m.update({"good": {"x": 0.9, "y": 0.1}, "bad": {"x": 0.2, "y": 0.8}}, {"y": 1.0})
    assert m.weights()["bad"] > 0.9
    m.update({"good": {"x": 1.0, "y": 0.0}, "bad": None}, {"x": 1.0})  # sleeping expert keeps its mass
    assert m.weights()["bad"] > 0.5


# --------------------------------------------------------- tree / drift / rules
def test_hoeffding_tree_learns():
    tree = HoeffdingTree(grace_period=50, delta=1e-5)
    rng = random.Random(0)
    for _ in range(3000):
        x = {"income": rng.uniform(0, 100), "city": rng.choice("abc")}
        tree.learn_one(x, "yes" if x["income"] > 50 or x["city"] == "c" else "no")
    assert tree.predict_one({"income": 90, "city": "a"}) == "yes"
    assert tree.predict_one({"income": 10, "city": "a"}) == "no"
    assert tree.rules()


def test_ddm_detects_rising_error_with_few_false_alarms():
    false_alarms = 0
    for seed in range(10):
        d = DDM()
        rng = random.Random(seed)
        false_alarms += sum(d.update(rng.random() < 0.05) == DRIFT for _ in range(3000))
        assert any(d.update(rng.random() < 0.3) == DRIFT for _ in range(500)), f"missed drift (seed {seed})"
    assert false_alarms <= 3


def test_rules():
    rule = validate_rule({"decision": "reject", "conditions": [{"feature": "applicant.credit_score", "op": "<", "value": 500}], "priority": 1})
    other = validate_rule({"decision": "review", "conditions": [{"feature": "$text", "op": "contains", "value": "lawyer"}], "priority": 5})
    assert first_match([rule, other], {"applicant.credit_score": 400, "$text": "my lawyer"})["decision"] == "review"
    assert first_match([rule, other], {"applicant.credit_score": 400})["decision"] == "reject"
    assert first_match([rule, other], {"applicant.credit_score": 900}) is None
    assert condition_matches({"feature": "x", "op": "is_missing"}, {})
    with pytest.raises(ValueError):
        validate_rule({"decision": "a", "conditions": [{"feature": "x", "op": "~"}]})


def test_simplify_conditions():
    out = simplify_conditions([
        {"feature": "x", "op": "<=", "value": 5}, {"feature": "x", "op": "<=", "value": 3},
        {"feature": "c", "op": "!=", "value": "a"}, {"feature": "c", "op": "!=", "value": "b"},
    ])
    assert {"feature": "x", "op": "<=", "value": 3} in out
    assert {"feature": "c", "op": "not_in", "value": ["a", "b"]} in out


def test_one_uncertain_teacher_label_does_not_create_certainty():
    """Regression (first user trial): a 55% teacher label made the student 94% sure,
    and spilled over to unrelated messages sharing one word."""
    rng = random.Random(1)
    t = trained("urgent", {"type": "noul", "instructions": "needs a response today"},
                [(s, str(u)) for s, _, u in (ticket(rng) for _ in range(600))])
    s = "Merhaba, erken rezervasyon yaptırmak istiyorum"
    other = "Rezervasyonumu iptal etmek istiyorum"
    before_same, before_other = (t.public(t.answer(x))["probability"] for x in (s, other))
    t.learn(s, {"true": 0.549, "false": 0.451}, source="teacher")
    # a coin-flip teacher label must not move the student much, here or elsewhere
    assert abs(t.public(t.answer(s))["probability"] - before_same) < 0.1
    assert abs(t.public(t.answer(other))["probability"] - before_other) < 0.05


def test_a_confident_teacher_label_still_teaches():
    rng = random.Random(1)
    t = trained("urgent", {"type": "noul", "instructions": "needs a response today"},
                [(s, str(u)) for s, _, u in (ticket(rng) for _ in range(600))])
    s = "Merhaba, erken rezervasyon yaptırmak istiyorum"
    before = t.public(t.answer(s))["probability"]
    t.learn(s, {"true": 0.9, "false": 0.1}, source="teacher")
    assert t.public(t.answer(s))["probability"] > before + 0.2


def test_a_human_correction_still_sticks():
    rng = random.Random(1)
    t = trained("urgent", {"type": "noul", "instructions": "needs a response today"},
                [(s, str(u)) for s, _, u in (ticket(rng) for _ in range(600))])
    s = "Merhaba, erken rezervasyon yaptırmak istiyorum"
    t.learn(s, "true")
    assert t.public(t.answer(s))["answer"] is True


def test_unfamiliar_states_abstain_even_when_confident():
    rng = random.Random(6)
    t = trained("department", DEPT, [ticket(rng)[:2] for _ in range(300)])
    a = t.answer("refund zorunlu yoksa mahkemeye başvuracağız avukatımız hazır bekliyor")
    assert a["familiarity"] < 0.5 and a["unfamiliar"]
    assert t.public(a)["abstain"] is True


def test_automatic_snapshots_are_pruned(tmp_path):
    from desic.storage import Storage

    st = Storage(tmp_path / "s.db")
    for v in range(15):
        st.save_snapshot("q", v, v, {}, {"v": v}, "automatic" if v != 3 else "manual")
    kept = [s["version"] for s in st.list_snapshots("q")]
    assert 3 in kept and len(kept) == Storage.KEEP_AUTOMATIC + 1 and max(kept) == 14
