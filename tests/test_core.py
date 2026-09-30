import pickle
import random

import pytest

from desic.core import DecisionModel, HoeffdingTree, Schema
from desic.core.drift import DDM, DRIFT
from desic.core.rules import condition_matches, first_match, validate_rule
from desic.core.schema import to_label, to_number
from desic.core.tree import simplify_conditions


def stream(n, seed=0, flip=False):
    rng = random.Random(seed)
    for _ in range(n):
        x = {"income": rng.uniform(0, 100), "age": rng.uniform(18, 80), "city": rng.choice(["a", "b", "c"])}
        y = "yes" if (x["income"] > 50 and x["age"] < 60) or x["city"] == "c" else "no"
        if flip:
            y = "no" if y == "yes" else "yes"
        yield x, y


def test_tree_learns_and_explains():
    tree = HoeffdingTree(grace_period=50, delta=1e-5)
    for x, y in stream(3000):
        tree.learn_one(x, y)
    correct = sum(tree.predict_one(x) == y for x, y in stream(500, seed=1))
    assert correct / 500 > 0.9
    exp = tree.explain_one({"income": 80, "age": 30, "city": "a"})
    assert exp["path"], "a trained tree should have a decision path"
    assert {s["feature"] for s in exp["path"]} <= {"income", "age", "city"}
    rules = tree.rules()
    assert rules and all("prediction" in r for r in rules)
    assert tree.stats()["nodes"] > 1


def test_tree_handles_missing_and_unseen_values():
    tree = HoeffdingTree(grace_period=50)
    for x, y in stream(2000):
        tree.learn_one(x, y)
    assert tree.predict_one({}) in ("yes", "no")
    assert tree.predict_one({"city": "never-seen", "income": None}) in ("yes", "no")


def test_empty_tree_predicts_nothing():
    assert HoeffdingTree().predict_one({"a": 1}) is None


@pytest.mark.parametrize("kind", ["tree", "forest"])
def test_model_adapts_to_drift(kind):
    m = DecisionModel(kind, {"n_trees": 5} if kind == "forest" else None)
    for x, y in stream(3000):
        m.learn(x, y)
    before = m.metrics.summary()["rolling_accuracy"]
    for x, y in stream(3000, seed=2, flip=True):
        m.learn(x, y)
    after = m.metrics.summary()["rolling_accuracy"]
    assert before > 0.85 and after > 0.85
    assert m.structure()["drifts"] >= 1
    assert m.feature_importance()


def test_model_pickles():
    m = DecisionModel("forest", {"n_trees": 3})
    for x, y in stream(500):
        m.learn(x, y)
    m2 = pickle.loads(pickle.dumps(m))
    x = {"income": 70, "age": 30, "city": "a"}
    assert m2.predict(x) == m.predict(x)


def test_feedback_prediction_is_what_gets_scored():
    m = DecisionModel()
    m.learn({"a": 1.0}, "x", evaluated_prediction="y")
    assert m.metrics.summary()["accuracy"] == 0.0
    assert m.metrics.confusion == {"x": {"y": 1}}


def test_ddm_detects_rising_error_with_few_false_alarms():
    false_alarms = 0
    for seed in range(10):
        d = DDM()
        rng = random.Random(seed)
        false_alarms += sum(d.update(rng.random() < 0.05) == DRIFT for _ in range(3000))
        assert any(d.update(rng.random() < 0.3) == DRIFT for _ in range(500)), f"missed drift (seed {seed})"
    assert false_alarms <= 3


def test_rules():
    rule = validate_rule({"decision": "reject", "conditions": [{"feature": "score", "op": "<", "value": 500}], "priority": 1})
    other = validate_rule({"decision": "vip", "conditions": [{"feature": "tier", "op": "in", "value": ["gold", "plat"]}], "priority": 5})
    assert first_match([rule, other], {"score": 400, "tier": "gold"})["decision"] == "vip"
    assert first_match([rule, other], {"score": 400, "tier": "std"})["decision"] == "reject"
    assert first_match([rule, other], {"score": 900}) is None
    assert condition_matches({"feature": "x", "op": "is_missing"}, {})
    assert condition_matches({"feature": "x", "op": "==", "value": "5"}, {"x": 5.0})
    with pytest.raises(ValueError):
        validate_rule({"decision": "a", "conditions": [{"feature": "x", "op": "~"}]})
    with pytest.raises(ValueError):
        validate_rule({"decision": "a", "conditions": []})


def test_simplify_conditions():
    out = simplify_conditions([
        {"feature": "x", "op": "<=", "value": 5}, {"feature": "x", "op": "<=", "value": 3},
        {"feature": "x", "op": ">", "value": 1}, {"feature": "c", "op": "!=", "value": "a"},
        {"feature": "c", "op": "!=", "value": "b"},
    ])
    assert {"feature": "x", "op": "<=", "value": 3} in out
    assert {"feature": "x", "op": ">", "value": 1} in out
    assert {"feature": "c", "op": "not_in", "value": ["a", "b"]} in out


def test_schema_inference_and_coercion():
    rows = [{"age": "31", "city": "izmir", "ok": "1"}, {"age": "", "city": "ankara", "ok": 0.0}, {"age": "4,5", "city": "izmir", "ok": "1"}]
    s = Schema.infer(rows, "ok")
    types = {f.name: f.type for f in s.features}
    assert types == {"age": "numeric", "city": "categorical"}
    assert s.classes == ["1", "0"]
    assert s.coerce({"age": "12", "city": 7, "extra": 1}) == {"age": 12.0, "city": "7"}
    assert to_number("nan") is None and to_number("inf") is None and to_number(True) is None
    assert to_label(1.0) == "1" and to_label(" yes ") == "yes" and to_label("") is None
