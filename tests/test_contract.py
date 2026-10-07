import random

import pytest

from desic.core import DecisionTask, QuestionSpec
from desic.core.calibration import TaskMetrics
from desic.core.contract import budget_threshold, decide, expected_costs, validate, wilson_upper
from desic.demo import ticket

DEPT = {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "technical", "sales"]}


def test_wilson_upper_bound():
    assert wilson_upper(0, 0) == 1.0
    assert 0.0 < wilson_upper(0, 100) < 0.03
    assert wilson_upper(5, 100) > 0.05  # an upper bound sits above the observed rate
    assert wilson_upper(5, 1000) < wilson_upper(5, 100)


def test_budget_threshold_picks_the_widest_threshold_that_fits():
    # confident answers are right, unsure ones are wrong half the time
    recs = [(0.95, True)] * 200 + [(0.55, i % 2 == 0) for i in range(100)]
    tau = budget_threshold(recs, 0.05)
    assert tau == 0.95
    assert budget_threshold(recs, 0.4) == 0.55  # a loose budget lets it answer everything
    assert budget_threshold(recs[:20], 0.05) is None  # too little evidence yet
    assert budget_threshold([(0.9, False)] * 50, 0.05) > 1  # nothing fits: answer nothing


def test_costs_set_the_threshold():
    d = decide({"a": 0.85, "b": 0.15}, {"cost_wrong": 10, "cost_abstain": 1})
    assert d["threshold"] == 0.9 and d["abstain"] and d["rule"] == "costs"
    d = decide({"a": 0.95, "b": 0.05}, {"cost_wrong": 10, "cost_abstain": 1})
    assert not d["abstain"] and d["expected_cost"] == pytest.approx(0.5)


def test_cost_matrix_can_change_the_answer():
    # approving a bad loan costs 10, rejecting a good one costs 1
    m = {"true": {"false": 10.0}, "false": {"true": 1.0}}
    probs = {"true": 0.7, "false": 0.3}
    assert expected_costs(probs, m, None) == pytest.approx({"true": 3.0, "false": 0.7})
    d = decide(probs, {"cost_matrix": m})
    assert d["answer"] == "false" and not d["abstain"]
    assert decide(probs, {"cost_matrix": m, "cost_abstain": 0.5})["abstain"]  # 0.7 > 0.5: escalate


def test_request_threshold_overrides_the_contract():
    d = decide({"a": 0.7, "b": 0.3}, {"cost_wrong": 10, "cost_abstain": 1}, abstain_threshold=0.5)
    assert d["rule"] == "request_threshold" and not d["abstain"]


def test_validate():
    assert validate({"risk_budget": 0.05, "cost_wrong": 5}, ["a", "b"]) == {"risk_budget": 0.05, "cost_wrong": 5.0}
    assert validate({"risk_budget": None}, ["a"]) == {"risk_budget": None}
    for bad in ({"risk_budget": 1.5}, {"cost_wrong": 0}, {"cost_matrix": {"z": {"a": 1}}}, {"cost_matrix": {"a": {"b": -1}}}):
        with pytest.raises(ValueError):
            validate(bad, ["a", "b"])


def test_risk_budget_keeps_answered_errors_down():
    rng = random.Random(4)
    t = DecisionTask(QuestionSpec.parse("dept", DEPT))
    stream = [ticket(rng)[:2] for _ in range(1500)]
    for s, y in stream[:150]:
        t.learn(s, y, source="dataset")
    t.policy.risk_budget = 0.02
    answered = wrong = 0
    for s, y in stream[150:]:
        a = t.answer(s)
        pub = t.public(a)
        if not pub["abstain"]:
            answered += 1
            wrong += pub["choice"] != y
        t.learn(s, y, source="dataset", served=a["probabilities"], served_raw=a["raw"], abstained=pub["abstain"])
    assert pub["decision"]["rule"] == "risk_budget"
    assert answered > 500 and wrong / answered <= 0.03


def test_expected_mode_trusts_calibration_when_labels_are_scarce():
    from desic.core.contract import expected_threshold

    served = [0.97] * 300 + [0.7] * 300  # calibrated: 3% and 30% wrong
    labels = [(0.97, i % 33 != 0) for i in range(15)]  # only 15 labels, none wrong among them
    assert budget_threshold(labels, 0.08) is None  # the guaranteed mode cannot even start
    assert expected_threshold(labels, served, 0.08) == 0.97
    # predicted (300·0.03 + 300·0.3)/600 = 16.5%, plus the labels' bias correction 1/15 − 0.03 = 3.7%
    assert expected_threshold(labels, served, 0.25) == 0.7
    assert expected_threshold(labels, served, 0.2) == 0.97
    # labels that contradict the confidences pull the estimate up
    bad = [(0.97, i % 2 == 0) for i in range(20)]
    assert expected_threshold(bad, served, 0.08) > 1


def test_validate_mode():
    assert validate({"risk_budget_mode": "expected"}, ["a"]) == {"risk_budget_mode": "expected"}
    with pytest.raises(ValueError):
        validate({"risk_budget_mode": "yolo"}, ["a"])
