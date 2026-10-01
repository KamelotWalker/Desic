import pickle
import random

from desic.core import DecisionTask, QuestionSpec
from desic.core.patches import PatchedTask
from desic.core.policy import DecisionPolicy, replay
from desic.demo import ticket

DEPT = {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "technical", "sales"]}


def trained(n=200, patched=False):
    t = DecisionTask(QuestionSpec.parse("dept", DEPT))
    t = PatchedTask(t) if patched else t
    rng = random.Random(1)
    for s, y in (ticket(rng)[:2] for _ in range(n)):
        t.learn(s, y, source="dataset")
    return t


def test_policy_changes_do_not_touch_the_model():
    """P1 (RESEARCH.md): policy and model state are versioned separately."""
    for patched in (False, True):
        t = trained(patched=patched)
        probe = "please refund my double charge"
        before, model_v, policy_v = t.answer(probe)["probabilities"], t.version, t.policy.version
        assert t.policy.update({"risk_budget": 0.05, "cost_wrong": 10, "cost_abstain": 1}, t.spec.option_names)
        assert t.version == model_v and t.policy.version == policy_v + 1
        assert t.answer(probe)["probabilities"] == before  # the model answers exactly as before
        assert not t.policy.update({"risk_budget": 0.05}, t.spec.option_names)  # no change, no new version
        t.learn(probe, "billing", source="human")
        assert t.version > model_v and t.policy.version == policy_v + 1  # learning leaves the policy alone
        pub = t.public(t.answer(probe))
        assert pub["decision"]["policy_version"] == policy_v + 1


def test_students_saved_with_contract_settings_get_a_policy():
    t = trained(n=50)
    state = dict(t.__dict__)
    del state["policy"]
    state["settings"] = {**t.settings, "abstain_threshold": 0.7, "risk_budget": 0.05, "cost_wrong": None}
    old = DecisionTask.__new__(DecisionTask)
    old.__setstate__(pickle.loads(pickle.dumps(state)))
    assert old.policy.abstain_threshold == 0.7 and old.policy.risk_budget == 0.05
    assert "risk_budget" not in old.settings and old.settings["teacher_weight"] == 0.5


def test_replay_a_candidate_policy_on_logged_decisions():
    logged = [({"a": 0.95, "b": 0.05}, "a")] * 80 + [({"a": 0.6, "b": 0.4}, "b")] * 20
    loose = replay(DecisionPolicy(abstain_threshold=0.5), logged)
    strict = replay(DecisionPolicy(abstain_threshold=0.9), logged)
    assert loose["coverage"] == 1.0 and loose["risk"] == 0.2
    assert strict["coverage"] == 0.8 and strict["risk"] == 0.0
    priced = replay(DecisionPolicy(cost_wrong=10, cost_abstain=1), logged)  # threshold 0.9
    assert priced["total_cost"] == 20.0 and priced["cost_per_decision"] == 0.2


def test_escalation_aware_answers_when_as_sure_as_the_teacher():
    """S2 (RESEARCH.md): escalating to a teacher that is right 80% of the time is worse
    than answering at 85% confidence."""
    from desic.core.calibration import TaskMetrics

    m = TaskMetrics()
    probs = {"a": 0.85, "b": 0.15}
    pol = DecisionPolicy(abstain_threshold=0.9, escalation_aware=True)
    assert pol.decide(probs, m)["abstain"]  # teacher accuracy unknown yet: the threshold rules
    for i in range(30):
        m.record_served("teacher", i % 5 != 0)  # 80% right
    d = pol.decide(probs, m)
    assert not d["abstain"] and d["rule"].endswith("+escalation_aware") and d["teacher_accuracy"] == 0.8
    assert pol.decide({"a": 0.7, "b": 0.3}, m)["abstain"]  # less sure than the teacher: still escalates
    low = DecisionPolicy(abstain_threshold=0.6, escalation_aware=True)
    assert low.decide({"a": 0.7, "b": 0.3}, m)["abstain"]  # a lax hand-set threshold is raised to the teacher's accuracy
    assert DecisionPolicy(abstain_threshold=0.9).decide(probs, m)["abstain"]  # off by default
    priced = DecisionPolicy(cost_wrong=10, cost_abstain=1, escalation_aware=True)
    assert priced.decide(probs, m)["abstain"]  # explicit costs already price the escalation
