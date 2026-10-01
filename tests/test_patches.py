import pickle
import random

from desic.core import DecisionTask, QuestionSpec
from desic.core.patches import PatchedTask
from desic.demo import ticket

DEPT = {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "technical", "sales"]}


def task(**kw):
    return PatchedTask(DecisionTask(QuestionSpec.parse("dept", DEPT)), **kw)


def rows(n, seed=0):
    rng = random.Random(seed)
    return [ticket(rng)[:2] for _ in range(n)]


def answers(t, states):
    return [t.answer(s)["probabilities"] for s in states]


def test_learns_immediately_before_the_base_does():
    t = task(probation=1000, min_probation=1000)
    for s, y in rows(200):
        t.learn(s, y, source="dataset")
    assert t.base.labels == 0 and len(t.pending) == 200  # nothing consolidated yet
    probe = rows(50, seed=1)
    acc = sum(max(a, key=a.get) == y for a, (_, y) in zip(answers(t, [s for s, _ in probe]), probe)) / 50
    assert acc > 0.8


def test_retracting_patches_on_probation_is_exact():
    data, probe = rows(300), [s for s, _ in rows(40, seed=2)]
    t = task(probation=1000, min_probation=1000)
    for s, y in data:
        t.learn(s, y, source="dataset")
    before = answers(t, probe)
    wrong = [s for s, y in data if y == "billing"][:15]
    for i, s in enumerate(wrong):
        t.learn(s, "sales", source="human", ref=f"bad-{i}")
    assert answers(t, probe) != before
    stats = t.retract(f"bad-{i}" for i in range(15))
    assert stats.pop("seconds") < 1.0
    assert stats == {"retracted": 15, "on_probation": 15, "consolidated": 0, "replayed": 0}
    after = answers(t, probe)
    assert all(abs(a[o] - b[o]) < 1e-9 for a, b in zip(after, before) for o in a)


def test_retracting_consolidated_labels_matches_never_having_seen_them():
    data = rows(400)
    bad = [(s, "sales") for s, y in data[:200] if y == "billing"][:10]
    stream = data[:200] + bad + data[200:]
    t = task(probation=20, replay=0, checkpoint_every=50)
    for i, (s, y) in enumerate(stream):
        t.learn(s, y, source="dataset", ref=f"e{i}")
    bad_refs = [f"e{i}" for i in range(200, 200 + len(bad))]
    stats = t.retract(bad_refs)
    assert stats["consolidated"] == len(bad) and 0 < stats["replayed"] < 400 - 20  # from a checkpoint, not from zero
    clean = task(probation=20, replay=0, checkpoint_every=50)
    for s, y in data:
        clean.learn(s, y, source="dataset")
    t.flush()
    clean.flush()
    probe = [s for s, _ in rows(30, seed=3)]
    for x in probe:
        a, b = t.base.answer(x)["probabilities"], clean.base.answer(x)["probabilities"]
        assert all(abs(a[o] - b[o]) < 1e-9 for o in a)


def test_a_contradicted_patch_loses_trust():
    t = task(probation=1000, min_probation=1000)
    t.learn("refund my double charge please", "sales", source="human")
    p = next(iter(t.store.entries.values()))
    for _ in range(3):
        t.learn("refund my double charge please now", "billing", source="human")
    assert p.trust(t.store.trust_prior) < 0.5


def test_memory_is_class_balanced():
    t = task(probation=0, min_probation=0, capacity=60)
    for s, y in rows(300):
        t.learn(s, y, source="dataset")
    sizes = sorted(len(v) for v in t.store.by_label.values())
    assert len(t.store) == 60 and sizes[-1] - sizes[0] <= 1


def test_survives_pickling_and_reports_patch_weight():
    t = task(probation=5, min_probation=5)
    for s, y in rows(50):
        t.learn(s, y, source="dataset")
    t2 = pickle.loads(pickle.dumps(t))
    a = t2.answer(rows(1, seed=9)[0][0])
    assert a["patch"] is not None and "patch" in a["weights"]
    assert t2.public(a)["choice"] in DEPT["criteria"]
    assert t2.summary()["patches"]["consolidated"] == 45


def test_gate_per_answer_falls_back_to_the_shared_cell():
    from desic.core.patches import Gate

    g = Gate()
    for i in range(20):  # patches keep beating the base when they propose "sales"
        g.update(i, "3≠:sales", 2.0, 0.1, 1.0)
    assert g.weights("3≠:sales")[1] > 0.9
    shared = g.weights("3≠")[1]
    assert g.weights("3≠:billing")[1] == shared  # an answer not seen yet starts from the shared cell
    g.forget(set(range(20)))
    assert g.weights("3≠:sales") == (0.5, 0.5)


def test_only_the_wrapped_base_checkpoint_is_saved():
    t = task(probation=5, min_probation=5, replay=0, checkpoint_every=20)
    data = rows(120)
    for i, (s, y) in enumerate(data):
        t.learn(s, y, source="dataset", ref=f"e{i}")
    assert len(t.checkpoints) > 1
    t2 = pickle.loads(pickle.dumps(t))
    assert [c[0] for c in t2.checkpoints] == [0]
    stats = t2.retract(["e10"])  # still correct after a restart, it just replays from the start
    assert stats["consolidated"] == 1 and stats["replayed"] == len(t2.log)


def test_fading_trust_is_undone_exactly():
    t = task(probation=1000, min_probation=1000, trust_halflife=50)
    t.learn("refund my double charge please", "sales", source="human")
    p = next(iter(t.store.entries.values()))
    for s, y in rows(60):
        t.learn(s, y, source="dataset")
    before = t.store.trust(p, t.t)
    t.learn("refund my double charge please now", "billing", source="human", ref="x")
    assert t.store.trust(p, t.t) < before
    t.retract(["x"])
    assert abs(t.store.trust(p, t.t) - before) < 1e-9


def test_gate_prior_and_sources():
    from desic.core.patches import Gate

    assert abs(Gate(prior=0.2).weights("2≠")[1] - 0.2) < 1e-9
    t = task(probation=1000, min_probation=1000, gate_sources=("human",))
    for s, y in rows(30):
        t.learn(s, y, source="dataset")
    assert t.gate.history == []  # dataset labels did not teach the gate


def test_retracted_labels_leave_the_risk_controller_too():
    """R1 (RESEARCH.md): after a retraction the risk budget sees exactly what a system
    that never got those labels would see."""
    from desic.core.contract import _budget_tau

    data, extra = rows(300), rows(60, seed=4)
    bad = [(s, "sales" if y != "sales" else "billing") for s, y in rows(40, seed=5)]

    def run(with_bad: bool):
        t = task(probation=1000, min_probation=1000)
        for s, y in data:
            t.learn(s, y, source="human")
        if with_bad:
            for i, (s, y) in enumerate(bad):
                t.learn(s, y, source="human", ref=f"bad-{i}")
        return t

    clean, dirty = run(False), run(True)
    assert dirty.metrics.n == clean.metrics.n + len(bad)
    tau_dirty = _budget_tau(dirty.metrics, 0.05)
    dirty.retract(f"bad-{i}" for i in range(len(bad)))
    m, c = dirty.metrics, clean.metrics
    assert (m.n, m.correct, len(m.records)) == (c.n, c.correct, len(c.records))
    assert abs(m.nll_sum - c.nll_sum) < 1e-9 and m.confusion == c.confusion
    assert [r[:2] for r in m.records] == [r[:2] for r in c.records]
    assert _budget_tau(m, 0.05) == _budget_tau(c, 0.05) != tau_dirty
    assert dirty.labels == clean.labels
