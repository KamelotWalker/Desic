"""Architectural guarantees of the runtime (RESEARCH.md, "Garantiler").

Each test states a guarantee as an equality with a counterfactual system. A test
marked xfail documents a guarantee the current architecture does NOT provide.
"""

import random
import tempfile

import pytest

from desic.core import DecisionTask, QuestionSpec
from desic.core.patches import PatchedTask
from desic.demo import ticket

DEPT = {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "technical", "sales"]}


def _stream():
    rng = random.Random(0)
    data = [ticket(rng)[:2] for _ in range(700)]
    bad = [(s, "sales") for s, y in data[:200] if y == "billing"][:15]
    clean = [(s, y, f"e{i}") for i, (s, y) in enumerate(data)]
    dirty = clean[:300] + [(s, y, f"bad{i}") for i, (s, y) in enumerate(bad)] + clean[300:]
    more = [(s, y, f"m{i}") for i, (s, y) in enumerate(ticket(random.Random(5))[:2] for _ in range(300))]
    return clean, dirty, more, [f"bad{i}" for i in range(len(bad))]


def _run(stream, **kw):
    t = PatchedTask(DecisionTask(QuestionSpec.parse("dept", DEPT)), **kw)
    for s, y, ref in stream:
        t.learn(s, y, source="dataset", ref=ref)
    return t


def _max_weight_gap(a, b):
    wa, wb = a.base.linear.w, b.base.linear.w
    return max((abs(wa.get(o, {}).get(f, 0) - wb.get(o, {}).get(f, 0))
                for o in set(wa) | set(wb) for f in set(wa.get(o, {})) | set(wb.get(o, {}))), default=0.0)


@pytest.mark.parametrize("replay", [0, 1])
@pytest.mark.xfail(strict=True, reason="G1 not provided yet: retracted labels keep their slot in logical time, "
                   "and rehearsal draws depend on them (RESEARCH.md G1)")
def test_retract_then_continue_equals_never_seen(replay):
    clean, dirty, more, bad_refs = _stream()
    c, d = _run(clean, replay=replay), _run(dirty, replay=replay)
    d.retract(bad_refs)
    for s, y, ref in more:
        c.learn(s, y, source="dataset", ref=ref)
        d.learn(s, y, source="dataset", ref=ref)
    assert _max_weight_gap(c, d) < 1e-9 and c.base.metrics.n == d.base.metrics.n


def test_model_operations_keep_the_decision_policy():
    from desic.service import Desic

    svc = Desic(tempfile.mkdtemp())
    svc.create_task(QuestionSpec.parse("dept", DEPT))
    rng = random.Random(1)
    svc.learn("dept", [{"state": s, "answer": y} for s, y in (ticket(rng)[:2] for _ in range(80))])
    snap = svc.snapshot("dept")
    svc.update_task("dept", {"settings": {"risk_budget": 0.05, "cost_wrong": 10, "cost_abstain": 1}})
    want = svc.get("dept").policy.settings()
    svc.reset_task("dept")
    assert svc.get("dept").policy.settings() == want
    svc.rollback("dept", snap["version"])  # the snapshot predates the policy change: the model rolls back, the policy does not
    assert svc.get("dept").policy.settings() == want
    import asyncio
    job = svc.new_job("rebuild")
    asyncio.run(svc.rebuild(job, "dept"))
    assert job.status == "done" and svc.get("dept").policy.settings() == want
    assert svc.get("dept").public(svc.get("dept").answer("refund please"))["decision"]["rule"].startswith("costs")
