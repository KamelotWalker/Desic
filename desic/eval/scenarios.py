"""Evaluation scenarios: how a learner behaves over a stream, not only where it ends.

Each scenario takes a learner factory ``make(classes) -> task`` (anything with
``answer`` / ``public`` / ``learn`` like :class:`DecisionTask`), labelled
``train`` and ``test`` pairs, the class list and a seed. The stream order, noise,
attacked classes and teacher mistakes all derive from the seed, so every learner
variant sees exactly the same stream.

Each returns a dict with a ``headline`` of the numbers worth comparing across
stages and the curves behind them.
"""

from __future__ import annotations

import copy
import random
import time
from typing import Any, Callable

from ..core.calibration import confidence_of
from ..core.patches import PatchedTask
from ..core.task import DecisionTask, QuestionSpec
from .metrics import Prequential, evaluate, forgetting_index, half_life

Item = tuple[str, str]
Make = Callable[[list[str]], Any]


def make_desic(classes: list[str], settings: dict | None = None) -> DecisionTask:
    return DecisionTask(QuestionSpec.parse("intent", {"type": "choice", "instructions": "Which intent is this?",
                                                      "criteria": list(classes)}), settings)


def make_patched(classes: list[str], settings: dict | None = None, **kw: Any) -> PatchedTask:
    return PatchedTask(make_desic(classes, settings), **kw)


LEARNERS: dict[str, Make] = {"desic": make_desic, "desic+patch": make_patched}


def _shuffled(train: list[Item], seed: int) -> list[Item]:
    stream = list(train)
    random.Random(seed).shuffle(stream)
    return stream


ANNOTATORS = 6  # burst and drift streams are labelled round-robin by this many annotators


def _who(i: int) -> str:
    return f"a{i % ANNOTATORS}"


def _step(task: Any, pq: Prequential | None, x: str, given: str, true: str | None = None, ref: str | None = None,
          annotator: str | None = None) -> dict:
    """Serve an answer, score it against the true label, then learn the given label."""
    a = task.answer(x)
    if pq is not None:
        pq.add(a["probabilities"], given if true is None else true)
    task.learn(x, given, source="dataset", served=a["probabilities"], served_raw=a["raw"], ref=ref, annotator=annotator)
    return a


def _top(task: Any, x: str) -> str:
    return confidence_of(task.answer(x)["probabilities"])[0]


def _share(task: Any, items: list[Item], label: str) -> float:
    return round(sum(_top(task, x) == label for x, _ in items) / len(items), 4) if items else 0.0


def _brief(ev: dict) -> dict:
    return {k: ev[k] for k in ("accuracy", "nll", "ece", "coverage", "answered_accuracy", "risk@80%", "aurc") if k in ev}


# --------------------------------------------------------------- scenarios
def shuffled(make: Make, train: list[Item], test: list[Item], classes: list[str], seed: int) -> dict:
    """The friendly case: a shuffled stream, with a learning curve on the test set."""
    stream = _shuffled(train, seed)
    task, pq = make(classes), Prequential()
    marks = {m for m in (500, 1000, 2500, 5000) if m < len(stream)}
    curve = []
    t0 = time.time()
    for i, (x, y) in enumerate(stream, 1):
        _step(task, pq, x, y)
        if i in marks:
            curve.append({"labels": i, **_brief(evaluate(task, test))})
    final = evaluate(task, test)
    return {
        "headline": {"accuracy": final["accuracy"], "nll": final["nll"], "ece": final["ece"],
                     "risk@80%": final["risk@80%"], "aurc": final["aurc"], "coverage": final["coverage"],
                     "answered_accuracy": final["answered_accuracy"], "cum_log_loss": pq.summary()["cum_log_loss"]},
        "final": final, "learning_curve": curve + [{"labels": len(stream), **_brief(final)}],
        "prequential": pq.summary(), "seconds": round(time.time() - t0, 1),
    }


def class_sorted(make: Make, train: list[Item], test: list[Item], classes: list[str], seed: int,
                 checkpoints: int = 11) -> dict:
    """Class-incremental stream: classes arrive one after another (like the Banking77
    train file). Measures how much of an earlier class is forgotten."""
    rng = random.Random(seed)
    order = list(classes)
    rng.shuffle(order)
    by: dict[str, list[Item]] = {}
    for it in train:
        by.setdefault(it[1], []).append(it)
    for rows in by.values():
        rng.shuffle(rows)
    k = min(checkpoints, len(order))
    groups = [order[i * len(order) // k:(i + 1) * len(order) // k] for i in range(k)]
    task, pq = make(classes), Prequential()
    per_class, learned_at, curve = [], {}, []
    t0 = time.time()
    for g, group in enumerate(groups):
        for c in group:
            for x, y in by.get(c, []):
                _step(task, pq, x, y)
            learned_at[c] = g
        ev = evaluate(task, test, per_class=True)
        per_class.append(ev["per_class"])
        seen = [c for grp in groups[:g + 1] for c in grp if c in ev["per_class"]]
        curve.append({"classes_seen": len(seen), "accuracy_all": ev["accuracy"],
                      "accuracy_seen": round(sum(ev["per_class"][c] for c in seen) / max(len(seen), 1), 4),
                      "accuracy_newest_group": round(sum(ev["per_class"].get(c, 0.0) for c in group) / len(group), 4)})
    final = {k_: v for k_, v in ev.items() if k_ != "per_class"}
    fi = forgetting_index(per_class, learned_at)
    return {
        "headline": {"accuracy": final["accuracy"], "nll": final["nll"], "ece": final["ece"], "forgetting_index": fi,
                     "cum_log_loss": pq.summary()["cum_log_loss"]},
        "final": final, "curve": curve, "prequential": pq.summary(), "seconds": round(time.time() - t0, 1),
    }


def noisy(make: Make, train: list[Item], test: list[Item], classes: list[str], seed: int, rate: float) -> dict:
    """Shuffled stream where a ``rate`` share of labels is silently wrong (a random other class)."""
    stream = _shuffled(train, seed)
    nrng = random.Random(seed * 1000 + 7)
    task, pq = make(classes), Prequential()
    poisoned: list[tuple[str, str, str]] = []
    t0 = time.time()
    for x, y in stream:
        given = y
        if nrng.random() < rate:
            given = nrng.choice([c for c in classes if c != y])
            poisoned.append((x, y, given))
        _step(task, pq, x, given, true=y)
    final = evaluate(task, test)
    # asked again about the very messages it was mislabelled on, does it repeat the lie?
    wrong = sum(_top(task, x) == g for x, _, g in poisoned)
    right = sum(_top(task, x) == y for x, y, _ in poisoned)
    n = max(len(poisoned), 1)
    return {
        "headline": {"accuracy": final["accuracy"], "nll": final["nll"], "ece": final["ece"],
                     "risk@80%": final["risk@80%"], "poison_repeated": round(wrong / n, 4),
                     "cum_log_loss": pq.summary()["cum_log_loss"]},
        "final": final, "poisoned_labels": len(poisoned), "poison_repeated": round(wrong / n, 4),
        "poison_state_correct": round(right / n, 4), "prequential": pq.summary(), "seconds": round(time.time() - t0, 1),
    }


def burst(make: Make, train: list[Item], test: list[Item], classes: list[str], seed: int, size: int = 30,
          at: float = 0.5, probe_every: int = 100, harm_at: tuple[int, ...] = (600, 1000)) -> dict:
    """A burst of ``size`` consecutive wrong labels: messages of a victim class A
    labelled as class B (a careless or malicious annotator), in the middle of a
    clean stream. Measures the damage, the recovery from later clean labels, and
    what undoing the burst costs today (rebuilding the student from the clean log)."""
    stream = _shuffled(train, seed)
    split = int(len(stream) * at)
    post = stream[split:]
    brng = random.Random(seed * 1000 + 11)
    counts = {c: sum(y == c for _, y in post) for c in classes}
    victims = [c for c in classes if counts[c] >= size + 10] or sorted(classes, key=lambda c: -counts[c])[:1]
    a_cls = brng.choice(victims)
    b_cls = brng.choice([c for c in classes if c != a_cls])
    take = [i for i, (_, y) in enumerate(post) if y == a_cls][:size]
    attack = [(post[i][0], b_cls) for i in take]
    taken = set(take)
    rest = [it for i, it in enumerate(post) if i not in taken]
    a_test = [it for it in test if it[1] == a_cls]

    task, pq = make(classes), Prequential()
    t0 = time.time()
    for i, (x, y) in enumerate(stream[:split]):
        _step(task, pq, x, y, annotator=_who(i))
    before, before_a = evaluate(task, test), evaluate(task, a_test)
    for i, (x, g) in enumerate(attack):
        _step(task, pq, x, g, true=a_cls, ref=f"burst-{i}", annotator="a0")  # one compromised annotator
    others_test = [it for it in test if it[1] != a_cls]
    after, after_a = evaluate(task, test), evaluate(task, a_test)
    others_before = round((before["accuracy"] * len(test) - before_a["accuracy"] * len(a_test)) / len(others_test), 4)
    others_after = round((after["accuracy"] * len(test) - after_a["accuracy"] * len(a_test)) / len(others_test), 4)
    as_b = _share(task, a_test, b_cls)

    # Undo: a learner that can retract labels does so on a copy; otherwise the only
    # way back is to throw the student away and replay the log without the burst.
    if hasattr(task, "retract"):
        clean = copy.deepcopy(task)
        u0 = time.time()
        stats = clean.retract(f"burst-{i}" for i in range(len(attack)))
        undo_seconds = round(time.time() - u0, 4)
        undo_events = stats["replayed"]
    else:
        u0 = time.time()
        clean = make(classes)
        for i, (x, y) in enumerate(stream[:split]):
            clean.learn(x, y, source="dataset", annotator=_who(i))
        undo_seconds = round(time.time() - u0, 4)
        undo_events = split
    undo, undo_a = evaluate(clean, test), evaluate(clean, a_test)

    # Recovery without undo: keep learning from the clean remainder of the stream. The undone
    # copy (a twin that never saw the burst) learns the same labels alongside, so the harm the
    # burst still does to the *other* classes can be measured at any later point, in particular
    # after the burst has left probation and reached the base model (B1 in RESEARCH.md).
    def others_acc(t: Any) -> float:
        return evaluate(t, others_test)["accuracy"]

    harm = {0: round(others_acc(clean) - others_acc(task), 4)}
    rec = [(0, after_a["accuracy"])]
    last = max(harm_at) if harm_at else 0
    for i, (x, y) in enumerate(rest, 1):
        _step(task, pq, x, y, annotator=_who(i))
        if i <= last:
            _step(clean, None, x, y, annotator=_who(i))
            if i in harm_at:
                harm[i] = round(others_acc(clean) - others_acc(task), 4)
        if i % probe_every == 0:
            rec.append((i, evaluate(task, a_test)["accuracy"]))
    del clean
    final, final_a = evaluate(task, test), evaluate(task, a_test)
    hl = half_life(rec, after_a["accuracy"], before_a["accuracy"])
    return {
        "headline": {"victim_before": before_a["accuracy"], "victim_after": after_a["accuracy"],
                     "victim_as_attack_label": as_b, "others_drop": round(others_before - others_after, 4),
                     "recovery_half_life": hl, "victim_final": final_a["accuracy"],
                     "victim_after_undo": undo_a["accuracy"], "accuracy_change_after_undo": round(undo["accuracy"] - before["accuracy"], 4),
                     "undo_events_replayed": undo_events, "undo_seconds": undo_seconds,
                     **{f"others_harm_at_{k}": v for k, v in harm.items()}},
        "victim": a_cls, "attack_label": b_cls, "burst_size": len(attack),
        "victim_labels_after_burst": sum(y == a_cls for _, y in rest),
        "before": _brief(before), "after": _brief(after), "final": _brief(final),
        "victim_curve": rec, "after_undo": _brief(undo), "prequential": pq.summary(),
        "seconds": round(time.time() - t0, 1),
    }


def drift(make: Make, train: list[Item], test: list[Item], classes: list[str], seed: int, moved: int = 10,
          at: float = 0.5, probe_every: int = 250, other_sample: int = 400) -> dict:
    """Concept drift: half-way through the stream the meaning of ``moved`` classes
    rotates (every message of class c_i is now labelled c_{i+1}); the rest stays.
    Measures how fast the new meaning is learned and whether the untouched classes
    stay stable meanwhile."""
    stream = _shuffled(train, seed)
    split = int(len(stream) * at)
    drng = random.Random(seed * 1000 + 13)
    rot = drng.sample(classes, min(moved, len(classes)))
    mapping = {c: rot[(i + 1) % len(rot)] for i, c in enumerate(rot)}
    aff_old = [it for it in test if it[1] in mapping]
    aff_new = [(x, mapping[y]) for x, y in aff_old]
    others = [it for it in test if it[1] not in mapping]
    other = drng.sample(others, min(other_sample, len(others)))

    task, pq = make(classes), Prequential()
    t0 = time.time()
    for i, (x, y) in enumerate(stream[:split]):
        _step(task, pq, x, y, annotator=_who(i))
    pre_aff, pre_other = evaluate(task, aff_old)["accuracy"], evaluate(task, other)["accuracy"]
    start = evaluate(task, aff_new)["accuracy"]
    curve, other_curve = [(0, start)], [(0, pre_other)]
    post_pq = Prequential()
    events_before = len(task.events)
    post = stream[split:]
    for i, (x, y) in enumerate(post, 1):
        _step(task, post_pq, x, mapping.get(y, y), annotator=_who(i))  # every annotator adopts the new meaning
        if i % probe_every == 0 or i == len(post):
            curve.append((i, evaluate(task, aff_new)["accuracy"]))
            other_curve.append((i, evaluate(task, other)["accuracy"]))
    hl = half_life(curve, start, pre_aff)
    return {
        "headline": {"moved_before": pre_aff, "moved_at_drift": start, "adaptation_half_life": hl,
                     "moved_final": curve[-1][1], "others_before": pre_other,
                     "others_min": min(v for _, v in other_curve), "others_final": other_curve[-1][1],
                     "post_drift_cum_log_loss": post_pq.summary()["cum_log_loss"]},
        "mapping": mapping, "moved_labels_after_drift": sum(y in mapping for _, y in post),
        "moved_curve": curve, "others_curve": other_curve,
        "detector_events": [e["type"] for e in task.events[events_before:]] if hasattr(task, "events") else [],
        "prequential_before": pq.summary(), "prequential_after": post_pq.summary(), "seconds": round(time.time() - t0, 1),
    }


TEACHER_CONFIDENCES = (0.6, 0.75, 0.9, 0.97)  # a calibrated teacher: right with exactly this probability


def teacher(make: Make, train: list[Item], test: list[Item], classes: list[str], seed: int, n: int = 5000,
            human_rate: float = 0.05, window: int = 500) -> dict:
    """Cold start with a teacher: nothing is labelled up front. Every message is
    answered by the student; when it abstains a simulated, calibrated but imperfect
    teacher labels it (soft label). A human spot-checks ``human_rate`` of all
    messages. Teacher dependency = share of messages that still need the teacher."""
    stream = _shuffled(train, seed)[:n]
    trng = random.Random(seed * 1000 + 17)
    task = make(classes)
    windows, cur = [], {"teacher": 0, "teacher_right": 0, "answered": 0, "answered_right": 0, "human": 0}
    served_wrong = [0, 0]  # whole stream, second half: what the user was actually given, whoever answered
    totals = dict.fromkeys(cur, 0)
    t0 = time.time()
    for i, (x, y) in enumerate(stream, 1):
        a = task.answer(x)
        pub = task.public(a)
        task.metrics.record_decision(pub["abstain"], pub["abstain"], pub["confidence"])  # as the service does
        if pub["abstain"]:
            c = trng.choice(TEACHER_CONFIDENCES)
            right = trng.random() < c
            others = [o for o in classes if o != y]
            said = y if right else trng.choice(others)
            second = trng.choice([o for o in classes if o != said])
            task.learn(x, {said: c, second: 1 - c}, source="teacher")
            cur["teacher"] += 1
            cur["teacher_right"] += right
            served_ok = right
        else:
            cur["answered"] += 1
            served_ok = confidence_of(a["probabilities"])[0] == y
            cur["answered_right"] += served_ok
        served_wrong[0] += not served_ok
        if i > len(stream) // 2:
            served_wrong[1] += not served_ok
        if trng.random() < human_rate:
            served_action = ("teacher", said) if pub["abstain"] else ("student", confidence_of(a["probabilities"])[0])
            task.learn(x, y, source="human", served=a["probabilities"], served_raw=a["raw"], abstained=pub["abstain"],
                       served_action=served_action)
            cur["human"] += 1
        if i % window == 0 or i == len(stream):
            size = i - (windows[-1]["decisions"] if windows else 0)
            windows.append({"decisions": i, "teacher_rate": round(cur["teacher"] / size, 4),
                            "coverage": round(cur["answered"] / size, 4),
                            "answered_accuracy": round(cur["answered_right"] / cur["answered"], 4) if cur["answered"] else None,
                            "served_error": round(1 - (cur["answered_right"] + cur["teacher_right"]) / size, 4),
                            "teacher_accuracy": round(cur["teacher_right"] / cur["teacher"], 4) if cur["teacher"] else None})
            for k in cur:
                totals[k] += cur[k]
                cur[k] = 0
    final = evaluate(task, test)
    return {
        "headline": {"teacher_rate_first": windows[0]["teacher_rate"], "teacher_rate_last": windows[-1]["teacher_rate"],
                     "answered_accuracy_last": windows[-1]["answered_accuracy"], "teacher_calls": totals["teacher"],
                     "human_labels": totals["human"], "test_accuracy": final["accuracy"], "test_ece": final["ece"],
                     "test_coverage": final["coverage"], "test_answered_accuracy": final["answered_accuracy"],
                     "served_error": round(served_wrong[0] / len(stream), 4),
                     "served_error_second_half": round(served_wrong[1] / (len(stream) - len(stream) // 2), 4)},
        "windows": windows, "teacher_accuracy": round(totals["teacher_right"] / max(totals["teacher"], 1), 4),
        "final": final, "seconds": round(time.time() - t0, 1),
    }


BUDGETS = (0.02, 0.05, 0.10)


def budget(make: Make, train: list[Item], test: list[Item], classes: list[str], seed: int, window: int = 1000) -> dict:
    """Does the risk budget hold? A shuffled stream where every decision is labelled
    afterwards; for each budget (and for the fixed default threshold) the decision rule
    runs side by side on the same student, and the realized error rate among the
    decisions it would have answered is compared with the budget. The oracle is the
    widest coverage any threshold could have had in that window, knowing the labels."""
    stream = _shuffled(train, seed)
    task = make(classes)
    policies = ({f"budget-{b:.0%}": (b, "guaranteed") for b in BUDGETS} | {f"expected-{b:.0%}": (b, "expected") for b in BUDGETS}
                | {"threshold-0.6": (None, None)})
    wins: dict[str, list[dict]] = {k: [] for k in policies}
    cur = {k: [0, 0] for k in policies}  # answered, wrong
    recs: list[tuple[float, bool]] = []
    t0 = time.time()
    for i, (x, y) in enumerate(stream, 1):
        a = task.answer(x)
        conf_ok = (confidence_of(a["probabilities"])[1], confidence_of(a["probabilities"])[0] == y)
        recs.append(conf_ok)
        task.metrics.record_decision(False, False, conf_ok[0])
        for name, (b, mode) in policies.items():
            task.policy.risk_budget, task.policy.risk_budget_mode = b, mode or "guaranteed"
            pub = task.public(a)
            if not pub["abstain"]:
                cur[name][0] += 1
                cur[name][1] += pub["choice"] != y
        task.policy.risk_budget = None
        task.learn(x, y, source="dataset", served=a["probabilities"], served_raw=a["raw"])
        if i % window == 0:
            for name, (b, _) in policies.items():
                n_ans, n_wrong = cur[name]
                row = {"decisions": i, "coverage": round(n_ans / window, 4),
                       "risk": round(n_wrong / n_ans, 4) if n_ans else None}
                if b is not None:
                    row["oracle_coverage"] = round(_oracle_coverage(recs, b), 4)
                wins[name].append(row)
                cur[name] = [0, 0]
            recs = []
    half = len(wins["threshold-0.6"]) // 2
    headline, detail = {}, {}
    for name, (b, _) in policies.items():
        late = wins[name][half:]
        answered = sum(w["coverage"] * window for w in late)
        wrong = sum((w["risk"] or 0) * w["coverage"] * window for w in late)
        key = name.replace("budget-", "b").replace("expected-", "e").replace("threshold-0.6", "t60")
        headline[f"{key}_risk"] = round(wrong / answered, 4) if answered else None
        headline[f"{key}_coverage"] = round(sum(w["coverage"] for w in late) / len(late), 4)
        if b is not None:
            headline[f"{key}_oracle_coverage"] = round(sum(w["oracle_coverage"] for w in late) / len(late), 4)
            headline[f"{key}_windows_over"] = sum(1 for w in wins[name][1:] if w["risk"] is not None and w["risk"] > b)
        detail[name] = wins[name]
    return {"headline": headline, "windows": detail, "seconds": round(time.time() - t0, 1)}


def _oracle_coverage(recs: list[tuple[float, bool]], b: float) -> float:
    """Widest share of these decisions a confidence threshold could answer with error rate ≤ b."""
    best, errors = 0, 0
    ranked = sorted(recs, key=lambda r: -r[0])
    for k, (c, ok) in enumerate(ranked, 1):
        errors += not ok
        if (k == len(ranked) or ranked[k][0] != c) and errors / k <= b:
            best = k
    return best / len(recs) if recs else 0.0


SCENARIOS: dict[str, Callable[..., dict]] = {
    "shuffled": shuffled,
    "sorted": class_sorted,
    "noise-1%": lambda *a: noisy(*a, rate=0.01),
    "noise-5%": lambda *a: noisy(*a, rate=0.05),
    "burst": burst,
    "drift": drift,
    "teacher": teacher,
    "budget": budget,
}
