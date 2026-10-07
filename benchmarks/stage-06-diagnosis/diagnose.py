"""Stage 06: where do the three remaining weaknesses of the patch layer come from?

Replays the drift, noise-5% and teacher scenarios of desic.eval (seed 0, same
streams) and, at each probe, scores the parts of the patched student separately:
the base model alone, the patches alone, and the combined answer.

    python benchmarks/stage-06-diagnosis/diagnose.py > diagnosis.json
"""

from __future__ import annotations

import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from desic.core.calibration import confidence_of
from desic.eval.run import _data
from desic.eval.scenarios import TEACHER_CONFIDENCES, _shuffled, _step, make_desic, make_patched

SEED = 0
LIMIT = int(os.environ.get("DIAG_LIMIT", 0)) or None  # subsample for a quick smoke run
top = lambda d: confidence_of(d)[0] if d else None


def learner(variant: str, classes):
    return {"base": make_desic, "patch": make_patched,
            "patch-noreplay": partial(make_patched, replay=0),
            "patch-probation20": partial(make_patched, probation=20, min_probation=20)}[variant](classes)


# ------------------------------------------------------------------ drift
def drift(variant: str) -> dict:
    train, test, classes = _data(LIMIT)
    stream = _shuffled(train, SEED)
    split = len(stream) // 2
    drng = random.Random(SEED * 1000 + 13)
    rot = drng.sample(classes, min(10, len(classes)))
    mapping = {c: rot[(i + 1) % len(rot)] for i, c in enumerate(rot)}
    aff_new = [(x, mapping[y]) for x, y in test if y in mapping]
    old_of = {v: k for k, v in mapping.items()}
    t = learner(variant, classes)
    for x, y in stream[:split]:
        _step(t, None, x, y)
    t_drift = getattr(t, "t", 0)
    curve = []
    post = stream[split:]
    for i, (x, y) in enumerate(post, 1):
        _step(t, None, x, mapping.get(y, y))
        if i % 500 and i != len(post):
            continue
        row = {"labels_after_drift": i, "combined": 0, "old_answer": 0}
        if variant != "base":
            row.update(base_alone=0, patches_awake=0, patches_alone=0, either=0, gate_weight=0.0, stale_vote_share=0.0)
        for x2, y2 in aff_new:
            a = t.answer(x2)
            row["combined"] += top(a["probabilities"]) == y2
            row["old_answer"] += top(a["probabilities"]) == old_of[y2]
            if variant == "base":
                continue
            f = t._last[2]
            ok_b = top(f["pb"]) == y2
            row["base_alone"] += ok_b
            if f["pq"] is not None:
                row["patches_awake"] += 1
                ok_p = top(f["pq"]) == y2
                row["patches_alone"] += ok_p
                row["either"] += ok_b or ok_p
                row["gate_weight"] += f["wp"]
                tot = stale = 0.0
                for s, p in f["near"]:
                    w = s * s * p.weight * t.store.trust(p, t.t)
                    tot += w
                    if p.t < t_drift and p.label in mapping:  # learned under the old meaning
                        stale += w
                row["stale_vote_share"] += stale / tot if tot else 0.0
            else:
                row["either"] += ok_b
        n = len(aff_new)
        awake = max(row.get("patches_awake", 0), 1)
        out = {"labels_after_drift": i, "combined": round(row["combined"] / n, 4), "old_answer": round(row["old_answer"] / n, 4)}
        if variant != "base":
            out.update(base_alone=round(row["base_alone"] / n, 4), patches_awake=round(row["patches_awake"] / n, 4),
                       patches_alone_when_awake=round(row["patches_alone"] / awake, 4), oracle_either=round(row["either"] / n, 4),
                       gate_weight_when_awake=round(row["gate_weight"] / awake, 4),
                       stale_vote_share_when_awake=round(row["stale_vote_share"] / awake, 4))
        curve.append(out)
    return {"scenario": "drift", "variant": variant, "curve": curve}


# ------------------------------------------------------------------ noise
def noise(variant: str) -> dict:
    train, test, classes = _data(LIMIT)
    stream = _shuffled(train, SEED)
    nrng = random.Random(SEED * 1000 + 7)
    t = learner(variant, classes)
    poisoned = []
    for x, y in stream:
        given = y
        if nrng.random() < 0.05:
            given = nrng.choice([c for c in classes if c != y])
            poisoned.append((x, y, given))
        _step(t, None, x, given, true=y)
    c = {"n": len(poisoned), "combined_repeats": 0, "combined_correct": 0}
    if variant != "base":
        c.update(base_repeats=0, patches_repeat=0, exact_match=0, gate_weight=[], gate_weight_when_repeated=[], entry_trust=[],
                 neighbours_agreeing_with_lie=[], cells={})
    for x, y, g in poisoned:
        a = t.answer(x)
        said = top(a["probabilities"])
        c["combined_repeats"] += said == g
        c["combined_correct"] += said == y
        if variant == "base":
            continue
        f = t._last[2]
        c["base_repeats"] += top(f["pb"]) == g
        if f["pq"] is not None:
            c["patches_repeat"] += top(f["pq"]) == g
            c["exact_match"] += f["strength"] >= 0.999
            c["gate_weight"].append(f["wp"])
            if said == g:
                c["gate_weight_when_repeated"].append(f["wp"])
            c["cells"][f["cell"].split(":")[0]] = c["cells"].get(f["cell"].split(":")[0], 0) + 1
            lie = [p for s, p in f["near"] if s >= 0.999 and p.label == g]
            if lie:
                c["entry_trust"].append(t.store.trust(lie[0], t.t))
            c["neighbours_agreeing_with_lie"].append(sum(p.label == g for s, p in f["near"] if s < 0.999))
    out = {k: (round(v / c["n"], 4) if isinstance(v, int) and k != "n" else v) for k, v in c.items()}
    for k in ("gate_weight", "gate_weight_when_repeated", "entry_trust", "neighbours_agreeing_with_lie"):
        if k in out:
            vals = out[k]
            out[k] = round(sum(vals) / len(vals), 4) if vals else None
    if variant != "base":
        exact_updates = sum(1 for h in t.gate.history if h[1].startswith("4"))
        out["gate_updates_in_exact_cells"] = exact_updates
        out["gate_updates_total"] = len(t.gate.history)
    return {"scenario": "noise-5%", "variant": variant, **out}


# ---------------------------------------------------------------- teacher
def teacher(variant: str) -> dict:
    train, test, classes = _data(LIMIT)
    stream = _shuffled(train, SEED)[:5000]
    last = len(stream) - 1500
    trng = random.Random(SEED * 1000 + 17)
    t = learner(variant, classes)
    rec = []
    for i, (x, y) in enumerate(stream, 1):
        a = t.answer(x)
        pub = t.public(a)
        f = t._last[2] if variant != "base" else None
        if pub["abstain"]:
            c = trng.choice(TEACHER_CONFIDENCES)
            right = trng.random() < c
            said = y if right else trng.choice([o for o in classes if o != y])
            second = trng.choice([o for o in classes if o != said])
            t.learn(x, {said: c, second: 1 - c}, source="teacher")
        elif i > last:  # the last 1,500 decisions
            r = {"ok": top(a["probabilities"]) == y, "conf": pub["confidence"]}
            if f is not None:
                r.update(base_fam=f["a"]["familiarity"], lifted=f["a"]["familiarity"] < 0.5,
                         base_ok=top(f["pb"]) == y, patch_ok=(top(f["pq"]) == y) if f["pq"] is not None else None,
                         wp=f["wp"], near_teacher=sum(p.source == "teacher" for _, p in f["near"]) / max(len(f["near"]), 1))
            rec.append(r)
        if trng.random() < 0.05:
            t.learn(x, y, source="human", served=a["probabilities"], served_raw=a["raw"], abstained=pub["abstain"])
    acc = lambda rs: (round(sum(r["ok"] for r in rs) / len(rs), 4), len(rs)) if rs else (None, 0)
    out = {"scenario": "teacher", "variant": variant, "answered_last_1500": acc(rec)}
    if variant != "base":
        lifted = [r for r in rec if r["lifted"]]
        plain = [r for r in rec if not r["lifted"]]
        wrong = [r for r in rec if not r["ok"]]
        out.update(
            answered_only_because_patches_made_it_familiar=acc(lifted),
            answered_familiar_to_base=acc(plain),
            wrong_answers=len(wrong),
            wrong_but_base_alone_right=sum(r["base_ok"] for r in wrong),
            wrong_but_patches_alone_right=sum(bool(r["patch_ok"]) for r in wrong),
            mean_gate_weight_on_wrong=round(sum(r["wp"] for r in wrong) / max(len(wrong), 1), 4),
            mean_gate_weight_on_right=round(sum(r["wp"] for r in rec if r["ok"]) / max(len(rec) - len(wrong), 1), 4),
            teacher_share_of_neighbours_wrong=round(sum(r["near_teacher"] for r in wrong) / max(len(wrong), 1), 4),
            teacher_share_of_neighbours_right=round(sum(r["near_teacher"] for r in rec if r["ok"]) / max(len(rec) - len(wrong), 1), 4),
        )
    return out


JOBS = [("drift", "base"), ("drift", "patch"), ("drift", "patch-noreplay"), ("drift", "patch-probation20"),
        ("noise", "base"), ("noise", "patch"), ("teacher", "base"), ("teacher", "patch")]


def run(job):
    fn = {"drift": drift, "noise": noise, "teacher": teacher}[job[0]]
    res = fn(job[1])
    print(json.dumps(res), file=sys.stderr, flush=True)
    return res


if __name__ == "__main__":
    _data(LIMIT)
    with ProcessPoolExecutor(4) as ex:
        print(json.dumps(list(ex.map(run, JOBS)), indent=1))
