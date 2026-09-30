"""Benchmark Desic on Banking77 (77 banking intents; 10,003 train / 3,080 test).

    python examples/benchmark_banking77.py            # online student only
    python examples/benchmark_banking77.py --neural   # + the neural student (needs desic[neural])

The training file is sorted by intent; it is shuffled like Desic's dataset
training does (``--sorted`` streams it in file order as a stress test).

Reports test accuracy, log loss, ECE, and — because Desic abstains — the
accuracy on the questions it chose to answer (coverage) at several abstain
thresholds. A learning curve shows accuracy after 10 / 25 / 50 / all
examples per intent. Data: github.com/PolyAI-LDN/task-specific-datasets.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import platform
import random
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from desic.core.calibration import TaskMetrics
from desic.core.task import DecisionTask, QuestionSpec

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"


def load(split: str) -> list[tuple[str, str]]:
    with urllib.request.urlopen(BASE + f"{split}.csv", timeout=60) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode("utf-8"))))
    return [(row["text"], row["category"]) for row in rows]


def evaluate(task: DecisionTask, test: list[tuple[str, str]]) -> dict:
    m = TaskMetrics(window=len(test) + 1)
    confs = []
    for text, label in test:
        a = task.answer(text)
        m.update(a["probabilities"], label, False)
        best = max(a["probabilities"], key=a["probabilities"].get)
        confs.append((a["probabilities"][best], best == label))
    s = m.summary()
    out = {k: s[k] for k in ("accuracy", "nll", "brier", "ece")}
    for t in (0.5, 0.7, 0.9):
        kept = [ok for c, ok in confs if c >= t]
        out[f"coverage@{t}"] = round(len(kept) / len(confs), 4)
        out[f"answered_acc@{t}"] = round(sum(kept) / len(kept), 4) if kept else None
    return out


def per_class_subset(train: list[tuple[str, str]], k: int, seed: int = 0) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    by: dict[str, list] = {}
    for x in train:
        by.setdefault(x[1], []).append(x)
    out = []
    for rows in by.values():
        out += rng.sample(rows, min(k, len(rows)))
    rng.shuffle(out)
    return out


def online(train: list, test: list, classes: list[str]) -> dict:
    task = DecisionTask(QuestionSpec.parse("intent", {"type": "choice", "instructions": "Which banking intent is this?",
                                                      "criteria": classes}))
    t0 = time.time()
    for text, label in train:
        task.learn(text, label, source="dataset")
    res = evaluate(task, test)
    res["train_seconds"] = round(time.time() - t0, 1)
    res["weights"] = task.summary()["expert_weights"]
    return res


def neural(train: list, test: list, classes: list[str], epochs: int, backbone: str) -> dict:
    from desic.service import Desic

    d = Desic(tempfile.mkdtemp())
    d.create_task(QuestionSpec.parse("intent", {"type": "choice", "instructions": "Which banking intent is this?",
                                                "criteria": classes}))
    d.learn("intent", [{"state": t, "answer": y} for t, y in train])
    d.neural.set_config({"epochs": epochs, "shadow_min": 0, "max_len": 512 if backbone != "scratch" else 384,
                         "backbone": backbone, "batch_size": 32})
    job = d.new_job("neural")
    t0 = time.time()
    asyncio.run(d.train_neural(job))
    if job.status != "done":
        return {"error": job.error}
    ck = d.storage.list_checkpoints()[0]
    task = d.get("intent")
    # the neural expert alone (its own calibrated probabilities)
    m = TaskMetrics(window=len(test) + 1)
    for text, label in test:
        dist, _ = d.neural.predict(task.spec, text, task.spec.option_names)
        m.update(dist, label, False)
    alone = {k: m.summary()[k] for k in ("accuracy", "nll", "ece")}
    mixed = evaluate(task, test)
    return {"status": ck["status"], "train_seconds": round(time.time() - t0, 1), "neural_alone": alone, "mixture": mixed,
            "weights": task.summary()["expert_weights"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neural", action="store_true")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--backbone", default="scratch")
    ap.add_argument("--sorted", action="store_true", help="stress test: stream the training file in its original, class-sorted order")
    ap.add_argument("--out", help="also write the results as JSON to this file")
    args = ap.parse_args()
    results: dict = {"benchmark": "banking77", "meta": _meta(args), "online": {}}
    train, test = load("train"), load("test")
    if not args.sorted:  # the file is sorted by intent; Desic's dataset training shuffles too
        random.Random(0).shuffle(train)
    classes = sorted({y for _, y in train})
    print(f"Banking77: {len(train)} train, {len(test)} test, {len(classes)} intents\n")
    print("Online student (learns one example at a time):")
    for k in (10, 25, 50):
        r = online(per_class_subset(train, k), test, classes)
        results["online"][f"{k}_per_intent"] = r
        print(f"  {k:>3} examples/intent  accuracy {r['accuracy']:.1%}  log loss {r['nll']:.3f}  ECE {r['ece']:.3f}")
    r = online(train, test, classes)
    results["online"]["all"] = r
    print(f"  all ({len(train)})       accuracy {r['accuracy']:.1%}  log loss {r['nll']:.3f}  ECE {r['ece']:.3f}  "
          f"({r['train_seconds']}s)")
    for t in (0.5, 0.7, 0.9):
        print(f"      answer only when ≥{t:.0%} sure: coverage {r[f'coverage@{t}']:.1%}, accuracy {r[f'answered_acc@{t}']:.1%}")
    print(f"      expert weights {r['weights']}")
    if args.neural:
        print(f"\nNeural student ({args.backbone}, {args.epochs} epochs) + online experts:")
        n = neural(train, test, classes, args.epochs, args.backbone)
        results["neural"] = n
        if "error" in n:
            print("  error:", n["error"])
            _write(args.out, results)
            return
        print(f"  neural alone  accuracy {n['neural_alone']['accuracy']:.1%}  log loss {n['neural_alone']['nll']:.3f}  "
              f"ECE {n['neural_alone']['ece']:.3f}  ({n['train_seconds']}s)")
        mx = n["mixture"]
        print(f"  mixture       accuracy {mx['accuracy']:.1%}  log loss {mx['nll']:.3f}  ECE {mx['ece']:.3f}")
        for t in (0.5, 0.7, 0.9):
            print(f"      answer only when ≥{t:.0%} sure: coverage {mx[f'coverage@{t}']:.1%}, accuracy {mx[f'answered_acc@{t}']:.1%}")
        print(f"      expert weights {n['weights']}")
    _write(args.out, results)


def _meta(args: argparse.Namespace) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        commit = ""
    try:
        import torch

        torch_version, gpu = torch.__version__, torch.cuda.is_available()
    except Exception:
        torch_version, gpu = None, False
    return {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "commit": commit, "python": platform.python_version(),
            "machine": platform.machine(), "cpu_count": __import__("os").cpu_count(), "torch": torch_version, "gpu": gpu,
            "args": vars(args)}


def _write(path: str | None, results: dict) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
