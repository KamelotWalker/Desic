"""Run the evaluation scenarios over several seeds and summarise them.

    desic eval --out results.json                 # all scenarios, 3 seeds, Banking77
    desic eval --scenarios burst,drift --seeds 1  # a subset
    desic eval --limit 2000                       # quick smoke run on a subsample
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .data import banking77
from .scenarios import LEARNERS, SCENARIOS

_DATA: dict = {}


def _data(limit: int | None) -> tuple[list, list, list[str]]:
    if limit not in _DATA:
        train, test = banking77()
        if limit:
            rng = random.Random(0)
            train, test = rng.sample(train, min(limit, len(train))), rng.sample(test, min(limit // 3, len(test)))
        _DATA[limit] = (train, test, sorted({y for _, y in train}))
    return _DATA[limit]


def run_one(scenario: str, seed: int, learner: str = "desic", limit: int | None = None) -> dict:
    train, test, classes = _data(limit)
    out = SCENARIOS[scenario](LEARNERS[learner], train, test, classes, seed)
    out["seed"] = seed
    return out


def aggregate(runs: list[dict]) -> dict:
    """Mean and sample standard deviation of every headline number across seeds."""
    out = {}
    for key in runs[0]["headline"]:
        vals = [r["headline"].get(key) for r in runs]
        nums = [v for v in vals if isinstance(v, (int, float))]
        row: dict = {"values": vals}
        if nums:
            mean = sum(nums) / len(nums)
            row["mean"] = round(mean, 4)
            row["std"] = round(math.sqrt(sum((v - mean) ** 2 for v in nums) / (len(nums) - 1)), 4) if len(nums) > 1 else 0.0
        if len(nums) < len(vals):
            row["missing"] = len(vals) - len(nums)  # e.g. a half-life never reached within the stream
        out[key] = row
    return out


def format_summary(results: dict) -> str:
    lines = []
    for name, sc in results["scenarios"].items():
        lines.append(f"{name}")
        for key, row in sc["summary"].items():
            if "mean" in row:
                v = f"{row['mean']:.4g} ± {row['std']:.2g}"
            else:
                v = "—"
            if row.get("missing"):
                v += f"  (not reached in {row['missing']} of {len(row['values'])} runs)"
            lines.append(f"  {key:<28} {v}")
    return "\n".join(lines)


def meta(args: argparse.Namespace) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        commit = ""
    return {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "commit": commit, "python": platform.python_version(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(), "args": vars(args)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="desic eval", description="Desic evaluation harness (Banking77 scenarios)")
    add_arguments(ap)
    run(ap.parse_args(argv))


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--scenarios", default="all", help=f"comma separated: {', '.join(SCENARIOS)} (default: all)")
    ap.add_argument("--seeds", type=int, default=3, help="seeds 0..N-1 (stream order, noise, attacked classes)")
    ap.add_argument("--learner", default="desic", choices=sorted(LEARNERS))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--limit", type=int, help="subsample the training data (quick smoke runs)")
    ap.add_argument("--out", help="write the full results as JSON")


def run(args: argparse.Namespace) -> dict:
    names = list(SCENARIOS) if args.scenarios == "all" else [s.strip() for s in args.scenarios.split(",")]
    unknown = [s for s in names if s not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenario(s): {', '.join(unknown)}")
    info = meta(args)  # before running: the commit that produced the numbers
    _data(args.limit)  # download once before forking
    jobs = [(s, seed) for s in names for seed in range(args.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {job: pool.submit(run_one, job[0], job[1], args.learner, args.limit) for job in jobs}
        done = {}
        for job, fut in futures.items():
            done[job] = fut.result()
            print(f"  {job[0]} seed {job[1]}: {done[job]['seconds']}s", flush=True)
    results = {"harness": "desic-eval/1", "data": "banking77", "learner": args.learner, "meta": info,
               "wall_seconds": round(time.time() - t0, 1), "scenarios": {}}
    for s in names:
        runs = [done[(s, seed)] for seed in range(args.seeds)]
        results["scenarios"][s] = {"summary": aggregate(runs), "runs": runs}
    print("\n" + format_summary(results))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    return results


if __name__ == "__main__":
    main()
