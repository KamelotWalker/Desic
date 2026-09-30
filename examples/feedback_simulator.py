"""Simulate users who ask Desic for decisions and then report the real outcome.

Run `desic demo && desic serve` first, open http://127.0.0.1:8000/#/models/loan_demo
and then start this script: the accuracy chart, live feed and tree update in real time.

    python examples/feedback_simulator.py --n 1000 --drift 500

With --drift N the bank's policy changes after N decisions (credit scores become
more important); watch Desic detect the drift and adapt.
"""

from __future__ import annotations

import argparse
import random
import time

import httpx

from desic.demo import loan_row


def new_policy(row: dict) -> str:
    # after the "policy change": high credit scores are approved even with a heavy loan burden
    if row["employment"] == "unemployed":
        return "reject"
    if row["credit_score"] > 1300:
        return "approve"
    return "review" if row["credit_score"] > 1000 else "reject"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="loan_demo")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--delay", type=float, default=0.05, help="seconds between decisions")
    ap.add_argument("--drift", type=int, default=0, help="change the ground truth after this many decisions")
    args = ap.parse_args()

    rng = random.Random()
    base = f"{args.url}/api/models/{args.model}"
    correct = 0
    with httpx.Client(timeout=10) as http:
        for i in range(1, args.n + 1):
            row = loan_row(rng)
            truth = row.pop("decision")
            if args.drift and i > args.drift:
                truth = new_policy(row)
            d = http.post(f"{base}/decide", json={"features": row}).raise_for_status().json()
            correct += d["prediction"] == truth
            http.post(f"{base}/feedback", json={"decision_id": d["id"], "label": truth}).raise_for_status()
            if i % 50 == 0:
                print(f"{i:5d} decisions · last-50 accuracy {correct / 50:.0%}")
                correct = 0
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
