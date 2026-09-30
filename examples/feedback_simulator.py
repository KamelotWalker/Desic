"""Simulate users who ask Desic for decisions and then report the real outcome.

Run `desic demo && desic serve` first, open http://127.0.0.1:8000/#/questions/loan_decision
and start this script: the learning curve, reliability diagram and live feed update in real time.

    python examples/feedback_simulator.py --n 1000 --drift 500

With --drift N the bank's policy changes after N decisions (credit score alone now
decides); watch Desic's accuracy dip and recover as the feedback comes in.
"""

from __future__ import annotations

import argparse
import random
import time

import httpx

from desic.demo import loan


def new_policy(state: dict) -> str:
    a = state["applicant"]
    if a["employment"] == "unemployed":
        return "reject"
    if a["credit_score"] > 1300:
        return "approve"
    return "review" if a["credit_score"] > 1000 else "reject"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--question", default="loan_decision")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--delay", type=float, default=0.05, help="seconds between decisions")
    ap.add_argument("--drift", type=int, default=0, help="change the ground truth after this many decisions")
    args = ap.parse_args()

    rng = random.Random()
    correct = abstained = 0
    with httpx.Client(base_url=args.url, timeout=30) as http:
        for i in range(1, args.n + 1):
            state, truth = loan(rng)
            if args.drift and i > args.drift:
                truth = new_policy(state)
            r = http.post("/v1/decide", json={"state": state, "questions": {args.question: {}}, "escalate": "never"}).raise_for_status().json()
            a = r["answers"][args.question]
            correct += a["choice"] == truth
            abstained += a["abstain"]
            http.post("/v1/feedback", json={"decision_id": r["id"], "answers": {args.question: truth}}).raise_for_status()
            if i % 50 == 0:
                print(f"{i:5d} decisions · last 50: accuracy {correct / 50:.0%}, abstained {abstained / 50:.0%}")
                correct = abstained = 0
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
