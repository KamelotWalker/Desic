"""A small synthetic loan-approval problem for trying Desic out."""

from __future__ import annotations

import random

from .core.schema import Schema
from .service import Desic

CITIES = ["istanbul", "ankara", "izmir", "bursa", "antalya"]
EMPLOYMENT = ["salaried", "self_employed", "unemployed", "retired"]


def loan_row(rng: random.Random) -> dict:
    income = round(rng.lognormvariate(10.3, 0.5))
    amount = round(rng.uniform(5_000, 400_000), -3)
    credit = int(rng.gauss(1350, 250))
    employment = rng.choices(EMPLOYMENT, weights=[6, 2, 1, 1])[0]
    debt_ratio = round(min(max(rng.gauss(0.35, 0.15), 0), 1), 2)
    row = {
        "monthly_income": income,
        "loan_amount": amount,
        "credit_score": max(0, min(1900, credit)),
        "employment": employment,
        "debt_ratio": debt_ratio,
        "city": rng.choice(CITIES),
    }
    burden = amount / max(income * 12, 1)
    if employment == "unemployed" or row["credit_score"] < 1000 or debt_ratio > 0.65:
        decision = "reject"
    elif row["credit_score"] > 1500 and burden < 1.5 and debt_ratio < 0.4:
        decision = "approve"
    elif burden > 3:
        decision = "reject"
    else:
        decision = "review"
    if rng.random() < 0.04:  # label noise, like real life
        decision = rng.choice(["approve", "review", "reject"])
    row["decision"] = decision
    return row


def build_demo(data_dir: str, rows: int = 2000) -> str:
    rng = random.Random(1)
    data = [loan_row(rng) for _ in range(rows)]
    desic = Desic(data_dir)
    columns = list(data[0])
    ds = desic.add_dataset("loan_applications (demo)", "upload", columns, data)
    msg = [f"dataset '{ds['name']}' ({ds['id']}) with {len(data)} rows"]
    if "loan_demo" not in desic.runtimes:
        schema = Schema.infer(data, "decision")
        desic.create_model("loan_demo", schema, "tree", description="Demo: approve / review / reject a loan")
        split = int(len(data) * 0.8)
        desic.learn("loan_demo", data[:split])
        for r in data[split:split + 25]:  # leave a few open decisions for the review queue
            desic.decide("loan_demo", r)
        desic.persist(desic.get("loan_demo"))
        msg.append("model 'loan_demo' trained on 80% of it, 25 decisions waiting for feedback")
    desic.storage.close()
    return "Created " + "; ".join(msg) + ". Now run: desic serve"
