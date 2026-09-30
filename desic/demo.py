"""Demo data: support tickets (text state) and loan applications (JSON state)."""

from __future__ import annotations

import asyncio
import random

from .core.task import QuestionSpec
from .service import Desic

TICKETS = {
    "billing": [
        "I was charged twice for my order", "kartımdan iki kez ödeme çekildi, iade istiyorum", "the invoice amount is wrong",
        "faturamda yanlış tutar var", "please refund my last payment", "aboneliğim onayım olmadan yenilendi",
        "why did my card get declined at checkout", "ödeme iadesi ne zaman hesabıma geçer", "I need a VAT invoice for March",
    ],
    "technical": [
        "the app crashes as soon as I open it", "uygulama açılmıyor, sürekli hata veriyor", "login page shows error 500",
        "şifre sıfırlama maili gelmiyor", "sync stopped working after the update", "site çok yavaş ve donuyor",
        "export to CSV produces an empty file", "bildirimler telefonuma düşmüyor", "API returns 401 with a valid token",
    ],
    "sales": [
        "do you offer discounts for teams", "kurumsal fiyat teklifi almak istiyorum", "what does the pro plan cost",
        "50 kişilik lisans için fiyat alabilir miyiz", "can we upgrade to enterprise this month", "demo talep ediyorum",
        "is there an annual billing discount", "eğitim kurumlarına indirim var mı", "we want to buy for 3 more offices",
    ],
}
OPENERS = ["", "Hi, ", "Merhaba, ", "Hello team, ", "Selam, ", "Good morning. "]
CALM = ["", " Thanks.", " Teşekkürler.", " No rush.", " Müsait olduğunuzda bakarsanız sevinirim.", " Have a nice day."]
URGENT = [" Acil!", " Acil dönüş lütfen.", " Bu çok acil.", " This is urgent, we are losing sales.", " asap", " Please help today!",
          " Bugün çözülmezse aboneliği iptal edeceğim.", " URGENT: customers are waiting.", " Hemen dönüş yapın lütfen."]


def ticket(rng: random.Random) -> tuple[str, str, bool]:
    dept = rng.choice(list(TICKETS))
    urgent = rng.random() < 0.4
    closer = rng.choice(URGENT if urgent else CALM)
    text = f"{rng.choice(OPENERS)}{rng.choice(TICKETS[dept])}.{closer}".replace("..", ".")
    return text, dept, urgent


def loan(rng: random.Random) -> tuple[dict, str]:
    income = round(rng.lognormvariate(10.3, 0.5))
    credit = max(300, min(1900, int(rng.gauss(1350, 250))))
    employment = rng.choices(["salaried", "self_employed", "unemployed", "retired"], weights=[6, 2, 1, 1])[0]
    debt = round(min(max(rng.gauss(0.35, 0.15), 0), 1), 2)
    amount = round(rng.uniform(5_000, 400_000), -3)
    state = {
        "applicant": {"monthly_income": income, "credit_score": credit, "employment": employment, "debt_ratio": debt,
                      "city": rng.choice(["istanbul", "ankara", "izmir", "bursa", "antalya"])},
        "loan": {"amount": amount, "purpose": rng.choice(["car", "home", "business", "education", "personal"])},
    }
    burden = amount / max(income * 12, 1)
    if employment == "unemployed" or credit < 1000 or debt > 0.65:
        y = "reject"
    elif credit > 1500 and burden < 1.5 and debt < 0.4:
        y = "approve"
    elif burden > 3:
        y = "reject"
    else:
        y = "review"
    if rng.random() < 0.04:
        y = rng.choice(["approve", "review", "reject"])
    return state, y


QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this ticket?",
                   "criteria": {"billing": "payments, invoices, refunds", "technical": "bugs, errors, outages",
                                "sales": "pricing, plans, purchasing"}},
    "urgent": {"type": "noul", "instructions": "The customer needs a response today."},
    "loan_decision": {"type": "choice", "instructions": "What should we do with this loan application?",
                      "criteria": {"approve": "", "review": "a credit officer should look at it", "reject": ""}},
}


def build_demo(data_dir: str, n: int = 600) -> str:
    rng = random.Random(1)
    desic = Desic(data_dir)
    created = []
    for name, raw in QUESTIONS.items():
        if name not in desic.tasks:
            desic.create_task(QuestionSpec.parse(name, raw))
            created.append(name)
    if created:
        tickets = [ticket(rng) for _ in range(n)]
        desic.learn("department", [{"state": t, "answer": d} for t, d, _ in tickets])
        desic.learn("urgent", [{"state": t, "answer": u} for t, _, u in tickets])
        loans = [loan(rng) for _ in range(n * 3)]
        desic.learn("loan_decision", [{"state": s, "answer": y} for s, y in loans])

        async def open_decisions() -> None:
            for _ in range(15):
                t, _, _ = ticket(rng)
                await desic.decide(t, {"department": {}, "urgent": {}})
                s, _ = loan(rng)
                await desic.decide(s, {"loan_decision": {}})

        asyncio.run(open_decisions())
        desic.persist_all()
        for name in created:
            desic.snapshot(name, note="demo baseline")
    desic.storage.close()
    if not created:
        return "Demo questions already exist. Run: desic serve"
    return (f"Created questions {', '.join(created)} (trained on synthetic tickets and loan applications, "
            "with 30 decisions waiting for feedback). Now run: desic serve")
