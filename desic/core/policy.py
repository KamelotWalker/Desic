"""Decision policy: *when* the student answers, kept apart from *what* it knows.

The model state (experts, patches, calibration) changes with every label; the
policy (abstain threshold, risk budget, costs) changes only when someone edits
it. Each has its own version, and every decision logs both, so a change in
behaviour can always be traced to one or the other.

Because the policy is a pure function of the served probabilities (plus the
risk controller's labelled history), any candidate policy can be replayed on
logged decisions that later got a label — "what would this policy have
answered, and how often would it have been wrong?" — without touching the
model. See :func:`replay`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from . import contract
from .calibration import confidence_of

Dist = dict[str, float]
KEYS = ("abstain_threshold", "risk_budget", "risk_budget_mode", "cost_wrong", "cost_abstain", "cost_matrix")


@dataclass
class DecisionPolicy:
    abstain_threshold: float = 0.6
    risk_budget: float | None = None
    risk_budget_mode: str = "guaranteed"
    cost_wrong: float | None = None
    cost_abstain: float | None = None
    cost_matrix: dict[str, dict[str, float]] | None = None
    version: int = 1
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def from_settings(cls, settings: dict | None) -> "DecisionPolicy":
        return cls(**{k: v for k, v in (settings or {}).items() if k in KEYS})

    def settings(self) -> dict:
        return {k: getattr(self, k) for k in KEYS}

    def update(self, changes: dict, options: list[str]) -> bool:
        """Apply validated changes; the version moves only if something actually changed."""
        clean = dict(contract.validate(changes, options))
        if "abstain_threshold" in changes:
            t = float(changes["abstain_threshold"])
            if not 0 <= t <= 1:
                raise ValueError("abstain_threshold must be between 0 and 1")
            clean["abstain_threshold"] = t
        changed = {k: v for k, v in clean.items() if getattr(self, k) != v}
        for k, v in changed.items():
            setattr(self, k, v)
        if changed:
            self.version += 1
            self.updated_at = time.time()
        return bool(changed)

    def decide(self, probs: Dist, metrics: Any = None, abstain_threshold: float | None = None) -> dict:
        d = contract.decide(probs, self.settings(), metrics, abstain_threshold)
        d["policy_version"] = self.version
        return d

    def to_dict(self) -> dict:
        return asdict(self)


def replay(policy: DecisionPolicy, logged: Iterable[tuple[Dist, str]], metrics: Any = None) -> dict:
    """Evaluate ``policy`` on logged (served probabilities, true label) pairs.

    Counts how many it would have answered, how many of those would have been wrong
    and, if the policy has costs, the total cost (a wrong answer costs ``cost_wrong``
    or the matrix cell, an escalation ``cost_abstain``). The risk budget's threshold
    is taken from ``metrics`` as it is now.
    """
    n = answered = wrong = 0
    cost = 0.0
    priced = policy.cost_matrix is not None or policy.cost_wrong is not None
    for probs, label in logged:
        if not probs or label is None:
            continue
        d = policy.decide(probs, metrics)
        n += 1
        if d["abstain"]:
            cost += policy.cost_abstain or 0.0
            continue
        answered += 1
        if d["answer"] != label:
            wrong += 1
            if policy.cost_matrix:
                default = 1.0 if policy.cost_wrong is None else policy.cost_wrong
                cost += policy.cost_matrix.get(d["answer"], {}).get(label, default)
            else:
                cost += policy.cost_wrong or 0.0
    return {"decisions": n, "answered": answered, "coverage": round(answered / n, 4) if n else None,
            "wrong": wrong, "risk": round(wrong / answered, 4) if answered else None,
            "total_cost": round(cost, 4) if priced else None,
            "cost_per_decision": round(cost / n, 4) if priced and n else None}


def top(probs: Dist) -> str:
    return confidence_of(probs)[0]
