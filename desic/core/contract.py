"""Decision contract: when the student answers and when it escalates.

A question's settings say what a mistake costs. Up to three rules apply; when
more than one is set, the strictest wins:

* **threshold** – answer when the calibrated confidence reaches
  ``abstain_threshold`` (the default when nothing else is set).
* **costs** – Chow's rule: answer only when the expected cost of answering is
  below the cost of escalating (``cost_wrong``, ``cost_abstain``). With a
  ``cost_matrix`` (asymmetric mistakes, e.g. approving a bad loan costs more
  than rejecting a good one) the answer itself is the one with the lowest
  expected cost, which need not be the most probable one.
* **risk budget** – keep the error rate among *answered* decisions at or below
  ``risk_budget``: the lowest confidence threshold whose error rate on recent
  labelled decisions still fits the budget with 90% confidence (Wilson upper
  bound). Until enough labelled decisions exist the other rules decide.

Inputs that are mostly new to the student are always escalated, whatever the
rules say. Every decision records what decided it (rule, threshold, expected
cost), so it can be audited and evaluated later. The policy is deterministic:
the propensity of the chosen action is 1.
"""

from __future__ import annotations

import math
from typing import Any

from .calibration import confidence_of

Dist = dict[str, float]
Z90 = 1.2816  # one-sided 90% normal quantile
MIN_LABELS = 30  # labelled decisions needed before the risk budget takes over
MIN_SUPPORT = 10  # a threshold must be backed by at least this many labelled decisions
REFIT_EVERY = 10  # labels between two threshold refits


def wilson_upper(errors: int, n: int, z: float = Z90) -> float:
    """One-sided upper confidence bound of an error rate (Wilson score interval)."""
    if n <= 0:
        return 1.0
    p = errors / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + margin) / denom)


def budget_threshold(records: list[tuple[float, bool]], budget: float) -> float | None:
    """Lowest confidence τ whose labelled decisions (confidence ≥ τ) fit ``budget`` at 90%.

    ``records`` are (served confidence, was it right) pairs. ``None`` = too little
    evidence yet; a value above 1 = no threshold fits, so nothing may be answered.
    """
    if len(records) < MIN_LABELS:
        return None
    ranked = sorted(records, key=lambda r: -r[0])
    errors, best = 0, None
    for k, (c, ok) in enumerate(ranked, 1):
        errors += not ok
        if k < len(ranked) and ranked[k][0] == c:
            continue  # a threshold at c admits every decision with confidence c: judge the whole group
        if k >= MIN_SUPPORT and wilson_upper(errors, k) <= budget:
            best = c  # keep going: a lower threshold that still fits means more coverage
    return best if best is not None else 1.01


def _budget_tau(metrics: Any, budget: float) -> float | None:
    cache = getattr(metrics, "_budget_cache", None)
    if not isinstance(cache, dict):
        cache = metrics._budget_cache = {}
    hit = cache.get(budget)
    if hit is not None and metrics.n - hit[0] < REFIT_EVERY:
        return hit[1]
    tau = budget_threshold([(r[0], r[1]) for r in metrics.records], budget)
    cache[budget] = (metrics.n, tau)
    return tau


def expected_costs(probs: Dist, matrix: dict[str, dict[str, float]], cost_wrong: float | None) -> Dist:
    """Expected cost of giving each answer; matrix[answer][truth], missing cells cost 0 if right else ``cost_wrong`` (1)."""
    default = 1.0 if cost_wrong is None else cost_wrong
    out = {}
    for a in probs:
        row = matrix.get(a, {})
        out[a] = sum(p * row.get(y, 0.0 if y == a else default) for y, p in probs.items())
    return out


def decide(probs: Dist, settings: dict, metrics: Any = None, abstain_threshold: float | None = None) -> dict:
    """Which answer to give and whether the rules allow giving it.

    An explicit ``abstain_threshold`` (per request) overrides the question's contract.
    """
    best, conf = confidence_of(probs)
    if abstain_threshold is not None:
        return {"rule": "request_threshold", "answer": best, "threshold": abstain_threshold,
                "abstain": conf < abstain_threshold, "expected_cost": None, "propensity": 1.0}
    cw, ca = settings.get("cost_wrong"), settings.get("cost_abstain")
    matrix = settings.get("cost_matrix")
    budget = settings.get("risk_budget")
    answer, exp_cost, rules = best, None, []
    abstain = False
    thresholds = []
    if matrix:
        costs = expected_costs(probs, matrix, cw)
        answer = min(costs, key=costs.get)
        exp_cost = costs[answer]
        if ca is not None:
            abstain = exp_cost > ca
        rules.append("cost_matrix")
    elif cw and ca is not None:
        thresholds.append(max(0.0, 1.0 - ca / cw))  # answer iff (1 - p) * cost_wrong <= cost_abstain
        exp_cost = (1.0 - conf) * cw
        rules.append("costs")
    if budget:
        tau = _budget_tau(metrics, budget) if metrics is not None else None
        if tau is not None:
            thresholds.append(tau)
            rules.append("risk_budget")
        else:
            rules.append("risk_budget_warming_up")  # too few labelled decisions: the other rules decide
    if not thresholds and not matrix:
        thresholds.append(settings.get("abstain_threshold", 0.6))
        rules.append("threshold")
    threshold = max(thresholds) if thresholds else None
    p_answer = probs.get(answer, 0.0)
    if threshold is not None and p_answer < threshold:
        abstain = True
    return {"rule": "+".join(rules), "answer": answer, "threshold": None if threshold is None else round(threshold, 4),
            "abstain": abstain, "expected_cost": None if exp_cost is None else round(exp_cost, 4), "propensity": 1.0}


def validate(settings: dict, options: list[str]) -> dict:
    """Check contract settings from a request; returns the cleaned values (``None`` clears one)."""
    out: dict[str, Any] = {}
    if "risk_budget" in settings:
        b = settings["risk_budget"]
        if b is not None:
            b = float(b)
            if not 0 < b < 1:
                raise ValueError("risk_budget must be between 0 and 1 (e.g. 0.05 = at most 5% wrong answers)")
        out["risk_budget"] = b
    for key in ("cost_wrong", "cost_abstain"):
        if key in settings:
            v = settings[key]
            if v is not None:
                v = float(v)
                if v < 0 or (key == "cost_wrong" and v == 0):
                    raise ValueError(f"{key} must be positive")
            out[key] = v
    if "cost_matrix" in settings:
        m = settings["cost_matrix"]
        if m is not None:
            if not isinstance(m, dict):
                raise ValueError("cost_matrix must be {answer: {true answer: cost}}")
            clean: dict[str, dict[str, float]] = {}
            for a, row in m.items():
                if a not in options or not isinstance(row, dict):
                    raise ValueError(f"cost_matrix: {a!r} is not an answer of this question")
                for y, c in row.items():
                    if y not in options:
                        raise ValueError(f"cost_matrix: {y!r} is not an answer of this question")
                    if float(c) < 0:
                        raise ValueError("cost_matrix: costs must not be negative")
                clean[a] = {y: float(c) for y, c in row.items()}
            m = clean or None
        out["cost_matrix"] = m
    return out
