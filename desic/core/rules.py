"""Hard (expert) rules that are evaluated before the learned model.

A rule is a list of conditions joined by AND. The first enabled rule that
matches (highest priority first) decides; otherwise the learned model does.
This gives the classic rule-engine behaviour (Drools/JESS style) on top of a
model that keeps learning underneath.
"""

from __future__ import annotations

import uuid
from typing import Any

OPERATORS = ("==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains", "is_missing")


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return None


def _eq(a: Any, b: Any) -> bool:
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        return na == nb
    return str(a) == str(b)


def condition_matches(cond: dict, x: dict) -> bool:
    op = cond.get("op", "==")
    v = x.get(cond["feature"])
    target = cond.get("value")
    if op == "is_missing":
        return v is None
    if v is None:
        return False
    if op == "==":
        return _eq(v, target)
    if op == "!=":
        return not _eq(v, target)
    if op in (">", ">=", "<", "<="):
        a, b = _num(v), _num(target)
        if a is None or b is None:
            return False
        return {">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b}[op]
    if op in ("in", "not_in"):
        values = target if isinstance(target, list) else [s.strip() for s in str(target).split(",")]
        hit = any(_eq(v, t) for t in values)
        return hit if op == "in" else not hit
    if op == "contains":
        return str(target).lower() in str(v).lower()
    raise ValueError(f"unknown operator {op!r}")


def validate_rule(rule: dict) -> dict:
    if not rule.get("decision"):
        raise ValueError("rule needs a 'decision'")
    conds = rule.get("conditions") or []
    if not conds:
        raise ValueError("rule needs at least one condition")
    for c in conds:
        if not c.get("feature"):
            raise ValueError("every condition needs a 'feature'")
        if c.get("op", "==") not in OPERATORS:
            raise ValueError(f"unknown operator {c.get('op')!r}; use one of {', '.join(OPERATORS)}")
    return {
        "id": rule.get("id") or uuid.uuid4().hex[:8],
        "name": rule.get("name") or "",
        "conditions": [{"feature": c["feature"], "op": c.get("op", "=="), "value": c.get("value")} for c in conds],
        "decision": str(rule["decision"]),
        "priority": int(rule.get("priority", 0)),
        "enabled": bool(rule.get("enabled", True)),
        "hits": int(rule.get("hits", 0)),
    }


def first_match(rules: list[dict], x: dict) -> dict | None:
    for rule in sorted(rules, key=lambda r: r.get("priority", 0), reverse=True):
        if rule.get("enabled", True) and all(condition_matches(c, x) for c in rule["conditions"]):
            return rule
    return None
