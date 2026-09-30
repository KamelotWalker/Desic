"""Metrics for the evaluation harness.

Everything is computed from what a user would actually have been served:
the calibrated (and familiarity-shrunk) probabilities of ``task.answer`` and
the abstain flag of ``task.public``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from ..core.calibration import confidence_of, reliability

EPS = 1e-6


def log_loss(p: dict[str, float], y: str) -> float:
    return -math.log(max(p.get(y, 0.0), EPS))


def risk_at_coverage(records: list[tuple[float, bool]], coverage: float) -> float | None:
    """Error rate among the ``coverage`` share of most confident answers."""
    if not records:
        return None
    k = max(1, math.ceil(coverage * len(records)))
    top = sorted(records, key=lambda r: -r[0])[:k]
    return 1.0 - sum(ok for _, ok in top) / k


def aurc(records: list[tuple[float, bool]]) -> float | None:
    """Area under the risk–coverage curve (lower is better; 0 = never wrong when it answers first)."""
    if not records:
        return None
    total = errors = 0.0
    for i, (_, ok) in enumerate(sorted(records, key=lambda r: -r[0]), 1):
        errors += not ok
        total += errors / i
    return total / len(records)


def evaluate(task: Any, items: Iterable[tuple[str, str]], per_class: bool = False) -> dict:
    """Score a task on held-out ``(state, true label)`` pairs without learning from them."""
    records: list[tuple[float, bool]] = []
    nll = brier = 0.0
    answered = answered_ok = 0
    by_class: dict[str, list[int]] = {}
    for x, y in items:
        a = task.answer(x)
        p = a["probabilities"]
        best, conf = confidence_of(p)
        ok = best == y
        records.append((conf, ok))
        nll += log_loss(p, y)
        brier += sum((v - (o == y)) ** 2 for o, v in p.items()) + (0.0 if y in p else 1.0)
        if not task.public(a)["abstain"]:
            answered += 1
            answered_ok += ok
        c = by_class.setdefault(y, [0, 0])
        c[0] += ok
        c[1] += 1
    n = len(records)
    if not n:
        return {"n": 0}
    out = {
        "n": n,
        "accuracy": round(sum(ok for _, ok in records) / n, 4),
        "nll": round(nll / n, 4),
        "brier": round(brier / n, 4),
        "ece": reliability(records)[0],
        # the served decision rule: answer unless the task abstains
        "coverage": round(answered / n, 4),
        "answered_accuracy": round(answered_ok / answered, 4) if answered else None,
        # threshold-free view of the same trade-off
        "risk@80%": round(risk_at_coverage(records, 0.8), 4),
        "aurc": round(aurc(records), 4),
    }
    if per_class:
        out["per_class"] = {k: round(v[0] / v[1], 4) for k, v in sorted(by_class.items())}
    return out


class Prequential:
    """Test-then-train bookkeeping over a stream: every label is first scored with
    the probabilities served *before* it was learned."""

    def __init__(self, every: int = 250, window: int = 500) -> None:
        self.n = 0
        self.loss = 0.0
        self.correct = 0
        self.every = every
        self.window = window
        self._recent: list[bool] = []
        self.curve: list[dict] = []

    def add(self, p: dict[str, float], y: str) -> None:
        best, _ = confidence_of(p)
        ok = best == y
        self.n += 1
        self.loss += log_loss(p, y)
        self.correct += ok
        self._recent.append(ok)
        if len(self._recent) > self.window:
            self._recent.pop(0)
        if self.n % self.every == 0:
            self.curve.append({"n": self.n, "cum_log_loss": round(self.loss / self.n, 4),
                               "window_accuracy": round(sum(self._recent) / len(self._recent), 4)})

    def summary(self) -> dict:
        return {"labels": self.n, "cum_log_loss": round(self.loss / self.n, 4) if self.n else None,
                "accuracy": round(self.correct / self.n, 4) if self.n else None, "curve": self.curve}


def forgetting_index(checkpoints: list[dict[str, float]], learned_at: dict[str, int]) -> float | None:
    """Average forgetting (Chaudhry et al., 2018).

    ``checkpoints[k][c]`` is the test accuracy on class ``c`` after checkpoint ``k``;
    ``learned_at[c]`` is the first checkpoint at which ``c`` had been taught. For every
    class taught before the last checkpoint: best accuracy it ever had minus the final one.
    """
    last = len(checkpoints) - 1
    drops = []
    for c, k0 in learned_at.items():
        if k0 >= last:
            continue
        best = max(checkpoints[k].get(c, 0.0) for k in range(k0, last))
        drops.append(best - checkpoints[last].get(c, 0.0))
    return round(sum(drops) / len(drops), 4) if drops else None


def half_life(curve: list[tuple[int, float]], low: float, high: float) -> int | None:
    """Labels needed to win back half of a drop from ``high`` to ``low``.

    ``curve`` holds ``(labels since the shock, value)`` points; ``None`` means the
    value never got half way back within the stream.
    """
    if high <= low:
        return 0
    target = low + 0.5 * (high - low)
    for t, v in curve:
        if v >= target:
            return t
    return None
