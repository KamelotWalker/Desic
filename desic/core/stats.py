"""Incremental statistics and per-attribute observers used by the Hoeffding tree.

Every structure here is updated one sample at a time in O(1) (or O(#classes))
so the learner never has to revisit old data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Hashable

Label = Hashable
Distribution = dict[Label, float]


def entropy(dist: Distribution) -> float:
    total = sum(dist.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for w in dist.values():
        if w > 0:
            p = w / total
            h -= p * math.log2(p)
    return h


def info_gain(parent: Distribution, children: list[Distribution], min_branch_frac: float) -> float:
    """Information gain of splitting ``parent`` into ``children``.

    Returns ``-inf`` when fewer than two branches hold at least
    ``min_branch_frac`` of the weight, so degenerate splits are never chosen.
    """
    total = sum(parent.values())
    if total <= 0:
        return float("-inf")
    weights = [sum(c.values()) for c in children]
    if sum(1 for w in weights if w / total >= min_branch_frac) < 2:
        return float("-inf")
    child_h = sum((w / total) * entropy(c) for w, c in zip(weights, children) if w > 0)
    return entropy(parent) - child_h


def hoeffding_bound(value_range: float, delta: float, n: float) -> float:
    return math.sqrt((value_range**2) * math.log(1.0 / delta) / (2.0 * n))


class Gaussian:
    """Weighted running mean/variance (Welford)."""

    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n = 0.0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float, w: float = 1.0) -> None:
        self.n += w
        delta = x - self.mean
        self.mean += w * delta / self.n
        self.m2 += w * delta * (x - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 0.0))

    def cdf(self, x: float) -> float:
        std = self.std
        if std <= 1e-12:
            return 1.0 if x >= self.mean else 0.0
        return 0.5 * (1.0 + math.erf((x - self.mean) / (std * math.sqrt(2.0))))

    def pdf(self, x: float) -> float:
        std = max(self.std, 1e-3)
        z = (x - self.mean) / std
        return math.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))


@dataclass
class SplitSuggestion:
    feature: str
    kind: str  # "numeric" | "nominal"
    merit: float
    value: Any  # threshold for numeric, category for nominal
    children: list[Distribution]


class NumericObserver:
    """Per-class Gaussian approximation of a numeric feature.

    Candidate thresholds are spread uniformly between the observed min and max;
    the class distribution on each side is estimated from the Gaussians' CDFs.
    """

    kind = "numeric"

    def __init__(self, n_splits: int = 10) -> None:
        self.n_splits = n_splits
        self.per_class: dict[Label, Gaussian] = {}
        self.min_per_class: dict[Label, float] = {}
        self.max_per_class: dict[Label, float] = {}

    def update(self, x: float, y: Label, w: float) -> None:
        g = self.per_class.get(y)
        if g is None:
            g = self.per_class[y] = Gaussian()
            self.min_per_class[y] = x
            self.max_per_class[y] = x
        else:
            self.min_per_class[y] = min(self.min_per_class[y], x)
            self.max_per_class[y] = max(self.max_per_class[y], x)
        g.update(x, w)

    def likelihood(self, x: float, y: Label) -> float:
        g = self.per_class.get(y)
        if g is None or g.n <= 0:
            return 0.0
        return g.pdf(x)

    def _split_dists(self, threshold: float) -> tuple[Distribution, Distribution]:
        left: Distribution = {}
        right: Distribution = {}
        for y, g in self.per_class.items():
            if threshold < self.min_per_class[y]:
                lw = 0.0
            elif threshold >= self.max_per_class[y]:
                lw = g.n
            else:
                lw = g.n * g.cdf(threshold)
            left[y] = lw
            right[y] = g.n - lw
        return left, right

    def best_split(self, feature: str, parent: Distribution, min_branch_frac: float) -> SplitSuggestion | None:
        if not self.per_class:
            return None
        lo = min(self.min_per_class.values())
        hi = max(self.max_per_class.values())
        if hi <= lo:
            return None
        best: SplitSuggestion | None = None
        step = (hi - lo) / (self.n_splits + 1)
        for i in range(1, self.n_splits + 1):
            t = lo + step * i
            left, right = self._split_dists(t)
            merit = info_gain(parent, [left, right], min_branch_frac)
            if best is None or merit > best.merit:
                best = SplitSuggestion(feature, "numeric", merit, t, [left, right])
        return best


class NominalObserver:
    """Counts of (category, class). Proposes binary ``x == v`` splits."""

    kind = "nominal"

    def __init__(self) -> None:
        self.counts: dict[Any, Distribution] = {}
        self.class_totals: Distribution = {}

    def update(self, x: Any, y: Label, w: float) -> None:
        dist = self.counts.setdefault(x, {})
        dist[y] = dist.get(y, 0.0) + w
        self.class_totals[y] = self.class_totals.get(y, 0.0) + w

    def likelihood(self, x: Any, y: Label) -> float:
        # Laplace-smoothed P(x | y)
        n_values = max(len(self.counts), 1)
        num = self.counts.get(x, {}).get(y, 0.0) + 1.0
        den = self.class_totals.get(y, 0.0) + n_values
        return num / den

    def best_split(self, feature: str, parent: Distribution, min_branch_frac: float) -> SplitSuggestion | None:
        if len(self.counts) < 2:
            return None
        best: SplitSuggestion | None = None
        for value, dist in self.counts.items():
            left = dict(dist)
            right = {y: self.class_totals.get(y, 0.0) - left.get(y, 0.0) for y in self.class_totals}
            merit = info_gain(parent, [left, right], min_branch_frac)
            if best is None or merit > best.merit:
                best = SplitSuggestion(feature, "nominal", merit, value, [left, right])
        return best


@dataclass
class RollingWindow:
    """Fixed-size window of 0/1 outcomes with O(1) mean."""

    size: int = 500
    values: list[int] = field(default_factory=list)
    _pos: int = 0
    _sum: int = 0

    def add(self, v: int) -> None:
        if len(self.values) < self.size:
            self.values.append(v)
        else:
            self._sum -= self.values[self._pos]
            self.values[self._pos] = v
            self._pos = (self._pos + 1) % self.size
        self._sum += v

    @property
    def mean(self) -> float | None:
        return self._sum / len(self.values) if self.values else None

    def __len__(self) -> int:
        return len(self.values)
