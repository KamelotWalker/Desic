"""Adaptive Hoeffding tree: an incremental tree that replaces itself on drift.

Used by the tree expert (see experts.py) on the structured fields of a state.
"""

from __future__ import annotations

import math
import random

from .drift import DDM, DRIFT, WARNING
from .stats import Label, RollingWindow
from .tree import HoeffdingTree

TREE_DEFAULTS = {
    "grace_period": 50,
    "delta": 1e-3,
    "tau": 0.1,
    "max_depth": 16,
    "leaf_prediction": "nba",
}


def _poisson(rng: random.Random, lam: float) -> int:
    # Knuth's algorithm; fine for the small lambdas used in online bagging.
    l, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= l:
            return k
        k += 1


class AdaptiveTree:
    """A Hoeffding tree guarded by DDM with background-tree replacement."""

    def __init__(self, tree_params: dict | None = None, seed: int | None = None, bagging_lambda: float | None = None) -> None:
        self.tree_params = {**TREE_DEFAULTS, **(tree_params or {})}
        self.rng = random.Random(seed)
        self.bagging_lambda = bagging_lambda
        self.tree = self._new_tree()
        self.background: HoeffdingTree | None = None
        self.detector = DDM()
        self.window = RollingWindow(200)
        self.warnings = 0
        self.drifts = 0

    def _new_tree(self) -> HoeffdingTree:
        return HoeffdingTree(**self.tree_params, seed=self.rng.randrange(1 << 30))

    @property
    def accuracy(self) -> float | None:
        return self.window.mean

    def learn_one(self, x: dict, y: Label, w: float = 1.0) -> str | None:
        event = None
        pred = self.tree.predict_one(x)
        if pred is not None:
            correct = pred == y
            self.window.add(int(correct))
            status = self.detector.update(not correct)
            if status == WARNING and self.background is None:
                self.background = self._new_tree()
                self.warnings += 1
                event = "warning"
            elif status == DRIFT:
                bg = self.background
                self.tree = bg if bg is not None and bg.n_seen > 0 else self._new_tree()
                self.background = None
                self.window = RollingWindow(200)
                self.drifts += 1
                event = "drift"
        if self.bagging_lambda:
            w *= _poisson(self.rng, self.bagging_lambda)
        if w > 0:
            self.tree.learn_one(x, y, w)
            if self.background is not None:
                self.background.learn_one(x, y, w)
        return event
