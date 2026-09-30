"""Self-learning decision models built from Hoeffding trees.

* ``tree``   – one adaptive Hoeffding tree (fully explainable).
* ``forest`` – an Adaptive Random Forest: several trees trained on
  Poisson-resampled streams with random feature subspaces, each with its own
  drift detector. More accurate, still explainable through its best member.

Both replace themselves when concept drift is detected, so the model follows
the user's feedback even when the "right answer" changes over time.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

from .drift import DDM, DRIFT, WARNING
from .stats import Distribution, Label, RollingWindow
from .tree import HoeffdingTree

TREE_DEFAULTS = {
    "grace_period": 50,
    "delta": 1e-3,
    "tau": 0.1,
    "max_depth": 16,
    "leaf_prediction": "nba",
}
FOREST_DEFAULTS = {"n_trees": 10, "lambda": 6.0, "max_features": "sqrt"}


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

    def __init__(self, tree_params: dict, seed: int | None = None, bagging_lambda: float | None = None) -> None:
        self.tree_params = dict(tree_params)
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

    def learn_one(self, x: dict, y: Label) -> str | None:
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
        w = _poisson(self.rng, self.bagging_lambda) if self.bagging_lambda else 1
        if w > 0:
            self.tree.learn_one(x, y, w)
            if self.background is not None:
                self.background.learn_one(x, y, w)
        return event


class PrequentialMetrics:
    """Test-then-train evaluation: every label is scored before it is learned."""

    def __init__(self, window: int = 500, max_points: int = 400) -> None:
        self.n = 0
        self.correct = 0
        self.window = RollingWindow(window)
        self.confusion: dict[str, dict[str, int]] = {}
        self.history: list[dict] = []
        self.max_points = max_points
        self._every = 1

    def update(self, predicted: Any, actual: Any) -> bool:
        ok = predicted is not None and predicted == actual
        self.n += 1
        self.correct += int(ok)
        self.window.add(int(ok))
        row = self.confusion.setdefault(str(actual), {})
        key = "∅" if predicted is None else str(predicted)
        row[key] = row.get(key, 0) + 1
        if self.n % self._every == 0:
            self.history.append({"n": self.n, "rolling": round(self.window.mean or 0.0, 4), "overall": round(self.correct / self.n, 4)})
            if len(self.history) > self.max_points:
                self.history = self.history[1::2]
                self._every *= 2
        return ok

    def summary(self) -> dict:
        per_class = {}
        labels = set(self.confusion) | {p for row in self.confusion.values() for p in row}
        for c in sorted(labels - {"∅"}):  # ∅ = the model had no prediction yet
            tp = self.confusion.get(c, {}).get(c, 0)
            support = sum(self.confusion.get(c, {}).values())
            predicted = sum(row.get(c, 0) for row in self.confusion.values())
            if support == 0 and predicted == 0:
                continue
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[c] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": support}
        return {
            "evaluated": self.n,
            "accuracy": round(self.correct / self.n, 4) if self.n else None,
            "rolling_accuracy": None if self.window.mean is None else round(self.window.mean, 4),
            "window": len(self.window),
            "per_class": per_class,
            "confusion": self.confusion,
        }


class DecisionModel:
    def __init__(self, kind: str = "tree", params: dict | None = None, seed: int = 42) -> None:
        if kind not in ("tree", "forest"):
            raise ValueError("model kind must be 'tree' or 'forest'")
        params = dict(params or {})
        self.kind = kind
        self.tree_params = {k: params.get(k, v) for k, v in TREE_DEFAULTS.items()}
        self.forest_params = {k: params.get(k, v) for k, v in FOREST_DEFAULTS.items()}
        self.seed = seed
        self.metrics = PrequentialMetrics()
        self.events: list[dict] = []
        self.n_learned = 0
        self.created_at = time.time()
        self.members = self._build_members()

    @property
    def params(self) -> dict:
        return {**self.tree_params, **(self.forest_params if self.kind == "forest" else {})}

    def _build_members(self) -> list[AdaptiveTree]:
        rng = random.Random(self.seed)
        if self.kind == "tree":
            return [AdaptiveTree(self.tree_params, seed=rng.randrange(1 << 30))]
        tp = {**self.tree_params, "max_features": self.forest_params["max_features"]}
        return [
            AdaptiveTree(tp, seed=rng.randrange(1 << 30), bagging_lambda=self.forest_params["lambda"])
            for _ in range(int(self.forest_params["n_trees"]))
        ]

    def reset(self) -> None:
        self.metrics = PrequentialMetrics()
        self.events = []
        self.n_learned = 0
        self.members = self._build_members()

    # ----------------------------------------------------------------- predict
    def predict_proba(self, x: dict) -> Distribution:
        if len(self.members) == 1:
            return self.members[0].tree.predict_proba_one(x)
        votes: Distribution = {}
        for m in self.members:
            proba = m.tree.predict_proba_one(x)
            weight = m.accuracy if m.accuracy is not None else 0.5
            for y, p in proba.items():
                votes[y] = votes.get(y, 0.0) + p * (weight + 1e-6)
        total = sum(votes.values())
        return {y: v / total for y, v in votes.items()} if total > 0 else {}

    def predict(self, x: dict) -> dict:
        proba = self.predict_proba(x)
        if not proba:
            return {"prediction": None, "confidence": 0.0, "probabilities": {}}
        pred, conf = max(proba.items(), key=lambda kv: kv[1])
        return {
            "prediction": pred,
            "confidence": round(conf, 4),
            "probabilities": {str(k): round(v, 4) for k, v in sorted(proba.items(), key=lambda kv: -kv[1])},
        }

    def best_member(self) -> AdaptiveTree:
        return max(self.members, key=lambda m: (m.accuracy or 0.0, m.tree.n_seen))

    def explain(self, x: dict) -> dict:
        member = self.best_member()
        exp = member.tree.explain_one(x)
        exp.pop("probabilities", None)
        if self.kind == "forest":
            votes: dict[str, int] = {}
            for m in self.members:
                p = m.tree.predict_one(x)
                if p is not None:
                    votes[str(p)] = votes.get(str(p), 0) + 1
            exp["votes"] = votes
            exp["explained_by"] = f"tree #{self.members.index(member) + 1} of {len(self.members)} (most accurate)"
        return exp

    # ------------------------------------------------------------------- learn
    _UNSET = object()

    def learn(self, x: dict, y: Label, evaluated_prediction: Any = _UNSET) -> list[dict]:
        """Score (prequentially) then learn one labelled example.

        ``evaluated_prediction`` is the prediction that was actually shown to
        the user when the label arrives later as feedback; when omitted the
        model predicts right now (test-then-train).
        """
        if evaluated_prediction is self._UNSET:
            evaluated_prediction = self.predict(x)["prediction"]
        self.metrics.update(evaluated_prediction, y)
        new_events = []
        for i, m in enumerate(self.members):
            ev = m.learn_one(x, y)
            if ev is not None:
                event = {"type": ev, "member": i, "at": self.n_learned, "time": time.time()}
                new_events.append(event)
        self.n_learned += 1
        if new_events:
            self.events = (self.events + new_events)[-100:]
        return new_events

    # ----------------------------------------------------------- introspection
    def feature_importance(self) -> dict[str, float]:
        total: dict[str, float] = {}
        for m in self.members:
            for f, g in m.tree.feature_gain.items():
                total[f] = total.get(f, 0.0) + g
        s = sum(total.values())
        if s <= 0:
            return {}
        return {f: round(g / s, 4) for f, g in sorted(total.items(), key=lambda kv: -kv[1])}

    def structure(self) -> dict:
        stats = [m.tree.stats() for m in self.members]
        return {
            "kind": self.kind,
            "members": len(self.members),
            "nodes": sum(s["nodes"] for s in stats),
            "leaves": sum(s["leaves"] for s in stats),
            "depth": max(s["depth"] for s in stats),
            "drifts": sum(m.drifts for m in self.members),
            "warnings": sum(m.warnings for m in self.members),
            "learned": self.n_learned,
        }

    def tree_dict(self, member: int | None = None) -> dict:
        m = self.members[member] if member is not None else self.best_member()
        return m.tree.to_dict()

    def rules(self, member: int | None = None) -> list[dict]:
        m = self.members[member] if member is not None else self.best_member()
        return m.tree.rules()
