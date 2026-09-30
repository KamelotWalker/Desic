"""Hoeffding tree (VFDT) — an incremental decision tree.

The tree learns from one example at a time and decides when to split a leaf
using the Hoeffding bound: once enough evidence has been collected that the
best split is better than the runner-up, the leaf is split. No example is
ever stored, so the model can keep learning forever from live feedback.

References:
    Domingos & Hulten, "Mining High-Speed Data Streams", KDD 2000.
    Gama et al., "Accurate decision trees for mining high-speed data streams", KDD 2003 (NB leaves).
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterator

from .stats import (
    Distribution,
    Label,
    NominalObserver,
    NumericObserver,
    hoeffding_bound,
)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _argmax(dist: Distribution) -> Label | None:
    if not dist:
        return None
    return max(dist.items(), key=lambda kv: kv[1])[0]


def _normalize(dist: Distribution) -> Distribution:
    total = sum(dist.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in dist.items()}


class Leaf:
    __slots__ = ("stats", "observers", "last_eval_weight", "mc_correct", "nb_correct", "depth", "features")

    def __init__(self, stats: Distribution | None = None, depth: int = 0) -> None:
        self.stats: Distribution = dict(stats or {})
        self.observers: dict[str, NumericObserver | NominalObserver] = {}
        self.last_eval_weight = self.weight
        self.mc_correct = 0.0
        self.nb_correct = 0.0
        self.depth = depth
        self.features: set[str] | None = None  # random subspace (forest members only)

    @property
    def weight(self) -> float:
        return sum(self.stats.values())


class Split:
    __slots__ = ("feature", "kind", "value", "children", "branch_weights", "stats", "merit", "depth")

    def __init__(self, feature: str, kind: str, value: Any, children: list[Leaf], stats: Distribution, merit: float, depth: int) -> None:
        self.feature = feature
        self.kind = kind
        self.value = value
        self.children: list[Leaf | Split] = list(children)
        self.branch_weights = [c.weight for c in children]
        self.stats = dict(stats)
        self.merit = merit
        self.depth = depth

    def branch(self, x: dict) -> int:
        v = x.get(self.feature)
        if v is not None:
            if self.kind == "numeric" and _is_number(v):
                return 0 if v <= self.value else 1
            if self.kind == "nominal":
                return 0 if v == self.value else 1
        # Missing / mistyped value: follow the heavier branch.
        return 0 if self.branch_weights[0] >= self.branch_weights[1] else 1

    def condition(self, branch: int) -> dict:
        if self.kind == "numeric":
            op = "<=" if branch == 0 else ">"
            value: Any = round(self.value, 6)
        else:
            op = "==" if branch == 0 else "!="
            value = self.value
        return {"feature": self.feature, "op": op, "value": value}


Node = Leaf | Split


class HoeffdingTree:
    def __init__(
        self,
        grace_period: int = 200,
        delta: float = 1e-7,
        tau: float = 0.05,
        max_depth: int = 20,
        leaf_prediction: str = "nba",
        min_branch_frac: float = 0.01,
        max_features: int | str | None = None,
        n_numeric_splits: int = 10,
        seed: int | None = None,
    ) -> None:
        if leaf_prediction not in ("mc", "nb", "nba"):
            raise ValueError("leaf_prediction must be 'mc', 'nb' or 'nba'")
        self.grace_period = grace_period
        self.delta = delta
        self.tau = tau
        self.max_depth = max_depth
        self.leaf_prediction = leaf_prediction
        self.min_branch_frac = min_branch_frac
        self.max_features = max_features
        self.n_numeric_splits = n_numeric_splits
        self.rng = random.Random(seed)
        self.root: Node = Leaf()
        self.n_seen = 0.0
        self.classes: dict[Label, None] = {}
        self.feature_gain: dict[str, float] = {}

    # ------------------------------------------------------------------ learning
    def learn_one(self, x: dict, y: Label, w: float = 1.0) -> None:
        if w <= 0:
            return
        self.classes.setdefault(y)
        self.n_seen += w
        node, parent, branch = self.root, None, None
        while isinstance(node, Split):
            b = node.branch(x)
            node.branch_weights[b] += w
            parent, branch = node, b
            node = node.children[b]
        leaf = node
        self._leaf_learn(leaf, x, y, w)
        if leaf.depth < self.max_depth and leaf.weight - leaf.last_eval_weight >= self.grace_period:
            self._attempt_split(leaf, parent, branch)
            leaf.last_eval_weight = leaf.weight

    def _leaf_features(self, leaf: Leaf, x: dict) -> Iterator[str]:
        if self.max_features is None:
            yield from x
            return
        if leaf.features is None:
            names = sorted(x)
            k = self.max_features
            if k == "sqrt":
                k = max(1, round(math.sqrt(len(names))))
            k = max(1, min(int(k), len(names)))
            leaf.features = set(self.rng.sample(names, k))
        yield from (f for f in x if f in leaf.features)

    def _leaf_learn(self, leaf: Leaf, x: dict, y: Label, w: float) -> None:
        if leaf.stats and self.leaf_prediction == "nba":
            if _argmax(leaf.stats) == y:
                leaf.mc_correct += w
            if _argmax(self._nb_proba(leaf, x)) == y:
                leaf.nb_correct += w
        leaf.stats[y] = leaf.stats.get(y, 0.0) + w
        for f in self._leaf_features(leaf, x):
            v = x[f]
            if v is None:
                continue
            obs = leaf.observers.get(f)
            if obs is None:
                obs = NumericObserver(self.n_numeric_splits) if _is_number(v) else NominalObserver()
                leaf.observers[f] = obs
            if obs.kind == "numeric" and not _is_number(v):
                continue
            obs.update(v, y, w)

    def _attempt_split(self, leaf: Leaf, parent: Split | None, branch: int | None) -> None:
        if len([c for c, w in leaf.stats.items() if w > 0]) < 2:
            return  # pure leaf
        suggestions = []
        for f, obs in leaf.observers.items():
            s = obs.best_split(f, leaf.stats, self.min_branch_frac)
            if s is not None and s.merit != float("-inf"):
                suggestions.append(s)
        if not suggestions:
            return
        suggestions.sort(key=lambda s: s.merit, reverse=True)
        best = suggestions[0]
        second = suggestions[1].merit if len(suggestions) > 1 else 0.0  # vs. "don't split"
        n = leaf.weight
        value_range = math.log2(max(len(leaf.stats), 2))
        eps = hoeffding_bound(value_range, self.delta, n)
        if best.merit <= 0 or not (best.merit - second > eps or eps < self.tau):
            return
        children = [Leaf(dist, leaf.depth + 1) for dist in best.children]
        node = Split(best.feature, best.kind, best.value, children, leaf.stats, best.merit, leaf.depth)
        if parent is None:
            self.root = node
        else:
            parent.children[branch] = node
        self.feature_gain[best.feature] = self.feature_gain.get(best.feature, 0.0) + best.merit * n

    # ---------------------------------------------------------------- inference
    def _sort(self, x: dict) -> tuple[Leaf, list[tuple[Split, int]]]:
        node, path = self.root, []
        while isinstance(node, Split):
            b = node.branch(x)
            path.append((node, b))
            node = node.children[b]
        return node, path

    def _nb_proba(self, leaf: Leaf, x: dict) -> Distribution:
        total = leaf.weight
        if total <= 0:
            return {}
        log_scores: dict[Label, float] = {}
        for y, cw in leaf.stats.items():
            if cw <= 0:
                continue
            score = math.log(cw / total)
            for f, obs in leaf.observers.items():
                v = x.get(f)
                if v is None or (obs.kind == "numeric" and not _is_number(v)):
                    continue
                score += math.log(max(obs.likelihood(v, y), 1e-300))
            log_scores[y] = score
        if not log_scores:
            return {}
        m = max(log_scores.values())
        return _normalize({y: math.exp(s - m) for y, s in log_scores.items()})

    def _leaf_proba(self, leaf: Leaf, x: dict) -> tuple[Distribution, str]:
        use_nb = self.leaf_prediction == "nb" or (
            self.leaf_prediction == "nba" and leaf.nb_correct > leaf.mc_correct
        )
        if use_nb and leaf.observers:
            proba = self._nb_proba(leaf, x)
            if proba:
                return proba, "naive_bayes"
        return _normalize(leaf.stats), "majority"

    def predict_proba_one(self, x: dict) -> Distribution:
        leaf, _ = self._sort(x)
        return self._leaf_proba(leaf, x)[0]

    def predict_one(self, x: dict) -> Label | None:
        return _argmax(self.predict_proba_one(x))

    def explain_one(self, x: dict) -> dict:
        leaf, path = self._sort(x)
        proba, method = self._leaf_proba(leaf, x)
        steps = []
        for split, b in path:
            cond = split.condition(b)
            cond["observed"] = x.get(split.feature)
            cond["missing"] = x.get(split.feature) is None
            steps.append(cond)
        return {
            "path": steps,
            "leaf_support": round(leaf.weight, 3),
            "leaf_distribution": {str(k): round(v, 3) for k, v in leaf.stats.items()},
            "leaf_method": method,
            "probabilities": proba,
        }

    # ------------------------------------------------------------ introspection
    def iter_nodes(self) -> Iterator[Node]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            if isinstance(node, Split):
                stack.extend(node.children)

    def stats(self) -> dict:
        n_nodes = n_leaves = depth = 0
        for node in self.iter_nodes():
            n_nodes += 1
            depth = max(depth, node.depth)
            if isinstance(node, Leaf):
                n_leaves += 1
        return {"nodes": n_nodes, "leaves": n_leaves, "depth": depth, "seen": round(self.n_seen, 1)}

    def to_dict(self, max_depth: int = 12) -> dict:
        def walk(node: Node) -> dict:
            if isinstance(node, Leaf) or node.depth >= max_depth:
                stats = node.stats
                proba = _normalize(stats)
                pred = _argmax(stats)
                return {
                    "type": "leaf",
                    "prediction": None if pred is None else str(pred),
                    "confidence": round(proba.get(pred, 0.0), 4) if pred is not None else 0.0,
                    "support": round(sum(stats.values()), 2),
                    "distribution": {str(k): round(v, 2) for k, v in stats.items()},
                    "truncated": isinstance(node, Split),
                }
            return {
                "type": "split",
                "feature": node.feature,
                "kind": node.kind,
                "value": round(node.value, 6) if node.kind == "numeric" else node.value,
                "merit": round(node.merit, 4),
                "support": round(sum(node.branch_weights), 2),
                "children": [walk(c) for c in node.children],
            }

        return walk(self.root)

    def rules(self) -> list[dict]:
        """Every root-to-leaf path as a simplified IF-THEN rule."""
        out: list[dict] = []

        def walk(node: Node, conds: list[dict]) -> None:
            if isinstance(node, Leaf):
                pred = _argmax(node.stats)
                if pred is None:
                    return
                proba = _normalize(node.stats)
                out.append({
                    "conditions": simplify_conditions(conds),
                    "prediction": str(pred),
                    "confidence": round(proba[pred], 4),
                    "support": round(node.weight, 2),
                })
                return
            for b, child in enumerate(node.children):
                walk(child, conds + [node.condition(b)])

        walk(self.root, [])
        out.sort(key=lambda r: r["support"], reverse=True)
        return out


def simplify_conditions(conds: list[dict]) -> list[dict]:
    """Merge redundant conditions on the same feature (x<=5 & x<=3 -> x<=3)."""
    upper: dict[str, float] = {}
    lower: dict[str, float] = {}
    equals: dict[str, Any] = {}
    not_equals: dict[str, list] = {}
    order: list[str] = []
    for c in conds:
        f, op, v = c["feature"], c["op"], c["value"]
        if f not in order:
            order.append(f)
        if op == "<=":
            upper[f] = min(upper.get(f, v), v)
        elif op == ">":
            lower[f] = max(lower.get(f, v), v)
        elif op == "==":
            equals[f] = v
        elif op == "!=":
            not_equals.setdefault(f, []).append(v)
    out = []
    for f in order:
        if f in lower:
            out.append({"feature": f, "op": ">", "value": lower[f]})
        if f in upper:
            out.append({"feature": f, "op": "<=", "value": upper[f]})
        if f in equals:
            out.append({"feature": f, "op": "==", "value": equals[f]})
        elif f in not_equals:
            vals = list(dict.fromkeys(not_equals[f]))
            if len(vals) == 1:
                out.append({"feature": f, "op": "!=", "value": vals[0]})
            else:
                out.append({"feature": f, "op": "not_in", "value": vals})
    return out
