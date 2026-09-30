"""The student's experts and how they are combined.

Each expert returns a probability distribution over the question's options,
or ``None`` when it has nothing to say (a "sleeping" expert). The mixture
weights experts by their running log loss — a strictly proper scoring rule, so
an expert is rewarded for honest probabilities, not just for being right.

Experts:
    prior   – base rate of each answer (always a sane fallback)
    linear  – online multinomial logistic regression over text + JSON features
    tree    – adaptive Hoeffding tree over structured fields (thresholds)
    memory  – nearest labelled examples; learns *instantly* from one correction
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from .features import Features
from .model import AdaptiveTree

Dist = dict[str, float]
EPS = 1e-6


def normalize(d: Dist) -> Dist:
    s = sum(d.values())
    if s <= 0:
        return {k: 1.0 / len(d) for k in d} if d else {}
    return {k: v / s for k, v in d.items()}


def uniform(options: list[str]) -> Dist:
    return {o: 1.0 / len(options) for o in options}


def smooth(d: Dist, options: list[str], eps: float = 0.01) -> Dist:
    k = len(options)
    return {o: (1 - eps) * d.get(o, 0.0) + eps / k for o in options}


def cross_entropy(target: Dist, p: Dist) -> float:
    return -sum(q * math.log(max(p.get(o, 0.0), EPS)) for o, q in target.items() if q > 0)


class Expert:
    name = "expert"

    def predict(self, x: Features, options: list[str]) -> Dist | None:
        raise NotImplementedError

    def learn(self, x: Features, target: Dist, weight: float) -> str | None:
        raise NotImplementedError

    def explain(self, x: Features, options: list[str], chosen: str) -> dict:
        return {}


class PriorExpert(Expert):
    name = "prior"

    def __init__(self) -> None:
        self.counts: Dist = {}

    def predict(self, x: Features, options: list[str]) -> Dist | None:
        if not self.counts:
            return None
        return normalize({o: self.counts.get(o, 0.0) + 0.5 for o in options})

    def learn(self, x: Features, target: Dist, weight: float) -> None:
        for o, q in target.items():
            self.counts[o] = self.counts.get(o, 0.0) + q * weight

    def explain(self, x: Features, options: list[str], chosen: str) -> dict:
        total = sum(self.counts.values()) or 1.0
        return {"base_rates": {o: round(self.counts.get(o, 0.0) / total, 3) for o in options}}


class LinearExpert(Expert):
    """Softmax regression trained by AdaGrad on the log loss (soft targets allowed).

    The AdaGrad accumulator is capped so the learning rate never decays below
    lr / sqrt(cap): the model keeps some plasticity for drifting feedback.
    """

    name = "linear"

    def __init__(self, lr: float = 0.5, g_cap: float = 25.0, l2: float = 1e-4) -> None:
        self.lr = lr
        self.g_cap = g_cap
        self.l2 = l2
        self.w: dict[str, dict[str, float]] = {}
        self.g: dict[str, dict[str, float]] = {}
        self.seen = 0

    def _logits(self, x: Features, options: list[str]) -> dict[str, float]:
        out = {}
        for o in options:
            wo = self.w.get(o)
            out[o] = sum(wo.get(f, 0.0) * v for f, v in x.sparse.items()) if wo else 0.0
        return out

    def predict(self, x: Features, options: list[str]) -> Dist | None:
        if self.seen == 0:
            return None
        z = self._logits(x, options)
        m = max(z.values())
        return normalize({o: math.exp(v - m) for o, v in z.items()})

    def learn(self, x: Features, target: Dist, weight: float, options: list[str] | None = None) -> None:
        options = list(dict.fromkeys([*(options or []), *self.w, *target]))
        z = self._logits(x, options)
        m = max(z.values())
        p = normalize({o: math.exp(v - m) for o, v in z.items()})
        for o in options:
            grad = (p[o] - target.get(o, 0.0)) * weight
            if abs(grad) < 1e-9:
                continue
            wo = self.w.setdefault(o, {})
            go = self.g.setdefault(o, {})
            for f, v in x.sparse.items():
                gi = grad * v + self.l2 * wo.get(f, 0.0)
                acc = min(go.get(f, 0.0) + gi * gi, self.g_cap)
                go[f] = acc
                wo[f] = wo.get(f, 0.0) - self.lr * gi / (math.sqrt(acc) + 1e-8)
        self.seen += 1

    def explain(self, x: Features, options: list[str], chosen: str) -> dict:
        others = [o for o in options if o != chosen]
        contrib = []
        wc = self.w.get(chosen, {})
        for f, v in x.sparse.items():
            if f == "bias":
                continue
            base = sum(self.w.get(o, {}).get(f, 0.0) for o in others) / len(others) if others else 0.0
            c = v * (wc.get(f, 0.0) - base)
            if abs(c) > 1e-3:
                contrib.append((f, c))
        contrib.sort(key=lambda t: -abs(t[1]))
        return {
            "for": [{"feature": f, "weight": round(c, 3)} for f, c in contrib if c > 0][:6],
            "against": [{"feature": f, "weight": round(c, 3)} for f, c in contrib if c < 0][:4],
        }


class TreeExpert(Expert):
    """Adaptive Hoeffding tree over the structured (scalar) fields of the state."""

    name = "tree"

    def __init__(self, seed: int = 7) -> None:
        self.model = AdaptiveTree(seed=seed)
        self.events: list[str] = []

    def predict(self, x: Features, options: list[str]) -> Dist | None:
        if not x.scalars or self.model.tree.n_seen == 0:
            return None
        proba = self.model.tree.predict_proba_one(x.scalars)
        if not proba:
            return None
        return normalize({o: proba.get(o, 0.0) + 0.02 for o in options})

    def learn(self, x: Features, target: Dist, weight: float) -> str | None:
        if not x.scalars:
            return None
        label, q = max(target.items(), key=lambda kv: kv[1])
        return self.model.learn_one(x.scalars, label, weight * q)

    def explain(self, x: Features, options: list[str], chosen: str) -> dict:
        exp = self.model.tree.explain_one(x.scalars)
        return {"path": exp["path"], "leaf_support": exp["leaf_support"]}


class MemoryExpert(Expert):
    """k-nearest labelled examples (cosine similarity on text / categorical features).

    This is the "hot" layer: one human correction changes the next answer for
    the same or a near-identical state immediately, before any weight update
    has had time to accumulate. An inverted index keeps lookups fast; features
    shared by more than a quarter of the memory (stop words, common categories)
    are not used to find candidates.
    """

    name = "memory"

    def __init__(self, capacity: int = 2000, k: int = 5, min_sim: float = 0.35) -> None:
        self.capacity = capacity
        self.k = k
        self.min_sim = min_sim
        self.items: dict[int, tuple] = {}  # id -> (vec, norm, target, weight, ref)
        self.order: deque = deque()
        self.index: dict[str, set[int]] = {}
        self.next_id = 0
        self.exact: dict[str, tuple[Dist, float]] = {}

    @staticmethod
    def _vec(x: Features) -> dict[str, float]:
        return {f: v for f, v in x.sparse.items() if f != "bias" and not f.startswith("n:")}

    def _neighbours(self, x: Features) -> list[tuple[float, tuple]]:
        vec = self._vec(x)
        if not vec or not self.items:
            return []
        limit = max(5, len(self.items) // 4)
        overlap: dict[int, int] = {}
        for f in vec:
            ids = self.index.get(f)
            if ids and len(ids) <= limit:
                for i in ids:
                    overlap[i] = overlap.get(i, 0) + 1
        if not overlap:
            return []
        candidates = sorted(overlap, key=lambda i: -overlap[i])[:200]
        norm = math.sqrt(sum(v * v for v in vec.values()))
        scored = []
        for i in candidates:
            item = self.items[i]
            ivec = item[0]
            dot = sum(v * ivec.get(f, 0.0) for f, v in vec.items())
            if dot > 0:
                scored.append((dot / (norm * item[1]), item))
        scored.sort(key=lambda t: -t[0])
        return scored[: self.k]

    def predict(self, x: Features, options: list[str], key: str | None = None) -> Dist | None:
        if key is not None and key in self.exact:
            dist, _ = self.exact[key]
            return smooth(dist, options, 0.1)
        near = [(s, it) for s, it in self._neighbours(x) if s >= self.min_sim]
        if not near:
            return None
        acc: Dist = {o: 0.0 for o in options}
        for s, (_, _, target, w, _) in near:
            for o, q in target.items():
                if o in acc:
                    acc[o] += s * s * w * q
        if sum(acc.values()) <= 0:
            return None
        return smooth(normalize(acc), options, 0.1)

    def learn(self, x: Features, target: Dist, weight: float, key: str | None = None, ref: str | None = None) -> None:
        vec = self._vec(x)
        if vec:
            i = self.next_id
            self.next_id += 1
            self.items[i] = (vec, math.sqrt(sum(v * v for v in vec.values())), dict(target), weight, ref)
            self.order.append(i)
            for f in vec:
                self.index.setdefault(f, set()).add(i)
            while len(self.order) > self.capacity:
                old = self.order.popleft()
                for f in self.items.pop(old)[0]:
                    ids = self.index.get(f)
                    if ids is not None:
                        ids.discard(old)
                        if not ids:
                            del self.index[f]
        if key is not None:
            prev = self.exact.get(key)
            if prev is None or weight >= prev[1]:  # a human label outranks a teacher label
                self.exact[key] = (dict(target), weight)
            if len(self.exact) > self.capacity * 4:
                for k in list(self.exact)[: self.capacity]:
                    del self.exact[k]

    def __len__(self) -> int:
        return len(self.items)

    def explain(self, x: Features, options: list[str], chosen: str, key: str | None = None) -> dict:
        if key is not None and key in self.exact:
            return {"exact_match": True}
        return {"neighbours": [
            {"similarity": round(s, 3), "answer": max(t.items(), key=lambda kv: kv[1])[0], "ref": ref}
            for s, (_, _, t, _, ref) in self._neighbours(x) if s >= self.min_sim
        ]}


class Mixture:
    """Hedge with sleeping experts and fixed share.

    * η = 1 on the log loss makes this the Bayesian mixture of the experts.
    * Only awake experts are re-weighted, and their total mass is preserved, so
      an expert that is rarely awake (memory) isn't punished while asleep.
    * Fixed share (α) lets a demoted expert recover quickly after concept drift.
    """

    def __init__(self, names: list[str], eta: float = 1.0, share: float = 0.02) -> None:
        self.eta = eta
        self.share = share
        self.logw = {n: 0.0 for n in names}

    def add(self, name: str, share: float) -> None:
        """Add an expert holding ``share`` of the total weight (the others keep their ratios)."""
        share = min(max(share, 1e-3), 0.9)
        m = max(self.logw.values())
        total = math.log(sum(math.exp(v - m) for v in self.logw.values())) + m
        self.logw[name] = total + math.log(share / (1 - share))

    def weights(self) -> dict[str, float]:
        m = max(self.logw.values())
        return normalize({n: math.exp(v - m) for n, v in self.logw.items()})

    def combine(self, preds: dict[str, Dist | None], options: list[str]) -> tuple[Dist, dict[str, float]]:
        awake = {n: p for n, p in preds.items() if p is not None}
        if not awake:
            return uniform(options), {}
        w = self.weights()
        wa = normalize({n: w[n] for n in awake})
        mix = {o: sum(wa[n] * awake[n].get(o, 0.0) for n in awake) for o in options}
        return smooth(normalize(mix), options, 0.005), wa

    def update(self, preds: dict[str, Dist | None], target: Dist) -> None:
        awake = [n for n, p in preds.items() if p is not None]
        if not awake:
            return
        w = self.weights()
        mass = sum(w[n] for n in awake)
        new = {n: w[n] * math.exp(-self.eta * cross_entropy(target, preds[n])) for n in awake}
        s = sum(new.values())
        if s > 0:
            for n in awake:
                w[n] = new[n] / s * mass
        k = len(w)
        w = {n: (1 - self.share) * v + self.share / k for n, v in w.items()}
        self.logw = {n: math.log(max(v, 1e-300)) for n, v in w.items()}
