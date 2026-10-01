"""Reversible patch layer: learn instantly, change the base model only after probation.

A plain online learner writes every label straight into its weights. One bad
batch of labels is then everywhere (benchmarks/stage-02: 30 wrong labels took
over a class and cost the others 3–5 points), and the only way back is to
replay the whole log.

:class:`PatchedTask` puts a reversible layer in front of a :class:`DecisionTask`
(the *base*):

* **Patches.** Every label becomes a patch entry at once: the state, the answer
  and its evidence (source, weight, ref, time). Patches act only *locally*, on
  inputs similar to them (cosine similarity, k nearest).
* **Trust.** Later labels next to a patch support it (same answer) or contradict
  it (another answer). A contradicted patch votes less and is rehearsed less
  (edited nearest neighbours), so a wrong or outdated label fades out locally.
* **Gate.** How much the patches outweigh the base is learned, not fixed: a
  Hedge mixture (base vs patches) per context cell; the cell is how close the
  nearest patch is and whether patches and base agree. A temperature on top
  keeps the combined answer calibrated.
* **Probation.** An entry reaches the base once ``probation`` further
  events. Until then retracting it is exact and O(1): the base never saw it.
* **Consolidation.** When an entry is consolidated the base learns it, together
  with ``replay`` class-balanced rehearsals of older, trusted entries, so a
  stream that arrives one class at a time does not wipe out the earlier ones.
* **Checkpoints.** The base is checkpointed every ``checkpoint_every``
  consolidations; retracting a consolidated entry replays from the nearest
  earlier checkpoint instead of from the beginning of the log.

Everything a label changed besides the base (the trust of its neighbours, the
gate, the calibration) is logged per entry, so a retraction undoes it too.
Consolidated entries stay in the store as a class-balanced episodic memory
(eviction takes the oldest entry of the most frequent answer).
"""

from __future__ import annotations

import math
import pickle
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from .calibration import TemperatureCalibrator, confidence_of
from .experts import Dist, cross_entropy, normalize, smooth
from .features import Features, state_hash
from .task import FAMILIAR, DecisionTask, SpecError


@dataclass
class Patch:
    id: int
    state: Any
    vec: dict[str, float]
    norm: float
    key: str
    target: Dist
    label: str
    weight: float
    source: str
    ref: str | None
    t: int
    consolidated: bool = False
    log_pos: int = -1
    support: float = 0.0  # later neighbours with the same answer (similarity² × weight)
    contra: float = 0.0   # later neighbours with another answer
    touched: list = field(default_factory=list)  # (neighbour id, d_support, d_contra) this entry caused

    def trust(self, prior: float) -> float:
        return (prior + self.support) / (prior + self.support + self.contra)


def _vec(x: Features) -> dict[str, float]:
    return {f: v for f, v in x.sparse.items() if f != "bias" and not f.startswith("n:")}


class PatchStore:
    """Patch entries with an inverted index for nearest-neighbour lookups."""

    def __init__(self, capacity: int = 20000, k: int = 10, min_sim: float = 0.35, df_cap: int = 1000,
                 trust_prior: float = 1.0) -> None:
        self.capacity = capacity
        self.k = k
        self.min_sim = min_sim
        self.df_cap = df_cap
        self.trust_prior = trust_prior
        self.entries: dict[int, Patch] = {}
        self.index: dict[str, set[int]] = {}
        self.by_label: dict[str, dict[int, None]] = {}  # insertion-ordered: oldest first
        self.by_key: dict[str, dict[int, None]] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, p: Patch) -> None:
        self.entries[p.id] = p
        for f in p.vec:
            self.index.setdefault(f, set()).add(p.id)
        self.by_label.setdefault(p.label, {})[p.id] = None
        self.by_key.setdefault(p.key, {})[p.id] = None
        while len(self.entries) > self.capacity and self._evict():
            pass

    def remove(self, pid: int) -> Patch | None:
        p = self.entries.pop(pid, None)
        if p is None:
            return None
        for f in p.vec:
            ids = self.index.get(f)
            if ids is not None:
                ids.discard(pid)
                if not ids:
                    del self.index[f]
        for group, k in ((self.by_label, p.label), (self.by_key, p.key)):
            g = group.get(k)
            if g is not None:
                g.pop(pid, None)
                if not g:
                    del group[k]
        return p

    def _evict(self) -> bool:
        """Drop the oldest consolidated entry of the most frequent answer (class-balanced memory)."""
        for label in sorted(self.by_label, key=lambda lb: -len(self.by_label[lb])):
            for pid in self.by_label[label]:
                if self.entries[pid].consolidated:
                    self.remove(pid)
                    return True
        return False

    def query(self, vec: dict[str, float], key: str) -> list[tuple[float, Patch]]:
        """The k most similar entries above ``min_sim``; an exact state match has similarity 1."""
        out: dict[int, float] = {pid: 1.0 for pid in self.by_key.get(key, {})}
        if vec and self.entries:
            limit = max(5, min(self.df_cap, len(self.entries) // 4))
            overlap: dict[int, int] = {}
            for f in vec:
                ids = self.index.get(f)
                if ids and len(ids) <= limit:
                    for i in ids:
                        overlap[i] = overlap.get(i, 0) + 1
            if overlap:
                norm = math.sqrt(sum(v * v for v in vec.values()))
                for i in sorted(overlap, key=lambda i: -overlap[i])[:200]:
                    if i in out:
                        continue
                    p = self.entries[i]
                    dot = sum(v * p.vec.get(f, 0.0) for f, v in vec.items())
                    s = dot / (norm * p.norm) if dot > 0 else 0.0
                    if s >= self.min_sim:
                        out[i] = s
        best = sorted(out.items(), key=lambda kv: -kv[1])[: self.k]
        return [(s, self.entries[i]) for i, s in best]

    def vote(self, near: list[tuple[float, Patch]], options: list[str]) -> Dist | None:
        acc = {o: 0.0 for o in options}
        for s, p in near:
            w = s * s * p.weight * p.trust(self.trust_prior)
            for o, q in p.target.items():
                if o in acc:
                    acc[o] += w * q
        if sum(acc.values()) <= 0:
            return None
        return smooth(normalize(acc), options, 0.05)


class Gate:
    """Per-context Hedge mixture of two forecasters: the base model and the patches.

    Context cell = similarity of the nearest patch (binned) × whether patches and
    base agree on the top answer. η = 1 on the log loss is the Bayesian mixture;
    fixed share keeps a demoted forecaster able to come back. Every update is
    logged with the entry that caused it, so it can be replayed without it.
    """

    BINS = (0.5, 0.7, 0.9, 0.999)

    def __init__(self, share: float = 0.02) -> None:
        self.share = share
        self.logw: dict[str, list[float]] = {}  # cell -> [log w_base, log w_patch]
        self.history: list[tuple[int, str, float, float, float]] = []  # (entry id, cell, loss base, loss patch, weight)

    def cell(self, strength: float, agree: bool) -> str:
        b = sum(strength >= t for t in self.BINS)
        return f"{b}{'=' if agree else '≠'}"

    def weights(self, cell: str) -> tuple[float, float]:
        lb, lp = self.logw.get(cell, [0.0, 0.0])
        m = max(lb, lp)
        eb, ep = math.exp(lb - m), math.exp(lp - m)
        return eb / (eb + ep), ep / (eb + ep)

    def update(self, pid: int, cell: str, loss_b: float, loss_p: float, weight: float) -> None:
        self.history.append((pid, cell, loss_b, loss_p, weight))
        self._apply(cell, loss_b, loss_p, weight)

    def _apply(self, cell: str, loss_b: float, loss_p: float, weight: float) -> None:
        wb, wp = self.weights(cell)
        nb, np_ = wb * math.exp(-weight * loss_b), wp * math.exp(-weight * loss_p)
        s = nb + np_
        if s <= 0:
            return
        wb, wp = nb / s, np_ / s
        wb, wp = (1 - self.share) * wb + self.share / 2, (1 - self.share) * wp + self.share / 2
        self.logw[cell] = [math.log(wb), math.log(wp)]

    def forget(self, ids: set[int]) -> None:
        self.history = [h for h in self.history if h[0] not in ids]
        self.logw = {}
        for _, cell, lb, lp, w in self.history:
            self._apply(cell, lb, lp, w)


class PatchedTask:
    """A :class:`DecisionTask` behind a reversible patch layer (same answer / public / learn interface)."""

    def __init__(self, base: DecisionTask, probation: int = 500, replay: int = 1, checkpoint_every: int = 1000,
                 keep_checkpoints: int = 4, trust_sim: float = 0.5, seed: int = 0, **store: Any) -> None:
        self.base = base
        self.probation = probation
        self.replay = replay
        self.checkpoint_every = checkpoint_every
        self.keep_checkpoints = keep_checkpoints
        self.trust_sim = trust_sim  # a later label counts for/against a patch only when at least this similar
        self.store = PatchStore(**store)
        self.gate = Gate()
        self.calibrator = TemperatureCalibrator()
        self.calib_log: deque = deque(maxlen=self.calibrator.samples.maxlen)  # (entry id, raw, label)
        self.pending: deque[Patch] = deque()
        self.log: list[Patch] = []  # consolidated entries, in consolidation order
        self.checkpoints: list[tuple[int, bytes]] = [(0, pickle.dumps(base))]
        self.refs: dict[str, int] = {}
        self.rng = random.Random(seed)
        self.t = 0
        self.next_id = 0
        self._last: tuple | None = None

    # ------------------------------------------------------------ delegation
    @property
    def spec(self):
        return self.base.spec

    @property
    def settings(self) -> dict:
        return self.base.settings

    @property
    def events(self) -> list[dict]:
        return self.base.events

    def public(self, internal: dict, abstain_threshold: float | None = None) -> dict:
        return self.base.public(internal, abstain_threshold)

    # ---------------------------------------------------------------- answer
    def _forecasts(self, state: Any, options: list[str] | None = None) -> dict:
        a = self.base.answer(state, options)
        opts = a["options"]
        pb = self.base.calibrator.apply(a["raw"])  # calibrated, before the familiarity shrink
        near = self.store.query(_vec(a["feats"]), a["key"])
        pq = self.store.vote(near, opts) if near else None
        f = {"a": a, "pb": pb, "pq": pq, "near": near, "cell": None, "strength": 0.0, "raw": pb, "wp": 0.0}
        if pq is not None:
            f["strength"] = near[0][0]
            f["cell"] = self.gate.cell(f["strength"], confidence_of(pb)[0] == confidence_of(pq)[0])
            wb, wp = self.gate.weights(f["cell"])
            f["raw"], f["wp"] = {o: wb * pb.get(o, 0.0) + wp * pq.get(o, 0.0) for o in opts}, wp
        return f

    def answer(self, state: Any, options: list[str] | None = None) -> dict:
        f = self._forecasts(state, options)
        a, opts = f["a"], f["a"]["options"]
        self._last = (a["key"], self.t, f)
        probs = self.calibrator.apply(f["raw"])
        familiarity = a["familiarity"]
        if f["pq"] is not None:  # a close labelled neighbour makes the input familiar before the base learns it
            familiarity = max(familiarity, min(1.0, f["strength"] / 0.5 * FAMILIAR))
        unfamiliar = familiarity < FAMILIAR
        if unfamiliar:
            lam = familiarity / FAMILIAR
            probs = {o: lam * p + (1 - lam) / len(opts) for o, p in probs.items()}
        weights = {k: v * (1 - f["wp"]) for k, v in a["weights"].items()}
        if f["pq"] is not None:
            weights["patch"] = f["wp"]
        return {**a, "raw": f["raw"], "probabilities": probs, "weights": weights, "familiarity": familiarity,
                "unfamiliar": unfamiliar,
                "patch": {"cell": f["cell"], "strength": round(f["strength"], 4), "weight": round(f["wp"], 4)}
                if f["pq"] is not None else None}

    # ----------------------------------------------------------------- learn
    def learn(self, state: Any, target: str | Dist, source: str = "human", weight: float | None = None,
              served: Dist | None = None, served_raw: Dist | None = None, abstained: bool = False,
              ref: str | None = None) -> list[dict]:
        spec = self.base.spec
        if not isinstance(target, dict):
            label = spec.label_of(target)
            if label not in spec.options:
                if spec.type != "choice":
                    raise SpecError(f"{label!r} is not an answer of {spec.name}")
                spec.options[label] = ""
            dist: Dist = {label: 1.0}
        else:
            dist = normalize({k: v for k, v in target.items() if k in spec.options and v > 0})
            if not dist:
                return []
        if weight is None:
            weight = self.base.settings["teacher_weight"] if source == "teacher" else 1.0
        key = state_hash(state)
        last = self._last
        f = last[2] if last is not None and last[0] == key and last[1] == self.t else self._forecasts(state)
        self._last = None
        vec = _vec(f["a"]["feats"])
        p = Patch(self.next_id, state, vec, math.sqrt(sum(v * v for v in vec.values())) or 1.0, key, dist,
                  confidence_of(dist)[0], weight, source, ref, self.t)
        self.next_id += 1
        # prequential: which forecaster would have been right, and was the combined answer calibrated?
        if f["pq"] is not None:
            self.gate.update(p.id, f["cell"], cross_entropy(dist, f["pb"]), cross_entropy(dist, f["pq"]), weight)
        if source in ("human", "dataset") and len(dist) == 1:
            self.calibrator.add(f["raw"], p.label)
            self.calib_log.append((p.id, f["raw"], p.label))
        # this label supports or contradicts the close patches it lands next to
        for s, nb in f["near"]:
            if s < self.trust_sim or nb.id == p.id:
                continue
            d = s * s * weight * max(dist.values())
            if nb.label == p.label:
                nb.support += d
                p.touched.append((nb.id, d, 0.0))
            else:
                nb.contra += d
                p.touched.append((nb.id, 0.0, d))
        self.store.add(p)
        self.pending.append(p)
        if ref is not None:
            self.refs[ref] = p.id
        self.t += 1
        return self._consolidate_due()

    def _consolidate_due(self) -> list[dict]:
        events: list[dict] = []
        while self.pending and self.t - 1 - self.pending[0].t >= self.probation:
            events += self._consolidate(self.pending.popleft())
        return events

    def _consolidate(self, p: Patch) -> list[dict]:
        events = self.base.learn(p.state, p.target, source=p.source, weight=p.weight, ref=p.ref)
        self._rehearse(exclude=p.id)
        p.consolidated = True
        p.log_pos = len(self.log)
        self.log.append(p)
        if len(self.log) % self.checkpoint_every == 0:
            self.checkpoints.append((len(self.log), pickle.dumps(self.base)))
            if len(self.checkpoints) > self.keep_checkpoints + 1:  # the empty base is always kept
                del self.checkpoints[1]
        return events

    def _rehearse(self, exclude: int) -> None:
        """Class-balanced rehearsal of consolidated entries, in proportion to their trust."""
        labels = [lb for lb, ids in self.store.by_label.items() if ids]
        for _ in range(self.replay):
            for _try in range(8):
                if not labels:
                    return
                ids = self.store.by_label[self.rng.choice(labels)]
                pid = next(iter(ids)) if len(ids) == 1 else list(ids)[self.rng.randrange(len(ids))]
                p = self.store.entries[pid]
                if p.consolidated and pid != exclude and self.rng.random() < p.trust(self.store.trust_prior):
                    self.base.learn(p.state, p.target, source=p.source, weight=p.weight, replay=True)
                    break

    def flush(self) -> None:
        """Consolidate everything still on probation (e.g. before saving a snapshot)."""
        while self.pending:
            self._consolidate(self.pending.popleft())

    # --------------------------------------------------------------- retract
    def retract(self, refs: Iterable[str]) -> dict:
        """Undo labels by ref. Their effect on neighbour trust, the gate and the
        calibration is undone; entries still on probation are simply dropped, and
        consolidated ones are removed from the base by restoring the nearest
        earlier checkpoint and replaying the consolidated log after it without them."""
        ids = {self.refs.pop(r) for r in refs if r in self.refs}
        ids &= set(self.store.entries)
        pend = {i for i in ids if not self.store.entries[i].consolidated}
        cons = ids - pend
        for i in ids:
            for nid, ds, dc in self.store.entries[i].touched:
                nb = self.store.entries.get(nid)
                if nb is not None:
                    nb.support -= ds
                    nb.contra -= dc
        if ids:
            self.gate.forget(ids)
            kept = [c for c in self.calib_log if c[0] not in ids]
            self.calib_log = deque(kept, maxlen=self.calib_log.maxlen)
            self.calibrator = TemperatureCalibrator()
            every, self.calibrator.refit_every = self.calibrator.refit_every, len(kept) + 1
            for _, raw, label in kept:
                self.calibrator.add(raw, label)
            self.calibrator.refit_every = every
            if len(self.calibrator.samples) >= self.calibrator.min_samples:
                self.calibrator.fit()  # once, instead of every 50 samples
            self.calibrator._since = len(kept) % every
        if pend:
            self.pending = deque(p for p in self.pending if p.id not in pend)
        replayed = 0
        if cons:
            first = min(self.store.entries[i].log_pos for i in cons)
            pos, blob = max((c for c in self.checkpoints if c[0] <= first), key=lambda c: c[0])
            self.base = pickle.loads(blob)
            self.checkpoints = [c for c in self.checkpoints if c[0] <= pos]
            tail = [p for p in self.log[pos:] if p.id not in cons]
            self.log = self.log[:pos]
            for i in cons:
                self.store.remove(i)
            for p in tail:
                self._consolidate(p)
                replayed += 1
        for i in pend:
            self.store.remove(i)
        self._last = None
        return {"retracted": len(ids), "on_probation": len(pend), "consolidated": len(cons), "replayed": replayed}

    def summary(self) -> dict:
        out = self.base.summary()
        out["patches"] = {"entries": len(self.store), "on_probation": len(self.pending), "consolidated": len(self.log),
                          "checkpoints": [c[0] for c in self.checkpoints], "temperature": self.calibrator.t,
                          "gate": {c: round(self.gate.weights(c)[1], 3) for c in sorted(self.gate.logw)}}
        return out
