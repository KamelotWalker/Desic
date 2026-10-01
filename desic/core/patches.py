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
  nearest patch is, whether patches and base agree and which answer the patches
  propose (an answer not seen yet starts from the shared cell). Keying it by
  answer keeps a burst of wrong labels from teaching the gate to trust patches
  for every other answer too. A temperature on top keeps the combined answer
  calibrated.
* **Probation.** An entry reaches the base once ``probation`` further labels
  have arrived (10% of the labels seen, 20–500). Until then retracting it is
  exact and O(1): the base never saw it.
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
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from .calibration import TaskMetrics, TemperatureCalibrator, confidence_of
from .experts import Dist, cross_entropy, normalize, smooth
from .features import Features, state_hash
from .task import FAMILIAR, GROUND_TRUTH_SOURCES, SCORE, DecisionTask, SpecError


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
    t_ref: int = 0        # time support/contra were last decayed to
    touched: list = field(default_factory=list)  # (neighbour id, d_support, d_contra, time) this entry caused
    metric: tuple | None = None  # its contribution to the served-quality metrics (undone on retract)
    annotator: str | None = None  # who gave the label (K1: source trust)
    source_touched: list = field(default_factory=list)  # (annotator, d_agree, d_disagree, time) it caused

    def _fade(self, now: int, halflife: float | None) -> float:
        return 0.5 ** ((now - self.t_ref) / halflife) if halflife and now > self.t_ref else 1.0

    def trust(self, prior: float, now: int = 0, halflife: float | None = None) -> float:
        """(prior + support) / (prior + support + contra); with a half-life, old evidence fades."""
        f = self._fade(now, halflife)
        return (prior + self.support * f) / (prior + (self.support + self.contra) * f)

    def add_evidence(self, d_support: float, d_contra: float, now: int, halflife: float | None) -> None:
        f = self._fade(now, halflife)
        self.support = self.support * f + d_support
        self.contra = self.contra * f + d_contra
        self.t_ref = max(self.t_ref, now)


def _vec(x: Features) -> dict[str, float]:
    return {f: v for f, v in x.sparse.items() if f != "bias" and not f.startswith("n:")}


class SourceTrust:
    """How well each annotator agrees with *other* annotators on similar inputs (K1).

    A careless or malicious annotator starts contradicting everyone else at once; under
    concept drift every annotator contradicts the old labels alike. So reliability is
    taken *relative to the median annotator*: a burst from one source loses weight, a
    shift everyone makes together does not. Counts fade with a half-life (in labels) so a
    compromised account is caught from its recent behaviour.
    """

    def __init__(self, halflife: float = 300.0, prior: float = 10.0, power: float = 1.0) -> None:
        self.halflife = halflife
        self.prior = prior  # pseudo-observations at the median: a new annotator is trusted like the median
        self.power = power
        self.stats: dict[str, list[float]] = {}  # annotator -> [agree, disagree, t_ref]
        self._cache: tuple[int, float] | None = None

    def _fade(self, a: str, now: int) -> list[float]:
        st = self.stats.setdefault(a, [0.0, 0.0, float(now)])
        if now > st[2]:
            f = 0.5 ** ((now - st[2]) / self.halflife)
            st[0] *= f
            st[1] *= f
            st[2] = float(now)
        return st

    def add(self, a: str, agree: float, disagree: float, now: int) -> None:
        st = self._fade(a, now)
        st[0] += agree
        st[1] += disagree
        self._cache = None

    def remove(self, a: str, agree: float, disagree: float, at: int, now: int) -> None:
        st = self._fade(a, now)
        f = 0.5 ** ((now - at) / self.halflife)
        st[0] = max(0.0, st[0] - agree * f)
        st[1] = max(0.0, st[1] - disagree * f)
        self._cache = None

    def _rate(self, a: str, now: int) -> float | None:
        """Share of this annotator's cross-annotator comparisons that disagreed (faded)."""
        st = self._fade(a, now)
        n = st[0] + st[1]
        return st[1] / n if n > 0 else None

    def median(self, now: int) -> float:
        if self._cache is not None and self._cache[0] == now:
            return self._cache[1]
        rates = sorted(r for r in (self._rate(a, now) for a in list(self.stats)) if r is not None)
        med = rates[len(rates) // 2] if rates else 0.0
        self._cache = (now, med)
        return med

    def weight(self, a: str | None, now: int) -> float:
        """Multiplier in (0, 1]: (median disagreement rate / this annotator's) ** power, so an
        annotator who disagrees with the others three times as often as the median counts a
        third; an unknown annotator, or one at or below the median, counts fully."""
        if a is None or a not in self.stats:
            return 1.0
        med = max(self.median(now), 0.02)  # floor: near-perfect agreement must not make every slip fatal
        st = self._fade(a, now)
        dis = (st[1] + self.prior * med) / (st[0] + st[1] + self.prior)
        return min(med / dis, 1.0) ** self.power if dis > 0 else 1.0

    def report(self, now: int) -> dict:
        return {a: {"agreement": None if self._rate(a, now) is None else round(1 - self._rate(a, now), 3),
                    "weight": round(self.weight(a, now), 3)} for a in sorted(self.stats)}


class PatchStore:
    """Patch entries with an inverted index for nearest-neighbour lookups."""

    def __init__(self, capacity: int = 20000, k: int = 10, min_sim: float = 0.35, df_cap: int = 1000,
                 trust_prior: float = 1.0, trust_halflife: float | None = None, supersede_below: float = 0.0) -> None:
        self.capacity = capacity
        self.k = k
        self.min_sim = min_sim
        self.df_cap = df_cap
        self.trust_prior = trust_prior
        self.trust_halflife = trust_halflife  # in labels; None = support and contradiction never fade
        # an entry whose trust fell below this has been superseded by newer labels around it: it no longer votes
        self.supersede_below = supersede_below
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

    def trust(self, p: Patch, now: int) -> float:
        return p.trust(self.trust_prior, now, getattr(self, "trust_halflife", None))

    def vote(self, near: list[tuple[float, Patch]], options: list[str], now: int = 0,
             source_weight: Any = None) -> Dist | None:
        acc = {o: 0.0 for o in options}
        cut = getattr(self, "supersede_below", 0.0)
        for s, p in near:
            tr = self.trust(p, now)
            if tr < cut:
                continue
            w = s * s * p.weight * tr
            if source_weight is not None:
                w *= source_weight(getattr(p, "annotator", None))
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

    def __init__(self, share: float = 0.02, prior: float = 0.5) -> None:
        self.share = share
        self.prior = prior  # weight of the patches in a context never seen yet
        self.logw: dict[str, list[float]] = {}  # cell -> [log w_base, log w_patch]
        self.history: list[tuple[int, str, float, float, float]] = []  # (entry id, cell, loss base, loss patch, weight)

    def cell(self, strength: float, agree: bool) -> str:
        b = sum(strength >= t for t in self.BINS)
        return f"{b}{'=' if agree else '≠'}"

    def weights(self, cell: str) -> tuple[float, float]:
        # a per-answer cell ("3≠:refund") not seen yet starts from its shared cell ("3≠")
        prior = getattr(self, "prior", 0.5)
        lb, lp = self.logw.get(cell) or self.logw.get(cell.split(":", 1)[0]) or [math.log(1 - prior), math.log(prior)]
        m = max(lb, lp)
        eb, ep = math.exp(lb - m), math.exp(lp - m)
        return eb / (eb + ep), ep / (eb + ep)

    def update(self, pid: int, cell: str, loss_b: float, loss_p: float, weight: float) -> None:
        self.history.append((pid, cell, loss_b, loss_p, weight))
        self._apply(cell, loss_b, loss_p, weight)

    def _apply(self, cell: str, loss_b: float, loss_p: float, weight: float) -> None:
        if ":" in cell:  # the shared cell keeps learning too: it is the prior for answers not seen yet
            self._apply_one(cell.split(":", 1)[0], loss_b, loss_p, weight)
        self._apply_one(cell, loss_b, loss_p, weight)

    def _apply_one(self, cell: str, loss_b: float, loss_p: float, weight: float) -> None:
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

    def __init__(self, base: DecisionTask, probation: int = 500, probation_share: float = 0.1, min_probation: int = 20,
                 replay: int = 1, checkpoint_every: int = 1000,
                 keep_checkpoints: int = 4, trust_sim: float = 0.5, gate_by_answer: bool = True,
                 gate_sources: tuple[str, ...] = ("human", "dataset", "teacher"), gate_prior: float = 0.5,
                 retire_below: float = 0.0, rehearse_min_support: float = 0.0, rehearse_if_base_agrees: float = 0.5,
                 consolidate_damping: float = 0.0, source_trust: bool = False, source_halflife: float = 300.0,
                 source_power: float = 1.0, seed: int = 0, **store: Any) -> None:
        self.base = base
        # Probation is ``probation_share`` of the labels seen, between ``min_probation`` and
        # ``probation``: a small question still trains its experts early (undoing an old label
        # replays only a few hundred events anyway), a large one keeps a long reversible window.
        self.max_probation = probation
        self.probation_share = probation_share
        self.min_probation = min_probation
        self.replay = replay
        self.checkpoint_every = checkpoint_every
        self.keep_checkpoints = keep_checkpoints
        self.trust_sim = trust_sim  # a later label counts for/against a patch only when at least this similar
        self.gate_by_answer = gate_by_answer  # one gate per answer the patches propose (contains a burst to its label)
        self.store = PatchStore(**store)
        self.gate = Gate(prior=gate_prior)
        self.gate_sources = tuple(gate_sources)  # whose labels teach the gate which forecaster to trust
        self.retire_below = retire_below  # entries trusted less than this are no longer rehearsed
        self.rehearse_min_support = rehearse_min_support  # rehearse only entries later labels have confirmed this much
        # rehearse only entries the base itself gives at least this probability (a lie or an outdated
        # meaning contradicts the rest of what the base knows, so it is not drilled in again)
        self.rehearse_if_base_agrees = rehearse_if_base_agrees
        # an entry the base finds less plausible than this is consolidated with proportionally less
        # weight, so a burst of wrong labels spreads less when it leaves probation (0 = off)
        self.consolidate_damping = consolidate_damping
        # K1: weigh labels by how well their annotator agrees with the other annotators
        self.sources = SourceTrust(source_halflife, power=source_power) if source_trust else None
        self.calibrator = TemperatureCalibrator()
        self.calib_log: deque = deque(maxlen=self.calibrator.samples.maxlen)  # (entry id, raw, label)
        # What users see is the patched task: it takes over the served-quality metrics, the
        # label counts and the version (a task wrapped after training keeps its history); the
        # base keeps private ones for what it learns at consolidation.
        self.metrics, base.metrics = base.metrics, TaskMetrics()
        self.labels_by_source = dict(base.labels_by_source)
        self.version = base.version
        self.pending: deque[Patch] = deque()
        self.log: list[Patch] = []  # consolidated entries, in consolidation order
        self.checkpoints: list[tuple[int, bytes]] = [(0, pickle.dumps(base))]  # the base as it was wrapped
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

    @property
    def rules(self) -> list[dict]:
        return self.base.rules

    @rules.setter
    def rules(self, value: list[dict]) -> None:
        self.base.rules = value

    @property
    def labels(self) -> int:
        return sum(self.labels_by_source.values())

    def __getattr__(self, name: str) -> Any:
        # everything else (attach_neural, mixture, featurizer, created_at, …) is the base's
        if name.startswith("__") or "base" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.base, name)

    def __getstate__(self) -> dict:
        # Checkpoints only speed up retracting consolidated labels; saving all of them would make
        # a 10k-label student ~5× larger on disk (243 vs 52 MB). Only the base as it was wrapped is
        # kept, so after a restart such a retraction replays from there until new checkpoints form.
        state = dict(self.__dict__)
        state["checkpoints"] = self.checkpoints[:1]
        state["_last"] = None
        return state

    def _source_weight(self, annotator: str | None) -> float:
        return self.sources.weight(annotator, self.t) if getattr(self, "sources", None) is not None else 1.0

    @property
    def probation(self) -> int:
        """How many of the latest labels are still on probation (reversible in O(1))."""
        return max(self.min_probation, min(self.max_probation, int(self.probation_share * self.t)))

    def public(self, internal: dict, abstain_threshold: float | None = None) -> dict:
        # the risk budget is checked against what users were served: this layer's metrics
        return self.base.public(internal, abstain_threshold, metrics=self.metrics)

    # ---------------------------------------------------------------- answer
    def _forecasts(self, state: Any, options: list[str] | None = None) -> dict:
        a = self.base.answer(state, options)
        opts = a["options"]
        pb = self.base.calibrator.apply(a["raw"])  # calibrated, before the familiarity shrink
        near = self.store.query(_vec(a["feats"]), a["key"])
        pq = self.store.vote(near, opts, self.t, self._source_weight if getattr(self, "sources", None) else None) if near else None
        f = {"a": a, "pb": pb, "pq": pq, "near": near, "cell": None, "strength": 0.0, "raw": pb, "wp": 0.0}
        if pq is not None:
            f["strength"] = near[0][0]
            top = confidence_of(pq)[0]
            f["cell"] = self.gate.cell(f["strength"], confidence_of(pb)[0] == top)
            if getattr(self, "gate_by_answer", False):
                f["cell"] += ":" + top
            wb, wp = self.gate.weights(f["cell"])
            f["raw"], f["wp"] = {o: wb * pb.get(o, 0.0) + wp * pq.get(o, 0.0) for o in opts}, wp
        return f

    def _serve(self, f: dict) -> tuple[Dist, float, bool]:
        """Calibrate the combined forecast and shrink it toward "don't know" when the input is unfamiliar."""
        a, opts = f["a"], f["a"]["options"]
        probs = self.calibrator.apply(f["raw"])
        familiarity = a["familiarity"]
        if f["pq"] is not None:  # a close labelled neighbour makes the input familiar before the base learns it
            familiarity = max(familiarity, min(1.0, f["strength"] / 0.5 * FAMILIAR))
        unfamiliar = familiarity < FAMILIAR
        if unfamiliar:
            lam = familiarity / FAMILIAR
            probs = {o: lam * p + (1 - lam) / len(opts) for o, p in probs.items()}
        return probs, familiarity, unfamiliar

    def answer(self, state: Any, options: list[str] | None = None) -> dict:
        f = self._forecasts(state, options)
        a = f["a"]
        self._last = (a["key"], self.t, f)
        probs, familiarity, unfamiliar = self._serve(f)
        weights = {k: v * (1 - f["wp"]) for k, v in a["weights"].items()}
        if f["pq"] is not None:
            weights["patch"] = f["wp"]
        return {**a, "raw": f["raw"], "probabilities": probs, "weights": weights, "familiarity": familiarity,
                "unfamiliar": unfamiliar, "patch_dist": f["pq"],
                "patch": {"cell": f["cell"], "strength": round(f["strength"], 4), "weight": round(f["wp"], 4)}
                if f["pq"] is not None else None}

    def explain(self, internal: dict) -> dict:
        out = self.base.explain(internal)
        out["familiarity"] = round(internal.get("familiarity", 1.0), 3)
        out["temperature"] = self.calibrator.t
        pq = internal.get("patch_dist")
        if pq is not None:
            top, tp = confidence_of(pq)
            out["experts"]["patch"] = {"awake": True, "weight": round(internal["weights"].get("patch", 0.0), 4),
                                       "answer": top, "p": round(tp, 4)}
            near = self.store.query(_vec(internal["feats"]), internal["key"])
            out["patches"] = [{"similarity": round(s, 3), "answer": p.label, "source": p.source, "ref": p.ref,
                               "trust": round(self.store.trust(p, self.t), 3), "on_probation": not p.consolidated}
                              for s, p in near[:5]]
        return out

    # ----------------------------------------------------------------- learn
    def learn(self, state: Any, target: str | Dist, source: str = "human", weight: float | None = None,
              served: Dist | None = None, served_raw: Dist | None = None, abstained: bool = False,
              ref: str | None = None, served_action: tuple[str, str] | None = None,
              annotator: str | None = None) -> list[dict]:
        """``served_action`` = (who answered: student/teacher/rule, the answer the user got),
        so the risk of what was actually served is tracked too (and undone with the label).
        ``annotator`` = who gave the label (a user id), for source trust."""
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
        p.annotator = annotator
        self.next_id += 1
        # prequential: which forecaster would have been right, and was the combined answer calibrated?
        if f["pq"] is not None and source in getattr(self, "gate_sources", (source,)):
            self.gate.update(p.id, f["cell"], cross_entropy(dist, f["pb"]), cross_entropy(dist, f["pq"]), weight)
        if source in GROUND_TRUTH_SOURCES and len(dist) == 1:
            if served is None:  # test-then-train: score what answer() would have served
                served = self._serve(f)[0]
            self.metrics.update(served, p.label, abstained, spec.option_names if spec.type == SCORE else None, ref=p.id)
            p.metric = self.metrics.last_contribution
            if served_action is not None:
                self.metrics.record_served(served_action[0], served_action[1] == p.label, ref=p.id)
            self.calibrator.add(f["raw"], p.label)
            self.calib_log.append((p.id, f["raw"], p.label))
        self.labels_by_source[source] = self.labels_by_source.get(source, 0) + 1
        self.version += 1
        # this label supports or contradicts the close patches it lands next to
        for s, nb in f["near"]:
            if s < self.trust_sim or nb.id == p.id:
                continue
            d = s * s * weight * max(dist.values())
            ds, dc = (d, 0.0) if nb.label == p.label else (0.0, d)
            nb.add_evidence(ds, dc, self.t, getattr(self.store, "trust_halflife", None))
            p.touched.append((nb.id, ds, dc, self.t))
            other = getattr(nb, "annotator", None)
            if getattr(self, "sources", None) is not None and annotator is not None and other is not None and other != annotator:
                agree, dis = (s * s, 0.0) if nb.label == p.label else (0.0, s * s)
                for who in (annotator, other):  # agreement is mutual
                    self.sources.add(who, agree, dis, self.t)
                    p.source_touched.append((who, agree, dis, self.t))
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
        weight, damp = p.weight, getattr(self, "consolidate_damping", 0.0)
        if getattr(self, "sources", None) is not None:
            weight *= self._source_weight(getattr(p, "annotator", None))
        if damp:
            plausible = self.base.answer(p.state)["raw"].get(p.label, 0.0)
            if plausible < damp:
                weight *= max(plausible / damp, 0.1)
        events = self.base.learn(p.state, p.target, source=p.source, weight=weight, ref=p.ref)
        self._rehearse(exclude=p.id)
        p.consolidated = True
        p.log_pos = len(self.log)
        self.log.append(p)
        if len(self.log) % self.checkpoint_every == 0:
            self.checkpoints.append((len(self.log), pickle.dumps(self.base)))
            if len(self.checkpoints) > self.keep_checkpoints + 1:  # the base as wrapped is always kept
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
                tr = self.store.trust(p, self.t)
                confirmed = p.support >= getattr(self, "rehearse_min_support", 0.0)
                if (p.consolidated and pid != exclude and confirmed and tr >= getattr(self, "retire_below", 0.0)
                        and self.rng.random() < tr * self._source_weight(getattr(p, "annotator", None))):
                    agree = getattr(self, "rehearse_if_base_agrees", 0.0)
                    if agree and self.base.answer(p.state)["raw"].get(p.label, 0.0) < agree:
                        break  # this rehearsal slot is skipped, not handed to another entry
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
        t0 = time.perf_counter()
        ids = {self.refs.pop(r) for r in refs if r in self.refs}
        ids &= set(self.store.entries)
        pend = {i for i in ids if not self.store.entries[i].consolidated}
        cons = ids - pend
        for i in ids:
            src = self.store.entries[i].source
            if self.labels_by_source.get(src):
                self.labels_by_source[src] -= 1
            if getattr(self, "sources", None) is not None:
                for who, ag, di, at in getattr(self.store.entries[i], "source_touched", ()):
                    self.sources.remove(who, ag, di, at, self.t)
            for nid, ds, dc, at in self.store.entries[i].touched:
                nb = self.store.entries.get(nid)
                if nb is not None:  # remove exactly what is left of that evidence today
                    hl = getattr(self.store, "trust_halflife", None)
                    nb.add_evidence(0.0, 0.0, self.t, hl)
                    f = 0.5 ** ((self.t - at) / hl) if hl else 1.0
                    nb.support -= ds * f
                    nb.contra -= dc * f
        if ids:
            # the served-quality metrics feed the risk budget: a retracted label must not count there either
            self.metrics.forget([self.store.entries[i].metric for i in ids if getattr(self.store.entries[i], "metric", None)])
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
        if ids:
            self.version += 1
        return {"retracted": len(ids), "on_probation": len(pend), "consolidated": len(cons), "replayed": replayed,
                "seconds": round(time.perf_counter() - t0, 4)}

    def summary(self) -> dict:
        out = self.base.summary()
        out.update(labels=self.labels, labels_by_source=self.labels_by_source, version=self.version)
        if getattr(self, "sources", None) is not None:
            out["annotators"] = self.sources.report(self.t)
        out["patches"] = {"probation": self.probation, "entries": len(self.store), "on_probation": len(self.pending), "consolidated": len(self.log),
                          "checkpoints": [c[0] for c in self.checkpoints], "temperature": self.calibrator.t,
                          "gate": {c: round(self.gate.weights(c)[1], 3) for c in sorted(self.gate.logw)}}
        return out
