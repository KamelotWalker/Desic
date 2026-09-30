"""Typed decision tasks: the System One contract plus a self-learning student.

A *question* is Jev-style: a name, a type and the allowed answers.

    choice – pick one of the options            -> choice + probabilities
    score  – ordered levels (low < mid < high)  -> expected score + probabilities
    noul   – is a yes/no proposition true?      -> P(true)

Every distinct question name is a task with its own student. The student can
only ever answer with one of the allowed options (no free text, nothing to
hallucinate); what it *can* get wrong is the probability, which is why each
answer carries a calibrated confidence and an ``abstain`` flag.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .calibration import TaskMetrics, TemperatureCalibrator, confidence_of
from .drift import DDM, DRIFT
from .experts import Dist, LinearExpert, MemoryExpert, Mixture, PriorExpert, TreeExpert, normalize
from .features import Featurizer, Features, state_hash

CHOICE, SCORE, NOUL = "choice", "score", "noul"
TYPES = (CHOICE, SCORE, NOUL)
NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
TRUE_WORDS = {"true", "yes", "1", "evet", "doğru", "dogru", "y", "t"}
FALSE_WORDS = {"false", "no", "0", "hayır", "hayir", "yanlış", "yanlis", "n", "f"}
GROUND_TRUTH_SOURCES = ("human", "dataset")
FAMILIAR = 0.5  # below this share of known evidence the answer is pulled toward "don't know"

DEFAULT_SETTINGS = {
    "abstain_threshold": 0.6,   # abstain when calibrated confidence is below this
    "teacher_mode": "on_abstain",  # off | on_abstain | always
    "teacher_weight": 0.5,      # how much a teacher label counts vs a human label (1.0)
}


class SpecError(ValueError):
    pass


@dataclass
class QuestionSpec:
    name: str
    type: str
    instructions: str = ""
    options: dict[str, str] = field(default_factory=dict)  # ordered; score = levels low→high

    @property
    def option_names(self) -> list[str]:
        return list(self.options)

    @classmethod
    def parse(cls, name: str, raw: dict) -> "QuestionSpec":
        if not NAME_RE.match(name or ""):
            raise SpecError(f"question name {name!r} must be 1-64 chars of letters, digits, _ . -")
        qtype = str(raw.get("type", CHOICE)).lower()
        if qtype not in TYPES:
            raise SpecError(f"question {name!r}: type must be one of {', '.join(TYPES)}")
        crit = raw.get("criteria", raw.get("options", raw.get("levels")))
        options: dict[str, str] = {}
        if isinstance(crit, dict):
            options = {str(k).strip(): "" if v is None else str(v) for k, v in crit.items() if str(k).strip()}
        elif isinstance(crit, list):
            for item in crit:
                if isinstance(item, dict):
                    key = str(item.get("name") or item.get("value") or item.get("label") or "").strip()
                    if key:
                        options[key] = str(item.get("description", ""))
                elif str(item).strip():
                    options[str(item).strip()] = ""
        if qtype == NOUL:
            options = {"true": options.get("true", ""), "false": options.get("false", "")}
        elif len(options) < 2:
            raise SpecError(f"question {name!r}: a {qtype} question needs at least two options in 'criteria'")
        if len(options) > 255:
            raise SpecError(f"question {name!r}: at most 255 options")
        return cls(name, qtype, str(raw.get("instructions", "")), options)

    def merge(self, other: "QuestionSpec") -> bool:
        """Adopt new options / descriptions from a request. Returns True if changed."""
        if other.type != self.type:
            raise SpecError(f"question {self.name!r} is a {self.type} question, not {other.type}")
        changed = False
        if other.instructions and other.instructions != self.instructions:
            self.instructions, changed = other.instructions, True
        for k, v in other.options.items():
            if k not in self.options:
                if self.type == SCORE:
                    raise SpecError(f"score question {self.name!r}: levels are fixed ({', '.join(self.options)})")
                self.options[k], changed = v, True
            elif v and v != self.options[k]:
                self.options[k], changed = v, True
        return changed

    def label_of(self, value: Any) -> str:
        """Normalise a feedback value to an option name (may be a *new* choice option)."""
        if self.type == NOUL:
            s = str(value).strip().lower()
            if isinstance(value, bool) or s in TRUE_WORDS | FALSE_WORDS:
                return "true" if (value is True or s in TRUE_WORDS) else "false"
            raise SpecError(f"{self.name}: a noul answer must be true or false, got {value!r}")
        s = str(value).strip()
        if s in self.options:
            return s
        low = {k.lower(): k for k in self.options}
        if s.lower() in low:
            return low[s.lower()]
        if self.type == SCORE:
            try:
                i = int(float(s))
                if 0 <= i < len(self.options):
                    return self.option_names[i]
            except ValueError:
                pass
            raise SpecError(f"{self.name}: unknown level {value!r}; use one of {', '.join(self.options)} or 0..{len(self.options) - 1}")
        if not s:
            raise SpecError(f"{self.name}: empty answer")
        return s  # open world: a new choice option learned from feedback

    def to_dict(self) -> dict:
        return asdict(self)


class DecisionTask:
    EXPERTS = ("prior", "linear", "tree", "memory")

    def __init__(self, spec: QuestionSpec, settings: dict | None = None) -> None:
        self.spec = spec
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        self.featurizer = Featurizer()
        self.prior = PriorExpert()
        self.linear = LinearExpert()
        self.tree = TreeExpert()
        self.memory = MemoryExpert()
        self.mixture = Mixture(list(self.EXPERTS))
        self.calibrator = TemperatureCalibrator()
        self.metrics = TaskMetrics()
        self.drift = DDM()
        self.events: list[dict] = []
        self.labels_by_source: dict[str, int] = {}
        self.awake_counts: dict[str, int] = {}
        self.version = 0
        self.rules: list[dict] = []
        self.created_at = time.time()
        self._neural: Callable[..., tuple[Dist, float] | None] | None = None  # runtime hook, never pickled

    # The neural student is a process-level resource: it is attached after
    # loading and must not be pickled together with the task.
    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_neural"] = None
        state["_neural_pretrained"] = False
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("_neural", None)

    def attach_neural(self, fn: Callable[..., tuple[Dist, float] | None] | None, share: float = 0.3,
                      pretrained: bool = False) -> None:
        """Plug in (or with ``None`` unplug) the neural expert; a new expert starts with ``share`` of the weight.

        ``pretrained`` says whether its encoder brings language knowledge of its own
        (ModernBERT, mmBERT …) rather than only what the labels taught it.
        """
        self._neural = fn
        self._neural_pretrained = pretrained
        if fn is not None and "neural" not in self.mixture.logw:
            self.mixture.add("neural", share)

    @property
    def experts(self) -> dict:
        return {"prior": self.prior, "linear": self.linear, "tree": self.tree, "memory": self.memory}

    @property
    def labels(self) -> int:
        return sum(self.labels_by_source.values())

    def _predict_experts(self, feats: Features, key: str, options: list[str], state: Any = None) -> dict[str, Dist | None]:
        preds: dict[str, Dist | None] = {
            "prior": self.prior.predict(feats, options),
            "linear": self.linear.predict(feats, options),
            "tree": self.tree.predict(feats, options),
            "memory": self.memory.predict(feats, options, key),
        }
        self._last_act = None
        if self._neural is not None and state is not None:
            try:
                res = self._neural(self.spec, state, options)
            except Exception:  # the neural student must never break a decision
                res = None
            if res is not None:
                dist, act = res
                preds["neural"] = {o: 0.99 * dist.get(o, 0.0) + 0.01 / len(options) for o in options}
                self._last_act = act
            else:
                preds["neural"] = None
        return preds

    # ----------------------------------------------------------------- answer
    def answer(self, state: Any, options: list[str] | None = None) -> dict:
        opts = [o for o in (options or self.spec.option_names) if o in self.spec.options] or self.spec.option_names
        feats = self.featurizer.extract(state)
        key = state_hash(state)
        preds = self._predict_experts(feats, key, opts, state)
        act = self._last_act
        raw, weights = self.mixture.combine(preds, opts)
        probs = self.calibrator.apply(raw)
        familiarity = self.familiarity(feats) if preds["memory"] is None or key not in self.memory.exact else 1.0
        # A *pretrained* neural encoder understands words the online experts never saw, so
        # it lifts the familiarity shrink; a scratch encoder only knows what the labels taught it.
        if familiarity < FAMILIAR and not (preds.get("neural") is not None and getattr(self, "_neural_pretrained", False)):
            lam = familiarity / FAMILIAR
            probs = {o: lam * p + (1 - lam) / len(opts) for o, p in probs.items()}
        return {"raw": raw, "probabilities": probs, "weights": weights, "preds": preds, "feats": feats, "key": key,
                "options": opts, "familiarity": familiarity, "act": act}

    def familiarity(self, feats: Features) -> float:
        """Share of the state's evidence (words, categories) the student has seen in labelled data."""
        total = known = 0.0
        vocab = self.linear.w.values()
        for f, v in feats.sparse.items():
            if f == "bias" or f.startswith("n:"):
                continue
            total += v * v
            if any(f in w for w in vocab):
                known += v * v
        return 1.0 if total == 0 else known / total

    def public(self, internal: dict, abstain_threshold: float | None = None) -> dict:
        probs = internal["probabilities"]
        best, conf = confidence_of(probs)
        threshold = self.settings["abstain_threshold"] if abstain_threshold is None else abstain_threshold
        knows = bool(internal["weights"])
        out: dict[str, Any] = {"type": self.spec.type}
        rounded = {o: round(p, 4) for o, p in sorted(probs.items(), key=lambda kv: -kv[1])}
        if self.spec.type == CHOICE:
            out.update(choice=best, probabilities=rounded)
        elif self.spec.type == SCORE:
            levels = self.spec.option_names
            out.update(score=round(sum(i * probs.get(lv, 0.0) for i, lv in enumerate(levels)), 4),
                       level=best, probabilities={lv: round(probs.get(lv, 0.0), 4) for lv in levels})
        else:
            p = probs.get("true", 0.5)
            out.update(probability=round(p, 4), answer=p >= 0.5)
        out["confidence"] = round(conf, 4)
        out["abstain"] = (not knows) or conf < threshold
        return out

    def explain(self, internal: dict) -> dict:
        probs = internal["probabilities"]
        best, _ = confidence_of(probs)
        feats, opts, key = internal["feats"], internal["options"], internal["key"]
        experts = {}
        for name, p in internal["preds"].items():
            if p is None:
                experts[name] = {"awake": False}
                continue
            top, tp = confidence_of(p)
            experts[name] = {"awake": True, "weight": round(internal["weights"].get(name, 0.0), 4), "answer": top, "p": round(tp, 4)}
        out: dict[str, Any] = {"experts": experts, "temperature": self.calibrator.t,
                               "familiarity": round(internal.get("familiarity", 1.0), 3)}
        if internal["preds"]["linear"] is not None:
            out["linear"] = self.linear.explain(feats, opts, best)
        if internal["preds"]["tree"] is not None:
            out["tree"] = self.tree.explain(feats, opts, best)
        if internal["preds"]["memory"] is not None:
            out["memory"] = self.memory.explain(feats, opts, best, key)
        if internal["preds"]["prior"] is not None:
            out["prior"] = self.prior.explain(feats, opts, best)
        if internal["preds"].get("neural") is not None:
            out["neural"] = {"act": round(internal["act"], 3) if internal.get("act") is not None else None}
        return out

    # ------------------------------------------------------------------ learn
    def learn(self, state: Any, target: str | Dist, source: str = "human", weight: float | None = None,
              served: Dist | None = None, served_raw: Dist | None = None, abstained: bool = False,
              ref: str | None = None) -> list[dict]:
        """Learn one example. ``served``/``served_raw`` are the (calibrated / raw)
        probabilities the user actually saw; when omitted the task predicts
        first (test-then-train) so the metrics stay prequential."""
        if not isinstance(target, dict):
            target = self.spec.label_of(target)
            if target not in self.spec.options:
                if self.spec.type != CHOICE:
                    raise SpecError(f"{target!r} is not an answer of {self.spec.name}")
                self.spec.options[target] = ""
            dist: Dist = {target: 1.0}
        else:
            dist = normalize({k: v for k, v in target.items() if k in self.spec.options and v > 0})
            if not dist:
                return []
        if weight is None:
            weight = self.settings["teacher_weight"] if source == "teacher" else 1.0
        opts = self.spec.option_names
        feats = self.featurizer.extract(state)
        key = state_hash(state)
        preds = self._predict_experts(feats, key, opts, state)
        events: list[dict] = []

        if source in GROUND_TRUTH_SOURCES and len(dist) == 1:
            label = next(iter(dist))
            if served is None:
                raw, _ = self.mixture.combine(preds, opts)
                served_raw, served = raw, self.calibrator.apply(raw)
            ok = self.metrics.update(served, label, abstained, opts if self.spec.type == SCORE else None)
            if served_raw is not None:
                self.calibrator.add(served_raw, label)
            if self.drift.update(not ok) == DRIFT:
                events.append({"type": "drift", "at": self.metrics.n, "time": time.time()})

        self.mixture.update(preds, dist)
        for name, p in preds.items():
            if p is not None:
                self.awake_counts[name] = self.awake_counts.get(name, 0) + 1
        self.prior.learn(feats, dist, weight)
        self.linear.learn(feats, dist, weight, opts)
        tree_event = self.tree.learn(feats, dist, weight)
        if tree_event == "drift":
            events.append({"type": "tree_replaced", "at": self.metrics.n, "time": time.time()})
        self.memory.learn(feats, dist, weight, key, ref)
        self.featurizer.observe(feats)
        self.labels_by_source[source] = self.labels_by_source.get(source, 0) + 1
        self.version += 1
        if events:
            self.events = (self.events + events)[-100:]
        return events

    # ---------------------------------------------------------------- summary
    def summary(self) -> dict:
        return {
            "name": self.spec.name,
            "type": self.spec.type,
            "instructions": self.spec.instructions,
            "options": self.spec.options,
            "settings": self.settings,
            "labels": self.labels,
            "labels_by_source": self.labels_by_source,
            "version": self.version,
            "temperature": self.calibrator.t,
            "expert_weights": {k: round(v, 4) for k, v in self.mixture.weights().items()},
            "expert_usage": {k: round(getattr(self, "awake_counts", {}).get(k, 0) / max(self.labels, 1), 4)
                             for k in self.mixture.logw},
            "neural_attached": self._neural is not None,
            "tree": self.tree.model.tree.stats(),
            "vocabulary": len({f for w in self.linear.w.values() for f in w}),
            "memory_size": len(self.memory),
            "created_at": self.created_at,
        }
