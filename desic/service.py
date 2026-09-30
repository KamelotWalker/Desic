"""Desic application service: typed decisions, feedback, teacher, data and jobs.

The HTTP layer (api.py) is a thin wrapper, so the engine can be embedded:

    desic = Desic("./data")
    r = await desic.decide("I was charged twice", {
        "department": {"type": "choice", "criteria": {"billing": "", "technical": "", "sales": ""}},
    })
    desic.feedback(r["id"], {"department": "billing"})
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core.features import state_hash as state_hash_of
from .core.features import state_text
from .core.rules import first_match, validate_rule
from .core.schema import to_label, to_number
from .core.task import CHOICE, NOUL, SCORE, DecisionTask, QuestionSpec, SpecError
from .generation import design_task, generate_examples, normalize_design
from .llm import LLMError, ProviderConfig, make_provider, teacher_answer
from .neural.runtime import NeuralError, NeuralManager
from .neural.text import question_text, state_to_text
from .storage import Storage

SAVE_EVERY = 20          # persist a task after this many changes
SNAPSHOT_EVERY = 250     # automatic snapshot every N labels
MAX_STATE_BYTES = 64_000


class NotFound(KeyError):
    pass


class Conflict(ValueError):
    pass


class EventBus:
    """Fan-out of JSON events to every connected dashboard (WebSocket)."""

    def __init__(self) -> None:
        self.subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)

    def publish(self, event: dict) -> None:
        event.setdefault("time", time.time())
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # slow client: drop rather than block learning


@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("id", "kind", "status", "progress", "message", "result", "error", "created_at")}


def teacher_from_env() -> ProviderConfig | None:
    provider = os.environ.get("DESIC_TEACHER_PROVIDER", "")
    key = os.environ.get("DESIC_TEACHER_API_KEY", "")
    if not provider and os.environ.get("ANTHROPIC_API_KEY"):
        provider = "anthropic"
    if not provider:
        return None
    return ProviderConfig(provider, key, os.environ.get("DESIC_TEACHER_MODEL", ""), os.environ.get("DESIC_TEACHER_BASE_URL", ""))


class Desic:
    def __init__(self, data_dir: str | Path = "data", db_name: str = "desic.db", teacher: ProviderConfig | None = None) -> None:
        path = ":memory:" if str(data_dir) == ":memory:" else Path(data_dir) / db_name
        self.storage = Storage(path)
        self.bus = EventBus()
        self.jobs: dict[str, Job] = {}
        self.tasks: dict[str, DecisionTask] = {t.spec.name: t for t in self.storage.load_tasks()}
        self.locks: dict[str, threading.RLock] = {}
        self.dirty: dict[str, int] = {}
        self.teacher_cfg: ProviderConfig | None = teacher if teacher is not None else teacher_from_env()
        self.teacher_stats = {"calls": 0, "errors": 0, "last_error": None, "last_call": None}
        self._provider = None
        self._background: set[asyncio.Task] = set()
        self.neural = NeuralManager(None if str(data_dir) == ":memory:" else Path(data_dir), self.storage)
        self.attach_neural()

    # ------------------------------------------------------------ neural hook
    def _neural_predict(self, spec: QuestionSpec, state: Any, options: list[str]):
        return self.neural.predict(spec, state, options)

    def attach_neural(self, reset: bool = False, task: DecisionTask | None = None) -> None:
        """Plug the active neural checkpoint into every student (or unplug it)."""
        fn = self._neural_predict if self.neural.active_id else None
        pretrained = self.neural.active_pretrained
        for t in [task] if task is not None else list(self.tasks.values()):
            with self.lock(t.spec.name):
                if reset:
                    t.mixture.logw.pop("neural", None)
                t.attach_neural(fn, pretrained=pretrained)

    # ------------------------------------------------------------- utilities
    def lock(self, name: str) -> threading.RLock:
        return self.locks.setdefault(name, threading.RLock())

    def get(self, name: str) -> DecisionTask:
        task = self.tasks.get(name)
        if task is None:
            raise NotFound(f"question {name!r} not found")
        return task

    def persist(self, task: DecisionTask) -> None:
        with self.lock(task.spec.name):
            self.storage.save_task(task.spec.name, task.spec.to_dict(), task)
            self.dirty[task.spec.name] = 0

    def persist_all(self) -> None:
        for name, n in list(self.dirty.items()):
            if n and name in self.tasks:
                self.persist(self.tasks[name])

    def _touch(self, task: DecisionTask, n: int = 1) -> None:
        name = task.spec.name
        before = task.labels - n
        self.dirty[name] = self.dirty.get(name, 0) + n
        if before // SNAPSHOT_EVERY != task.labels // SNAPSHOT_EVERY:
            self.snapshot(name, note="automatic")
        if self.dirty[name] >= SAVE_EVERY:
            self.persist(task)

    @staticmethod
    def check_state(state: Any) -> Any:
        if state is None or (isinstance(state, str) and not state.strip()):
            raise ValueError("state is required (text or a JSON object)")
        if not isinstance(state, (str, dict, list)):
            state = str(state)
        if len(json.dumps(state, ensure_ascii=False)) > MAX_STATE_BYTES:
            raise ValueError(f"state is larger than {MAX_STATE_BYTES // 1000} kB; send only what the decision needs")
        return state

    # ------------------------------------------------------------------ tasks
    def create_task(self, spec: QuestionSpec, settings: dict | None = None) -> DecisionTask:
        if spec.name in self.tasks:
            raise Conflict(f"question {spec.name!r} already exists")
        task = DecisionTask(spec, settings)
        self.tasks[spec.name] = task
        self.attach_neural(task=task)
        self.persist(task)
        self.bus.publish({"type": "task_created", "task": spec.name})
        return task

    def ensure_task(self, name: str, raw: dict | None) -> tuple[DecisionTask, list[str] | None]:
        """Resolve a question from a request, registering or extending it on the fly."""
        if not raw:
            return self.get(name), None
        spec = QuestionSpec.parse(name, raw)
        task = self.tasks.get(name)
        if task is None:
            return self.create_task(spec), spec.option_names
        with self.lock(name):
            if task.spec.merge(spec):
                self.dirty[name] = self.dirty.get(name, 0) + SAVE_EVERY  # persist the new spec soon
        return task, spec.option_names

    def update_task(self, name: str, patch: dict) -> dict:
        task = self.get(name)
        with self.lock(name):
            if "instructions" in patch:
                task.spec.instructions = str(patch["instructions"] or "")
            if isinstance(patch.get("add_options"), list):
                if task.spec.type != CHOICE:
                    raise SpecError("only choice questions can gain options")
                for o in patch["add_options"]:
                    if str(o).strip():
                        task.spec.options.setdefault(str(o).strip(), "")
            if isinstance(patch.get("descriptions"), dict):
                for k, v in patch["descriptions"].items():
                    if k in task.spec.options:
                        task.spec.options[k] = str(v or "")
            settings = patch.get("settings") or {}
            if "abstain_threshold" in settings:
                t = float(settings["abstain_threshold"])
                if not 0 <= t <= 1:
                    raise ValueError("abstain_threshold must be between 0 and 1")
                task.settings["abstain_threshold"] = t
            if "teacher_mode" in settings:
                if settings["teacher_mode"] not in ("off", "on_abstain", "always"):
                    raise ValueError("teacher_mode must be off, on_abstain or always")
                task.settings["teacher_mode"] = settings["teacher_mode"]
            if "teacher_weight" in settings:
                w = float(settings["teacher_weight"])
                if not 0 <= w <= 1:
                    raise ValueError("teacher_weight must be between 0 and 1")
                task.settings["teacher_weight"] = w
        self.persist(task)
        self.bus.publish({"type": "task_updated", "task": name})
        return self.task_detail(name)

    def delete_task(self, name: str) -> None:
        self.get(name)
        del self.tasks[name]
        self.storage.delete_task(name)
        self.bus.publish({"type": "task_deleted", "task": name})

    def reset_task(self, name: str) -> None:
        old = self.get(name)
        with self.lock(name):
            fresh = DecisionTask(copy.deepcopy(old.spec), dict(old.settings))
            fresh.rules = old.rules
            self.tasks[name] = fresh
        self.attach_neural(task=fresh)
        self.persist(fresh)
        self.bus.publish({"type": "task_reset", "task": name})

    def task_card(self, task: DecisionTask) -> dict:
        m = task.metrics.summary()
        return {
            "name": task.spec.name, "type": task.spec.type, "instructions": task.spec.instructions,
            "options": task.spec.option_names, "labels": task.labels, "accuracy": m["accuracy"], "ece": m["ece"],
            "abstain_rate": m["abstain_rate"], "teacher_rate": m["teacher_rate"], "decisions": m["decisions"],
            "created_at": task.created_at,
        }

    def list_tasks(self) -> list[dict]:
        return [self.task_card(t) for t in self.tasks.values()]

    def task_detail(self, name: str) -> dict:
        task = self.get(name)
        with self.lock(name):
            return {
                **task.summary(),
                "metrics": task.metrics.summary(),
                "history": task.metrics.history,
                "events": task.events[-30:],
                "rules": task.rules,
                "served": getattr(task, "served", {"labels": 0, "correct": 0}),
                "pending": self.storage.count_pending(name),
                "log": self.storage.event_counts(name),
            }

    # -------------------------------------------------------------- decisions
    async def decide(self, state: Any, questions: dict[str, Any], record: bool = True, explain: bool = False,
                     escalate: str = "auto", abstain_threshold: float | None = None) -> dict:
        state = self.check_state(state)
        if not questions:
            raise ValueError("ask at least one question")
        if escalate not in ("auto", "never", "always"):
            raise ValueError("escalate must be auto, never or always")
        decision_id = "dec_" + uuid.uuid4().hex[:20]
        answers: dict[str, dict] = {}
        stored: dict[str, dict] = {}
        teacher_jobs: list[str] = []
        for name, raw in questions.items():
            task, options = self.ensure_task(name, raw if isinstance(raw, dict) else None)
            with self.lock(name):
                internal = task.answer(state, options)
                pub = task.public(internal, abstain_threshold)
                exp = task.explain(internal) if explain else None
                rule = first_match(task.rules, {**internal["feats"].flat, "$text": state_text(state)})
            student = {"probabilities": internal["probabilities"], "raw": internal["raw"], "abstain": pub["abstain"],
                       "answer": _answer_label(pub)}
            source = "student"
            if rule is not None and rule["decision"] in internal["options"]:
                pub = _force_answer(task, pub, rule["decision"])
                source = "rule"
                with self.lock(name):
                    rule["hits"] = rule.get("hits", 0) + 1
            pub["source"] = source
            if rule is not None and source == "rule":
                pub["rule"] = {"id": rule["id"], "name": rule["name"]}
            if exp is not None:
                pub["explanation"] = exp
            answers[name] = pub
            stored[name] = {"student": student, "options": internal["options"]}
            mode = task.settings.get("teacher_mode", "on_abstain")
            wants = escalate == "always" or (escalate == "auto" and (mode == "always" or (mode == "on_abstain" and pub["abstain"])))
            if source != "rule" and wants and self.teacher_cfg is not None:
                teacher_jobs.append(name)

        if teacher_jobs:
            results = await asyncio.gather(*(self._ask_teacher(state, self.tasks[n]) for n in teacher_jobs), return_exceptions=True)
            for name, res in zip(teacher_jobs, results):
                task = self.tasks[name]
                if isinstance(res, BaseException):
                    answers[name]["teacher_error"] = str(res)
                    continue
                dist, rationale = res
                opts = stored[name]["options"]
                sub = {o: dist.get(o, 0.0) for o in opts}
                s = sum(sub.values()) or 1.0
                internal = {"probabilities": {o: v / s for o, v in sub.items()}, "weights": {"teacher": 1.0}}
                pub = task.public(internal, 0.0)
                pub["source"] = "teacher"
                if rationale:
                    pub["rationale"] = rationale
                if "explanation" in answers[name]:
                    pub["explanation"] = answers[name]["explanation"]
                pub["student"] = {k: answers[name][k] for k in ("confidence", "abstain") if k in answers[name]}
                answers[name] = pub
                with self.lock(name):
                    task.learn(state, dist, source="teacher", ref=decision_id)
                self.storage.add_event(name, state, dist, "teacher", task.settings["teacher_weight"], decision_id)
                self._touch(task)

        for name, pub in answers.items():
            task = self.tasks[name]
            with self.lock(name):
                task.metrics.record_decision(stored[name]["student"]["abstain"], pub["source"] == "teacher")
            stored[name].update({k: v for k, v in pub.items() if k != "explanation"})
            stored[name]["answer_label"] = _answer_label(pub)
        if record:
            rows = [(n, stored[n]["answer_label"], pub["confidence"], stored[n]["student"]["abstain"], pub["source"])
                    for n, pub in answers.items()]
            self.storage.insert_decision(decision_id, time.time(), state, stored, rows)
            for n in answers:
                self.dirty[n] = self.dirty.get(n, 0) + 1
            self.bus.publish({"type": "decision", "id": decision_id, "state": _preview(state),
                              "answers": {n: {k: v for k, v in a.items() if k != "explanation"} for n, a in answers.items()}})
        return {"id": decision_id if record else None, "answers": answers}

    async def _ask_teacher(self, state: Any, task: DecisionTask) -> tuple[dict, str]:
        self.teacher_stats["calls"] += 1
        self.teacher_stats["last_call"] = time.time()
        try:
            return await teacher_answer(self.provider(), state, task.spec)
        except Exception as e:
            self.teacher_stats["errors"] += 1
            self.teacher_stats["last_error"] = str(e)[:300]
            raise

    def feedback(self, decision_id: str, answers: dict[str, Any], source: str = "human") -> dict:
        if source not in ("human", "system"):
            raise ValueError("feedback source must be human or system")
        d = self.storage.get_decision(decision_id)
        if d is None:
            raise NotFound(f"decision {decision_id!r} not found")
        if not answers:
            raise ValueError("give the correct answer for at least one question")
        out = {}
        for name, value in answers.items():
            if name not in d["answers"]:
                raise ValueError(f"decision {decision_id} did not ask {name!r}")
            if d["labels"].get(name) is not None:
                raise Conflict(f"{name!r} already has feedback for this decision")
            task = self.get(name)
            stored = d["answers"][name]
            with self.lock(name):
                label = task.spec.label_of(value)
                student = stored.get("student", {})
                events = task.learn(d["state"], label, source="human" if source == "human" else "dataset",
                                    served=student.get("probabilities"), served_raw=student.get("raw"),
                                    abstained=bool(student.get("abstain")), ref=decision_id)
                served = task.__dict__.setdefault("served", {"labels": 0, "correct": 0})
                served["labels"] += 1
                correct = stored.get("answer_label") == label
                served["correct"] += int(correct)
                metrics = task.metrics.summary()
            self.storage.add_event(name, d["state"], label, source, 1.0, decision_id)
            self.storage.set_answer_label(decision_id, name, label)
            self._touch(task)
            self._shadow_observe(task, d["state"], stored.get("options") or task.spec.option_names, label,
                                 student.get("probabilities") or {})
            for ev in events:
                self.bus.publish({"type": "drift", "task": name, "event": ev})
            self.bus.publish({"type": "feedback", "task": name, "decision_id": decision_id, "label": label, "correct": correct,
                              "metrics": _brief(metrics), "point": task.metrics.history[-1] if task.metrics.history else None})
            out[name] = {"label": label, "correct": correct,
                         "student_correct": student.get("answer") == label, "metrics": _brief(metrics)}
        self._maybe_auto_train()
        return {"decision_id": decision_id, "answers": out}

    def learn(self, name: str, examples: list[dict], source: str = "dataset", record: bool = True) -> dict:
        """Learn from labelled examples: [{"state": ..., "answer": ...}, ...]."""
        task = self.get(name)
        n = skipped = 0
        rows = []
        events: list[dict] = []
        if self.neural.active_id:  # one batched forward pass instead of one per example
            self.neural.prefetch(task.spec, [ex.get("state") for ex in examples if ex.get("state") not in (None, "")],
                                 task.spec.option_names)
        with self.lock(name):
            for ex in examples:
                state = ex.get("state")
                value = ex.get("answer", ex.get("label"))
                if state in (None, "") or value is None:
                    skipped += 1
                    continue
                try:
                    label = task.spec.label_of(value)
                except SpecError:
                    skipped += 1
                    continue
                events += task.learn(state, label, source=source)
                rows.append((name, None, state, label, source, 1.0))
                n += 1
            metrics = task.metrics.summary()
        if record and rows:
            self.storage.add_events(rows)
        self._touch(task, n)
        for ev in events:
            self.bus.publish({"type": "drift", "task": name, "event": ev})
        self.bus.publish({"type": "learned", "task": name, "n": n, "metrics": _brief(metrics),
                          "point": task.metrics.history[-1] if task.metrics.history else None})
        return {"learned": n, "skipped": skipped, "metrics": _brief(metrics)}

    def answers(self, name: str, limit: int = 50, pending: bool = False, uncertain_first: bool = False) -> list[dict]:
        self.get(name)
        return self.storage.list_answers(name, limit, pending, uncertain_first)

    # ------------------------------------------------------------------- rules
    def set_rules(self, name: str, rules: list[dict]) -> list[dict]:
        task = self.get(name)
        validated = [validate_rule(r) for r in rules]
        for r in validated:
            if r["decision"] not in task.spec.options:
                raise ValueError(f"rule {r['name'] or r['id']}: {r['decision']!r} is not an answer of {name}")
        with self.lock(name):
            old = {r["id"]: r for r in task.rules}
            for r in validated:
                prev = old.get(r["id"])
                if prev:
                    r["hits"] = prev.get("hits", 0)
            task.rules = validated
        self.persist(task)
        self.bus.publish({"type": "task_updated", "task": name})
        return task.rules

    # ------------------------------------------------ snapshots / audit / replay
    def snapshot(self, name: str, note: str = "") -> dict:
        task = self.get(name)
        with self.lock(name):
            m = task.metrics.summary()
            brief = {k: m[k] for k in ("accuracy", "nll", "ece", "labels")}
            self.storage.save_snapshot(name, task.version, task.labels, brief, task, note)
        return {"task": name, "version": task.version, "labels": task.labels, "metrics": brief, "note": note}

    def rollback(self, name: str, version: int) -> dict:
        self.get(name)
        restored = self.storage.load_snapshot(name, version)
        if restored is None:
            raise NotFound(f"snapshot {version} of {name!r} not found")
        with self.lock(name):
            self.tasks[name] = restored
        self.attach_neural(task=restored)
        self.persist(restored)
        self.bus.publish({"type": "task_reset", "task": name})
        return self.task_detail(name)

    def retract(self, event_id: int, retracted: bool = True) -> dict:
        ev = self.storage.get_event(event_id)
        if ev is None:
            raise NotFound(f"feedback event {event_id} not found")
        self.storage.set_retracted(event_id, retracted)
        return {**ev, "retracted": retracted, "hint": "run a rebuild to remove its influence from the student"}

    async def rebuild(self, job: Job, name: str, exclude_sources: list[str] | None = None) -> None:
        """Replay the feedback log into a fresh student (skipping retracted / excluded events)."""
        try:
            old = self.get(name)
            exclude = set(exclude_sources or [])
            events = [e for e in self.storage.iter_events(name) if e["source"] not in exclude]
            fresh = DecisionTask(copy.deepcopy(old.spec), dict(old.settings))
            fresh.rules = old.rules
            self.attach_neural(task=fresh)

            def replay(chunk: list[dict]) -> None:
                for e in chunk:
                    source = {"system": "dataset"}.get(e["source"], e["source"])
                    fresh.learn(e["state"], e["label"], source=source, weight=e["weight"])

            for i in range(0, len(events), 250):
                await asyncio.to_thread(replay, events[i:i + 250])
                self.update_job(job, progress=round(min((i + 250) / max(len(events), 1), 1.0), 4),
                                message=f"replayed {min(i + 250, len(events))}/{len(events)} events")
            self.snapshot(name, note="before rebuild")
            with self.lock(name):
                self.tasks[name] = fresh
            self.attach_neural(task=fresh)
            self.persist(fresh)
            self.bus.publish({"type": "task_reset", "task": name})
            self.update_job(job, status="done", progress=1.0, message="rebuild finished",
                            result={"task": name, "replayed": len(events), "metrics": _brief(fresh.metrics.summary())})
        except Exception as e:
            self.update_job(job, status="error", error=str(e) or type(e).__name__)

    # ----------------------------------------------------------------- teacher
    def set_teacher(self, cfg: ProviderConfig | None) -> dict:
        if cfg is not None:
            make_provider(cfg)  # validate
        self.teacher_cfg = cfg
        self._provider = None
        self.bus.publish({"type": "teacher_updated"})
        return self.teacher_public()

    def teacher_public(self) -> dict:
        return {"configured": self.teacher_cfg is not None,
                **(self.teacher_cfg.public() if self.teacher_cfg else {}), "stats": self.teacher_stats}

    def provider(self):
        if self.teacher_cfg is None:
            raise LLMError("no teacher configured: set an AI provider on the Teacher page or via DESIC_TEACHER_* env vars")
        if self._provider is None:
            self._provider = make_provider(self.teacher_cfg)
        return self._provider

    async def test_teacher(self) -> dict:
        spec = QuestionSpec.parse("desic_self_test", {"type": NOUL, "instructions": "The message is a greeting."})
        t0 = time.time()
        dist, rationale = await teacher_answer(self.provider(), "Hello there, nice to meet you!", spec)
        return {"ok": True, "p_true": round(dist["true"], 3), "rationale": rationale, "latency_ms": round((time.time() - t0) * 1000)}

    # ------------------------------------------------------------ neural student
    def _shadow_observe(self, task: DecisionTask, state: Any, options: list[str], label: str, served: dict) -> None:
        if not self.neural.shadow_id:
            return
        ck = self.neural.shadow_id
        try:
            outcome = self.neural.observe(task.spec, state, options, label, served)
        except Exception as e:  # never let the shadow break feedback
            self.bus.publish({"type": "neural_error", "error": str(e)[:300]})
            return
        if outcome == "promoted":
            self.attach_neural(reset=True)
        if outcome:
            self.bus.publish({"type": "neural_updated", "checkpoint": ck, "outcome": outcome})

    def neural_examples(self) -> list:
        """The training set: every non-retracted label in the log, one per (question, state).

        Priority when a state was labelled more than once: human > system/dataset > teacher,
        and the most recent label within the same priority wins.
        """
        from .neural.train import Example

        rank = {"human": 3, "system": 2, "dataset": 2, "teacher": 1}
        tw = float(self.neural.config["teacher_weight"])
        best: dict[str, tuple[int, Any]] = {}
        for name, task in self.tasks.items():
            spec = task.spec
            first = question_text(spec.type, spec.instructions or spec.name, spec.options)
            options = spec.option_names
            for e in self.storage.iter_events(name):
                label = e["label"]
                if isinstance(label, dict):
                    target = [float(label.get(o, 0.0)) for o in options]
                elif label in spec.options:
                    target = [1.0 if o == label else 0.0 for o in options]
                else:
                    continue
                total = sum(target)
                if total <= 0:
                    continue
                target = [v / total for v in target]
                key = name + ":" + state_hash_of(e["state"])
                r = rank.get(e["source"], 1)
                if key in best and best[key][0] > r:
                    continue
                weight = tw if e["source"] == "teacher" else 1.0
                best[key] = (r, Example(name, spec.type, first, state_to_text(e["state"]), options, target, weight,
                                        e["source"], key))
        return [ex for _, ex in best.values()]

    async def train_neural(self, job: Job) -> None:
        loop = asyncio.get_running_loop()

        def progress(frac: float, msg: str) -> None:
            loop.call_soon_threadsafe(lambda: self.update_job(job, progress=round(frac, 4), message=msg))

        self.neural.training = {"job": job.id, "started": time.time()}
        self.bus.publish({"type": "neural_updated"})
        try:
            self.storage.set_setting("neural_last_train_labels", sum(t.labels for t in self.tasks.values()))
            examples = await asyncio.to_thread(self.neural_examples)
            self.update_job(job, message=f"training on {len(examples)} labelled states")
            ck = await asyncio.to_thread(self.neural.train, examples, progress)
            if ck["status"] == "active":
                self.attach_neural(reset=True)
            self.update_job(job, status="done", progress=1.0, message=f"checkpoint {ck['id']}: {ck['status']}",
                            result={"checkpoint": ck["id"], "status": ck["status"], "note": ck["note"],
                                    "test": ck["report"]["test"]["overall"]})
        except Exception as e:
            self.update_job(job, status="error", error=str(e) or type(e).__name__)
        finally:
            self.neural.training = None
            self.bus.publish({"type": "neural_updated"})

    def start_neural_training(self) -> Job:
        if not self.neural.ok:
            raise NeuralError(self.neural.reason)
        if self.neural.training:
            raise Conflict("a neural training job is already running")
        job = self.new_job("neural", "preparing neural training")
        self.spawn(self.train_neural(job))
        return job

    def _maybe_auto_train(self) -> None:
        every = int(self.neural.config.get("auto_train_every", 0))
        if every <= 0 or not self.neural.ok or self.neural.training:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # called from a worker thread; the next feedback will check again
        total = sum(t.labels for t in self.tasks.values())
        if total - int(self.storage.get_setting("neural_last_train_labels", 0) or 0) >= every:
            self.start_neural_training()

    def neural_action(self, ck_id: str, action: str) -> dict:
        try:
            if action == "promote":
                self.neural.promote(ck_id, note="promoted manually")
                self.attach_neural(reset=True)
            elif action in ("reject", "retire"):
                was_active = self.neural.active_id == ck_id
                self.neural.set_status(ck_id, "rejected" if action == "reject" else "retired", note=f"{action}ed manually")
                if was_active:
                    self.attach_neural()
            elif action == "delete":
                self.neural.delete(ck_id)
            else:
                raise ValueError(f"unknown action {action!r}")
        except KeyError as e:
            raise NotFound(str(e.args[0])) from None
        self.bus.publish({"type": "neural_updated"})
        return self.neural.status()

    # ---------------------------------------------------------------- datasets
    def add_dataset(self, name: str, source: str, columns: list[str], rows: list[dict], meta: dict | None = None) -> dict:
        ds_id = self.storage.create_dataset(name, source, columns, rows, meta)
        ds = self.storage.get_dataset(ds_id)
        self.bus.publish({"type": "dataset_created", "dataset": ds})
        return ds

    def dataset_profile(self, ds_id: str, preview: int = 20) -> dict:
        ds = self.storage.get_dataset(ds_id)
        if ds is None:
            raise NotFound(f"dataset {ds_id!r} not found")
        rows = self.storage.dataset_rows(ds_id, 2000)
        profile = {}
        for col in ds["columns"]:
            vals = [r.get(col) for r in rows if r.get(col) not in (None, "")]
            distinct = list(dict.fromkeys(str(v) if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False) for v in vals))
            numeric = bool(vals) and all(to_number(v) is not None for v in vals if not isinstance(v, (dict, list)))
            avg_len = sum(len(str(v)) for v in vals) / len(vals) if vals else 0
            kind = "object" if any(isinstance(v, (dict, list)) for v in vals) else (
                "number" if numeric else ("text" if avg_len > 40 else "category"))
            profile[col] = {"type": kind, "distinct": len(distinct), "values": distinct[:12] if kind in ("category", "number") else []}
        return {**ds, "profile": profile, "preview": rows[:preview]}

    def _examples(self, rows: list[dict], answer_col: str, state_cols: list[str], mode: str) -> list[dict]:
        out = []
        for r in rows:
            value = r.get(answer_col)
            if value in (None, ""):
                continue
            single = r.get(state_cols[0]) if len(state_cols) == 1 else None
            # one text column stays plain text whatever its length, so a short message is
            # represented the same way in training and at decision time
            if mode == "text" or (mode == "auto" and isinstance(single, str) and to_number(single) is None):
                state: Any = " \n".join(str(r.get(c, "")) for c in state_cols if r.get(c) not in (None, ""))
            elif mode == "auto" and len(state_cols) == 1 and isinstance(r.get(state_cols[0]), (dict, list)):
                state = r.get(state_cols[0])
            else:
                state = {c: _coerce(r.get(c)) for c in state_cols if r.get(c) not in (None, "")}
            if state in ("", {}):
                continue
            out.append({"state": state, "answer": value})
        return out

    def _task_for_dataset(self, name: str, examples: list[dict], qtype: str, levels: list[str] | None,
                          instructions: str) -> DecisionTask:
        if name in self.tasks:
            return self.tasks[name]
        labels = list(dict.fromkeys(to_label(e["answer"]) for e in examples if to_label(e["answer"]) is not None))
        if qtype == NOUL:
            criteria: Any = ["true", "false"]
        elif qtype == SCORE:
            criteria = levels or sorted(labels, key=lambda v: (to_number(v) is None, to_number(v) or 0, v))
        else:
            criteria = labels
        if len(criteria) > 255:
            raise ValueError(f"the answer column has {len(criteria)} distinct values; a question allows at most 255")
        spec = QuestionSpec.parse(name, {"type": qtype, "instructions": instructions, "criteria": criteria})
        return self.create_task(spec)

    async def train_from_dataset(self, job: Job, ds_id: str, task_name: str, answer_column: str,
                                 state_columns: list[str] | None = None, state_mode: str = "auto", qtype: str = CHOICE,
                                 levels: list[str] | None = None, instructions: str = "", holdout: float = 0.2) -> None:
        try:
            ds = self.storage.get_dataset(ds_id)
            if ds is None:
                raise NotFound(f"dataset {ds_id!r} not found")
            if answer_column not in ds["columns"]:
                raise ValueError(f"dataset has no column {answer_column!r}")
            cols = [c for c in (state_columns or ds["columns"]) if c != answer_column and c in ds["columns"]]
            if not cols:
                raise ValueError("choose at least one state column")
            examples = self._examples(self.storage.dataset_rows(ds_id), answer_column, cols, state_mode)
            if not examples:
                raise ValueError("no usable rows")
            task = self._task_for_dataset(task_name, examples, qtype, levels, instructions)
            random.Random(7).shuffle(examples)
            holdout = min(max(float(holdout), 0.0), 0.5)
            n_test = int(len(examples) * holdout)
            train, test = examples[: len(examples) - n_test], examples[len(examples) - n_test:]
            self.update_job(job, message=f"training {task.spec.name} on {len(train)} examples")
            for i in range(0, len(train), 200):
                await asyncio.to_thread(self.learn, task.spec.name, train[i:i + 200])
                self.update_job(job, progress=round(min((i + 200) / len(train), 1.0) * (0.9 if test else 1.0), 4),
                                message=f"learned {min(i + 200, len(train))}/{len(train)}")
            result: dict[str, Any] = {"task": task.spec.name, "trained": len(train)}
            if test:
                result["holdout"] = await asyncio.to_thread(self.evaluate, task.spec.name, test)
            self.persist(task)
            self.update_job(job, status="done", progress=1.0, result=result, message="training finished")
        except Exception as e:
            self.update_job(job, status="error", error=str(e) or type(e).__name__)

    def evaluate(self, name: str, examples: list[dict]) -> dict:
        """Score examples without learning from them (accuracy, NLL, ECE)."""
        from .core.calibration import TaskMetrics

        task = self.get(name)
        m = TaskMetrics(window=len(examples) + 1)
        with self.lock(name):
            for ex in examples:
                try:
                    label = task.spec.label_of(ex["answer"])
                except SpecError:
                    continue
                internal = task.answer(ex["state"])
                m.update(internal["probabilities"], label, task.public(internal)["abstain"],
                         task.spec.option_names if task.spec.type == SCORE else None)
        s = m.summary()
        return {k: s[k] for k in ("labels", "accuracy", "nll", "brier", "ece", "answered_accuracy")} | {
            "coverage": round(1 - sum(r[2] for r in m.records) / max(len(m.records), 1), 4)}

    async def distill(self, job: Job, ds_id: str, task_name: str, state_columns: list[str] | None = None,
                      state_mode: str = "auto", limit: int = 200) -> None:
        """Let the teacher label unlabelled rows and teach the student with them."""
        try:
            ds = self.storage.get_dataset(ds_id)
            if ds is None:
                raise NotFound(f"dataset {ds_id!r} not found")
            task = self.get(task_name)
            provider = self.provider()
            cols = [c for c in (state_columns or ds["columns"]) if c in ds["columns"]]
            rows = self.storage.dataset_rows(ds_id, max(1, min(int(limit), 5000)))
            states = [e["state"] for e in self._examples([{**r, "__x": 1} for r in rows], "__x", cols, state_mode)]
            sem = asyncio.Semaphore(4)
            done = failed = 0

            async def one(state: Any) -> None:
                nonlocal done, failed
                async with sem:
                    try:
                        self.teacher_stats["calls"] += 1
                        dist, _ = await teacher_answer(provider, state, task.spec)
                    except Exception as e:
                        failed += 1
                        self.teacher_stats["errors"] += 1
                        self.teacher_stats["last_error"] = str(e)[:300]
                        return
                with self.lock(task_name):
                    task.learn(state, dist, source="teacher")
                self.storage.add_event(task_name, state, dist, "teacher", task.settings["teacher_weight"])
                self._touch(task)
                done += 1
                self.update_job(job, progress=round((done + failed) / len(states), 4), message=f"teacher labelled {done}/{len(states)}")

            await asyncio.gather(*(one(s) for s in states))
            if done == 0 and failed:
                raise LLMError(f"all {failed} teacher calls failed: {self.teacher_stats['last_error']}")
            self.persist(task)
            self.update_job(job, status="done", progress=1.0, message="distillation finished",
                            result={"task": task_name, "labelled": done, "failed": failed})
        except Exception as e:
            self.update_job(job, status="error", error=str(e) or type(e).__name__)

    # --------------------------------------------------------------- generation
    async def design(self, description: str) -> dict:
        return await design_task(self.provider(), description)

    async def generate(self, job: Job, description: str, design: dict, n: int, name: str = "", train: bool = True) -> None:
        try:
            design = normalize_design({**design, "answers": [{"name": k, "description": v} for k, v in design.get("options", {}).items()]}
                                      if "options" in design and "answers" not in design else design)

            async def progress(done: int, total: int) -> None:
                self.update_job(job, progress=round(min(done / total, 1.0) * 0.9, 4), message=f"{done}/{total} examples written")

            rows = await generate_examples(self.provider(), design, description, n, progress)
            ds = self.add_dataset(name or design["name"], "generated", ["state", "answer"], rows,
                                  {"description": description, "design": design})
            result: dict[str, Any] = {"dataset": ds["id"], "examples": len(rows)}
            if train:
                spec = QuestionSpec.parse(design["name"], {"type": design["type"], "instructions": design["instructions"],
                                                           "criteria": design["options"]})
                if spec.name not in self.tasks:
                    self.create_task(spec)
                tj = self.new_job("train", f"training {spec.name}")
                result["train_job"] = tj.id
                self.spawn(self.train_from_dataset(tj, ds["id"], spec.name, "answer", ["state"], "auto",
                                                   spec.type, None, spec.instructions))
            self.update_job(job, status="done", progress=1.0, result=result, message=f"{len(rows)} examples generated")
        except Exception as e:
            self.update_job(job, status="error", error=str(e) or type(e).__name__)

    # -------------------------------------------------------------------- jobs
    def spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    def new_job(self, kind: str, message: str = "") -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, message=message)
        self.jobs[job.id] = job
        self._publish_job(job)
        return job

    def _publish_job(self, job: Job) -> None:
        self.bus.publish({"type": "job", "job": job.to_dict()})

    def update_job(self, job: Job, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(job, k, v)
        self._publish_job(job)


# ------------------------------------------------------------------ helpers
def _answer_label(pub: dict) -> str:
    if pub["type"] == CHOICE:
        return pub["choice"]
    if pub["type"] == SCORE:
        return pub["level"]
    return "true" if pub["answer"] else "false"


def _force_answer(task: DecisionTask, pub: dict, label: str) -> dict:
    pub = dict(pub)
    if task.spec.type == CHOICE:
        pub["choice"] = label
    elif task.spec.type == SCORE:
        pub["level"] = label
        pub["score"] = float(task.spec.option_names.index(label))
    else:
        pub["answer"] = label == "true"
        pub["probability"] = 1.0 if label == "true" else 0.0
    pub["confidence"] = 1.0
    pub["abstain"] = False
    return pub


def _coerce(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        n = to_number(s)
        # keep identifiers like "0042" or phone numbers as text
        if n is not None and not (len(s) > 1 and s[0] == "0" and s[1].isdigit()):
            return n
        return s
    return v


def _preview(state: Any) -> str:
    text = state_text(state)
    return text if len(text) <= 240 else text[:237] + "…"


def _brief(metrics: dict) -> dict:
    return {k: metrics[k] for k in ("labels", "accuracy", "nll", "ece")}
