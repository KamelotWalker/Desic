"""Desic application service: models, decisions, feedback, datasets and jobs.

The HTTP layer (api.py) is a thin wrapper around this class, so the same
engine can be embedded in any Python program:

    desic = Desic("./data")
    desic.create_model("loan", schema)
    d = desic.decide("loan", {"income": 4200, "age": 31})
    desic.feedback("loan", d["id"], "approve")
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core.model import DecisionModel
from .core.rules import first_match, validate_rule
from .core.schema import Schema, to_label
from .storage import Storage

SAVE_EVERY = 25  # persist after this many learning steps


def _new_counters() -> dict:
    return {"decisions": 0, "feedback": 0, "final_correct": 0, "rule_decisions": 0}


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


@dataclass
class ModelRuntime:
    name: str
    schema: Schema
    model: DecisionModel
    description: str = ""
    rules: list[dict] = field(default_factory=list)
    counters: dict = field(default_factory=_new_counters)
    created_at: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    dirty: int = 0


class Desic:
    def __init__(self, data_dir: str | Path = "data", db_name: str = "desic.db") -> None:
        path = ":memory:" if str(data_dir) == ":memory:" else Path(data_dir) / db_name
        self.storage = Storage(path)
        self.bus = EventBus()
        self.jobs: dict[str, Job] = {}
        self.runtimes: dict[str, ModelRuntime] = {}
        for rec in self.storage.load_models():
            model = rec["state"] or DecisionModel(rec["kind"])
            self.runtimes[rec["name"]] = ModelRuntime(
                name=rec["name"],
                schema=Schema.from_dict(rec["schema"]),
                model=model,
                description=rec["description"],
                rules=rec["rules"],
                counters={**_new_counters(), **rec["meta"].get("counters", {})},
                created_at=rec["created_at"],
            )

    # ------------------------------------------------------------- persistence
    def persist(self, rt: ModelRuntime) -> None:
        with rt.lock:
            self.storage.save_model(
                rt.name, rt.model.kind, rt.description, rt.schema.to_dict(), rt.rules,
                {"counters": rt.counters}, rt.model,
            )
            rt.dirty = 0

    def persist_all(self) -> None:
        for rt in self.runtimes.values():
            if rt.dirty:
                self.persist(rt)

    def _touch(self, rt: ModelRuntime, n: int = 1) -> None:
        rt.dirty += n
        if rt.dirty >= SAVE_EVERY:
            self.persist(rt)

    # ------------------------------------------------------------------ models
    def get(self, name: str) -> ModelRuntime:
        rt = self.runtimes.get(name)
        if rt is None:
            raise NotFound(f"model {name!r} not found")
        return rt

    def create_model(self, name: str, schema: Schema, kind: str = "tree", params: dict | None = None,
                     description: str = "") -> ModelRuntime:
        name = name.strip()
        if not name:
            raise ValueError("model name is required")
        if name in self.runtimes:
            raise Conflict(f"model {name!r} already exists")
        if not schema.features:
            raise ValueError("a model needs at least one feature")
        if schema.target in schema.feature_names:
            raise ValueError("the target cannot also be a feature")
        rt = ModelRuntime(name=name, schema=schema, model=DecisionModel(kind, params), description=description)
        self.runtimes[name] = rt
        self.persist(rt)
        self.bus.publish({"type": "model_created", "model": name})
        return rt

    def delete_model(self, name: str) -> None:
        self.get(name)
        del self.runtimes[name]
        self.storage.delete_model(name)
        self.bus.publish({"type": "model_deleted", "model": name})

    def reset_model(self, name: str) -> None:
        rt = self.get(name)
        with rt.lock:
            rt.model.reset()
            rt.counters = _new_counters()
        self.persist(rt)
        self.bus.publish({"type": "model_reset", "model": name})

    def list_models(self) -> list[dict]:
        return [self.model_card(rt) for rt in self.runtimes.values()]

    def model_card(self, rt: ModelRuntime) -> dict:
        m = rt.model.metrics.summary()
        return {
            "name": rt.name,
            "description": rt.description,
            "kind": rt.model.kind,
            "target": rt.schema.target,
            "classes": rt.schema.classes,
            "n_features": len(rt.schema.features),
            "learned": rt.model.n_learned,
            "accuracy": m["accuracy"],
            "rolling_accuracy": m["rolling_accuracy"],
            "decisions": rt.counters["decisions"],
            "created_at": rt.created_at,
        }

    def model_detail(self, name: str) -> dict:
        rt = self.get(name)
        with rt.lock:
            metrics = rt.model.metrics.summary()
            fb = rt.counters["feedback"]
            return {
                **self.model_card(rt),
                "params": rt.model.params,
                "schema": rt.schema.to_dict(),
                "structure": rt.model.structure(),
                "metrics": metrics,
                "history": rt.model.metrics.history,
                "importance": rt.model.feature_importance(),
                "events": rt.model.events[-30:],
                "rules": rt.rules,
                "counters": {
                    **rt.counters,
                    "pending": self.storage.count_pending(name),
                    "decision_accuracy": round(rt.counters["final_correct"] / fb, 4) if fb else None,
                },
            }

    def tree(self, name: str, member: int | None = None) -> dict:
        rt = self.get(name)
        with rt.lock:
            return rt.model.tree_dict(member)

    def learned_rules(self, name: str, limit: int = 100) -> list[dict]:
        rt = self.get(name)
        with rt.lock:
            return rt.model.rules()[:limit]

    # --------------------------------------------------------------- decisions
    def decide(self, name: str, raw: dict, record: bool = True) -> dict:
        rt = self.get(name)
        x = rt.schema.coerce(raw)
        with rt.lock:
            result = rt.model.predict(x)
            explanation = rt.model.explain(x) if rt.model.n_learned else {"path": []}
            rule = first_match(rt.rules, x)
            if rule is not None:
                rule["hits"] = rule.get("hits", 0) + 1
                source, prediction = "rule", rule["decision"]
            elif result["prediction"] is not None:
                source, prediction = "model", result["prediction"]
            else:
                source, prediction = "none", None
            if record:
                rt.counters["decisions"] += 1
                if source == "rule":
                    rt.counters["rule_decisions"] += 1
        decision = {
            "id": uuid.uuid4().hex,
            "model": name,
            "created_at": time.time(),
            "features": x,
            "prediction": prediction,
            "model_prediction": result["prediction"],
            "confidence": 1.0 if source == "rule" else result["confidence"],
            "probabilities": result["probabilities"],
            "source": source,
            "rule_id": rule["id"] if rule else None,
        }
        if record:
            self.storage.insert_decision(decision)
            self._touch(rt)
            self.bus.publish({"type": "decision", "model": name, "decision": decision})
        decision["explanation"] = explanation
        if rule:
            decision["rule"] = {"id": rule["id"], "name": rule["name"]}
        return decision

    def feedback(self, name: str, decision_id: str, label: Any) -> dict:
        rt = self.get(name)
        label = to_label(label)
        if label is None:
            raise ValueError("label is required")
        d = self.storage.get_decision(decision_id)
        if d is None or d["model"] != name:
            raise NotFound(f"decision {decision_id!r} not found")
        if d["label"] is not None:
            raise Conflict("this decision already has feedback")
        correct = d["prediction"] == label
        with rt.lock:
            rt.schema.observe_label(label)
            rt.schema.observe_row(d["features"])
            events = rt.model.learn(d["features"], label, evaluated_prediction=d["model_prediction"])
            rt.counters["feedback"] += 1
            rt.counters["final_correct"] += int(correct)
            if d["rule_id"]:
                for r in rt.rules:
                    if r["id"] == d["rule_id"]:
                        key = "confirmed" if correct else "overridden"
                        r[key] = r.get(key, 0) + 1
            metrics = rt.model.metrics.summary()
        self.storage.set_feedback(decision_id, label, correct)
        self._touch(rt)
        self._publish_learning(rt, events, metrics)
        self.bus.publish({"type": "feedback", "model": name, "decision_id": decision_id, "label": label, "correct": correct})
        return {"decision_id": decision_id, "label": label, "correct": correct, "metrics": _brief(metrics)}

    def decisions(self, name: str, limit: int = 50, pending: bool = False, uncertain_first: bool = False) -> list[dict]:
        self.get(name)
        return self.storage.list_decisions(name, limit, pending, uncertain_first)

    def learn(self, name: str, rows: list[dict]) -> dict:
        """Learn from already-labelled rows (each row contains the target column)."""
        rt = self.get(name)
        n, skipped, events = self._learn_rows(rt, rows)
        metrics = rt.model.metrics.summary()
        self._touch(rt, n)
        self._publish_learning(rt, events, metrics)
        return {"learned": n, "skipped": skipped, "metrics": _brief(metrics)}

    def _learn_rows(self, rt: ModelRuntime, rows: list[dict]) -> tuple[int, int, list[dict]]:
        n = skipped = 0
        events: list[dict] = []
        with rt.lock:
            for row in rows:
                y = to_label(row.get(rt.schema.target))
                if y is None:
                    skipped += 1
                    continue
                x = rt.schema.coerce(row)
                rt.schema.observe_label(y)
                rt.schema.observe_row(x)
                events += rt.model.learn(x, y)
                n += 1
        return n, skipped, events

    def _publish_learning(self, rt: ModelRuntime, events: list[dict], metrics: dict) -> None:
        for ev in events:
            self.bus.publish({"type": "drift", "model": rt.name, "event": ev})
        self.bus.publish({
            "type": "metrics",
            "model": rt.name,
            "learned": rt.model.n_learned,
            "metrics": _brief(metrics),
            "point": rt.model.metrics.history[-1] if rt.model.metrics.history else None,
        })

    # ------------------------------------------------------------------- rules
    def set_rules(self, name: str, rules: list[dict]) -> list[dict]:
        rt = self.get(name)
        validated = [validate_rule(r) for r in rules]
        with rt.lock:
            old = {r["id"]: r for r in rt.rules}
            for r in validated:  # keep live counters for rules that already existed
                prev = old.get(r["id"])
                if prev:
                    for key in ("hits", "confirmed", "overridden"):
                        r[key] = prev.get(key, 0)
            rt.rules = validated
        self.persist(rt)
        self.bus.publish({"type": "rules", "model": name})
        return rt.rules

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
            try:
                s = Schema.infer(rows, target="\0", features=[col])
                f = s.features[0]
                profile[col] = {"type": f.type, "values": f.values[:20]}
            except ValueError:
                profile[col] = {"type": "categorical", "values": []}
            distinct = {to_label(r.get(col)) for r in rows} - {None}
            profile[col]["distinct"] = len(distinct)
        return {**ds, "profile": profile, "preview": rows[:preview]}

    # -------------------------------------------------------------------- jobs
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

    async def train_from_dataset(self, job: Job, ds_id: str, model_name: str, target: str | None = None,
                                 features: list[str] | None = None, kind: str = "tree", params: dict | None = None,
                                 holdout: float = 0.2, shuffle: bool = True, passes: int = 1) -> None:
        try:
            ds = self.storage.get_dataset(ds_id)
            if ds is None:
                raise NotFound(f"dataset {ds_id!r} not found")
            rows = list(self.storage.iter_dataset_rows(ds_id))
            if model_name in self.runtimes:
                rt = self.get(model_name)
                if target and target != rt.schema.target:
                    raise ValueError(f"model {model_name!r} predicts {rt.schema.target!r}, not {target!r}")
                if rt.schema.target not in ds["columns"]:
                    raise ValueError(f"dataset has no column {rt.schema.target!r}")
            else:
                if not target or target not in ds["columns"]:
                    raise ValueError("choose the target (decision) column")
                schema = Schema.infer(rows, target, features)
                rt = self.create_model(model_name, schema, kind, params, description=f"trained from dataset {ds['name']}")
            if shuffle:
                random.Random(7).shuffle(rows)
            holdout = min(max(float(holdout), 0.0), 0.5)
            n_test = int(len(rows) * holdout)
            train, test = rows[: len(rows) - n_test], rows[len(rows) - n_test:]
            passes = max(1, min(int(passes), 10))
            total = len(train) * passes
            done = 0
            chunk = 250
            self.update_job(job, message=f"training {rt.name} on {len(train)} rows")
            for p in range(passes):
                order = train if p == 0 else random.Random(p).sample(train, len(train))
                for i in range(0, len(order), chunk):
                    _, _, events = await asyncio.to_thread(self._learn_rows, rt, order[i:i + chunk])
                    done += len(order[i:i + chunk])
                    self._publish_learning(rt, events, rt.model.metrics.summary())
                    self.update_job(job, progress=round(done / max(total, 1), 4))
            result: dict[str, Any] = {"model": rt.name, "trained_rows": len(train), "passes": passes}
            if test:
                result["holdout"] = await asyncio.to_thread(self.evaluate, rt, test)
            result["prequential"] = _brief(rt.model.metrics.summary())
            self.persist(rt)
            self.update_job(job, status="done", progress=1.0, result=result, message="training finished")
        except Exception as e:  # surfaced to the dashboard through the job
            self.update_job(job, status="error", error=str(e) or type(e).__name__)

    def evaluate(self, rt: ModelRuntime, rows: list[dict]) -> dict:
        correct = n = 0
        confusion: dict[str, dict[str, int]] = {}
        with rt.lock:
            for row in rows:
                y = to_label(row.get(rt.schema.target))
                if y is None:
                    continue
                pred = rt.model.predict(rt.schema.coerce(row))["prediction"]
                n += 1
                correct += int(pred == y)
                r = confusion.setdefault(y, {})
                r[str(pred)] = r.get(str(pred), 0) + 1
        return {"rows": n, "accuracy": round(correct / n, 4) if n else None, "confusion": confusion}


def _brief(metrics: dict) -> dict:
    return {k: metrics[k] for k in ("evaluated", "accuracy", "rolling_accuracy")}
