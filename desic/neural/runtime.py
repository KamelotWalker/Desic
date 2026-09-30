"""Neural checkpoint lifecycle: train → offline gate → shadow → active.

    feedback log ─► candidate training (thread) ─► temperature fit ─► offline test
         │                                                               │ beats the prior and
         │                                                               │ the active checkpoint?
         ▼                                                               ▼
    live feedback ──────────────────────────────────────────────►  SHADOW: predicts on every
                                                                   labelled decision, never served
                                                                         │ shadow log loss ≤ live
                                                                         ▼ log loss + margin
                                                                   ACTIVE: joins every question's
                                                                   expert mixture as "neural"

Adding an expert to a log-loss Hedge mixture is low-risk (its regret is
bounded), which is why the live gate only asks the shadow not to be clearly
worse than what users are currently served.

Torch is imported lazily so Desic runs without it.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from ..core.features import state_hash
from . import available
from .text import question_text, state_to_text

CONFIG_DEFAULTS: dict[str, Any] = {
    "backbone": "scratch",       # scratch | any Hugging Face encoder id or local path
    "max_len": 256,
    "epochs": 4,
    "batch_size": 16,
    "objective": "ce",           # ce | rlcd
    "lr_backbone": 3e-5,
    "lr_head": 1e-3,
    "device": "auto",
    "auto_train_every": 0,       # new labels between automatic trainings (0 = manual only)
    "shadow_min": 30,            # labelled decisions a shadow must see before promotion (0 = promote immediately)
    "promote_margin": 0.1,       # shadow log loss may exceed the live log loss by at most this
    "auto_promote": True,
    "init_from_active": True,    # continue fine-tuning the active checkpoint when the backbone matches
    "teacher_weight": 0.5,
    "max_examples": 20000,
}


class NeuralError(RuntimeError):
    pass


class NeuralManager:
    def __init__(self, data_dir: Path | None, storage: Any) -> None:
        self.storage = storage
        self.dir = (data_dir / "neural") if data_dir is not None else Path(tempfile.mkdtemp(prefix="desic-neural-"))
        self.config = {**CONFIG_DEFAULTS, **(storage.get_setting("neural", {}) or {})}
        self.ok, self.reason = available()
        self._lock = threading.RLock()
        self._active: tuple[str, Any, dict] | None = None  # (id, net, temperatures)
        self._shadow: tuple[str, Any, dict] | None = None
        self._device = None
        self.cache: OrderedDict = OrderedDict()
        self.training: dict | None = None
        self.cancel = False
        if self.ok:
            for status, slot in (("active", "_active"), ("shadow", "_shadow")):
                ck = storage.checkpoint_with_status(status)
                if ck is not None:
                    try:
                        setattr(self, slot, self._load(ck))
                    except Exception as e:  # a missing / broken checkpoint must not stop the server
                        storage.update_checkpoint(ck["id"], status="failed", note=f"could not load: {e}")

    # ------------------------------------------------------------------ config
    def set_config(self, patch: dict) -> dict:
        cfg = dict(self.config)
        for k, v in patch.items():
            if k not in CONFIG_DEFAULTS:
                raise ValueError(f"unknown neural setting {k!r}")
            default = CONFIG_DEFAULTS[k]
            if isinstance(default, bool):
                v = bool(v)
            elif isinstance(default, int):
                v = int(v)
            elif isinstance(default, float):
                v = float(v)
            else:
                v = str(v).strip()
            cfg[k] = v
        if cfg["objective"] not in ("ce", "rlcd"):
            raise ValueError("objective must be ce or rlcd")
        if not cfg["backbone"]:
            raise ValueError("choose a backbone")
        if not (1 <= cfg["epochs"] <= 100 and 1 <= cfg["batch_size"] <= 512 and 32 <= cfg["max_len"] <= 8192):
            raise ValueError("epochs 1-100, batch_size 1-512, max_len 32-8192")
        self.config = cfg
        self.storage.set_setting("neural", cfg)
        return cfg

    # ---------------------------------------------------------------- status
    @property
    def active_id(self) -> str | None:
        return self._active[0] if self._active else None

    @property
    def active_pretrained(self) -> bool:
        """True when the active checkpoint's encoder was pretrained (not the built-in scratch encoder)."""
        return self._active is not None and getattr(self._active[1].backbone, "kind", "scratch") != "scratch"

    @property
    def shadow_id(self) -> str | None:
        return self._shadow[0] if self._shadow else None

    def status(self) -> dict:
        return {
            "available": self.ok,
            "reason": self.reason,
            "config": self.config,
            "active": self.active_id,
            "shadow": self.shadow_id,
            "training": self.training,
            "checkpoints": self.storage.list_checkpoints(),
            "backbones": BACKBONES,
        }

    # ------------------------------------------------------------- inference
    def _device_obj(self):
        if self._device is None:
            from .model import pick_device

            self._device = pick_device(self.config.get("device", "auto"))
        return self._device

    def _load(self, ck: dict) -> tuple[str, Any, dict]:
        from .model import load_net

        net, _ = load_net(Path(ck["path"]), self._device_obj())
        return ck["id"], net, ck["report"].get("temperatures", {})

    @staticmethod
    def _inputs(spec: Any, state: Any, options: list[str]) -> tuple[str, str]:
        opts = {o: spec.options.get(o, "") for o in options}
        return question_text(spec.type, spec.instructions or spec.name, opts), state_to_text(state)

    def _run(self, slot: tuple[str, Any, dict], items: list[tuple[Any, Any, list[str]]]) -> list[tuple[dict, float] | None]:
        """Batched forward pass for [(spec, state, options)] with one model."""
        from .train import predict_logits, softmax

        ck_id, net, temps = slot
        out: list[tuple[dict, float] | None] = [None] * len(items)
        todo = []
        for i, (spec, state, options) in enumerate(items):
            first, second = self._inputs(spec, state, options)
            key = (ck_id, spec.name, first, state_hash(state))
            hit = self.cache.get(key)
            if hit is not None:
                self.cache.move_to_end(key)
                out[i] = hit
                continue
            ids = net.backbone.encode(first, second)
            if ids is not None:
                todo.append((i, key, spec, options, ids))
        if todo:
            class _Item:
                def __init__(self, options):
                    self.options = options

            with self._lock:
                preds = predict_logits(net, [(_Item(o), ids) for _, _, _, o, ids in todo], self._device_obj())
            for (i, key, spec, options, _), (z, act) in zip(todo, preds):
                p = softmax(z, temps.get(spec.name, 1.0))
                res = ({o: pv for o, pv in zip(options, p)}, act)
                out[i] = res
                self.cache[key] = res
                if len(self.cache) > 4000:
                    self.cache.popitem(last=False)
        return out

    def predict(self, spec: Any, state: Any, options: list[str]) -> tuple[dict, float] | None:
        slot = self._active
        if slot is None:
            return None
        return self._run(slot, [(spec, state, options)])[0]

    def prefetch(self, spec: Any, states: list[Any], options: list[str], batch: int = 64) -> None:
        slot = self._active
        if slot is None:
            return
        for i in range(0, len(states), batch):
            self._run(slot, [(spec, s, options) for s in states[i:i + batch]])

    # ---------------------------------------------------------------- shadow
    def observe(self, spec: Any, state: Any, options: list[str], label: str, served: dict) -> str | None:
        """Score the shadow checkpoint on a labelled decision. Returns 'promoted' / 'rejected' / None."""
        slot = self._shadow
        if slot is None or label not in options:
            return None
        res = self._run(slot, [(spec, state, options)])[0]
        if res is None:
            return None
        probs, _ = res
        ck = self.storage.get_checkpoint(slot[0])
        if ck is None:
            return None
        sh = ck["shadow"] or {"n": 0, "nll": 0.0, "correct": 0, "live_nll": 0.0, "live_correct": 0}
        sh["n"] += 1
        sh["nll"] += -math.log(max(probs.get(label, 0.0), 1e-6))
        sh["correct"] += int(max(probs, key=probs.get) == label)
        sh["live_nll"] += -math.log(max(served.get(label, 0.0), 1e-6))
        sh["live_correct"] += int(max(served, key=served.get) == label) if served else 0
        self.storage.update_checkpoint(slot[0], shadow=sh)
        if sh["n"] >= max(int(self.config["shadow_min"]), 1) and self.config["auto_promote"]:
            if sh["nll"] / sh["n"] <= sh["live_nll"] / sh["n"] + float(self.config["promote_margin"]):
                self.promote(slot[0], note=f"shadow log loss {sh['nll'] / sh['n']:.3f} vs live {sh['live_nll'] / sh['n']:.3f}")
                return "promoted"
            self.set_status(slot[0], "rejected", note="worse than the live system in shadow")
            return "rejected"
        return None

    # ------------------------------------------------------------- lifecycle
    def promote(self, ck_id: str, note: str = "") -> None:
        ck = self.storage.get_checkpoint(ck_id)
        if ck is None:
            raise KeyError(f"checkpoint {ck_id!r} not found")
        if ck["status"] in ("failed",):
            raise ValueError("a failed checkpoint cannot be promoted")
        slot = self._shadow if self.shadow_id == ck_id else self._load(ck)
        with self._lock:
            if self._active is not None:
                self.storage.update_checkpoint(self._active[0], status="retired")
            self._active = slot
            if self.shadow_id == ck_id:
                self._shadow = None
            self.cache.clear()
        self.storage.update_checkpoint(ck_id, status="active", note=note or "promoted")

    def set_status(self, ck_id: str, status: str, note: str = "") -> None:
        if status not in ("rejected", "retired"):
            raise ValueError("status must be rejected or retired")
        with self._lock:
            if self.active_id == ck_id:
                self._active = None
                self.cache.clear()
            if self.shadow_id == ck_id:
                self._shadow = None
        self.storage.update_checkpoint(ck_id, status=status, **({"note": note} if note else {}))

    def delete(self, ck_id: str) -> None:
        ck = self.storage.get_checkpoint(ck_id)
        if ck is None:
            raise KeyError(f"checkpoint {ck_id!r} not found")
        if ck["status"] in ("active", "shadow"):
            raise ValueError("retire or reject the checkpoint before deleting it")
        shutil.rmtree(ck["path"], ignore_errors=True)
        self.storage.update_checkpoint(ck_id, status="deleted")

    # --------------------------------------------------------------- training
    def train(self, examples: list, progress: Callable[[float, str], None] | None = None) -> dict:
        """Blocking: run in a worker thread. Returns the checkpoint record."""
        if not self.ok:
            raise NeuralError(self.reason)
        from .model import DecisionNet, backbone_spec, build_backbone, load_net, save_net
        from .train import evaluate
        from .train import train as run_training

        cfg = dict(self.config)
        device = self._device_obj()
        spec = backbone_spec(cfg["backbone"], cfg["max_len"])
        if progress:
            progress(0.01, f"loading backbone {cfg['backbone']}")
        active = self._active
        init_note = "fresh"
        net = None
        if cfg["init_from_active"] and active is not None:
            ck = self.storage.get_checkpoint(active[0])
            if ck and ck["backbone"] == cfg["backbone"]:
                net, _ = load_net(Path(ck["path"]), device)
                init_note = f"continued from {active[0]}"
        if net is None:
            try:
                net = DecisionNet(build_backbone(spec))
            except Exception as e:
                raise NeuralError(f"could not load backbone {cfg['backbone']!r}: {e}") from None
        self.cancel = False
        report = run_training(net, examples, {**cfg, "max_examples": cfg["max_examples"]}, device, progress,
                              stop=lambda: self.cancel)
        test = [ex for ex in examples if ex.split == "test"]
        report["prior"] = _prior_metrics(examples, test)
        if active is not None:
            report["baseline"] = {"checkpoint": active[0], **evaluate(active[1], test, active[2], device)}
        report["init"] = init_note
        ck_id = "nn-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        path = self.dir / ck_id
        save_net(net, path, {"id": ck_id})
        status, note = self._gate(report)
        self.storage.add_checkpoint(ck_id, "candidate", cfg["backbone"], str(path), report, note)
        if status == "rejected":
            self.storage.update_checkpoint(ck_id, status="rejected")
        elif self.config["shadow_min"] <= 0 and self.config["auto_promote"]:
            self.promote(ck_id, note=note + " · promoted without shadow (shadow_min = 0)")
        else:
            with self._lock:
                if self._shadow is not None:
                    self.storage.update_checkpoint(self._shadow[0], status="rejected", note="replaced by a newer candidate")
                net.eval()
                self._shadow = (ck_id, net, report["temperatures"])
            self.storage.update_checkpoint(ck_id, status="shadow")
        return self.storage.get_checkpoint(ck_id)

    @staticmethod
    def _gate(report: dict) -> tuple[str, str]:
        cand = report["test"]["overall"]
        if not cand.get("n"):
            return "shadow", "no labelled test examples; judged in shadow only"
        prior = report["prior"]
        if prior.get("nll") is not None and cand["nll"] >= prior["nll"]:
            return "rejected", f"test log loss {cand['nll']:.3f} is no better than the base rates ({prior['nll']:.3f})"
        base = report.get("baseline", {}).get("overall", {})
        if base.get("n") and cand["nll"] > base["nll"] + 0.02:
            return "rejected", f"test log loss {cand['nll']:.3f} is worse than the active checkpoint ({base['nll']:.3f})"
        return "shadow", f"passed the offline gate (test log loss {cand['nll']:.3f}, accuracy {cand['accuracy']:.1%})"


def _prior_metrics(train_examples: list, test: list) -> dict:
    """Log loss / accuracy of always answering with the training base rates."""
    from .train import GROUND_TRUTH

    rates: dict[str, dict[str, float]] = {}
    for ex in train_examples:
        if ex.split != "train":
            continue
        r = rates.setdefault(ex.task, {})
        for o, q in zip(ex.options, ex.target):
            r[o] = r.get(o, 0.0) + q
    n = correct = 0
    nll = 0.0
    for ex in test:
        if ex.source not in GROUND_TRUTH:
            continue
        r = rates.get(ex.task, {})
        total = sum(r.get(o, 0.0) + 0.5 for o in ex.options)
        p = [(r.get(o, 0.0) + 0.5) / total for o in ex.options]
        truth = max(range(len(ex.target)), key=lambda i: ex.target[i])
        n += 1
        nll -= math.log(p[truth])
        correct += int(max(range(len(p)), key=lambda i: p[i]) == truth)
    return {"n": n, "nll": round(nll / n, 4) if n else None, "accuracy": round(correct / n, 4) if n else None}


BACKBONES = {
    "scratch": "Built-in tiny encoder (trains from zero, no download)",
    "answerdotai/ModernBERT-large": "ModernBERT-large — Laya's English backbone (395M)",
    "answerdotai/ModernBERT-base": "ModernBERT-base — English (150M)",
    "jhu-clsp/mmBERT-base": "mmBERT-base — 100+ languages incl. Turkish (Laya multilingual)",
    "jhu-clsp/mmBERT-small": "mmBERT-small — multilingual, fast",
}
