"""Training the neural student from the feedback log (requires PyTorch).

Objective (in the order the Jev/Laya research recommends):

1. supervised proper scoring: soft cross-entropy (log score) against human
   labels and the teacher's probabilities (distillation), plus the Ranked
   Probability Score for ordered ``score`` questions;
2. an act/escalate head trained to predict whether the top answer is right;
3. optional RLCD-style stage: Gaussian noise on the logits, a group of noisy
   samples per example, reward = log score of the sampled distribution,
   REINFORCE with the group-mean baseline (GRPO-style);
4. post-training temperature per question, fitted on a held-out slice.

The split is a stable hash of (question, state), so the test slice is never
trained on by any checkpoint — candidates can be compared fairly.
"""

from __future__ import annotations

import copy
import math
import random
import time
import zlib
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from ..core.calibration import reliability
from .model import DecisionNet, collate

GROUND_TRUTH = ("human", "dataset", "system")


@dataclass
class Example:
    task: str
    qtype: str
    first: str               # question + answers with markers
    second: str              # state text
    options: list[str]
    target: list[float]      # aligned with options
    weight: float
    source: str
    key: str                 # question + state hash, for the split

    @property
    def split(self) -> str:
        b = zlib.crc32(self.key.encode()) % 10
        return "test" if b == 0 else "val" if b == 1 else "train"


DEFAULTS = {
    "epochs": 4,
    "batch_size": 16,
    "lr_backbone": 3e-5,      # pretrained encoders move slowly …
    "lr_scratch": 1e-3,       # … a scratch encoder needs a larger step
    "lr_head": 1e-3,
    "weight_decay": 0.01,
    "rps_weight": 1.0,
    "act_weight": 0.2,
    "objective": "ce",        # ce | rlcd
    "rlcd_group": 4,
    "rlcd_sigma": 0.5,
    "rlcd_ce_anchor": 0.5,
    "max_examples": 20000,
    "seed": 13,
}


def _batches(items: list, size: int, rng: random.Random | None = None) -> list[list]:
    items = list(items)
    if rng is not None:
        rng.shuffle(items)
    return [items[i:i + size] for i in range(0, len(items), size)]


def _targets(batch: list[tuple[Example, list[int]]], k: int, device: torch.device) -> torch.Tensor:
    t = torch.zeros((len(batch), k), device=device)
    for i, (ex, _) in enumerate(batch):
        t[i, : len(ex.target)] = torch.tensor(ex.target, device=device)
    return t


def losses(logits: torch.Tensor, act_logit: torch.Tensor, target: torch.Tensor, batch: list[tuple[Example, list[int]]],
           cfg: dict) -> tuple[torch.Tensor, dict]:
    w = torch.tensor([ex.weight for ex, _ in batch], device=logits.device)
    valid = torch.isfinite(logits)
    logp = torch.log_softmax(logits, dim=-1).masked_fill(~valid, 0.0)
    ce = -(target * logp).sum(-1)
    loss = ce
    parts = {"ce": float((ce.detach() * w).sum() / w.sum())}
    # RPS for ordered questions
    is_score = torch.tensor([ex.qtype == "score" for ex, _ in batch], device=logits.device)
    if bool(is_score.any()):
        p = logp.exp() * valid
        diff = (p.cumsum(-1) - target.cumsum(-1)) ** 2
        k = valid.sum(-1).clamp(min=2)
        rps = (diff.sum(-1) - diff.gather(1, (k - 1).unsqueeze(1)).squeeze(1)) / (k - 1)
        loss = loss + cfg["rps_weight"] * rps * is_score
        parts["rps"] = float((rps.detach() * is_score).sum() / is_score.sum())
    # act / escalate head: is the top answer the (soft) target's top answer?
    with torch.no_grad():
        right = (logits.argmax(-1) == target.argmax(-1)).float()
    act = F.binary_cross_entropy_with_logits(act_logit, right, reduction="none")
    loss = loss + cfg["act_weight"] * act
    # RLCD-style exploration
    if cfg["objective"] == "rlcd":
        g, sigma = int(cfg["rlcd_group"]), float(cfg["rlcd_sigma"])
        base = logits.masked_fill(~valid, 0.0)
        eps = torch.randn((g, *base.shape), device=base.device) * sigma
        sampled = (base.detach().unsqueeze(0) + eps).masked_fill(~valid.unsqueeze(0), float("-inf"))
        reward = (target.unsqueeze(0) * torch.log_softmax(sampled, -1).masked_fill(~valid.unsqueeze(0), 0.0)).sum(-1)  # log score
        adv = reward - reward.mean(0, keepdim=True)
        log_pi = -(((sampled.masked_fill(~valid.unsqueeze(0), 0.0) - base.unsqueeze(0)) ** 2).sum(-1)) / (2 * sigma ** 2)
        rl = -(adv.detach() * log_pi).mean(0)
        loss = cfg["rlcd_ce_anchor"] * loss + rl
        parts["rlcd"] = float((rl.detach() * w).sum() / w.sum())
    return (loss * w).sum() / w.sum(), parts


@torch.no_grad()
def predict_logits(net: DecisionNet, encoded: list[tuple[Any, list[int]]], device: torch.device, batch_size: int = 32) -> list[tuple[list[float], float]]:
    net.eval()
    out: list[tuple[list[float], float]] = []
    for batch in _batches(encoded, batch_size):
        ids, attn, markers = collate([x for _, x in batch], net.backbone.mask_token_id, device)
        logits, act = net(ids, attn, markers)
        for i, (item, _) in enumerate(batch):
            k = len(item.options) if hasattr(item, "options") else int(torch.isfinite(logits[i]).sum())
            out.append((logits[i, :k].float().cpu().tolist(), float(torch.sigmoid(act[i]).cpu())))
    return out


def softmax(z: list[float], t: float = 1.0) -> list[float]:
    m = max(z)
    e = [math.exp((v - m) / t) for v in z]
    s = sum(e)
    return [v / s for v in e]


def fit_temperature(pairs: list[tuple[list[float], list[float]]]) -> float:
    """Temperature in [0.5, 4] minimising soft NLL on (logits, target) pairs."""
    if len(pairs) < 10:
        return 1.0
    best_t, best = 1.0, float("inf")
    for i in range(41):
        t = math.exp(math.log(0.5) + i / 40 * (math.log(4.0) - math.log(0.5)))
        nll = 0.0
        for z, y in pairs:
            p = softmax(z, t)
            nll -= sum(q * math.log(max(pv, 1e-12)) for q, pv in zip(y, p) if q > 0)
        if nll < best:
            best, best_t = nll, t
    return round(best_t, 4)


def metrics(rows: list[tuple[list[float], list[float], str]]) -> dict:
    """rows: (calibrated probs, target, qtype) — ground-truth labels only."""
    if not rows:
        return {"n": 0}
    n = len(rows)
    acc = nll = brier = 0.0
    rel = []
    for p, y, _ in rows:
        pred = max(range(len(p)), key=lambda i: p[i])
        truth = max(range(len(y)), key=lambda i: y[i])
        ok = pred == truth
        acc += ok
        nll -= math.log(max(p[truth], 1e-12))
        brier += sum((pv - (1.0 if i == truth else 0.0)) ** 2 for i, pv in enumerate(p))
        rel.append((p[pred], ok))
    ece, _ = reliability(rel)
    return {"n": n, "accuracy": round(acc / n, 4), "nll": round(nll / n, 4), "brier": round(brier / n, 4), "ece": ece}


def evaluate(net: DecisionNet, examples: list[Example], temps: dict[str, float], device: torch.device) -> dict:
    enc = [(ex, ids) for ex in examples if (ids := net.backbone.encode(ex.first, ex.second)) is not None]
    preds = predict_logits(net, enc, device)
    by_task: dict[str, list] = {}
    for (ex, _), (z, _) in zip(enc, preds):
        if ex.source not in GROUND_TRUTH:
            continue
        p = softmax(z, temps.get(ex.task, 1.0))
        by_task.setdefault(ex.task, []).append((p, ex.target, ex.qtype))
    return {"overall": metrics([r for rows in by_task.values() for r in rows]),
            "tasks": {t: metrics(rows) for t, rows in by_task.items()}}


def train(net: DecisionNet, examples: list[Example], cfg: dict, device: torch.device,
          progress: Callable[[float, str], None] | None = None, stop: Callable[[], bool] | None = None) -> dict:
    cfg = {**DEFAULTS, **cfg}
    rng = random.Random(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    examples = examples[-int(cfg["max_examples"]):]
    encoded = [(ex, ids) for ex in examples if (ids := net.backbone.encode(ex.first, ex.second)) is not None]
    skipped = len(examples) - len(encoded)
    train_set = [e for e in encoded if e[0].split == "train"]
    val_set = [e for e in encoded if e[0].split == "val"]
    test_set = [e[0] for e in encoded if e[0].split == "test"]
    if len(train_set) < 10:
        raise ValueError(f"not enough training examples ({len(train_set)}); give more feedback or train on a dataset first")

    lr_bb = cfg["lr_scratch"] if net.backbone.kind == "scratch" else cfg["lr_backbone"]
    head_params = [p for n, p in net.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([
        {"params": list(net.backbone.parameters()), "lr": lr_bb},
        {"params": head_params, "lr": cfg["lr_head"]},
    ], weight_decay=cfg["weight_decay"])
    net.to(device)
    epochs = int(cfg["epochs"])
    batches_per_epoch = math.ceil(len(train_set) / cfg["batch_size"])
    total = epochs * batches_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, total // 10)) * max(0.05, 1 - s / total))
    history = []
    best_state, best_val = None, float("inf")
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        net.train()
        run = 0.0
        for batch in _batches(train_set, cfg["batch_size"], rng):
            if stop and stop():
                raise InterruptedError("training was cancelled")
            ids, attn, markers = collate([x for _, x in batch], net.backbone.mask_token_id, device)
            logits, act = net(ids, attn, markers)
            loss, parts = losses(logits, act, _targets(batch, logits.shape[1], device), batch, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run += float(loss.detach())
            if progress and (step % 5 == 0 or step == total):
                progress(step / total * 0.9, f"epoch {epoch + 1}/{epochs} · step {step}/{total} · loss {float(loss.detach()):.3f}")
        val_nll = _soft_nll(net, val_set, device) if val_set else run / max(batches_per_epoch, 1)
        history.append({"epoch": epoch + 1, "train_loss": round(run / max(batches_per_epoch, 1), 4), "val_nll": round(val_nll, 4)})
        if val_nll < best_val:
            best_val = val_nll
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
    if best_state is not None:
        net.load_state_dict(best_state)
    if progress:
        progress(0.93, "fitting temperatures on the validation slice")
    temps = fit_temperatures(net, [e for e, _ in val_set], device)
    if progress:
        progress(0.97, "evaluating on the held-out test slice")
    report = {
        "temperatures": temps,
        "test": evaluate(net, test_set, temps, device),
        "train": {"examples": len(train_set), "val": len(val_set), "test": len(test_set), "skipped": skipped,
                  "epochs": epochs, "history": history, "seconds": round(time.time() - t0, 1),
                  "objective": cfg["objective"], "device": str(device)},
    }
    return report


def _soft_nll(net: DecisionNet, enc: list[tuple[Example, list[int]]], device: torch.device) -> float:
    preds = predict_logits(net, enc, device)
    total = 0.0
    for (ex, _), (z, _) in zip(enc, preds):
        p = softmax(z)
        total -= sum(q * math.log(max(pv, 1e-12)) for q, pv in zip(ex.target, p) if q > 0)
    return total / max(len(enc), 1)


def fit_temperatures(net: DecisionNet, val: list[Example], device: torch.device) -> dict[str, float]:
    enc = [(ex, ids) for ex in val if (ids := net.backbone.encode(ex.first, ex.second)) is not None]
    preds = predict_logits(net, enc, device)
    by_task: dict[str, list] = {}
    for (ex, _), (z, _) in zip(enc, preds):
        by_task.setdefault(ex.task, []).append((z, ex.target))
    return {t: fit_temperature(pairs) for t, pairs in by_task.items()}
