"""Calibration: temperature scaling and the metrics that keep it honest.

A model is calibrated when, among all answers it gives with probability 0.8,
about 80% are correct. Desic fits one temperature per question online on the
most recent labelled answers (prequential: each answer is scored with the
probabilities that were actually served, before the label is learned) and
reports ECE, Brier score, NLL, a reliability diagram and the risk–coverage
curve that tells you where to put the abstain threshold.
"""

from __future__ import annotations

import math
from collections import deque

Dist = dict[str, float]
EPS = 1e-6


def temper(p: Dist, t: float) -> Dist:
    if abs(t - 1.0) < 1e-6:
        return dict(p)
    logs = {o: math.log(max(v, 1e-12)) / t for o, v in p.items()}
    m = max(logs.values())
    ex = {o: math.exp(v - m) for o, v in logs.items()}
    s = sum(ex.values())
    return {o: v / s for o, v in ex.items()}


class TemperatureCalibrator:
    def __init__(self, window: int = 500, refit_every: int = 50, min_samples: int = 30) -> None:
        self.samples: deque = deque(maxlen=window)  # (log raw probabilities, index of the true answer)
        self.t = 1.0
        self.refit_every = refit_every
        self.min_samples = min_samples
        self._since = 0

    def apply(self, p: Dist) -> Dist:
        return temper(p, self.t)

    def add(self, raw: Dist, label: str) -> None:
        if label not in raw or len(raw) < 2:
            return
        logs = [math.log(max(v, 1e-12)) for v in raw.values()]
        self.samples.append((logs, list(raw).index(label)))
        self._since += 1
        if self._since >= self.refit_every and len(self.samples) >= self.min_samples:
            self.fit()

    def _nll(self, t: float) -> float:
        total = 0.0
        for logs, y in self.samples:
            scaled = [v / t for v in logs]
            m = max(scaled)
            total += m + math.log(sum(math.exp(v - m) for v in scaled)) - scaled[y]
        return total / len(self.samples)

    def fit(self) -> float:
        self._since = 0
        # Golden-section search on log T. T is bounded to [0.5, 4]: calibration may at
        # most double the sharpness, because a temperature fitted on familiar data would
        # otherwise make the student absurdly sure about inputs it has never seen.
        lo, hi = math.log(0.5), math.log(4.0)
        g = (math.sqrt(5) - 1) / 2
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        fa, fb = self._nll(math.exp(a)), self._nll(math.exp(b))
        for _ in range(22):
            if fa < fb:
                hi, b, fb = b, a, fa
                a = hi - g * (hi - lo)
                fa = self._nll(math.exp(a))
            else:
                lo, a, fa = a, b, fb
                b = lo + g * (hi - lo)
                fb = self._nll(math.exp(b))
        self.t = round(math.exp((lo + hi) / 2), 4)
        return self.t


def confidence_of(p: Dist) -> tuple[str, float]:
    best = max(p.items(), key=lambda kv: kv[1])
    return best[0], best[1]


def reliability(records: list[tuple[float, bool]], bins: int = 10) -> tuple[float | None, list[dict]]:
    """ECE and reliability-diagram bins from (confidence, correct) pairs."""
    if not records:
        return None, []
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for c, ok in records:
        buckets[min(int(c * bins), bins - 1)].append((c, ok))
    ece = 0.0
    out = []
    n = len(records)
    for i, b in enumerate(buckets):
        if not b:
            continue
        conf = sum(c for c, _ in b) / len(b)
        acc = sum(ok for _, ok in b) / len(b)
        ece += len(b) / n * abs(acc - conf)
        out.append({"lo": i / bins, "hi": (i + 1) / bins, "count": len(b), "confidence": round(conf, 4), "accuracy": round(acc, 4)})
    return round(ece, 4), out


def risk_coverage(records: list[tuple[float, bool]], points: int = 20) -> list[dict]:
    """Accuracy you get if you only answer the top-x% most confident decisions."""
    if not records:
        return []
    ranked = sorted(records, key=lambda r: -r[0])
    out = []
    n = len(ranked)
    correct = 0
    marks = {max(1, round(n * (i + 1) / points)) for i in range(points)}
    for i, (c, ok) in enumerate(ranked, 1):
        correct += ok
        if i in marks:
            out.append({"coverage": round(i / n, 4), "accuracy": round(correct / i, 4), "threshold": round(c, 4)})
    return out


class TaskMetrics:
    """Prequential quality of the *served* answers, scored against human labels."""

    def __init__(self, window: int = 1000, max_points: int = 300) -> None:
        self.records: deque = deque(maxlen=window)  # (confidence, correct, abstained, nll, brier, rps|None)
        self.n = 0
        self.correct = 0
        self.nll_sum = 0.0
        self.brier_sum = 0.0
        self.confusion: dict[str, dict[str, int]] = {}
        self.history: list[dict] = []
        self.max_points = max_points
        self._every = 1
        self.decisions: deque = deque(maxlen=500)  # (abstained, teacher_called) per decision
        self.served: deque = deque(maxlen=1000)  # confidence of each recent decision, labelled or not
        self.teacher_calls = 0
        self.decisions_total = 0

    def record_decision(self, abstained: bool, teacher: bool, confidence: float | None = None) -> None:
        self.decisions.append((abstained, teacher))
        if confidence is not None:
            if not hasattr(self, "served"):  # metrics saved before this field existed
                self.served = deque(maxlen=1000)
            self.served.append(confidence)
        self.decisions_total += 1
        self.teacher_calls += int(teacher)

    def update(self, served: Dist, label: str, abstained: bool, levels: list[str] | None = None) -> bool:
        pred, conf = confidence_of(served)
        ok = pred == label
        nll = -math.log(max(served.get(label, 0.0), EPS))
        brier = sum((served.get(o, 0.0) - (1.0 if o == label else 0.0)) ** 2 for o in set(served) | {label})
        rps = None
        if levels and label in levels:
            cum_p = cum_y = 0.0
            rps = 0.0
            for lv in levels[:-1]:
                cum_p += served.get(lv, 0.0)
                cum_y += 1.0 if lv == label else 0.0
                rps += (cum_p - cum_y) ** 2
            rps /= max(len(levels) - 1, 1)
        self.n += 1
        self.correct += ok
        self.nll_sum += nll
        self.brier_sum += brier
        self.records.append((conf, ok, abstained, nll, brier, rps))
        row = self.confusion.setdefault(label, {})
        row[pred] = row.get(pred, 0) + 1
        if self.n % self._every == 0:
            recent = list(self.records)[-200:]
            ece, _ = reliability([(c, o) for c, o, *_ in recent])
            dec = list(self.decisions)[-200:]
            self.history.append({
                "n": self.n,
                "accuracy": round(sum(r[1] for r in recent) / len(recent), 4),
                "ece": ece,
                "teacher_rate": round(sum(t for _, t in dec) / len(dec), 4) if dec else 0.0,
            })
            if len(self.history) > self.max_points:
                self.history = self.history[1::2]
                self._every *= 2
        return ok

    def summary(self) -> dict:
        recs = list(self.records)
        pairs = [(c, o) for c, o, *_ in recs]
        ece, bins = reliability(pairs)
        answered = [r for r in recs if not r[2]]
        rps = [r[5] for r in recs if r[5] is not None]
        dec = list(self.decisions)
        w = len(recs)
        return {
            "labels": self.n,
            "window": w,
            "accuracy": round(sum(o for _, o in pairs) / w, 4) if w else None,
            "accuracy_all_time": round(self.correct / self.n, 4) if self.n else None,
            "nll": round(sum(r[3] for r in recs) / w, 4) if w else None,
            "brier": round(sum(r[4] for r in recs) / w, 4) if w else None,
            "rps": round(sum(rps) / len(rps), 4) if rps else None,
            "ece": ece,
            "reliability": bins,
            "risk_coverage": risk_coverage(pairs),
            "answered_accuracy": round(sum(r[1] for r in answered) / len(answered), 4) if answered else None,
            "abstain_rate": round(sum(a for a, _ in dec) / len(dec), 4) if dec else None,
            "teacher_rate": round(sum(t for _, t in dec) / len(dec), 4) if dec else None,
            "teacher_calls": self.teacher_calls,
            "decisions": self.decisions_total,
            "confusion": self.confusion,
        }
