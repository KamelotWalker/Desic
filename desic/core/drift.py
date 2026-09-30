"""Concept drift detection.

DDM (Gama et al., 2004) watches the stream of prediction errors. When the
error rate rises significantly above the best rate seen so far it first
raises a *warning* (start training a replacement model in the background)
and then a *drift* (swap the replacement in).
"""

from __future__ import annotations

import math

NONE, WARNING, DRIFT = "none", "warning", "drift"


class DDM:
    def __init__(self, min_instances: int = 100, warning_level: float = 2.0, drift_level: float = 3.0,
                 min_errors: int = 5) -> None:
        self.min_instances = min_instances
        self.min_errors = min_errors
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.errors = 0
        self.p = 1.0
        self.s = 0.0
        self.p_min = float("inf")
        self.s_min = float("inf")
        self.ps_min = float("inf")

    def update(self, error: bool) -> str:
        self.n += 1
        self.errors += int(error)
        # Laplace-smoothed error rate: a perfect start (0 errors) must not give
        # s_min == 0, otherwise the very next mistake would look like a drift.
        self.p = (self.errors + 1) / (self.n + 2)
        self.s = math.sqrt(self.p * (1.0 - self.p) / self.n)
        if self.n < self.min_instances:
            return NONE
        if self.errors < self.min_errors:  # too little evidence either way
            if self.p + self.s <= self.ps_min:
                self.p_min, self.s_min, self.ps_min = self.p, self.s, self.p + self.s
            return NONE
        if self.p + self.s <= self.ps_min:
            self.p_min, self.s_min = self.p, self.s
            self.ps_min = self.p + self.s
        if self.p + self.s > self.p_min + self.drift_level * self.s_min:
            self.reset()
            return DRIFT
        if self.p + self.s > self.p_min + self.warning_level * self.s_min:
            return WARNING
        return NONE
