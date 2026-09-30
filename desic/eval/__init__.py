"""Evaluation harness: scenarios that test *how* a learner learns over a stream.

Scenarios (see :mod:`desic.eval.scenarios`): shuffled, class-sorted, label
noise, a burst of wrong labels, concept drift and a cold start with a teacher.
Metrics (see :mod:`desic.eval.metrics`): cumulative prequential log loss,
selective risk (risk@coverage, AURC), forgetting index, adaptation half-life
and teacher dependency.
"""

from .metrics import Prequential, aurc, evaluate, forgetting_index, half_life, risk_at_coverage
from .scenarios import LEARNERS, SCENARIOS

__all__ = ["LEARNERS", "SCENARIOS", "Prequential", "aurc", "evaluate", "forgetting_index", "half_life",
           "risk_at_coverage"]
