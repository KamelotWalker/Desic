import random

import pytest

from desic.demo import ticket
from desic.eval import SCENARIOS, aurc, forgetting_index, half_life, risk_at_coverage
from desic.eval.metrics import Prequential
from desic.eval.run import aggregate
from desic.eval.scenarios import budget, burst, class_sorted, drift, make_desic, noisy, shuffled, teacher


def test_selective_risk_and_aurc():
    recs = [(0.9, True), (0.8, True), (0.7, False), (0.6, True)]
    assert risk_at_coverage(recs, 0.5) == 0.0
    assert risk_at_coverage(recs, 1.0) == 0.25
    # errors ranked first are worse than errors ranked last
    assert aurc([(0.9, False), (0.1, True)]) > aurc([(0.9, True), (0.1, False)])
    assert aurc([(0.5, True)] * 3) == 0.0


def test_forgetting_index():
    ck = [{"a": 0.9}, {"a": 0.6, "b": 0.8}, {"a": 0.3, "b": 0.7, "c": 0.9}]
    # a: best 0.9 → 0.3 ; b: best 0.8 → 0.7 ; c is the newest, not counted
    assert forgetting_index(ck, {"a": 0, "b": 1, "c": 2}) == pytest.approx(0.35)
    assert forgetting_index([{"a": 1.0}], {"a": 0}) is None


def test_half_life():
    curve = [(0, 0.0), (100, 0.3), (200, 0.5), (300, 0.8)]
    assert half_life(curve, low=0.0, high=0.8) == 200
    assert half_life(curve, low=0.0, high=2.0) is None
    assert half_life(curve, low=0.5, high=0.5) == 0


def test_prequential_scores_before_learning():
    pq = Prequential(every=2)
    pq.add({"a": 0.5, "b": 0.5}, "a")
    pq.add({"a": 1.0, "b": 0.0}, "a")
    s = pq.summary()
    assert s["labels"] == 2 and s["accuracy"] == 1.0
    assert s["cum_log_loss"] == pytest.approx(0.3466, abs=1e-3)
    assert s["curve"][0]["n"] == 2


def test_aggregate_marks_unreached_values():
    agg = aggregate([{"headline": {"x": 1.0, "hl": None}}, {"headline": {"x": 3.0, "hl": 200}}])
    assert agg["x"]["mean"] == 2.0 and agg["x"]["std"] == pytest.approx(1.4142, abs=1e-3)
    assert agg["hl"]["missing"] == 1 and agg["hl"]["mean"] == 200


@pytest.fixture(scope="module")
def tickets():
    rng = random.Random(3)
    rows = [ticket(rng)[:2] for _ in range(500)]
    return rows[:400], rows[400:], sorted({y for _, y in rows})


def test_every_scenario_runs_and_reports_a_headline(tickets):
    train, test, classes = tickets
    runs = {
        "shuffled": shuffled(make_desic, train, test, classes, 0),
        "sorted": class_sorted(make_desic, train, test, classes, 0),
        "noise": noisy(make_desic, train, test, classes, 0, rate=0.1),
        "burst": burst(make_desic, train, test, classes, 0, size=15, probe_every=25),
        "drift": drift(make_desic, train, test, classes, 0, moved=2, probe_every=25),
        "teacher": teacher(make_desic, train, test, classes, 0, n=300, window=100),
        "budget": budget(make_desic, train, test, classes, 0, window=100),
    }
    for name, r in runs.items():
        assert r["headline"], name
    assert runs["shuffled"]["headline"]["accuracy"] > 0.8
    assert runs["noise"]["poisoned_labels"] > 0
    b = runs["burst"]
    assert b["burst_size"] == 15 and b["victim"] != b["attack_label"]
    assert b["headline"]["victim_after"] <= b["headline"]["victim_before"]
    assert b["headline"]["victim_after_undo"] == b["headline"]["victim_before"]  # undo = replaying the clean log
    assert b["headline"]["accuracy_change_after_undo"] == 0
    assert len(runs["teacher"]["windows"]) == 3
    bh = runs["budget"]["headline"]
    assert bh["b5%_coverage"] <= bh["b10%_coverage"] <= bh["t60_coverage"] + 1e-9
    assert set(SCENARIOS) >= {"shuffled", "sorted", "noise-1%", "noise-5%", "burst", "drift", "teacher", "budget"}


def test_scenarios_are_deterministic(tickets):
    train, test, classes = tickets
    a = burst(make_desic, train, test, classes, 1, size=15, probe_every=25)
    b = burst(make_desic, train, test, classes, 1, size=15, probe_every=25)
    a.pop("seconds"), b.pop("seconds")
    a["headline"].pop("undo_seconds"), b["headline"].pop("undo_seconds")
    assert a == b


def test_cli_help_lists_the_scenarios(capsys):
    from desic.cli import main

    with pytest.raises(SystemExit):
        main(["eval", "--help"])
    assert "noise-1%" in capsys.readouterr().out
