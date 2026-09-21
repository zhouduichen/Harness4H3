from __future__ import annotations

from harness4h3.archive.pareto import ParetoArchive, dominates
from harness4h3.controller.schemas import EvaluationRecord, HardwareMetrics
from harness4h3.target.profile import ObjectiveSpec


def evaluation(quality, latency, memory=4.0, feasible=True):
    return EvaluationRecord(
        quality_score=quality,
        quality_metrics={"quality_score": quality},
        hardware=HardwareMetrics(latency_s=latency, peak_memory_gb=memory),
        feasible=feasible,
    )


def test_pareto_uses_target_objective_directions(tmp_path):
    objectives = (
        ObjectiveSpec("quality_score", "maximize", 2),
        ObjectiveSpec("latency_s", "minimize", 1),
    )
    archive = ParetoArchive(tmp_path)
    archive.update("S0000", evaluation(0.80, 40), objectives)
    archive.update("S0001", evaluation(0.81, 45), objectives)
    assert [entry.candidate_id for entry in archive.front()] == ["S0000", "S0001"]
    assert archive.front()[0].objectives == objectives


def test_feasible_candidate_dominates_infeasible_candidate_before_objectives():
    assert dominates(evaluation(0.70, 100, feasible=True), evaluation(0.90, 10, feasible=False))


def test_old_model_archive_entries_remain_readable(tmp_path):
    archive = ParetoArchive(tmp_path)
    archive.update("M0000", evaluation(0.80, 40))
    assert archive.front()[0].candidate_id == "M0000"
