from harness4h3.controller.loop import OptimizationLoop
from harness4h3.controller.schemas import EvaluationResult


def test_decision_semantics_keep_legacy_pointer_and_split_statuses():
    accepted = EvaluationResult(
        quality_score=0.9,
        quality_metrics={},
        hardware=None,
        feasible=True,
        violations=[],
        critical_regression=False,
    )
    search = OptimizationLoop._decision(True, "M0001", None)
    final = OptimizationLoop._decision(True, "M0001", accepted)
    reject = OptimizationLoop._decision(False, "M0000", accepted)

    assert search["keep"] is True and search["continue_from"] == "M0001"
    assert search["status"] == "search_keep" and search["search_keep"] is True
    assert final["status"] == "final_accept" and final["final_accept"] is True
    assert reject["status"] == "reject" and reject["reject"] is True
