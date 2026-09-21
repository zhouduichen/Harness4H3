from harness4h3.remote.round_gate import evaluate_round_gate


def _worker(**overrides):
    value = {
        "status": "success",
        "experiment_id": "exp_0001",
        "operator": "quantize",
        "metrics": {"real_worker": True, "offline_simulation": False},
    }
    value.update(overrides)
    return value


def _evaluation(**overrides):
    value = {"quality_score": 0.9, "feasible": True, "violations": []}
    value.update(overrides)
    return value


def test_round_gate_rejects_missing_child_and_keeps_evidence():
    result = evaluate_round_gate(
        worker_result=_worker(),
        evaluation=_evaluation(),
        target={"min_quality_score": 0.8},
        capability_evidence={},
        lane_evidence={"disjoint": True},
        parent_digest="sha256:p",
        child_digest=None,
        retention={"outcome": "rejected_candidate"},
    )
    assert result.status == "rejected"
    assert "child_digest_missing" in result.reasons
    assert result.retention["outcome"] == "rejected_candidate"


def test_round_gate_accepts_real_feasible_candidate():
    result = evaluate_round_gate(
        worker_result=_worker(),
        evaluation=_evaluation(),
        target={"min_quality_score": 0.8},
        capability_evidence={},
        lane_evidence={"disjoint": True, "worker_gpus": [1, 2]},
        parent_digest="sha256:p",
        child_digest="sha256:c",
        retention={"outcome": "accepted_candidate", "retained": True},
    )
    assert result.status == "accepted"
    assert result.evidence["child_digest"] == "sha256:c"


def test_round_gate_replans_when_runtime_capability_is_not_verified():
    result = evaluate_round_gate(
        worker_result=_worker(
            operator="step_distill",
            operator_args={"lpl_target_steps": 8},
        ),
        evaluation=_evaluation(),
        target={"min_quality_score": 0.8},
        capability_evidence={"lpl": {"safe_to_plan": False}},
        lane_evidence={"disjoint": True},
        parent_digest="sha256:p",
        child_digest="sha256:c",
        retention={},
    )
    assert result.status == "replan"
    assert result.reasons == ("capability_unverified:lpl",)


def test_round_gate_replans_when_verified_name_has_no_execution_contract():
    result = evaluate_round_gate(
        worker_result=_worker(
            operator="step_distill",
            operator_args={"tdtm_merge_steps": 4},
        ),
        evaluation=_evaluation(),
        target={"min_quality_score": 0.8},
        capability_evidence={"tdtm": {"safe_to_plan": True}},
        lane_evidence={"disjoint": True},
        parent_digest="sha256:p",
        child_digest="sha256:c",
        retention={},
    )
    assert result.status == "replan"
    assert result.reasons == ("capability_contract_missing:tdtm",)


def test_round_gate_waits_for_resource_dependency_before_quality_decision():
    result = evaluate_round_gate(
        worker_result={},
        evaluation={},
        target={},
        capability_evidence={},
        lane_evidence={"status": "waiting", "reason": "foreign_compute_process"},
        parent_digest=None,
        child_digest=None,
        retention={},
    )
    assert result.status == "waiting"
    assert result.reasons == ("resource_dependency_wait",)
