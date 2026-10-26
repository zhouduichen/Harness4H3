from harness4h3.evolution.selection import select_representatives
from harness4h3.evolution.campaign import BranchCampaign


def test_selects_distinct_lite_medium_heavy_representatives():
    entries = [
        {"candidate_id": "M0001", "valid": True, "quality_score": 0.90, "inference_wall_time_s_mean": 10},
        {"candidate_id": "M0002", "valid": True, "quality_score": 0.95, "inference_wall_time_s_mean": 20},
        {"candidate_id": "M0003", "valid": True, "quality_score": 0.99, "inference_wall_time_s_mean": 40},
    ]
    selected = select_representatives(entries, baseline_quality=0.90)
    assert [item.mode for item in selected] == ["lite", "medium", "heavy"]
    assert len({item.candidate_id for item in selected}) == 3
    # With the paper's squared-distance weights, the middle point is a
    # better cost-focused compromise than the absolute minimum-cost extreme.
    assert selected[0].candidate_id == "M0002"
    assert selected[-1].candidate_id == "M0003"


def test_excludes_unverified_offline_below_floor_and_missing_cost():
    entries = [
        {"candidate_id": "M0001", "valid": True, "quality_score": .99, "hardware_metrics": {"inference_wall_time_s_mean": 10}},
        {"candidate_id": "M0002", "valid": False, "quality_score": 1, "inference_wall_time_s_mean": 1},
        {"candidate_id": "M0003", "valid": True, "offline_simulation": True, "quality_score": 1, "inference_wall_time_s_mean": 1},
        {"candidate_id": "M0004", "valid": True, "quality_score": .89, "inference_wall_time_s_mean": 1},
        {"candidate_id": "M0005", "valid": True, "quality_score": 1},
    ]
    selected = select_representatives(entries, baseline_quality=.90)
    assert [item.candidate_id for item in selected] == ["M0001"]


def test_handles_constant_objectives_and_stable_ties():
    entries = [
        {"candidate_id": "M0002", "valid": True, "quality_score": .9, "inference_wall_time_s_mean": 2},
        {"candidate_id": "M0001", "valid": True, "quality_score": .9, "inference_wall_time_s_mean": 2},
    ]
    selected = select_representatives(entries, baseline_quality=.9)
    assert len(selected) == 2
    assert [item.candidate_id for item in selected] == ["M0001", "M0002"]


def test_campaign_round_robins_branches_and_resumes_costs(tmp_path):
    front = [
        {"candidate_id": "M0001", "valid": True, "quality_score": .90, "inference_wall_time_s_mean": 10},
        {"candidate_id": "M0002", "valid": True, "quality_score": .95, "inference_wall_time_s_mean": 20},
        {"candidate_id": "M0003", "valid": True, "quality_score": .99, "inference_wall_time_s_mean": 40},
    ]
    campaign = BranchCampaign(tmp_path, baseline_quality=.90, budget_seconds=100, protocol={"controller": "fixed"})
    first = campaign.next_branch(front)
    resumed_before_round_finishes = BranchCampaign(tmp_path, baseline_quality=.90, budget_seconds=100, protocol={"controller": "fixed"})
    assert resumed_before_round_finishes.next_branch(front) == first
    campaign.record_round(
        round_id="R0001",
        branch_id=first["mode"], parent_model_id=first["head_candidate_id"],
        child_model_id="M0004", status="promoted",
        costs={"generation_seconds": 2, "training_seconds": 3, "evaluation_seconds": 4},
    )
    second = campaign.next_branch(front)
    assert first["mode"] != second["mode"]
    resumed = BranchCampaign(tmp_path, baseline_quality=.90, budget_seconds=100, protocol={"controller": "fixed"})
    assert resumed.remaining_seconds == 91
    assert resumed.state["branches"][first["mode"]]["head_candidate_id"] == "M0004"
    duplicate = resumed.record_round(
        round_id="R0001", branch_id=first["mode"], parent_model_id=first["head_candidate_id"],
        child_model_id="M0004", status="promoted",
        costs={"generation_seconds": 2, "training_seconds": 3, "evaluation_seconds": 4},
    )
    assert duplicate["round_id"] == "R0001"
    assert resumed.remaining_seconds == 91


def test_campaign_rejects_protocol_drift_on_resume(tmp_path):
    BranchCampaign(tmp_path, baseline_quality=.90, budget_seconds=100, protocol={"evaluator": "v1"})
    try:
        BranchCampaign(tmp_path, baseline_quality=.90, budget_seconds=100, protocol={"evaluator": "v2"})
    except ValueError as exc:
        assert "immutable" in str(exc)
    else:
        raise AssertionError("protocol drift should fail closed")
