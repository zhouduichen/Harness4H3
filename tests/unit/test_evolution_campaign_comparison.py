import json

import pytest

from tools.compare_evolution_campaigns import compare


def _campaign(root, protocol_hash="same"):
    root.mkdir()
    (root / "branch-campaign.json").write_text(json.dumps({
        "protocol_sha256": protocol_hash,
        "budget_seconds": 100,
        "baseline_quality": .9,
    }))
    (root / "branch-events.jsonl").write_text(
        json.dumps({"round_id": "R0001", "status": "rejected", "child_model_id": "M0001", "total_seconds": 8,
                    "costs_seconds": {"generation_seconds": 1, "training_seconds": 3, "evaluation_seconds": 4, "retry_seconds": 0}}) + "\n" +
        json.dumps({"round_id": "R0002", "status": "promoted", "child_model_id": "M0002", "total_seconds": 12,
                    "costs_seconds": {"generation_seconds": 2, "training_seconds": 4, "evaluation_seconds": 6, "retry_seconds": 0}}) + "\n"
    )
    (root / "pareto-front.json").write_text(json.dumps({"entries": [{"candidate_id": "M0002"}]}))


def test_comparison_checks_matched_protocol_and_reports_costs(tmp_path):
    single, pareto = tmp_path / "single", tmp_path / "pareto"
    _campaign(single)
    _campaign(pareto)
    (pareto / "accounting-reconciliation.jsonl").write_text(json.dumps({
        "round_id": "R0001", "measured_round_wall_seconds": 15,
    }) + "\n")
    result = compare(single, pareto)
    assert result["single"]["first_qualified"]["cumulative_search_seconds"] == 20
    assert result["pareto"]["failed_or_rejected_seconds"] == 15
    assert result["single"]["pareto_front"][0]["candidate_id"] == "M0002"
    assert result["matched_compute_seconds"] == 20
    assert result["pareto_at_matched_compute"]["rounds"] == 1
    assert result["pareto_at_matched_compute"]["first_qualified"] is None
    assert result["pareto_at_matched_compute"]["failed_or_rejected_seconds"] == 15


def test_comparison_rejects_protocol_mismatch(tmp_path):
    single, pareto = tmp_path / "single", tmp_path / "pareto"
    _campaign(single)
    _campaign(pareto, "changed")
    with pytest.raises(ValueError, match="not matched"):
        compare(single, pareto)
