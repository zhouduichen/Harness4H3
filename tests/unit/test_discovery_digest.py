import json

from harness4h3.memory.discovery_digest import DiscoveryDigest


def test_digest_keeps_failure_recipe_and_bounds_long_history():
    records = [
        {
            "experiment_id": "exp-%04d" % index,
            "plan": {"operator": "lpl", "operator_args": {"target_steps": 8}},
            "execution": {"status": "failed" if index == 0 else "success"},
            "failure_type": "oom" if index == 0 else None,
            "evaluation": {"quality_score": 0.8 + index / 1000.0},
            "created_at": "2026-09-18T00:%02d:00+00:00" % index,
            "checkpoint_path": "/data/secret/full-model.safetensors",
            "stdout": "x" * 10000,
        }
        for index in range(40)
    ]
    digest = DiscoveryDigest.build(records)
    context = digest.to_context()
    assert len(context["recent_experiments"]) == 8
    assert any(item["experiment_id"] == "exp-0000" for item in context["operator_findings"]["lpl"])
    assert all(len(json.dumps(item, ensure_ascii=False)) <= 2048 for item in context["recent_experiments"])
    encoded = json.dumps(context, ensure_ascii=False).lower()
    assert "checkpoint_path" not in encoded
    assert "full-model.safetensors" not in encoded
    assert "source_digest" in context


def test_digest_records_observation_ids_and_failure_counts():
    digest = DiscoveryDigest.build(
        [{"experiment_id": "e1", "operator": "quantize", "failure_type": "worker_oom"}],
        observations=[{"observation_id": "obs-1"}, {"observation_id": "obs-1"}, {"observation_id": "obs-2"}],
    )
    context = digest.to_context()
    assert context["source_observation_ids"] == ["obs-1", "obs-2"]
    assert context["failure_counts"] == {"worker_oom": 1}
    assert context["operator_findings"]["quantize"][0]["experiment_id"] == "e1"


def test_digest_bounds_observation_id_history():
    observations = [{"observation_id": "obs-%02d" % index} for index in range(40)]

    context = DiscoveryDigest.build([], observations=observations).to_context()

    assert len(context["source_observation_ids"]) == 24
    assert context["source_observation_ids"][0] == "obs-16"
    assert context["source_observation_ids"][-1] == "obs-39"
