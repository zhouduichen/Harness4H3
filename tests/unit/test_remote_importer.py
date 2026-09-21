from harness4h3.memory.experience import ExperienceStore
from harness4h3.remote.importer import RemoteResultImporter, normalize_trainer_result


def test_successful_training_result_is_unvalidated_without_evaluation():
    record = normalize_trainer_result(
        "ssh://Jiayu-intern/data/m0006.json",
        "c" * 64,
        {
            "status": "success",
            "metrics": {"optimizer_steps": 1, "gradient_norm": 0.2, "child_sha256": "d" * 64},
            "cost": {"wall_time_s": 12.0, "gpu_hours": 1.5},
            "output_state": {
                "model_id": "M0006",
                "parent_model_id": "M0005",
                "checkpoint_path": "/data/models/MiniMax-H3/harness4h3/distill16/M0006.safetensors",
                "algorithm_state": {"operator": "step_distill", "target_steps": 16},
            },
        },
    )
    assert record.status == "training_only_unvalidated"
    assert record.operator == "step_distill"
    assert record.reward is None
    assert record.training["cost"]["gpu_hours"] == 1.5


def test_failed_result_preserves_failure_and_does_not_create_child():
    record = normalize_trainer_result(
        "ssh://host/failure.json",
        "e" * 64,
        {"status": "failed", "failure_type": "training_oom", "message": "OOM"},
    )
    assert record.status == "failed"
    assert record.child_model_id is None
    assert record.decision["failure_type"] == "training_oom"


def test_importer_is_idempotent_and_keeps_corrupt_items_separate(tmp_path):
    store = ExperienceStore(tmp_path / "experience.jsonl")
    importer = RemoteResultImporter(store)
    item = {
        "source_uri": "ssh://host/result.json",
        "source_sha256": "f" * 64,
        "result": {"status": "success", "output_state": {"model_id": "M0001"}},
    }
    first = importer.import_results([item, {"source_uri": "ssh://host/bad.json", "source_sha256": "bad", "result": {}}])
    second = importer.import_results([item])
    assert first.imported == 1
    assert len(first.corrupt) == 1
    assert second.skipped_duplicates == 1
    assert len(list(store.read())) == 1
