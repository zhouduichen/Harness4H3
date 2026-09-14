from harness4h3.memory.experience import ExperienceRecord, ExperienceStore


def test_experience_round_trips_and_preserves_unvalidated_status(tmp_path):
    record = ExperienceRecord(
        experience_id="xp-m0006",
        source_uri="ssh://Jiayu-intern/data/m0006.json",
        source_sha256="a" * 64,
        source_kind="trainer_result",
        experiment_id="distill16-m0006",
        parent_model_id="M0005",
        child_model_id="M0006",
        operator="step_distill",
        operator_args={"target_steps": 16},
        training={"optimizer_steps": 1, "gradient_norm": 1.2},
        evaluation=None,
        decision={"status": "not_evaluated"},
        reward=None,
        status="training_only_unvalidated",
        provenance={"remote_path": "/data/m0006.json"},
        created_at="2026-09-14T00:00:00+00:00",
    )
    store = ExperienceStore(tmp_path / "experience.jsonl")
    assert store.append(record) is True
    loaded = list(store.read())
    assert loaded == [record]
    assert store.source_hashes() == {record.source_uri: record.source_sha256}
    assert loaded[0].to_dict()["status"] == "training_only_unvalidated"


def test_experience_store_rejects_duplicate_source_hash(tmp_path):
    record = ExperienceRecord.minimal("xp-1", "ssh://host/result.json", "b" * 64)
    store = ExperienceStore(tmp_path / "experience.jsonl")
    assert store.append(record) is True
    assert store.append(record) is False
    assert len(list(store.read())) == 1
