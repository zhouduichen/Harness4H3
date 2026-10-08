import hashlib
import json
from pathlib import Path

from harness4h3.student.wan_campaign import sha256_file
from harness4h3.memory.experience import ExperienceStore
from tools import wan_vllm_campaign


def _v2(parent_digest: str, manifest_digest: str) -> dict:
    return {
        "schema_version": 2,
        "proposal_id": "wan_1000",
        "parent_checkpoint_sha256": parent_digest,
        "training": {
            "method": "progressive_distillation",
            "source_steps": 32,
            "target_steps": 8,
            "train_steps": 2,
            "learning_rate": 1e-4,
            "critic_learning_rate": 1e-4,
            "velocity_weight": 1.0,
            "clean_anchor_weight": 0.5,
            "rollout_steps": 2,
            "rollout_every": 1,
            "rollout_endpoint_weight": 1.0,
            "rollout_clean_weight": 0.5,
            "trainable_scope": "all",
            "fp32_optimizer_state": False,
        },
        "initialization": {"mode": "inherit"},
        "precision": {"loss_reduction": "fp32", "optimizer_state": "bf16"},
        "curriculum": {"fidelity_schedule": ["F1", "F2", "F3"]},
        "data": {"manifest_digest": manifest_digest, "prompt_policy": "balanced", "seed_policy": "fixed"},
        "continuation": {"overlap_latent_frames": 7, "anchor_schedule": "linear"},
        "focus": "semantic",
        "hypothesis": "Shared rollout improves semantic alignment without weakening continuation.",
    }


def test_v2_policy_reaches_shared_algorithm_and_persists_campaign_state(tmp_path, monkeypatch):
    parent = tmp_path / "parent.safetensors"
    parent.write_bytes(b"parent")
    teacher_baseline = tmp_path / "teacher.json"
    teacher_baseline.write_text(json.dumps({"status": "success", "optimization_metrics": {"semantic": 0.636}}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    clip = tmp_path / "clip.bin"
    clip.write_bytes(b"clip")
    wan_config = tmp_path / "wan.yaml"
    wan_config.write_text("wan_student: {}\n", encoding="utf-8")
    training_entrypoint = tmp_path / "worker.py"
    training_entrypoint.write_text("# trusted fake worker\n", encoding="utf-8")
    output_root = tmp_path / "campaign"
    parent_digest = sha256_file(parent)
    manifest_digest = sha256_file(manifest)
    child = output_root / "round-0001" / "training" / "student.safetensors"

    config = {
        "training_config": str(tmp_path / "training.yaml"),
        "wan_config": str(wan_config),
        "evaluation_manifest": str(manifest),
        "clip_model_path": str(clip),
        "teacher_baseline_result": str(teacher_baseline),
        "parent_checkpoint": str(parent),
        "output_root": str(output_root),
        "worker_python": "/opt/python",
        "training_entrypoint": str(training_entrypoint),
        "vllm_base_url": "http://127.0.0.1:8000/v1",
        "vllm_model": "fake-controller",
        "max_rounds": 1,
        "max_cases": 1,
        "seed_count": 1,
        "algorithm_capabilities": ["progressive_distillation"],
        "dmd2_memory_admitted": False,
        "wan_dmd2_adapter_ready": False,
        "proposal_schema_version": 2,
        "fidelity_schedule": ["F1", "F2", "F3"],
        "continuation_contract": "30s",
        "offline_simulation": False,
        "semantic_floor_ratio": 0.95,
        "quality_floor_ratio": 0.90,
        "min_semantic_delta": 0.005,
        "no_improvement_patience": 2,
        "training_timeout_s": 30,
        "vllm_timeout_s": 30,
        "gpu_wait_s": 1,
    }
    evaluations = [{
        "status": "success",
        "valid": True,
        "promotable": True,
        "offline_simulation": False,
        "continuation_verified": True,
        "quality_score": 0.60,
        "quality_metrics": {"semantic": 0.55},
    }, {
        "status": "success",
        "valid": True,
        "promotable": True,
        "offline_simulation": False,
        "continuation_verified": True,
        "quality_score": 0.66,
        "quality_metrics": {"semantic": 0.61},
    }]

    monkeypatch.setattr(wan_vllm_campaign, "ensure_controller", lambda config: None)
    monkeypatch.setattr(wan_vllm_campaign, "wait_for_gpu_quiet", lambda config: None)
    monkeypatch.setattr(wan_vllm_campaign, "acquire_controller_handoff", lambda *args: None)
    monkeypatch.setattr(wan_vllm_campaign, "release_controller_handoff", lambda *args: None)
    monkeypatch.setattr(wan_vllm_campaign, "_request_vllm", lambda config, context: _v2(parent_digest, manifest_digest))
    monkeypatch.setattr(wan_vllm_campaign, "evaluate_checkpoint", lambda config, checkpoint, output: evaluations.pop(0))

    def fake_training(config, proposal, parent_path, round_dir):
        child.parent.mkdir(parents=True, exist_ok=True)
        child.write_bytes(b"child")
        checkpoint = round_dir / "training" / "training-checkpoint-stage-00.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return {
            "status": "success",
            "offline_simulation": False,
            "student_checkpoint": str(child),
            "student_checkpoint_sha256": sha256_file(child),
            "parent_checkpoint_sha256": sha256_file(parent_path),
            "teacher_checkpoint_sha256": "b" * 64,
            "algorithm_path": "h3_training.algorithms.progressive_distillation.ProgressiveDistillation",
            "training_checkpoint_evidence": str(checkpoint),
            "training_checkpoint_sha256": sha256_file(checkpoint),
            "resume_config_digest": "c" * 64,
        }

    monkeypatch.setattr(wan_vllm_campaign, "run_training", fake_training)
    result = wan_vllm_campaign.run_campaign(config, max_rounds=1)
    assert result["rounds_completed"] == 1
    assert result["accepted_parent_checkpoint_sha256"] == sha256_file(child)
    assert (output_root / "campaign-base.json").is_file()
    assert (output_root / "campaign-state.json").is_file()
    assert (output_root / "fingerprints.jsonl").is_file()
    experiences = list(ExperienceStore(output_root / "experiences.jsonl").read())
    assert len(experiences) == 1
    assert experiences[0].status == "accepted"
    assert experiences[0].operator == "progressive_distillation"
    state = json.loads((output_root / "campaign-state.json").read_text(encoding="utf-8"))
    assert state["parent_checkpoint_sha256"] == sha256_file(child)
