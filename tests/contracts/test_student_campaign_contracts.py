from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from harness4h3.campaign.base import ActorIdentity, CampaignBase, CampaignBaseError, canonical_digest, sha256_path
from harness4h3.campaign.capabilities import Capability, CapabilitySnapshot
from harness4h3.campaign.gates import AcceptanceGate, MetricEvidence
from harness4h3.campaign.reviews import ReviewPipeline, review_json_schema
from harness4h3.student.campaign import StudentCampaign, student_proposal_json_schema
from harness4h3.student.fidelity import FidelityGate, stage_spec
from harness4h3.student.proposal import ProposalValidationError, StudentProposal, StudentTarget
from harness4h3.student.edge import FakeTargetDeviceRunner, TargetDeviceEvaluator
from harness4h3.student.target import TargetDeviceProfile
from harness4h3.student.worker import StudentTrainWorker, TrainingResult
from tests.unit.test_student_proposal import valid_payload


def _base() -> CampaignBase:
    target = {"id": "contract-target"}
    verifier = {"version": "contract-verifier"}
    capabilities = {"capabilities": []}
    return CampaignBase(
        schema_version=1,
        campaign_id="contract-campaign",
        target_profile=target,
        target_profile_hash=canonical_digest(target),
        verifier_bank=verifier,
        verifier_bank_hash=canonical_digest(verifier),
        dataset_manifest_hash=canonical_digest({"manifest": "contract"}),
        evaluation_recipe_hash=canonical_digest({"recipe": "contract"}),
        controller_identity=ActorIdentity("controller", "model", "v1"),
        critic_identity=ActorIdentity("critic", "model", "v1"),
        evaluator_identity=ActorIdentity("evaluator", "fixed", "v1"),
        prompt_version="contract-prompt-v1",
        capability_snapshot=capabilities,
        capability_snapshot_hash=canonical_digest(capabilities),
    )


def _candidate_payload(index: int) -> dict:
    payload = valid_payload(depth=24)
    payload["proposal_id"] = "student_%04d" % index
    payload["training"].pop("max_steps", None)
    payload["training"].pop("batch_size", None)
    return payload


def test_controller_action_schema_has_no_inert_budget_actions_and_normalizes_method():
    schema = student_proposal_json_schema(StudentTarget())
    training = schema["properties"]["training"]["properties"]
    assert "max_steps" not in training
    assert "batch_size" not in training

    payload = _candidate_payload(1)
    proposal = StudentProposal.from_dict(payload)
    assert proposal.training.method == "dmd2"
    assert "max_steps" not in proposal.to_dict()["training"]
    assert "batch_size" not in proposal.to_dict()["training"]


def test_progressive_distillation_binary_halving_is_checked_before_training():
    payload = _candidate_payload(1)
    payload["training"].update({"method": "progressive_distillation", "source_steps": 12, "target_steps": 4})
    with pytest.raises(ProposalValidationError, match="binary halving"):
        StudentProposal.from_dict(payload)
    payload["training"].update({"source_steps": 16, "target_steps": 4})
    assert StudentProposal.from_dict(payload).training.method == "progressive_distillation"


def test_fidelity_is_cumulative_and_gate_failure_is_fail_closed():
    f1, f2, f3 = (stage_spec(256, name) for name in ("F1", "F2", "F3"))
    assert (f1.train_steps, f2.train_steps, f3.train_steps) == (64, 64, 128)
    assert (f1.cumulative_train_steps, f2.cumulative_train_steps, f3.cumulative_train_steps) == (64, 128, 256)
    assert (f1.evaluation_cases, f2.seed_count, f3.verifier_strength) == (1, 2, "full")

    evidence = (
        MetricEvidence("video_decodable", "v1", "video", 1.0, True, "evaluator", "server", True),
        MetricEvidence("parent_checkpoint_bound", "v1", "parent", 1.0, True, "worker", "server", True),
        MetricEvidence("fidelity_executed", "v1", "f1", 1.0, True, "worker", "server", True),
    )
    decision = FidelityGate().evaluate(f1, evidence)
    assert decision.passed is False
    assert decision.violations == ("algorithm_dispatch",)


def test_campaign_does_not_start_f2_after_f1_gate_failure(tmp_path):
    from tests.integration.test_campaign_control_plane import Advocate, BatchProvider, Critical, FakeCompiler, FakeEvaluator, Modifier

    capabilities = CapabilitySnapshot((Capability("dmd2", "training", "scripted", {}, "V4", True, ""),))
    base = replace(
        _base(),
        critic_identity=Critical.identity,
        capability_snapshot=capabilities.to_dict(),
        capability_snapshot_hash=capabilities.digest,
    )

    class FailF1Worker:
        def __init__(self):
            self.fidelities = []

        def run(self, manifest, round_dir, *, fidelity="F1", **kwargs):
            self.fidelities.append(fidelity)
            checkpoint = Path(round_dir) / "student.safetensors"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"child")
            return TrainingResult(
                status="success", proposal_digest=manifest.proposal_digest, compiler_digest=manifest.manifest_digest,
                parent_sha256="a" * 64, child_sha256="b" * 64, child_checkpoint=str(checkpoint),
                optimizer_steps=1, initial_loss=1.0, final_loss=0.5, gradient_norm=1.0,
                wall_time_s=0.1, peak_memory_gb=1.0, changed_parameter_count=1,
                offline_simulation=False, algorithm_dispatch="failed",
            )

    worker = FailF1Worker()
    result = StudentCampaign(
        BatchProvider(), FakeCompiler(), worker, FakeEvaluator(), output_root=tmp_path,
        campaign_base=base, capability_snapshot=capabilities,
        review_pipeline=ReviewPipeline(Advocate(), Critical(), Modifier(), base, max_rounds=1),
        fidelity_schedule=("F1", "F2", "F3"),
    ).run(max_rounds=1)
    assert result.status == "BUDGET_EXHAUSTED"
    assert worker.fidelities and set(worker.fidelities) == {"F1"}


def test_edge_evidence_is_required_for_target_satisfied(tmp_path: Path):
    candidate = _candidate_envelope()
    evidence = {
        "quality": MetricEvidence("quality", "v1", "quality", 0.8, True, "server", "server"),
        "video_decodable": MetricEvidence("video_decodable", "v1", "video", 1.0, True, "server", "server", True),
        "evaluation_promotable": MetricEvidence("evaluation_promotable", "v1", "evaluation", 1.0, True, "server", "server", True),
    }
    gate = AcceptanceGate()
    result = gate.evaluate(candidate, evidence, hard_constraints={"min_quality_score": 0.7}, objectives={"quality": "maximize"}, min_rounds_met=True)
    assert result.promotable is True
    assert result.target_satisfied is False
    checkpoint = tmp_path / "contract-student.safetensors"
    checkpoint.write_bytes(b"contract-student")
    try:
        edge = TargetDeviceEvaluator(FakeTargetDeviceRunner(), target_device_id="edge-device").evaluate(
            checkpoint, {"proposal_digest": "contract"}, tmp_path / "contract-edge"
        )
        evidence.update({item.metric_name: item.to_metric_evidence() for item in edge})
    finally:
        checkpoint.unlink(missing_ok=True)
    profile = TargetDeviceProfile(
        id="edge-device",
        runtime_backend="fake-runtime",
        max_latency_s=0.02,
        max_memory_gb=1.0,
        max_energy_j=2.0,
        max_thermal_c=60.0,
        max_model_size_gb=1.0,
        supported_precision=("bf16",),
        supported_quantization=("none",),
        resolution=(512, 512),
        frames=5,
        sampling_steps=1,
    )
    result = gate.evaluate(candidate, evidence, hard_constraints={"min_quality_score": 0.7}, objectives={"quality": "maximize"}, min_rounds_met=True, target_device_profile=profile)
    assert result.target_satisfied is True


def test_campaign_base_rejects_content_hash_change_and_file_identity_changes(tmp_path):
    base = _base()
    tampered = base.to_dict()
    tampered["target_profile"]["id"] = "changed"
    with pytest.raises(CampaignBaseError, match="immutable content"):
        CampaignBase.from_dict(tampered)

    content = tmp_path / "manifest.json"
    content.write_text("one\n", encoding="utf-8")
    first = sha256_path(content)
    content.write_text("two\n", encoding="utf-8")
    assert sha256_path(content) != first


def test_parent_inheritance_reports_parameter_counts_and_real_sha(tmp_path):
    student = torch.nn.Linear(2, 2, bias=True)
    parent = tmp_path / "parent.safetensors"
    save_file({"weight": torch.ones_like(student.weight)}, str(parent))
    parent_sha, inherited, total = StudentTrainWorker._inherit_parent(student, parent)
    assert parent_sha == __import__("hashlib").sha256(parent.read_bytes()).hexdigest()
    assert inherited == student.weight.numel()
    assert total == student.weight.numel() + student.bias.numel()


def test_review_roles_have_closed_schemas_and_opposite_contracts():
    advocate = review_json_schema("advocate")
    critical = review_json_schema("critical")
    revision = review_json_schema("revision")
    assert advocate["additionalProperties"] is False
    assert critical["additionalProperties"] is False
    assert revision["additionalProperties"] is False
    assert "hard_objection" in critical["required"]
    assert "base_digest" in revision["required"]


def _candidate_envelope():
    from harness4h3.campaign.proposals import CandidateEnvelope

    return CandidateEnvelope(
        candidate_id="candidate-1",
        parent_candidate_id="M0000",
        generation=1,
        experiment_id="experiment-1",
        proposal_digest="proposal-1",
        mutation_fields=("training.method",),
        architecture={"family": "test"},
        training_recipe={"method": "dmd2"},
        deployment_recipe={"precision": "bf16"},
        provenance={"source": "contract"},
        predicted_metric_delta={"quality": 0.1},
    )
