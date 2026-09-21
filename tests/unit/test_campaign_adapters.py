from __future__ import annotations

from pathlib import Path

from harness4h3.campaign.adapters import StudentCampaignAdapter
from harness4h3.campaign.adapters import LegacyH3CampaignAdapter
from harness4h3.operators.model_evolution import build_model_evolution_registry
from harness4h3.target.profile import TargetProfile
from harness4h3.student.evaluator import StudentEvaluation
from harness4h3.student.worker import TrainingResult
from tests.unit.test_student_proposal import valid_payload


class FakeCompiler:
    target = None

    def compile(self, proposal, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return type(
            "Manifest",
            (),
            {
                "graph_status": "compiled",
                "parameter_count": 1_200_000_000,
                "manifest_digest": "sha256:manifest",
                "proposal_digest": proposal.digest,
                "to_dict": lambda self: {
                    "graph_status": self.graph_status,
                    "parameter_count": self.parameter_count,
                    "manifest_digest": self.manifest_digest,
                    "proposal_digest": self.proposal_digest,
                },
            },
        )()


class FakeWorker:
    def run(self, manifest, round_dir):
        child = Path(round_dir) / "student.safetensors"
        child.parent.mkdir(parents=True, exist_ok=True)
        child.write_bytes(b"child")
        return TrainingResult(
            status="success", proposal_digest=manifest.proposal_digest,
            compiler_digest=manifest.manifest_digest, parent_sha256="0" * 64,
            child_sha256="1" * 64, child_checkpoint=str(child), optimizer_steps=1,
            initial_loss=1.0, final_loss=0.5, gradient_norm=1.0, wall_time_s=0.1,
            peak_memory_gb=4.0, changed_parameter_count=1, offline_simulation=True,
        )


class FakeEvaluator:
    def evaluate(self, checkpoint, round_dir):
        return StudentEvaluation(True, True, None, "evaluation_ok", str(checkpoint), quality_score=0.8)


def test_student_adapter_maps_compile_train_eval_to_shared_evidence(tmp_path):
    adapter = StudentCampaignAdapter(FakeCompiler(), FakeWorker(), FakeEvaluator())
    result = adapter.run_proposal(valid_payload(), tmp_path / "round", fidelity="F1")
    assert result["v1"]["graph_status"] == "compiled"
    assert result["v2"]["status"] == "success"
    assert result["v3"]["video_decodable"] is True
    evidence = adapter.verify(None, result, tmp_path / "round")
    assert any(item.metric_name == "video_decodable" for item in evidence)


def test_legacy_adapter_persists_immutable_base_and_trace(tmp_path):
    class Controller:
        provider_name = "controller"
        model_name = "model"

    class Evaluator:
        version = "evaluator-v1"

    target = TargetProfile("edge", "mobile", "scripted")
    adapter = LegacyH3CampaignAdapter(build_model_evolution_registry(), tmp_path, target)
    trace = adapter.trace("session-1", Controller(), Evaluator())
    assert trace.read()[0].event_type == "campaign.created"
    assert (tmp_path / "campaign-session-1-base.json").is_file()
    assert trace.base.target_profile_hash
