"""Deterministic real-PyTorch model-evolution loop using TinyH3.

The experiment validates the training and Harness contracts on CPU. It makes
no claim about MiniMax-H3 checkpoint compatibility or achievable quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.archive.model_store import ModelStore
from harness4h3.controller.schemas import EvaluationResult, ExperimentPlan, HardwareMetrics
from harness4h3.h3.state import ModelState
from harness4h3.operators.model_evolution import build_external_model_evolution_registry
from harness4h3.target.profile import TargetProfile
from h3_training.tiny.evaluator import TinyCheckpointEvaluator
from h3_training.tiny.factory import create_tiny_checkpoint
from research.experiments.a0_model_evolution import A0Budget, A0CampaignResult, run_campaign


ROOT = Path(__file__).resolve().parents[2]


class TinySequenceController:
    provider_name = "tiny-deterministic"
    model_name = "recovery-then-binary-distillation"

    def plan(self, context) -> ExperimentPlan:
        number = context.budget_state.used_iterations + 1
        if number == 1:
            operator = "recovery_finetune"
            args = {"training_steps": 4}
            diagnosis = "validate real optimizer and checkpoint lineage"
            estimated_wall_time, estimated_gpu_hours = 3600.0, 2.0
        else:
            operator = "step_distill"
            current_nfe = int(context.current_model_state.sampling_steps)
            args = {"target_steps": current_nfe // 2}
            diagnosis = "validate one evaluator-gated binary distillation stage"
            estimated_wall_time, estimated_gpu_hours = 7200.0, 4.0
        return ExperimentPlan(
            experiment_id=f"exp_{number:04d}",
            parent_model_id=context.current_model_state.model_id,
            diagnosis=diagnosis,
            objective="exercise a real PyTorch child through the frozen Harness contract",
            hypothesis=f"{operator} will produce a distinct reloadable TinyH3 child",
            operator=operator,
            operator_args=args,
            expected_effects={"checkpoint": "changed", "lineage": "extended"},
            risks=["reference-model-only", "quality regression"],
            required_budget={
                "wall_time_s": estimated_wall_time,
                "gpu_hours": estimated_gpu_hours,
                "controller_calls": 0,
                "tier": 1,
            },
            acceptance={"max_quality_drop": 1.0},
            stop_conditions={"critical_regression": True, "budget_exhausted": True},
            rationale="use the smallest real model that exercises training, evidence, and worker boundaries",
        )


class TinyCampaignEvaluator:
    def __init__(self, seed: int) -> None:
        self.inner = TinyCheckpointEvaluator(dataset_size=4, base_seed=seed + 20_000)

    def evaluate(self, state: ModelState, target: TargetProfile, baseline_quality: float) -> EvaluationResult:
        measured = self.inner.evaluate_checkpoint(Path(state.checkpoint_path), target=None)
        metrics = dict(measured.metrics)
        hardware = HardwareMetrics(
            latency_s=float(metrics["latency_s"]),
            peak_memory_gb=0.0,
            model_size_gb=float(metrics["model_size_gb"]),
            throughput=1.0 / max(float(metrics["latency_s"]), 1e-12),
        )
        return EvaluationResult(
            quality_score=measured.score,
            quality_metrics={**metrics, "tiny_reference": True, "offline_simulation": False},
            hardware=hardware,
            feasible=True,
            violations=[],
            critical_regression=False,
        )


@dataclass(frozen=True)
class TinyClosedLoopResult:
    campaign: A0CampaignResult
    model_store: ModelStore

    @property
    def report(self) -> Mapping[str, Any]:
        return self.campaign.report

    @property
    def status(self) -> str:
        return self.campaign.status


def run_tiny_real_closed_loop(
    output_root: Path,
    max_experiments: int = 2,
    seed: int = 11,
) -> TinyClosedLoopResult:
    if max_experiments != 2:
        raise ValueError("the reference closed loop is intentionally fixed to two experiments")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    initial_path = create_tiny_checkpoint(
        output_root / "initial" / "M0000.pt", "M0000", sampling_nfe=4, seed=seed
    )
    initial_evaluation = TinyCheckpointEvaluator(dataset_size=4, base_seed=seed + 20_000).evaluate_checkpoint(initial_path)
    initial_state = ModelState(
        model_id="M0000",
        parent_model_id=None,
        checkpoint_path=str(initial_path),
        architecture_name="TinyH3",
        parameter_count=None,
        trainable_parameter_count=0,
        num_blocks=2,
        hidden_size=32,
        num_attention_heads=4,
        ffn_width=64,
        dtype="float32",
        sampling_steps=4,
        components={"video": True, "audio": True},
        algorithm_state={"algorithm": "untrained_tiny_baseline"},
        runtime_state={"backend": "tiny_cpu_reference", "metrics_stale": False},
        measured_metrics=dict(initial_evaluation.metrics),
        provenance={"kind": "tiny_reference", "offline_simulation": False},
    )
    initial = ModelCandidate("M0000", None, 0, str(initial_path), initial_state, None, "baseline")
    worker_config = output_root / "tiny-worker.json"
    worker_config.write_text(
        json.dumps(
            {
                "trainer_command": [
                    sys.executable,
                    str(ROOT / "tools" / "tiny_training_worker.py"),
                    "--max-training-steps",
                    "4",
                    "--dataset-size",
                    "4",
                    "--seed",
                    str(seed + 100),
                ],
                "trainer_timeout_s": 60,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    registry = build_external_model_evolution_registry(
        (sys.executable, str(ROOT / "tools" / "h3_model_worker.py"), "--config", str(worker_config)),
        timeout_s=60,
    )
    target = TargetProfile("tiny_cpu_contract", "cpu", "TinyH3 reference CPU")
    campaign = run_campaign(
        target,
        TinySequenceController(),
        registry,
        TinyCampaignEvaluator(seed),
        output_root,
        initial_candidate=initial,
        budget=A0Budget(max_gpu_hours=16.0, max_experiments=2, max_failed_experiments=2),
        stop_on_target=False,
        offline_simulation=False,
    )
    return TinyClosedLoopResult(campaign, ModelStore(output_root / "models"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="run the real-PyTorch TinyH3 Harness loop")
    parser.add_argument("--output-root", default="var/tiny-real-closed-loop")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    result = run_tiny_real_closed_loop(Path(args.output_root), seed=args.seed)
    print(json.dumps({"status": result.status, "report": result.report}, indent=2, sort_keys=True))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
