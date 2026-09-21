from __future__ import annotations

from pathlib import Path

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.archive.model_store import ModelStore
from harness4h3.archive.pareto import ParetoArchive
from harness4h3.archive.system_store import SystemCandidateStore
from harness4h3.controller.loop import OptimizationLoop
from harness4h3.controller.schemas import BudgetState, CostEstimate, EvaluationRecord, ExperimentPlan, HardwareMetrics, OperatorResult
from harness4h3.h3.state import ModelState
from harness4h3.memory.experiment_store import ExperimentStore
from harness4h3.operators.base import ExecutionContext, OperatorRegistry
from harness4h3.target.profile import TargetProfile


class ContractRecoveryOperator:
    name = "recovery_finetune"
    description = "contract-only external recovery worker"

    def schema(self):
        return {"training_steps": "int"}

    def validate(self, parent, args, target):
        if set(args) != {"training_steps"} or int(args["training_steps"]) <= 0:
            raise ValueError("training_steps must be positive")

    def estimate_cost(self, parent, args, target):
        return CostEstimate(wall_time_s=1.0, gpu_hours=0.1)

    def execute(self, parent, args, runtime):
        artifacts = Path(runtime.experiment_dir) / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        child_path = artifacts / (runtime.child_model_id + ".safetensors")
        child_path.write_bytes(Path(parent.checkpoint_path).read_bytes() + b"child")
        state = parent.state.derive(
            runtime.child_model_id,
            checkpoint_path=str(child_path),
            provenance={"real_worker": True, "offline_simulation": False, "evidence_kind": "contract"},
        )
        return OperatorResult(
            "success",
            state,
            CostEstimate(wall_time_s=0.1, gpu_hours=0.01),
            artifacts=[str(child_path)],
            metrics={"real_worker": True, "offline_simulation": False, "evidence_kind": "contract"},
        )


class ContractController:
    provider_name = "contract"
    model_name = "contract-controller"

    def __init__(self):
        self.contexts = []

    def plan(self, context):
        self.contexts.append(context)
        number = context.budget_state.used_iterations + 1
        return ExperimentPlan(
            experiment_id="exp_%04d" % number,
            parent_model_id=context.current_model_state.model_id,
            diagnosis="real evidence contract test",
            objective="produce and independently evaluate one child",
            hypothesis="the worker result must be judged by the evaluator",
            operator="recovery_finetune",
            operator_args={"training_steps": number},
            expected_effects={"quality_score": "measure"},
            risks=["quality regression"],
            required_budget={"wall_time_s": 1.0, "gpu_hours": 0.1, "controller_calls": 0},
            acceptance={"max_quality_drop": None, "min_quality_score": None},
            stop_conditions={"critical_regression": False, "target_satisfied": False, "budget_exhausted": True},
            rationale="contract test",
            parent_system_id=str(context.current_system["id"]),
        )


class RejectingBenchmarkEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, state, target, baseline_quality, system=None, **kwargs):
        self.calls.append((state.model_id, system.id if system else None))
        return EvaluationRecord(
            quality_score=0.70 if state.model_id != "M0000" else 0.90,
            quality_metrics={"raw": {"decodable": 1.0}, "real_benchmark": True},
            hardware=HardwareMetrics(latency_s=20.0, peak_memory_gb=4.0, model_size_gb=2.0),
            feasible=False,
            critical_regression=state.model_id != "M0000",
            model_id=state.model_id,
            system_id=system.id if system else None,
            device_id="contract-device",
            task_split="heldout",
            validity={"benchmark_ran": True, "generation_valid": True},
            provenance={"real_benchmark": True, "offline_simulation": False, "evidence_kind": "contract"},
        )


def test_real_loop_rejects_measured_child_and_generates_second_plan(tmp_path):
    parent_path = tmp_path / "M0000.safetensors"
    parent_path.write_bytes(b"parent-checkpoint")
    state = ModelState(
        model_id="M0000",
        parent_model_id=None,
        checkpoint_path=str(parent_path),
        architecture_name="MiniMax-H3-FL2VA",
        measured_metrics={"quality_score": 0.90, "latency_s": 20.0, "peak_memory_gb": 4.0, "model_size_gb": 2.0},
        provenance={"real_h3": True, "offline_simulation": False},
    )
    initial = ModelCandidate("M0000", None, 0, str(parent_path), state, None, "baseline")
    registry = OperatorRegistry()
    registry.register(ContractRecoveryOperator())
    controller = ContractController()
    evaluator = RejectingBenchmarkEvaluator()
    models = ModelStore(tmp_path / "models")
    systems = SystemCandidateStore(tmp_path / "systems", model_store=models)
    loop = OptimizationLoop(
        controller,
        registry,
        evaluator,
        models,
        ParetoArchive(tmp_path / "pareto"),
        ExperimentStore(tmp_path / "experiments.jsonl"),
        tmp_path / "runs",
        tmp_path / "session.json",
        systems=systems,
    )
    result = loop.run(
        "real-contract",
        TargetProfile("gpu", "gpu", "L40", max_quality_drop=0.05),
        BudgetState(max_iterations=2, max_failed_experiments=2, max_gpu_hours=1, max_controller_calls=2),
        initial,
    )

    assert result.status == "no_valid_plan"
    assert result.current_model_id == "M0000"
    assert result.current_system_id == "S0000"
    assert [item.current_model_state.model_id for item in controller.contexts] == ["M0000", "M0000"]
    assert all(item.current_system["id"] == "S0000" for item in controller.contexts)
    records = list(loop.experiments.read())
    assert len(records) == 2
    assert records[0].evaluation["evaluation_id"] == "E0001"
    assert records[1].evaluation["evaluation_id"] == "E0002"
    assert all(record.fingerprint for record in records)
    assert len({record.fingerprint for record in records}) == 2
    assert all(item.evaluation["provenance"]["offline_simulation"] is False for item in records if item.evaluation)
    assert all(item.decision["status"] == "reject" for item in records)
    assert len(loop.models.lineage()) == 3
    assert len(loop.systems.lineage()) == 3
    assert (tmp_path / "runs" / "exp_0002" / "controller_request.json").is_file()
