#!/usr/bin/env python3
"""Trusted CPU reference trainer implementing the H3 worker JSON contract."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from h3_training.adapters.tiny import TinyH3Adapter
from h3_training.algorithms.progressive_distillation import (
    DistillationStage,
    ProgressiveDistillation,
    ProgressiveDistillationConfig,
)
from h3_training.algorithms.recovery_finetune import RecoveryConfig, RecoveryFineTune
from h3_training.data.dataset import SyntheticH3Dataset
from h3_training.engine.evidence import capture_parent, save_verified_child, sha256_file
from h3_training.engine.state import TrainingFailure
from h3_training.engine.trainer import TrainerConfig, TrainerEngine
from h3_training.tiny.evaluator import TinyCheckpointEvaluator
from h3_training.tiny.factory import load_tiny_checkpoint


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _positive_int(value: Any, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
        raise TrainingFailure("invalid_training_config", f"{name} must be in [1, {maximum}]")
    return value


def run(request: Mapping[str, Any], result_path: Path, maximum_steps: int, dataset_size: int, seed: int) -> int:
    started = time.perf_counter()
    try:
        operator = str(request.get("operator", ""))
        if operator not in {"recovery_finetune", "step_distill"}:
            raise TrainingFailure("unsupported_training_operator", operator)
        parent = request.get("parent")
        args = request.get("operator_args")
        if not isinstance(parent, Mapping) or not isinstance(args, Mapping):
            raise TrainingFailure("invalid_training_config", "parent and operator_args are required")
        parent_path = Path(str(parent.get("checkpoint_path", ""))).resolve()
        if not parent_path.is_file():
            raise TrainingFailure("checkpoint_corrupt", "parent checkpoint is missing")
        parent_hash = sha256_file(parent_path)
        parent_model, metadata = load_tiny_checkpoint(parent_path)
        adapter = TinyH3Adapter()
        teacher = copy.deepcopy(parent_model)
        if operator == "recovery_finetune":
            requested_steps = _positive_int(args.get("training_steps"), "training_steps", 10_000_000)
            training_steps = min(requested_steps, maximum_steps)
            method = RecoveryFineTune(
                parent_model,
                adapter,
                RecoveryConfig(learning_rate=3e-3, trainable_scope="heads", drift_weight=0.05),
                teacher,
                metadata,
            )
            child_nfe = int(metadata["sampling_nfe"])
        else:
            target_nfe = _positive_int(args.get("target_steps"), "target_steps", 1024)
            teacher_nfe = int(metadata["sampling_nfe"])
            try:
                stage = DistillationStage(teacher_nfe, target_nfe)
            except ValueError as exc:
                raise TrainingFailure("invalid_training_config", str(exc)) from exc
            training_steps = maximum_steps
            requested_steps = maximum_steps
            method = ProgressiveDistillation(
                parent_model,
                teacher,
                adapter,
                ProgressiveDistillationConfig(stage=stage, learning_rate=2e-3, trainable_scope="all"),
                metadata,
            )
            child_nfe = target_nfe
        child_id = str(request.get("child_model_id", ""))
        parent_id = str(parent.get("id", ""))
        if not child_id or not parent_id or child_id == parent_id:
            raise TrainingFailure("invalid_training_config", "distinct parent and child IDs are required")
        dataset = SyntheticH3Dataset(dataset_size, base_seed=seed, config=parent_model.config)
        batches = [[dataset[index]] for index in range(len(dataset))]
        engine = TrainerEngine(
            TrainerConfig(seed=seed + 1, parent_sha256=parent_hash, max_gradient_norm=10.0)
        )
        method.prepare()
        evidence = capture_parent(parent_path, method.student, method.trainable_parameter_names)
        result = engine.run(method, batches, max_steps=training_steps)
        method.student.scheduler_state.update(
            model_id=child_id,
            parent_id=parent_id,
            sampling_nfe=child_nfe,
            provenance={
                "kind": "tiny_reference_training",
                "real_worker": True,
                "offline_simulation": False,
                "operator": operator,
            },
        )
        child_path = Path(str(request["artifacts_dir"])).resolve() / "trainer-child.pt"
        child = save_verified_child(method.student, evidence, child_path)
        evaluation = TinyCheckpointEvaluator(dataset_size=4, base_seed=seed + 10_000).evaluate_checkpoint(child_path)
        parent_state = dict(parent.get("state") or {})
        parameter_count = sum(parameter.numel() for parameter in method.student_model.parameters())
        trainable_parameter_count = sum(
            parameter.numel()
            for name, parameter in method.student_model.named_parameters()
            if name in method.trainable_parameter_names
        )
        output_state = {
            **parent_state,
            "model_id": child_id,
            "parent_model_id": parent_id,
            "checkpoint_path": str(child_path),
            "architecture_name": "TinyH3",
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "num_blocks": parent_model.config.num_layers,
            "hidden_size": parent_model.config.hidden_size,
            "num_attention_heads": parent_model.config.num_heads,
            "ffn_width": parent_model.config.ffn_width,
            "dtype": "float32",
            "sampling_steps": child_nfe,
            "algorithm_state": {
                "algorithm": method.algorithm_name,
                "requested_training_steps": requested_steps,
                "effective_training_steps": training_steps,
            },
            "runtime_state": {"backend": "tiny_cpu_reference", "metrics_stale": False},
            "measured_metrics": dict(evaluation.metrics),
            "provenance": dict(method.student.scheduler_state["provenance"]),
        }
        _write(
            result_path,
            {
                "status": "success",
                "output_state": output_state,
                "cost": {"wall_time_s": time.perf_counter() - started, "gpu_hours": 0.0, "controller_calls": 0},
                "metrics": {
                    "real_worker": True,
                    "offline_simulation": False,
                    "algorithm": method.algorithm_name,
                    "optimizer_steps": result.loop_state.global_step,
                    "optimizer_steps_by_role": result.optimizer_steps,
                    "initial_loss": result.initial_loss,
                    "final_loss": result.final_loss,
                    "max_gradient_norm": result.max_gradient_norm,
                    "parent_sha256": child.parent_sha256,
                    "child_sha256": child.child_sha256,
                    "changed_trainable_tensors": child.changed_trainable,
                    "unchanged_frozen_tensors": child.unchanged_frozen,
                    "child_reloaded": child.reloaded,
                    "evaluation": dict(evaluation.metrics),
                },
            },
        )
        return 0
    except TrainingFailure as exc:
        _write(
            result_path,
            {
                "status": "failed",
                "failure_type": exc.code,
                "message": str(exc),
                "cost": {"wall_time_s": time.perf_counter() - started, "gpu_hours": 0.0, "controller_calls": 0},
                "metrics": {"real_worker": True, "offline_simulation": False},
            },
        )
        return 1
    except Exception as exc:
        _write(
            result_path,
            {
                "status": "failed",
                "failure_type": "checkpoint_corrupt",
                "message": str(exc),
                "cost": {"wall_time_s": time.perf_counter() - started, "gpu_hours": 0.0, "controller_calls": 0},
                "metrics": {"real_worker": True, "offline_simulation": False},
            },
        )
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="real TinyH3 reference training worker")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--max-training-steps", type=int, default=8)
    parser.add_argument("--dataset-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=700)
    args = parser.parse_args(argv)
    result_path = Path(args.result).resolve()
    try:
        if args.max_training_steps <= 0 or args.dataset_size <= 0:
            raise TrainingFailure("invalid_training_config", "worker limits must be positive")
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        if not isinstance(request, Mapping):
            raise TrainingFailure("invalid_training_config", "request must be an object")
        return run(request, result_path, args.max_training_steps, args.dataset_size, args.seed)
    except TrainingFailure as exc:
        _write(result_path, {"status": "failed", "failure_type": exc.code, "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
