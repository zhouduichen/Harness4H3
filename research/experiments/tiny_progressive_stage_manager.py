"""Formal TinyH3 4->2->1 progressive-distillation experiment.

The stage manager owns teacher promotion and the evaluator gates every
promotion. This is intentionally a separate experiment from the generic
Controller campaign, so the multi-stage protocol cannot be accidentally
reduced to a temporary sequence of Controller decisions.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.h3.state import ModelState
from harness4h3.operators.base import ExecutionContext
from harness4h3.operators.model_evolution import build_external_model_evolution_registry
from harness4h3.target.profile import TargetProfile
from h3_training.algorithms.progressive_stage_manager import ProgressiveStageManager
from h3_training.tiny.evaluator import TinyCheckpointEvaluator
from h3_training.tiny.factory import create_tiny_checkpoint


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TinyProgressiveStageResult:
    manager: Mapping[str, Any]
    stages: List[Mapping[str, Any]]
    final_model_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_tiny_progressive_stage_experiment(
    output_root: Path,
    seed: int = 31,
) -> TinyProgressiveStageResult:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    initial_path = create_tiny_checkpoint(output_root / "initial" / "M0000.pt", "M0000", sampling_nfe=4, seed=seed)
    evaluator = TinyCheckpointEvaluator(dataset_size=4, base_seed=seed + 20_000)
    initial_eval = evaluator.evaluate_checkpoint(initial_path)
    initial_state = ModelState(
        model_id="M0000",
        parent_model_id=None,
        checkpoint_path=str(initial_path),
        architecture_name="TinyH3",
        num_blocks=2,
        hidden_size=32,
        num_attention_heads=4,
        ffn_width=64,
        dtype="float32",
        sampling_steps=4,
        components={"video": True, "audio": True},
        algorithm_state={"algorithm": "untrained_tiny_baseline"},
        runtime_state={"backend": "tiny_cpu_reference", "metrics_stale": False},
        measured_metrics=dict(initial_eval.metrics),
        provenance={"kind": "tiny_reference", "offline_simulation": False},
    )
    current = ModelCandidate("M0000", None, 0, str(initial_path), initial_state, None, "baseline")

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
    target = TargetProfile("tiny_progressive_cpu", "cpu", "TinyH3 progressive reference")
    manager = ProgressiveStageManager(4, 1, current.id, current.checkpoint_path)
    records: List[Mapping[str, Any]] = []
    while not manager.complete:
        request = manager.next_stage_request()
        assert request is not None
        stage_index = int(request["stage_index"])
        child_id = f"M{stage_index + 1:04d}"
        experiment_dir = output_root / f"stage-{stage_index:02d}"
        result = registry.execute(
            "step_distill",
            current,
            {"target_steps": int(request["student_nfe"])},
            target,
            ExecutionContext(experiment_dir, child_id),
        )
        if not result.ok or result.output_state is None:
            raise RuntimeError(f"stage {stage_index} failed: {result.failure_type}: {result.message}")
        child_state = result.output_state
        child = ModelCandidate(
            child_id,
            current.id,
            current.generation + 1,
            child_state.checkpoint_path,
            child_state,
            f"stage-{stage_index:02d}",
            "candidate",
        )
        evaluation = evaluator.evaluate_checkpoint(Path(child.checkpoint_path), target=None)
        accepted = not evaluation.critical_regression
        manager.promote(
            child.id,
            Path(child.checkpoint_path),
            int(request["student_nfe"]),
            accepted=accepted,
        )
        records.append(
            {
                "stage_index": stage_index,
                "teacher_model_id": request["teacher_model_id"],
                "student_model_id": child.id,
                "teacher_nfe": request["teacher_nfe"],
                "student_nfe": request["student_nfe"],
                "accepted": accepted,
                "evaluation": dict(evaluation.metrics),
                "operator_metrics": dict(result.metrics),
            }
        )
        current = child

    final = TinyProgressiveStageResult(manager.state_dict(), records, current.id)
    (output_root / "progressive-stage-manifest.json").write_text(
        json.dumps(final.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return final


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="run the formal TinyH3 4->2->1 stage-manager experiment")
    parser.add_argument("--output-root", default="var/tiny-progressive-stage-manager")
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args(argv)
    result = run_tiny_progressive_stage_experiment(Path(args.output_root), args.seed)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
