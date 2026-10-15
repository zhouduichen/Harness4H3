"""Frozen, non-adaptive final evaluation boundary."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ..adapters.wan.contracts import CheckpointArtifact, stable_digest


@dataclass(frozen=True)
class FinalEvaluationRequest:
    model_id: str
    algorithm_id: str
    model_checkpoint_sha256: str
    algorithm_source_sha256: str
    source_revision: str
    protocol_digest: str

    def __post_init__(self) -> None:
        for name in asdict(self):
            if not str(getattr(self, name)).strip():
                raise ValueError("final evaluation requires %s" % name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FinalEvaluationRunner:
    """Run one held-out Test evaluation without adaptive-state writes."""

    def __init__(self, evaluator: Any, output_path: Path):
        self.evaluator = evaluator
        self.output_path = Path(output_path)

    def run(self, request: FinalEvaluationRequest, artifact: CheckpointArtifact) -> Mapping[str, Any]:
        config = getattr(self.evaluator, "config", None)
        if str(getattr(config, "evaluation_split", "")).lower() != "test":
            raise ValueError("FinalEvaluation requires evaluator.evaluation_split=test")
        if artifact.sha256 != request.model_checkpoint_sha256:
            raise ValueError("final evaluation model checkpoint hash mismatch")
        evaluate_final = getattr(self.evaluator, "evaluate_final", None)
        if not callable(evaluate_final):
            raise TypeError("final evaluator must expose evaluate_final(artifact)")
        measured = evaluate_final(artifact)
        if not isinstance(measured, Mapping):
            raise ValueError("final evaluator returned a non-mapping result")
        frozen_measurement = {
            **dict(measured),
            "evaluation_split": "test",
            "promotable": False,
            "adaptive_state_mutated": False,
        }
        payload = {
            "phase": "final_evaluation",
            "evaluation_split": "test",
            "adaptive_state_mutated": False,
            "frozen_request": request.to_dict(),
            "protocol_digest": request.protocol_digest,
            "request_digest": stable_digest(request.to_dict()),
            "evaluation": frozen_measurement,
        }
        self._write_json(payload)
        return payload

    def _write_json(self, payload: Mapping[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=self.output_path.name + ".tmp-", dir=str(self.output_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.output_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


__all__ = ["FinalEvaluationRequest", "FinalEvaluationRunner"]
