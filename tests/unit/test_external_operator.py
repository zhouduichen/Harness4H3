from __future__ import annotations

import sys
from pathlib import Path

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.controller.schemas import CostEstimate
from harness4h3.executor.local import LocalProcessExecutor
from harness4h3.h3.state import ModelState
from harness4h3.operators.base import ExecutionContext
from harness4h3.operators.external import ExternalScriptOperator
from harness4h3.target.profile import TargetProfile


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "external_operator.py"


def test_external_operator_uses_machine_readable_contract_and_new_checkpoint(tmp_path):
    parent_path = tmp_path / "parent.safetensors"
    parent_path.write_bytes(b"immutable parent")
    state = ModelState.fake_baseline()
    state = ModelState.from_dict({**state.to_dict(), "checkpoint_path": str(parent_path)})
    parent = ModelCandidate("M0000", None, 0, str(parent_path), state, None, "baseline")
    target = TargetProfile("target", "gpu", "local", max_model_size_gb=5)
    operator = ExternalScriptOperator(
        "quantize",
        "fixture quantizer",
        (sys.executable, str(FIXTURE)),
        {"bits": (int,)},
        LocalProcessExecutor(),
        CostEstimate(wall_time_s=1.0),
        timeout_s=5,
    )

    assert operator.dry_run(state, {"bits": 4}, target) == CostEstimate(wall_time_s=1.0)
    result = operator.execute(parent, {"bits": 4}, ExecutionContext(tmp_path / "exp", "M0001"))

    assert result.ok
    assert result.output_state is not None
    assert result.output_state.model_id == "M0001"
    assert result.output_state.parent_model_id == "M0000"
    assert result.output_state.checkpoint_path != str(parent_path)
    assert Path(result.output_state.checkpoint_path).is_file()
    assert Path(result.output_state.checkpoint_path).is_relative_to((tmp_path / "exp" / "artifacts").resolve())
    assert parent_path.read_bytes() == b"immutable parent"
    assert result.metrics["process"]["pid"] is not None
    assert "created child.safetensors" in (tmp_path / "exp" / "stdout.log").read_text(encoding="utf-8")
