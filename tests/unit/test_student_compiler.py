from __future__ import annotations

from pathlib import Path

import pytest

from harness4h3.student.compiler import CompileError, StudentCompiler
from harness4h3.student.proposal import StudentProposal, StudentTarget
from tests.unit.test_student_proposal import valid_payload


def test_production_proposal_compiles_on_meta_without_allocating_weights(tmp_path):
    proposal = StudentProposal.from_dict(valid_payload(hidden_size=2048, depth=24))
    manifest = StudentCompiler(StudentTarget()).compile(proposal, tmp_path)
    assert 1_000_000_000 <= manifest.parameter_count <= 2_000_000_000
    assert manifest.graph_status == "compiled"
    assert manifest.output_shape == manifest.input_shape
    assert Path(manifest.path).is_file()
    assert manifest.manifest_digest


def test_invalid_memory_budget_fails_before_graph_build(tmp_path):
    target = StudentTarget(max_peak_memory_gb=1.0)
    with pytest.raises(CompileError, match="peak memory"):
        StudentCompiler(target).compile(StudentProposal.from_dict(valid_payload()), tmp_path)


def test_materially_different_architectures_produce_different_graph_digests(tmp_path):
    first = StudentCompiler(StudentTarget()).compile(
        StudentProposal.from_dict(valid_payload(hidden_size=2048, depth=24)), tmp_path / "a"
    )
    second = StudentCompiler(StudentTarget()).compile(
        StudentProposal.from_dict(valid_payload(hidden_size=1792, depth=32)), tmp_path / "b"
    )
    assert first.graph_digest != second.graph_digest


def test_manifest_digest_rejects_tampering(tmp_path):
    proposal = StudentProposal.from_dict(valid_payload())
    manifest = StudentCompiler().compile(proposal, tmp_path)
    path = Path(manifest.path)
    path.write_text(path.read_text(encoding="utf-8").replace("\"depth\": 24", "\"depth\": 25"), encoding="utf-8")
    with pytest.raises(CompileError, match="manifest_digest_mismatch"):
        type(manifest).from_path(path)
