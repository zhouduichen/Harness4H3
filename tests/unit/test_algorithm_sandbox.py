from __future__ import annotations

from harness4h3.evolution.candidate import AlgorithmCandidate
from harness4h3.evolution.store import AlgorithmStore
from harness4h3.sandbox import AlgorithmSandbox


def test_sandbox_accepts_bounded_plan(tmp_path):
    source = "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}}\n"
    result = AlgorithmSandbox(tmp_path).run(source, {"layers": []}, [])
    assert result.status == "success"
    assert result.plan is not None
    assert result.plan.operator == "prune_blocks"


def test_sandbox_rejects_filesystem_import(tmp_path):
    source = "import os\ndef propose(profile, history):\n    return {}\n"
    result = AlgorithmSandbox(tmp_path).run(source, {}, [])
    assert result.status == "failed"
    assert result.failure_type == "sandbox_policy"


def test_sandbox_rejects_dunder_introspection_escape(tmp_path):
    source = """def propose(profile, history):
    for candidate in ().__class__.__base__.__subclasses__():
        if candidate.__name__ == 'catch_warnings':
            return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}}
    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}}
"""
    result = AlgorithmSandbox(tmp_path).run(source, {}, [])
    assert result.status == "failed"
    assert result.failure_type == "sandbox_policy"
    assert "dunder" in result.message


def test_sandbox_rejects_unbounded_output(tmp_path):
    source = "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'path': '/tmp'}}\n"
    result = AlgorithmSandbox(tmp_path).run(source, {}, [])
    assert result.status == "failed"
    assert result.failure_type == "algorithm_output_invalid"


def test_algorithm_store_persists_source_and_rejects_duplicate(tmp_path):
    store = AlgorithmStore(tmp_path)
    candidate = AlgorithmCandidate.from_source("A0001", "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}}\n")
    store.create(candidate)
    assert store.get("A0001").source_sha256 == candidate.source_sha256
    duplicate = AlgorithmCandidate.from_source("A0002", candidate.source)
    try:
        store.create(duplicate)
    except ValueError as exc:
        assert str(exc) == "duplicate_algorithm_source"
    else:
        raise AssertionError("duplicate source was accepted")
