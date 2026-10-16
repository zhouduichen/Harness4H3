from __future__ import annotations

import harness4h3.sandbox.runner as sandbox_runner

from harness4h3.evolution.candidate import AlgorithmCandidate
from harness4h3.evolution.store import AlgorithmStore
from harness4h3.operators.wan import OperatorSpec
from harness4h3.sandbox import AlgorithmSandbox


def test_sandbox_accepts_bounded_plan(tmp_path):
    source = "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}}\n"
    result = AlgorithmSandbox(tmp_path).run(source, {"layers": []}, [])
    assert result.status == "success"
    assert result.plan is not None
    assert result.plan.operator == "prune_blocks"
    assert result.metadata["runtime_audit"] == "deny-file-network-process-dynamic-code"


def test_sandbox_allows_safe_iteration_helpers(tmp_path):
    source = """def propose(profile, history):
    selected = next(item for item in profile['items'] if item['enabled'])
    return {'operator': 'prune_blocks', 'args': {'block_ids': [selected['block_id']]}}
"""
    result = AlgorithmSandbox(tmp_path).run(source, {"items": [{"block_id": "block.1", "enabled": True}]}, [])
    assert result.status == "success"
    assert result.plan is not None
    assert result.plan.args["block_ids"] == ["block.1"]


def test_sandbox_normalizes_persisted_operator_args_alias(tmp_path):
    source = "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'operator_args': {'block_ids': ['block.1']}}\n"
    result = AlgorithmSandbox(tmp_path).run(source, {"layers": []}, [])
    assert result.status == "success"
    assert result.plan is not None
    assert result.plan.to_dict() == {"operator": "prune_blocks", "args": {"block_ids": ["block.1"]}}


def test_sandbox_normalizes_top_level_operator_alias(tmp_path):
    source = "def propose(profile, history):\n    return {'prune_blocks': {'block_ids': ['block.1']}}\n"
    result = AlgorithmSandbox(tmp_path).run(source, {}, [])
    assert result.status == "success"
    assert result.plan is not None
    assert result.plan.to_dict() == {"operator": "prune_blocks", "args": {"block_ids": ["block.1"]}}


def test_sandbox_accepts_bounded_composite_plan(tmp_path):
    source = """def propose(profile, history):
    return {
        'operator': 'compose',
        'args': {'steps': [
            {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}},
            {'operator': 'prune_ffn_channels', 'args': {'block_ids': ['block.2'], 'retained_indices': [0, 1]}},
        ]},
    }
"""

    result = sandbox_runner.AlgorithmSandbox(tmp_path).run(source, {}, [])

    assert result.status == "success"
    assert result.plan is not None
    assert result.plan.operator == "compose"
    assert result.plan.to_dict()["args"]["steps"][1]["operator"] == "prune_ffn_channels"


def test_sandbox_accepts_only_registered_search_space_extension(tmp_path):
    source = "def propose(profile, history):\n    return {'operator': 'prune_ffn_by_weight_threshold', 'args': {'block_ids': ['block.0'], 'threshold': 0.3}}\n"
    spec = OperatorSpec(
        "prune_ffn_by_weight_threshold",
        "1",
        {"block_ids": "list[str]", "threshold": "float"},
        "channel",
        trusted_factory_key="prune_ffn_by_weight_threshold",
    )
    denied = AlgorithmSandbox(tmp_path / "denied").run(source, {}, [])
    assert denied.status == "failed"
    assert denied.failure_type == "algorithm_output_invalid"
    allowed = AlgorithmSandbox(tmp_path / "allowed", operator_specs=(spec,)).run(source, {}, [])
    assert allowed.status == "success"
    assert allowed.plan is not None
    assert allowed.plan.operator == "prune_ffn_by_weight_threshold"


def test_sandbox_allows_registered_extension_inside_ordered_composite(tmp_path):
    source = """def propose(profile, history):
    return {'operator': 'compose', 'args': {'steps': [
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}},
        {'operator': 'prune_ffn_by_weight_threshold', 'args': {'block_ids': ['block.1'], 'threshold': 0.3}},
    ]}}
"""
    spec = OperatorSpec(
        "prune_ffn_by_weight_threshold",
        "1",
        {"block_ids": "list[str]", "threshold": "float"},
        "channel",
        trusted_factory_key="prune_ffn_by_weight_threshold",
    )

    result = AlgorithmSandbox(tmp_path, operator_specs=(spec,)).run(source, {}, [])

    assert result.status == "success"
    assert result.plan is not None
    assert result.plan.to_dict()["args"]["steps"][1]["operator"] == "prune_ffn_by_weight_threshold"


def test_sandbox_rejects_nested_composite_plan(tmp_path):
    source = """def propose(profile, history):
    return {'operator': 'compose', 'args': {'steps': [
        {'operator': 'compose', 'args': {'steps': [
            {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}},
            {'operator': 'prune_blocks', 'args': {'block_ids': ['block.2']}},
        ]}},
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.3']}},
    ]}}
"""

    result = sandbox_runner.AlgorithmSandbox(tmp_path).run(source, {}, [])

    assert result.status == "failed"
    assert result.failure_type == "algorithm_output_invalid"
    assert "compose" in result.message


def test_sandbox_rejects_composite_plan_longer_than_six_steps(tmp_path):
    source = """def propose(profile, history):
    return {'operator': 'compose', 'args': {'steps': [
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}},
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.2']}},
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.3']}},
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.4']}},
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.5']}},
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.6']}},
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.7']}},
    ]}}
"""

    result = sandbox_runner.AlgorithmSandbox(tmp_path).run(source, {}, [])

    assert result.status == "failed"
    assert result.failure_type == "algorithm_output_invalid"
    assert "6" in result.message


def test_sandbox_rejects_unsafe_composite_step_keys(tmp_path):
    source = """def propose(profile, history):
    return {'operator': 'compose', 'args': {'steps': [
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1'], 'device': 'cuda:0'}},
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.2']}},
    ]}}
"""

    result = sandbox_runner.AlgorithmSandbox(tmp_path).run(source, {}, [])

    assert result.status == "failed"
    assert result.failure_type == "algorithm_output_invalid"
    assert "device" in result.message


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


def test_sandbox_fails_closed_when_resource_limits_are_unavailable(tmp_path, monkeypatch):
    source = "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}}\n"
    patched_runner = sandbox_runner.RUNNER_SOURCE.replace(
        "try:\n    if resource is None:",
        "try:\n    resource = None\n    if resource is None:",
        1,
    )
    monkeypatch.setattr(sandbox_runner, "RUNNER_SOURCE", patched_runner)

    result = AlgorithmSandbox(tmp_path).run(source, {"layers": []}, [])

    assert result.status == "failed"
    assert result.failure_type == "process_exit"
    assert result.plan is None


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
