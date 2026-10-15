from __future__ import annotations

import pytest

from harness4h3.evolution.runtime_operator import RuntimeCompressionRequest, SparseAttentionRuntimeOperator
from harness4h3.evolution.taxonomy import classify_algorithm_change


def test_sparse_attention_runtime_is_explicitly_disabled_and_weight_neutral():
    request = RuntimeCompressionRequest(
        operator_name="sparse_attention_runtime",
        model_checkpoint_sha256="a" * 64,
        runtime_version="proposal-v1",
        calibration_digest="b" * 64,
    )
    result = SparseAttentionRuntimeOperator().prepare(request)
    assert result["status"] == "disabled"
    assert result["evidence_level"] == "interface_only"
    assert result["changes_weights"] is False
    assert result["hardware_metrics"]["available"] is False


def test_runtime_request_rejects_weight_mutation():
    with pytest.raises(ValueError, match="cannot mutate"):
        RuntimeCompressionRequest(
            operator_name="sparse_attention_runtime",
            model_checkpoint_sha256="a" * 64,
            runtime_version="proposal-v1",
            calibration_digest="b" * 64,
            changes_weights=True,
        )


def test_tica_wrapper_is_classified_as_algorithm_rule_over_existing_closure():
    result = classify_algorithm_change(
        "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n",
        "def propose(profile, history):\n    return {'operator': 'tica_prune', 'args': {'mode': 'block_plus_ffn'}}\n",
        {"operator": "prune_blocks", "args": {"block_ids": ["block.0"]}},
        {"operator": "tica_prune", "args": {"mode": "block_plus_ffn"}},
        {"operators": [{"name": "prune_blocks"}, {"name": "prune_ffn_channels"}]},
    )
    assert result["change_class"] == "algorithm_rule_evolution"
    assert result["reachable_by_existing_closure"] is True
