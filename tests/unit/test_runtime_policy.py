from __future__ import annotations

import copy

import pytest

from harness4h3.backends.comfyui import BackendError
from harness4h3.benchmark.runtime_policy import apply_runtime_policy


def workflow():
    return {
        "122": {"class_type": "VAEDecode", "inputs": {"samples": ["125", 0], "vae": ["119", 0]}},
        "131": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {"clip": ["128", 0], "vae": ["119", 0], "prompt": "x", "width": 352, "height": 640, "length": 22},
        },
        "134": {"class_type": "MiniMaxH3TurboLoRA", "inputs": {"model": ["127", 0], "low_vram": False}},
        "137": {"class_type": "PathchSageAttentionKJ", "inputs": {"model": ["134", 0], "allow_compile": True}},
    }


def test_vae_tiling_rewrites_decoder_without_mutating_source():
    source = workflow()
    candidate = copy.deepcopy(source)
    result = apply_runtime_policy(candidate, {"kind": "vae_tiling", "args": {"tile_size": 256, "overlap": 32}})
    assert result["122"]["class_type"] == "VAEDecodeTiled"
    assert result["122"]["inputs"]["tile_size"] == 256
    assert result["122"]["inputs"]["overlap"] == 32
    assert result["122"]["inputs"]["temporal_size"] == 64
    assert result["122"]["inputs"]["temporal_overlap"] == 8
    assert source["122"]["class_type"] == "VAEDecode"


def test_runtime_offload_sets_only_existing_low_vram_controls():
    candidate = workflow()
    apply_runtime_policy(candidate, {"kind": "runtime_offload", "args": {"mode": "aggressive"}})
    assert candidate["134"]["inputs"]["low_vram"] is True
    assert candidate["137"]["inputs"]["allow_compile"] is False


def test_chunking_fails_explicitly_when_node_has_no_chunk_input():
    with pytest.raises(BackendError) as error:
        apply_runtime_policy(workflow(), {"kind": "inference_chunking", "args": {"chunk_size": 4}})
    assert error.value.failure_type == "runtime_policy_unsupported"


def test_unknown_policy_fails_explicitly():
    with pytest.raises(BackendError) as error:
        apply_runtime_policy(workflow(), {"kind": "unknown", "args": {}})
    assert error.value.failure_type == "runtime_policy_unsupported"


def test_component_lifecycle_fails_when_workflow_has_no_stage_controls():
    with pytest.raises(BackendError) as error:
        apply_runtime_policy(
            workflow(),
            {
                "kind": "component_lifecycle_optimize",
                "args": {
                    "unload_text_encoder_after_encode": True,
                    "offload_vae_until_decode": True,
                    "free_cache_before_decode": True,
                },
            },
        )
    assert error.value.failure_type == "runtime_policy_unsupported"


def test_cache_release_is_a_guarded_backend_boundary_policy():
    candidate = workflow()
    result = apply_runtime_policy(candidate, {"kind": "cache_release", "args": {"stage": "before_decode"}})
    assert result == candidate
