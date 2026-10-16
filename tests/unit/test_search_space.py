from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution.agent import AlgorithmAgent
from harness4h3.evolution.taxonomy import classify_search_space_proposal
from harness4h3.operators.wan import (
    OperatorSpec,
    SearchSpaceVersion,
    SearchSpaceManager,
    ThresholdFFNChannelOperator,
    TimestepBlockGatingOperator,
    build_wan_compression_registry,
)
from harness4h3.operators.wan.search_space import _assert_finite


def _parent(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    return adapter, parent


def _threshold_spec():
    return OperatorSpec(
        name="prune_ffn_by_weight_threshold",
        version="1",
        argument_schema={"block_ids": "list[str]", "threshold": "float"},
        granularity="channel",
        constraints=("wan_ffn_width_aligned", "threshold_must_change_width"),
        limitations=("weight_proxy_only", "not_real_wan_validated"),
        trusted_factory_key="prune_ffn_by_weight_threshold",
    )


def test_finite_check_uses_transformer_for_diffusion_pipeline():
    class PipelineLike:
        parameters = {"pipeline_component": "not-an-iterator"}

        def __init__(self):
            self.transformer = torch.nn.Linear(2, 2)

    _assert_finite(PipelineLike())


def test_unvalidated_operator_is_described_but_not_executable(tmp_path):
    manager = SearchSpaceManager(tmp_path / "space")
    proposal = manager.propose(_threshold_spec(), "weight proxy plateau", ["R0001"])
    registry = build_wan_compression_registry()
    assert proposal.status == "proposed"
    assert proposal.spec.name not in registry.names()
    with pytest.raises(ValueError, match="unsupported_compression_operator"):
        registry.get(proposal.spec.name)


def test_fake_threshold_operator_validates_registers_and_reloads(tmp_path):
    adapter, parent = _parent(tmp_path)
    manager = SearchSpaceManager(tmp_path / "space")
    proposal = manager.propose(_threshold_spec(), "weight proxy plateau", ["R0001"])
    validated = manager.validate(proposal, adapter, parent, output_dir=tmp_path / "validation")
    assert validated.status == "validated"
    assert validated.validation["offline_simulation"] is True
    version = manager.register(validated)
    assert version.status == "validated"
    assert proposal.spec.name in version.operator_names()

    registry = manager.registry_for(version, build_wan_compression_registry())
    assert proposal.spec.name in registry.names()
    result = registry.execute(
        proposal.spec.name,
        adapter,
        parent,
        {"block_ids": ["block.0"], "threshold": float(validated.validation["threshold"])},
        tmp_path / "child",
    )
    assert result.status == "success"
    assert result.child_checkpoint is not None
    child = adapter.reload_model(result.child_checkpoint)
    assert child.blocks[0].up_proj.out_features < 16
    assert all(block.up_proj.out_features == 16 for block in child.blocks[1:])
    assert result.structural_change["operator"] == "prune_ffn_by_weight_threshold"
    assert result.structural_change["block_ids"] == ["block.0"]

    composite = registry.execute(
        "compose",
        adapter,
        parent,
        {
            "steps": [
                {"operator": proposal.spec.name, "args": {"block_ids": ["block.0"], "threshold": float(validated.validation["threshold"])}},
                {"operator": "prune_blocks", "args": {"block_ids": ["block.0"]}},
            ]
        },
        tmp_path / "composite-child",
    )
    assert composite.status == "success", composite.message


def test_rejected_proposal_cannot_register_or_mutate_registry(tmp_path):
    adapter, parent = _parent(tmp_path)
    manager = SearchSpaceManager(tmp_path / "space")
    invalid = OperatorSpec(
        name="unknown_operator",
        version="1",
        argument_schema={"x": "int"},
        granularity="unknown",
        trusted_factory_key="not_trusted",
    )
    proposal = manager.propose(invalid, "invalid factory", [])
    rejected = manager.validate(proposal, adapter, parent, output_dir=tmp_path / "bad")
    assert rejected.status == "rejected"
    assert rejected.rejection_reason
    with pytest.raises(ValueError, match="only_validated"):
        manager.register(rejected)
    assert "unknown_operator" not in manager.registry_for().names()


def test_threshold_operator_rejects_non_structural_threshold(tmp_path):
    adapter, parent = _parent(tmp_path)
    result = ThresholdFFNChannelOperator().execute(
        adapter, parent, {"block_ids": ["block.0"], "threshold": 0.0000001}, tmp_path / "unchanged"
    )
    assert result.status == "failed"
    assert result.child_checkpoint is None


def test_search_space_active_pointer_can_roll_back_without_deleting_new_version(tmp_path):
    adapter, parent = _parent(tmp_path)
    manager = SearchSpaceManager(tmp_path / "space")
    proposal = manager.propose(_threshold_spec(), "weight proxy plateau", ["R0001"])
    validated = manager.validate(proposal, adapter, parent, output_dir=tmp_path / "validation")
    version = manager.register(validated)
    assert manager.active.version_id == version.version_id
    rolled_back = manager.rollback("S0001")
    assert rolled_back.version_id == "S0001"
    assert manager.active.version_id == "S0001"
    assert manager.store.latest().version_id == version.version_id
    assert (tmp_path / "space" / "active.json").is_file()


def test_active_registry_contains_only_active_search_space_operators(tmp_path):
    manager = SearchSpaceManager(tmp_path / "space")
    limited = SearchSpaceVersion(
        "S0098",
        manager.active.version_id,
        (OperatorSpec("prune_blocks", "1", {"block_ids": "list[str]"}, "block"),),
    )
    manager.store.write_version(limited)
    manager.activate(limited.version_id)

    registry = manager.registry_for(limited, build_wan_compression_registry(include_multigranular=True))

    assert registry.names() == ("prune_blocks",)
    with pytest.raises(ValueError, match="unsupported_compression_operator"):
        registry.get("attention_window")


def test_search_space_register_is_idempotent_after_resume(tmp_path):
    adapter, parent = _parent(tmp_path)
    manager = SearchSpaceManager(tmp_path / "space")
    validated = manager.validate(manager.propose(_threshold_spec(), "resume-safe proposal"), adapter, parent, output_dir=tmp_path / "validation")
    first = manager.register(validated)
    second = manager.register(validated)

    assert second.version_id == first.version_id
    assert len(manager.store.read_versions()) == 2


def test_validating_search_space_proposal_can_resume(tmp_path):
    adapter, parent = _parent(tmp_path)
    manager = SearchSpaceManager(tmp_path / "space")
    manager.propose(_threshold_spec(), "recover an interrupted validation")
    pending = manager.store.read_proposals()[0]
    validating = type(pending)(
        proposal_id=pending.proposal_id,
        parent_version_id=pending.parent_version_id,
        spec=pending.spec,
        rationale=pending.rationale,
        evidence_ids=pending.evidence_ids,
        trigger_evidence=pending.trigger_evidence,
        status="validating",
    )
    manager.store.write_proposal(validating)

    resumed = manager.validate(validating, adapter, parent, output_dir=tmp_path / "validation")

    assert resumed.status == "validated"


def test_versioned_implementation_ref_loads_and_validates_without_factory_key(tmp_path):
    adapter, parent = _parent(tmp_path)
    manager = SearchSpaceManager(tmp_path / "space")
    spec = OperatorSpec(
        name="timestep_block_gate",
        version="1",
        argument_schema={"rules": "list[object]", "policy": "str"},
        granularity="runtime_timestep",
        constraints=("normalized_timestep",),
        limitations=("runtime_only",),
        implementation_ref="harness4h3.operators.wan.compression.TimestepBlockGatingOperator",
        execution_domain="runtime",
    )
    proposal = manager.propose(spec, "load the verified runtime implementation by reference", ["R0001"])
    validated = manager.validate(proposal, adapter, parent, output_dir=tmp_path / "validation")
    assert validated.status == "validated"

    version = manager.register(validated)
    registry = manager.registry_for(version, build_wan_compression_registry())
    assert isinstance(registry.get("timestep_block_gate"), TimestepBlockGatingOperator)


def test_search_space_version_round_trips_implementation_reference():
    spec = OperatorSpec(
        name="example_runtime",
        version="1",
        argument_schema={"keep_ratio": "float"},
        granularity="runtime",
        implementation_ref="harness4h3.operators.wan.compression.SparseRuntimeOperator",
        execution_domain="runtime",
    )
    version = SearchSpaceVersion("S0099", "S0001", (spec,))
    restored = SearchSpaceVersion.from_dict(version.to_dict())
    assert restored.spec("example_runtime").implementation_ref == spec.implementation_ref


def test_agent_generated_rule_is_sandboxed_validated_registered_and_reused(tmp_path):
    adapter, parent = _parent(tmp_path)
    rule_source = """def propose(profile, history):
    channels = profile['channel_sensitivity'][0]['channels']
    ordered = sorted(channels, key=lambda item: float(item['importance_score']))
    retained = sorted([int(item['channel']) for item in ordered[:8]])
    block_ids = [item['block_id'] for item in profile['layers']]
    return {'operator': 'prune_ffn_channels', 'args': {'block_ids': block_ids, 'retained_indices': retained}}
"""
    spec = OperatorSpec(
        name="agent_rule_low_noise_channel_rank",
        version="1",
        argument_schema={"block_ids": "list[str]"},
        granularity="channel",
        constraints=("delegates_to_trusted_ffn_operator",),
        limitations=("cpu_fake_rule_evidence_only",),
        trusted_factory_key="generated_channel_rule",
        implementation_source=rule_source,
    )
    manager = SearchSpaceManager(tmp_path / "space")
    agent = AlgorithmAgent(
        provider=lambda payload: {
            "spec": spec.to_dict(),
            "rationale": "the fixed space cannot express this profile-ranked rule",
            "evidence_ids": ["R0001"],
        }
    )
    suggestion = agent.propose_search_space_change(search_space=manager.active.to_dict(), history=[], failures=["channel_importance_plateau"])
    proposal = manager.propose(
        OperatorSpec.from_dict(suggestion["spec"]),
        suggestion["rationale"],
        suggestion["evidence_ids"],
    )
    validated = manager.validate(proposal, adapter, parent, output_dir=tmp_path / "validation")
    assert validated.status == "validated"
    assert validated.validation["offline_simulation"] is True
    assert validated.validation["implementation_source_sha256"]
    version = manager.register(validated)
    registry = manager.registry_for(version, build_wan_compression_registry())
    assert spec.name in registry.names()
    result = registry.execute(spec.name, adapter, parent, {"block_ids": ["block.0", "block.1", "block.2", "block.3"]}, tmp_path / "child")
    assert result.status == "success"
    assert result.structural_change["delegated_operator"] == "prune_ffn_channels"
    assert result.structural_change["generated_rule_source_sha256"]
    restored = SearchSpaceManager(tmp_path / "space")
    assert spec.name in restored.registry_for(restored.active, build_wan_compression_registry()).names()


def test_generated_rule_is_classified_as_rule_evolution_until_structural_counterexample_exists():
    active = {
        "operators": [
            {"name": "prune_ffn_channels"},
            {"name": "compose"},
        ]
    }
    result = classify_search_space_proposal(
        {
            "name": "agent_rule_low_noise_channel_rank",
            "trusted_factory_key": "generated_channel_rule",
        },
        active,
    )
    assert result["change_class"] == "algorithm_rule_evolution"
    assert result["reachable_by_existing_closure"] is True
    assert result["evidence_status"] == "delegates_to_existing_structural_operator"


def test_agent_timestep_gate_extension_is_validated_and_enters_active_registry(tmp_path):
    adapter, parent = _parent(tmp_path)
    manager = SearchSpaceManager(tmp_path / "space")
    suggestion = AlgorithmAgent().propose_search_space_change(
        search_space=manager.active.to_dict(),
        history=[],
        failures=["timestep_latency_plateau"],
    )
    spec = OperatorSpec.from_dict(suggestion["spec"])
    proposal = manager.propose(spec, suggestion["rationale"], suggestion["evidence_ids"])
    validated = manager.validate(proposal, adapter, parent, output_dir=tmp_path / "validation")
    assert validated.status == "validated"
    assert validated.validation["structure_outside_active_closure"] is True
    assert validated.validation["structural_counterexample"]["dynamic_compute"] is True
    version = manager.register(validated)
    registry = manager.registry_for(version, build_wan_compression_registry())
    assert isinstance(registry.get("timestep_block_gate"), TimestepBlockGatingOperator)
    assert "timestep_block_gate" in registry.names()

    # The next Algorithm Program reads the activated version, generates the
    # newly reachable operator, and passes it through the same plan validator.
    profile = {
        "model": {"ffn_width": 16},
        "layers": [{"block_id": "block.0"}, {"block_id": "block.1"}],
        "block_sensitivity": [{"block_id": "block.0", "importance_score": 0.1}],
    }
    candidate = AlgorithmAgent().generate("A0002", parent_id=None, profile=profile, history=[])
    from harness4h3.sandbox import AlgorithmSandbox
    from harness4h3.sandbox.runner import CompressionPlan

    sandbox = AlgorithmSandbox(tmp_path / "next-generation-sandbox", operator_specs=version.operators)
    program = sandbox.run_program(
        candidate.source,
        entrypoint="search",
        payload={
            "profile": profile,
            "history": [],
            "state": {"depth": 1, "proposed_plan": None},
            "search_space": version.to_dict(),
        },
    )
    assert program.status == "success"
    gate_plan = next(item for item in program.value["candidates"] if item["operator"] == "timestep_block_gate")
    normalized = CompressionPlan.from_mapping(gate_plan, operator_specs={item.name: item.to_dict() for item in version.operators})
    assert normalized.operator == "timestep_block_gate"
    window_plan = next(item for item in program.value["candidates"] if item["operator"] == "attention_window")
    window = CompressionPlan.from_mapping(window_plan, operator_specs={item.name: item.to_dict() for item in version.operators})
    assert window.operator == "attention_window"
