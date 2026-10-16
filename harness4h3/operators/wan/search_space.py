"""Versioned, trusted extension points for Wan compression search spaces."""

from __future__ import annotations

import json
import hashlib
import importlib
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from ...adapters.wan.contracts import CheckpointArtifact, CompressionResult, OperatorCapability, WanModelAdapter
from ...sandbox import AlgorithmSandbox
from .compression import AttentionHeadPruningOperator, AttentionWindowOperator, FFNChannelPruningOperator, HeadGatingOperator, HeadPruningOperator, KVTokenSelectionOperator, LayerPruningOperator, SamplingStepSkipOperator, SparseRuntimeOperator, TimestepAttentionPolicyOperator, TimestepBlockGatingOperator, TokenMergeOperator, TokenPruningOperator, WanCompressionRegistry, build_wan_compression_registry


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    version: str
    argument_schema: Mapping[str, Any]
    granularity: str
    constraints: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    trusted_factory_key: str = ""
    implementation_source: str = ""
    execution_domain: str = "structural"
    implementation_ref: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip() or not self.granularity.strip():
            raise ValueError("operator specification requires name, version, and granularity")
        if not isinstance(self.argument_schema, Mapping):
            raise ValueError("operator argument_schema must be a mapping")
        if self.execution_domain not in {"structural", "runtime", "sampler", "composite"}:
            raise ValueError("unsupported operator execution domain")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OperatorSpec":
        return cls(
            name=str(raw["name"]),
            version=str(raw.get("version", "1")),
            argument_schema=dict(raw.get("argument_schema") or {}),
            granularity=str(raw.get("granularity", "unknown")),
            constraints=tuple(str(item) for item in raw.get("constraints", [])),
            limitations=tuple(str(item) for item in raw.get("limitations", [])),
            trusted_factory_key=str(raw.get("trusted_factory_key", "")),
            implementation_source=str(raw.get("implementation_source", "")),
            execution_domain=str(raw.get("execution_domain", "structural")),
            implementation_ref=str(raw.get("implementation_ref", "")),
        )

    def capability(self) -> OperatorCapability:
        """Return the execution metadata consumed by registry/worker code."""

        implementation_ref = self.implementation_ref or self.trusted_factory_key or self.name
        return OperatorCapability(
            name=self.name,
            version=self.version,
            granularity=self.granularity,
            execution_domain=self.execution_domain,
            argument_schema=self.argument_schema,
            implementation_ref=implementation_ref,
            constraints=self.constraints,
        )


@dataclass(frozen=True)
class SearchSpaceVersion:
    version_id: str
    parent_version_id: Optional[str]
    operators: tuple[OperatorSpec, ...]
    constraints: Mapping[str, Any] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    status: str = "validated"

    def __post_init__(self) -> None:
        if not self.version_id.startswith("S") or not self.operators:
            raise ValueError("search-space version requires an ID and operators")
        if len({item.name for item in self.operators}) != len(self.operators):
            raise ValueError("search-space operator names must be unique")
        if self.status not in {"validated", "rejected"}:
            raise ValueError("unsupported search-space version status")

    def operator_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.operators)

    def spec(self, name: str) -> OperatorSpec:
        for item in self.operators:
            if item.name == name:
                return item
        raise ValueError("operator_not_in_search_space: %s" % name)

    def capabilities(self) -> tuple[OperatorCapability, ...]:
        """Expose execution-domain metadata without exposing factory internals."""

        return tuple(item.capability() for item in self.operators)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SearchSpaceVersion":
        return cls(
            version_id=str(raw["version_id"]),
            parent_version_id=str(raw["parent_version_id"]) if raw.get("parent_version_id") else None,
            operators=tuple(OperatorSpec.from_dict(item) for item in raw.get("operators", [])),
            constraints=dict(raw.get("constraints") or {}),
            evidence_ids=tuple(str(item) for item in raw.get("evidence_ids", [])),
            status=str(raw.get("status", "validated")),
        )


@dataclass(frozen=True)
class SearchSpaceProposal:
    proposal_id: str
    parent_version_id: str
    spec: OperatorSpec
    rationale: str
    evidence_ids: tuple[str, ...] = ()
    trigger_evidence: Mapping[str, Any] = field(default_factory=dict)
    status: str = "proposed"
    rejection_reason: Optional[str] = None
    validation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.proposal_id.startswith("P") or not self.parent_version_id.startswith("S"):
            raise ValueError("search-space proposal requires proposal and parent version IDs")
        if self.status not in {"proposed", "validating", "validated", "rejected"}:
            raise ValueError("unsupported search-space proposal status")
        if self.status == "rejected" and not self.rejection_reason:
            raise ValueError("rejected search-space proposal requires a reason")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SearchSpaceProposal":
        return cls(
            proposal_id=str(raw["proposal_id"]),
            parent_version_id=str(raw["parent_version_id"]),
            spec=OperatorSpec.from_dict(raw["spec"]),
            rationale=str(raw.get("rationale", "")),
            evidence_ids=tuple(str(item) for item in raw.get("evidence_ids", [])),
            trigger_evidence=dict(raw.get("trigger_evidence") or {}),
            status=str(raw.get("status", "proposed")),
            rejection_reason=str(raw["rejection_reason"]) if raw.get("rejection_reason") else None,
            validation=dict(raw.get("validation") or {}),
        )


class SearchSpaceStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.proposals_dir = self.root / "proposals"
        self.versions_dir = self.root / "versions"
        self.active_path = self.root / "active.json"

    def _ensure(self) -> None:
        self.proposals_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def next_proposal_id(self) -> str:
        numbers = []
        for path in self.proposals_dir.glob("P*.json"):
            try:
                numbers.append(int(path.stem[1:]))
            except ValueError:
                continue
        return "P%04d" % (max(numbers, default=0) + 1)

    def next_version_id(self) -> str:
        numbers = []
        for path in self.versions_dir.glob("S*.json"):
            try:
                numbers.append(int(path.stem[1:]))
            except ValueError:
                continue
        return "S%04d" % (max(numbers, default=0) + 1)

    def write_proposal(self, proposal: SearchSpaceProposal) -> SearchSpaceProposal:
        self._ensure()
        self._atomic_write(self.proposals_dir / (proposal.proposal_id + ".json"), proposal.to_dict())
        return proposal

    def read_proposals(self) -> tuple[SearchSpaceProposal, ...]:
        if not self.proposals_dir.exists():
            return ()
        return tuple(SearchSpaceProposal.from_dict(json.loads(path.read_text(encoding="utf-8"))) for path in sorted(self.proposals_dir.glob("P*.json")))

    def write_version(self, version: SearchSpaceVersion) -> SearchSpaceVersion:
        self._ensure()
        self._atomic_write(self.versions_dir / (version.version_id + ".json"), version.to_dict())
        return version

    def read_versions(self) -> tuple[SearchSpaceVersion, ...]:
        if not self.versions_dir.exists():
            return ()
        return tuple(SearchSpaceVersion.from_dict(json.loads(path.read_text(encoding="utf-8"))) for path in sorted(self.versions_dir.glob("S*.json")))

    def latest(self) -> Optional[SearchSpaceVersion]:
        versions = self.read_versions()
        return versions[-1] if versions else None

    def read_active_id(self) -> Optional[str]:
        try:
            payload = json.loads(self.active_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        value = payload.get("version_id") if isinstance(payload, Mapping) else None
        return str(value) if value else None

    def write_active_id(self, version_id: str) -> None:
        self._ensure()
        self._atomic_write(self.active_path, {"version_id": str(version_id)})

    @staticmethod
    def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class OperatorImplementationRegistry:
    """Resolve versioned Wan operator implementations without per-operator branches.

    Search-space JSON is data, not executable Python.  A persisted
    ``implementation_ref`` may point only into the repository's Wan operator
    namespace, while tests and deployment code can register an explicit
    factory under a stable reference.  The implementation is still activated
    only after ``SearchSpaceManager.validate`` runs its real adapter/reload
    checks.
    """

    _ALLOWED_MODULE_PREFIX = "harness4h3.operators.wan"

    def __init__(self, factories: Optional[Mapping[str, Callable[..., Any]]] = None) -> None:
        self._factories = dict(factories or {})

    def register(self, implementation_ref: str, factory: Callable[..., Any]) -> None:
        reference = str(implementation_ref).strip()
        if not reference or not callable(factory):
            raise ValueError("operator implementation registration is invalid")
        self._factories[reference] = factory

    def _resolve_factory(self, implementation_ref: str) -> Callable[..., Any]:
        reference = str(implementation_ref).strip()
        registered = self._factories.get(reference)
        if registered is not None:
            return registered
        module_name, separator, attribute = reference.replace(":", ".").rpartition(".")
        if not separator or not module_name.startswith(self._ALLOWED_MODULE_PREFIX + "."):
            raise ValueError("unavailable_operator_implementation: %s" % reference)
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute, None)
        if not callable(factory):
            raise ValueError("invalid_operator_implementation: %s" % reference)
        self._factories[reference] = factory
        return factory

    def instantiate(self, spec: OperatorSpec) -> Any:
        reference = str(spec.implementation_ref or spec.trusted_factory_key).strip()
        if not reference:
            raise ValueError("operator_implementation_ref_required")
        factory = self._resolve_factory(reference)
        try:
            return factory(spec)
        except TypeError:
            # Built-in operator classes use a zero-argument constructor.
            return factory()


def default_search_space() -> SearchSpaceVersion:
    return SearchSpaceVersion(
        version_id="S0001",
        parent_version_id=None,
        operators=(
            OperatorSpec("prune_blocks", "1", {"block_ids": "list[str]"}, "block", ("cannot_remove_all_blocks",), (), ""),
            OperatorSpec("prune_ffn_channels", "1", {"block_ids": "list[str]", "retained_indices": "list[int]"}, "channel", ("wan_ffn_width_aligned",), ("explicit_indices_only",), ""),
            OperatorSpec("compose", "1", {"steps": "list[2..6]"}, "composite", ("no_nested_compose",), (), "", execution_domain="composite", implementation_ref="harness4h3.operators.wan.compression.CompositeCompressionOperator"),
            OperatorSpec("prune_layers", "1", {"block_ids": "list[str]"}, "layer", ("cannot_remove_all_layers",), (), "prune_layers", implementation_ref="harness4h3.operators.wan.compression.LayerPruningOperator"),
            OperatorSpec("prune_attention_heads", "1", {"head_indices": "list[int]", "attention_names": "list[str]"}, "attention_head", ("retain_at_least_one_head",), (), "prune_attention_heads", implementation_ref="harness4h3.operators.wan.compression.AttentionHeadPruningOperator"),
            OperatorSpec("prune_heads", "1", {"head_indices": "list[int]", "attention_names": "list[str]"}, "head", ("retain_at_least_one_head",), (), "prune_heads", implementation_ref="harness4h3.operators.wan.compression.HeadPruningOperator"),
            OperatorSpec("head_gating", "1", {"attention": "str", "block_ids": "list[str]", "keep_ratio": "float", "head_indices": "list[int]", "score": "str", "rules": "list[object]"}, "head_runtime", ("self_attention_only", "reduced_qkv_projection", "retain_at_least_one_head"), ("dynamic_policy", "timestep_rules_optional"), "head_gating", execution_domain="runtime", implementation_ref="harness4h3.operators.wan.compression.HeadGatingOperator"),
            OperatorSpec("prune_tokens", "2", {"keep_ratio": "float", "block_ids": "list[str]", "score": "str", "grid_shape": "list[int3]"}, "token", ("keep_at_least_one_token", "preserve_temporal_coverage_when_grid_is_supplied"), ("dynamic_policy", "spatiotemporal_grid_optional"), "prune_tokens", execution_domain="runtime", implementation_ref="harness4h3.operators.wan.compression.TokenPruningOperator"),
            OperatorSpec("token_merge", "2", {"keep_ratio": "float", "block_ids": "list[str]", "mode": "str", "grid_shape": "list[int3]"}, "token", ("self_attention_pre_rope_merge", "unmerge_to_residual_shape"), ("dynamic_policy", "adjacent_groups_only", "spatiotemporal_grid_optional"), "token_merge", execution_domain="runtime", implementation_ref="harness4h3.operators.wan.compression.TokenMergeOperator"),
            OperatorSpec("kv_token_select", "1", {"keep_ratio": "float", "block_ids": "list[str]", "score": "str"}, "token", ("full_query_length", "reduced_kv_length", "self_attention_only"), ("dynamic_policy", "shared_batch_index_set"), "kv_token_select", execution_domain="runtime", implementation_ref="harness4h3.operators.wan.compression.KVTokenSelectionOperator"),
            OperatorSpec("sparse_runtime", "1", {"keep_ratio": "float", "attention": "str"}, "sparse_runtime", ("self_attention_top_k",), ("dynamic_policy",), "sparse_runtime", execution_domain="runtime", implementation_ref="harness4h3.operators.wan.compression.SparseRuntimeOperator"),
            OperatorSpec("attention_window", "2", {"window_size": "int", "attention": "str", "grid_shape": "list[int3]", "window_radius": "list[int3]"}, "attention", ("self_attention_local_window",), ("dynamic_policy", "requires_sequence_order", "spatiotemporal_grid_optional"), "attention_window", execution_domain="runtime", implementation_ref="harness4h3.operators.wan.compression.AttentionWindowOperator"),
            OperatorSpec("sampling_step_skip", "1", {"target_steps": "int"}, "sampling_step", ("preserve_endpoints",), ("dynamic_policy",), "sampling_step_skip", execution_domain="sampler", implementation_ref="harness4h3.operators.wan.compression.SamplingStepSkipOperator"),
        ),
        constraints={"wan_channel_pruning_requires_all_blocks": True},
        status="validated",
    )


class SearchSpaceManager:
    def __init__(self, root: Path) -> None:
        self.store = SearchSpaceStore(Path(root))
        self.implementation_registry = OperatorImplementationRegistry(_TRUSTED_FACTORIES)
        if self.store.latest() is None:
            self.store.write_version(default_search_space())
        if self.store.read_active_id() is None:
            latest = self.store.latest()
            if latest is not None:
                self.store.write_active_id(latest.version_id)

    @property
    def active(self) -> SearchSpaceVersion:
        active_id = self.store.read_active_id()
        versions = {item.version_id: item for item in self.store.read_versions()}
        selected = versions.get(active_id) if active_id else None
        if selected is not None and selected.status == "validated":
            return selected
        validated = [item for item in self.store.read_versions() if item.status == "validated"]
        return validated[-1] if validated else default_search_space()

    def propose(
        self,
        spec: OperatorSpec,
        rationale: str,
        evidence_ids: Sequence[str] = (),
        *,
        trigger_evidence: Optional[Mapping[str, Any]] = None,
    ) -> SearchSpaceProposal:
        if spec.name in self.active.operator_names():
            raise ValueError("duplicate_search_space_operator")
        proposal = SearchSpaceProposal(
            proposal_id=self.store.next_proposal_id(),
            parent_version_id=self.active.version_id,
            spec=spec,
            rationale=str(rationale),
            evidence_ids=tuple(str(item) for item in evidence_ids),
            trigger_evidence=dict(trigger_evidence or {}),
        )
        return self.store.write_proposal(proposal)

    def validate(
        self,
        proposal: SearchSpaceProposal,
        adapter: WanModelAdapter,
        parent: CheckpointArtifact,
        *,
        diagnostic: Optional[Callable[[CompressionResult], Mapping[str, Any]]] = None,
        output_dir: Optional[Path] = None,
    ) -> SearchSpaceProposal:
        if proposal.status not in {"proposed", "validating"}:
            raise ValueError("search_space_proposal_not_pending")
        base = replace(proposal, status="validating")
        self.store.write_proposal(base)
        started = time.monotonic()
        try:
            operator = self.implementation_registry.instantiate(proposal.spec)
            if operator.name != proposal.spec.name:
                raise ValueError("operator_implementation_name_mismatch")
            parent_model = adapter.reload_model(parent)
            block_ids = [item.block_id for item in adapter.get_blocks(parent_model)]
            threshold = None
            plan_args: dict[str, Any] = {"block_ids": block_ids}
            implementation_key = proposal.spec.trusted_factory_key or proposal.spec.name
            if implementation_key == "prune_ffn_by_weight_threshold":
                scores = _channel_scores(adapter, parent_model, block_ids)
                lower, upper = min(scores), max(scores)
                if upper <= lower:
                    raise ValueError("threshold_validation_requires_nonuniform_channel_scores")
                threshold = lower + (upper - lower) * 0.5
                plan_args["threshold"] = threshold
            elif implementation_key == "timestep_block_gate" or proposal.spec.name == "timestep_block_gate":
                if not block_ids:
                    raise ValueError("timestep_gate_requires_blocks")
                plan_args = {
                    "rules": [{
                        "block_id": block_ids[0],
                        "min_timestep": 0.0,
                        "max_timestep": 0.3,
                    }],
                    "policy": "skip_when_in_range",
                }
            elif implementation_key == "timestep_attention_policy" or proposal.spec.name == "timestep_attention_policy":
                if not block_ids:
                    raise ValueError("timestep_attention_requires_blocks")
                plan_args = {
                    "rules": [
                        {"min_timestep": 0.0, "max_timestep": 0.3, "keep_ratio": 0.5, "block_ids": [block_ids[0]], "score": "l2_norm"},
                        {"min_timestep": 0.7, "max_timestep": 1.0, "keep_ratio": 0.75, "block_ids": [block_ids[0]], "score": "l2_norm"},
                    ],
                    "policy": "select_kv_tokens",
                }
            elif implementation_key in {"prune_attention_heads", "prune_heads"}:
                plan_args = {"head_indices": [0], "attention_names": ["attn1", "attn2"]}
            elif implementation_key == "head_gating":
                plan_args = {"attention": "attn1", "keep_ratio": 0.75, "score": "weight_l2"}
            elif implementation_key == "prune_tokens":
                plan_args = {"keep_ratio": 0.75}
            elif implementation_key == "token_merge":
                plan_args = {"keep_ratio": 0.75, "mode": "adjacent"}
            elif implementation_key == "kv_token_select":
                plan_args = {"keep_ratio": 0.75, "score": "l2_norm"}
            elif implementation_key == "sparse_runtime":
                plan_args = {"keep_ratio": 0.75, "attention": "attn1"}
            elif implementation_key == "attention_window":
                plan_args = {"window_size": 32, "attention": "attn1"}
            elif implementation_key == "sampling_step_skip":
                plan_args = {"target_steps": 8}
            if proposal.spec.implementation_source:
                # An authored implementation is still untrusted.  Execute it
                # only to derive the validation plan, with the proposed spec
                # as the sole extension visible at the sandbox boundary.
                generated = AlgorithmSandbox(
                    Path(output_dir or (self.store.root / "validation" / proposal.proposal_id)) / "implementation-sandbox",
                    operator_specs=(proposal.spec,),
                ).run(
                    proposal.spec.implementation_source,
                    _rule_profile(adapter, parent_model, plan_args),
                    (),
                )
                if generated.status != "success" or generated.plan is None:
                    raise ValueError(generated.failure_type or "implementation_source_validation_failed")
                if proposal.spec.name == "timestep_block_gate" and generated.plan.operator != proposal.spec.name:
                    raise ValueError("timestep_gate_source_must_return_declared_operator")
                if proposal.spec.name == "timestep_attention_policy" and generated.plan.operator != proposal.spec.name:
                    raise ValueError("timestep_attention_source_must_return_declared_operator")
                plan_args = dict(generated.plan.args)
            result = operator.execute(
                adapter,
                parent,
                plan_args,
                Path(output_dir or (self.store.root / "validation" / proposal.proposal_id)),
            )
            if result.status != "success" or result.child_checkpoint is None:
                raise ValueError(result.failure_type or "operator_validation_failed")
            reloaded = adapter.reload_model(result.child_checkpoint)
            structure = adapter.inspect(reloaded)
            _assert_finite(reloaded)
            metrics = dict(diagnostic(result) if diagnostic is not None else {"valid": True, "offline_simulation": True, "search_score": 0.9})
            if metrics.get("valid") is not True:
                raise ValueError("diagnostic_validation_failed")
            validation = {
                "status": "validated",
                "elapsed_s": time.monotonic() - started,
                "offline_simulation": bool(metrics.get("offline_simulation", True)),
                "child_checkpoint_sha256": result.child_checkpoint.sha256,
                "block_count": int(structure.block_count),
                "ffn_width": int(structure.ffn_width),
                "metrics": metrics,
                "trigger_evidence": dict(proposal.trigger_evidence),
                "operator": proposal.spec.name,
                "plan_args": _jsonable(plan_args),
                "parameter_reduction_ratio": float(result.structural_change.get("parameter_reduction_ratio", 0.0)),
                "dynamic_compute": bool(result.structural_change.get("dynamic_compute", False)),
                "child_reload_verified": bool(result.structural_change.get("child_reload_verified", True)),
            }
            if threshold is not None:
                validation["threshold"] = float(threshold)
            if proposal.spec.name == "timestep_block_gate":
                validation["structure_outside_active_closure"] = True
                validation["structural_counterexample"] = {
                    "kind": "dynamic_timestep_block_execution",
                    "operator": proposal.spec.name,
                    "parameter_reduction_ratio": float(result.structural_change.get("parameter_reduction_ratio", 0.0)),
                    "dynamic_compute": bool(result.structural_change.get("dynamic_compute", False)),
                    "child_reload_verified": bool(result.structural_change.get("child_reload_verified", False)),
                }
            if proposal.spec.name == "timestep_attention_policy":
                validation["structure_outside_active_closure"] = True
                validation["structural_counterexample"] = {
                    "kind": "dynamic_timestep_attention_execution",
                    "operator": proposal.spec.name,
                    "rules": _jsonable(plan_args.get("rules", [])),
                    "parameter_reduction_ratio": float(result.structural_change.get("parameter_reduction_ratio", 0.0)),
                    "dynamic_compute": bool(result.structural_change.get("dynamic_compute", False)),
                    "per_sample_timestep": bool(result.structural_change.get("per_sample_timestep", False)),
                    "child_reload_verified": bool(result.structural_change.get("child_reload_verified", False)),
                }
            if proposal.spec.implementation_source:
                validation["implementation_source_sha256"] = hashlib.sha256(proposal.spec.implementation_source.encode("utf-8")).hexdigest()
            validated = replace(base, status="validated", validation=validation)
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            validated = replace(
                base,
                status="rejected",
                rejection_reason=str(exc),
                validation={"status": "rejected", "elapsed_s": time.monotonic() - started, "offline_simulation": True},
            )
        return self.store.write_proposal(validated)

    def register(self, proposal: SearchSpaceProposal) -> SearchSpaceVersion:
        if proposal.status != "validated":
            raise ValueError("only_validated_search_space_proposals_can_register")
        existing = next(
            (
                item
                for item in self.store.read_versions()
                if proposal.proposal_id in item.evidence_ids and item.status == "validated"
            ),
            None,
        )
        if existing is not None:
            self.store.write_active_id(existing.version_id)
            return existing
        active = self.active
        if proposal.parent_version_id != active.version_id:
            raise ValueError("search_space_parent_version_mismatch")
        version = SearchSpaceVersion(
            version_id=self.store.next_version_id(),
            parent_version_id=active.version_id,
            operators=tuple((*active.operators, proposal.spec)),
            constraints=dict(active.constraints),
            evidence_ids=tuple((*active.evidence_ids, *proposal.evidence_ids, proposal.proposal_id)),
            status="validated",
        )
        version = self.store.write_version(version)
        self.store.write_active_id(version.version_id)
        return version

    def activate(self, version_id: str) -> SearchSpaceVersion:
        """Select a previously validated version without deleting newer branches."""

        version = next((item for item in self.store.read_versions() if item.version_id == str(version_id)), None)
        if version is None:
            raise ValueError("search_space_version_unavailable: %s" % version_id)
        if version.status != "validated":
            raise ValueError("search_space_version_not_validated: %s" % version_id)
        self.store.write_active_id(version.version_id)
        return version

    def rollback(self, version_id: str) -> SearchSpaceVersion:
        """Revert execution to an older validated version while retaining the newer version."""

        return self.activate(version_id)

    def registry_for(self, version: Optional[SearchSpaceVersion] = None, base: Optional[WanCompressionRegistry] = None) -> WanCompressionRegistry:
        target = version or self.active
        available = base or build_wan_compression_registry(include_multigranular=True)
        factories = {key: factory for key, factory in _TRUSTED_FACTORIES.items()}
        for spec in target.operators:
            reference = str(spec.implementation_ref or spec.trusted_factory_key).strip()
            if reference and reference not in factories:
                factories[reference] = lambda current_spec, registry=self.implementation_registry: registry.instantiate(current_spec)
        return available.for_search_space(target, factories)


class ThresholdFFNChannelOperator:
    name = "prune_ffn_by_weight_threshold"

    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        threshold = plan.get("threshold")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0.0 < float(threshold):
            return CompressionResult("failed", parent.sha256, None, failure_type="invalid_threshold", message="threshold must be positive")
        try:
            model = adapter.reload_model(parent)
            block_refs = adapter.get_blocks(model)
            requested = plan.get("block_ids")
            requested_ids = tuple(str(item) for item in requested) if isinstance(requested, (list, tuple)) else tuple(item.block_id for item in block_refs)
            if getattr(adapter, "channel_pruning_requires_all_blocks", False):
                requested_ids = tuple(item.block_id for item in block_refs)
            scores = _channel_scores(adapter, model, requested_ids)
            retained = [index for index, value in enumerate(scores) if value >= float(threshold)]
            width = len(scores)
            if not retained or len(retained) >= width:
                return CompressionResult("failed", parent.sha256, None, failure_type="threshold_no_structural_change", message="threshold retained zero or all channels")
            result = FFNChannelPruningOperator().execute(
                adapter,
                parent,
                {"block_ids": list(requested_ids), "retained_indices": retained},
                Path(output_dir),
            )
            if result.status != "success":
                return result
            return replace(result, structural_change={**dict(result.structural_change), "operator": self.name, "threshold": float(threshold), "derived_retained_indices": retained})
        except (AttributeError, IndexError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return CompressionResult("failed", parent.sha256, None, failure_type="threshold_operator_failed", message=str(exc))


class GeneratedChannelRuleOperator:
    """Execute Agent-authored channel-ranking rules through AlgorithmSandbox."""

    def __init__(self, name: str, source: str) -> None:
        self.name = str(name)
        self.source = str(source)

    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        try:
            model = adapter.reload_model(parent)
            profile = _rule_profile(adapter, model, plan)
            sandbox = AlgorithmSandbox(Path(output_dir) / "rule-sandbox")
            generated = sandbox.run(self.source, profile, [])
            if generated.status != "success" or generated.plan is None:
                return CompressionResult(
                    "failed",
                    parent.sha256,
                    None,
                    failure_type=generated.failure_type or "generated_rule_failed",
                    message=generated.message,
                )
            if generated.plan.operator != "prune_ffn_channels":
                return CompressionResult("failed", parent.sha256, None, failure_type="generated_rule_wrong_base_operator", message="generated rule must delegate to prune_ffn_channels")
            result = FFNChannelPruningOperator().execute(
                adapter,
                parent,
                generated.plan.args,
                Path(output_dir) / "mutation",
            )
            if result.status != "success":
                return result
            return replace(
                result,
                structural_change={
                    **dict(result.structural_change),
                    "operator": self.name,
                    "generated_rule_source_sha256": hashlib.sha256(self.source.encode("utf-8")).hexdigest(),
                    "delegated_operator": generated.plan.operator,
                },
            )
        except (AttributeError, IndexError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return CompressionResult("failed", parent.sha256, None, failure_type="generated_rule_failed", message=str(exc))


def _resolve(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = getattr(value, part) if not part.isdigit() else value[int(part)]
    return value


def _paired_ffn(block: Any) -> tuple[Any, Any]:
    for up_path, down_path in (("up_proj", "down_proj"), ("ffn.up_proj", "ffn.down_proj"), ("ffn.net.0.proj", "ffn.net.2")):
        try:
            up, down = _resolve(block, up_path), _resolve(block, down_path)
        except (AttributeError, IndexError, TypeError):
            continue
        if hasattr(up, "weight") and hasattr(down, "weight"):
            return up, down
    raise ValueError("unsupported_model_layout: paired FFN projections unavailable")


def _channel_scores(adapter: WanModelAdapter, model: Any, block_ids: Sequence[str]) -> list[float]:
    refs = {item.block_id: item for item in adapter.get_blocks(model)}
    selected = [refs[item] for item in block_ids if item in refs]
    if not selected:
        raise ValueError("invalid_channel_selection")
    per_block: list[list[float]] = []
    for ref in selected:
        up, down = _paired_ffn(_resolve(model, ref.module_path))
        width = int(up.out_features)
        per_block.append([
            float(up.weight[index].detach().float().abs().mean().item() + down.weight[:, index].detach().float().abs().mean().item())
            for index in range(width)
        ])
    width = len(per_block[0])
    if any(len(values) != width for values in per_block):
        raise ValueError("unsupported_model_layout: FFN widths are not aligned")
    return [sum(values[index] for values in per_block) / len(per_block) for index in range(width)]


def _assert_finite(model: Any) -> None:
    for parameter in model.parameters():
        if not bool(parameter.detach().isfinite().all()):
            raise ValueError("model_integrity_failed: non-finite parameter")


_TRUSTED_FACTORIES: dict[str, Callable[[], Any]] = {
    "prune_ffn_by_weight_threshold": ThresholdFFNChannelOperator,
    "timestep_block_gate": TimestepBlockGatingOperator,
    "timestep_attention_policy": TimestepAttentionPolicyOperator,
    "generated_channel_rule": lambda spec=None: GeneratedChannelRuleOperator(
        str(getattr(spec, "name", "generated_channel_rule")),
        str(getattr(spec, "implementation_source", "")),
    ),
    "prune_layers": LayerPruningOperator,
    "prune_attention_heads": AttentionHeadPruningOperator,
    "prune_heads": HeadPruningOperator,
    "head_gating": HeadGatingOperator,
    "prune_tokens": TokenPruningOperator,
    "token_merge": TokenMergeOperator,
    "kv_token_select": KVTokenSelectionOperator,
    "sparse_runtime": SparseRuntimeOperator,
    "attention_window": AttentionWindowOperator,
    "sampling_step_skip": SamplingStepSkipOperator,
}


def _make_trusted_operator(spec: OperatorSpec, factory: Callable[..., Any]) -> Any:
    if spec.trusted_factory_key == "generated_channel_rule":
        if not spec.implementation_source.strip():
            raise ValueError("generated_rule_source_required")
        return factory(spec)
    return factory()


def _rule_profile(adapter: WanModelAdapter, model: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
    block_refs = {item.block_id: item for item in adapter.get_blocks(model)}
    requested = plan.get("block_ids") if isinstance(plan, Mapping) else None
    block_ids = [str(item) for item in requested] if isinstance(requested, (list, tuple)) and requested else list(block_refs)
    scores = _channel_scores(adapter, model, block_ids)
    structure = adapter.inspect(model)
    return {
        "model": {
            "architecture": structure.architecture,
            "block_count": int(structure.block_count),
            "ffn_width": int(structure.ffn_width),
        },
        "layers": [{"block_id": item} for item in block_ids if item in block_refs],
        "channel_sensitivity": [{
            "block_id": item,
            "width": len(scores),
            "available": True,
            "channels": [
                {"channel": index, "importance_score": float(value), "available": True}
                for index, value in enumerate(scores)
            ],
        } for item in block_ids if item in block_refs],
        "timestep_sensitivity": {
            "high_noise": {"importance_score": 0.8, "available": True},
            "middle": {"importance_score": 0.5, "available": True},
            "low_noise": {"importance_score": 0.2, "available": True},
        },
        "provenance": {"offline_simulation": bool(structure.metadata.get("offline_simulation", False)), "method": "validated_generated_rule_profile"},
    }


__all__ = [
    "GeneratedChannelRuleOperator",
    "OperatorImplementationRegistry",
    "OperatorSpec",
    "SearchSpaceManager",
    "SearchSpaceProposal",
    "SearchSpaceStore",
    "SearchSpaceVersion",
    "ThresholdFFNChannelOperator",
    "default_search_space",
]
