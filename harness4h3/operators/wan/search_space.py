"""Versioned, trusted extension points for Wan compression search spaces."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from ...adapters.wan.contracts import CheckpointArtifact, CompressionResult, WanModelAdapter
from .compression import FFNChannelPruningOperator, WanCompressionRegistry


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

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip() or not self.granularity.strip():
            raise ValueError("operator specification requires name, version, and granularity")
        if not isinstance(self.argument_schema, Mapping):
            raise ValueError("operator argument_schema must be a mapping")

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
            status=str(raw.get("status", "proposed")),
            rejection_reason=str(raw["rejection_reason"]) if raw.get("rejection_reason") else None,
            validation=dict(raw.get("validation") or {}),
        )


class SearchSpaceStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.proposals_dir = self.root / "proposals"
        self.versions_dir = self.root / "versions"

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


def default_search_space() -> SearchSpaceVersion:
    return SearchSpaceVersion(
        version_id="S0001",
        parent_version_id=None,
        operators=(
            OperatorSpec("prune_blocks", "1", {"block_ids": "list[str]"}, "block", ("cannot_remove_all_blocks",), (), ""),
            OperatorSpec("prune_ffn_channels", "1", {"block_ids": "list[str]", "retained_indices": "list[int]"}, "channel", ("wan_ffn_width_aligned",), ("explicit_indices_only",), ""),
            OperatorSpec("compose", "1", {"steps": "list[2..3]"}, "composite", ("no_nested_compose",), (), ""),
        ),
        constraints={"wan_channel_pruning_requires_all_blocks": True},
        status="validated",
    )


class SearchSpaceManager:
    def __init__(self, root: Path) -> None:
        self.store = SearchSpaceStore(Path(root))
        if self.store.latest() is None:
            self.store.write_version(default_search_space())

    @property
    def active(self) -> SearchSpaceVersion:
        return self.store.latest() or default_search_space()

    def propose(self, spec: OperatorSpec, rationale: str, evidence_ids: Sequence[str] = ()) -> SearchSpaceProposal:
        if spec.name in self.active.operator_names():
            raise ValueError("duplicate_search_space_operator")
        proposal = SearchSpaceProposal(
            proposal_id=self.store.next_proposal_id(),
            parent_version_id=self.active.version_id,
            spec=spec,
            rationale=str(rationale),
            evidence_ids=tuple(str(item) for item in evidence_ids),
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
        if proposal.status != "proposed":
            raise ValueError("search_space_proposal_not_pending")
        base = replace(proposal, status="validating")
        self.store.write_proposal(base)
        started = time.monotonic()
        try:
            factory = _TRUSTED_FACTORIES[proposal.spec.trusted_factory_key]
            operator = factory()
            if operator.name != proposal.spec.name:
                raise ValueError("trusted_factory_name_mismatch")
            parent_model = adapter.reload_model(parent)
            block_ids = [item.block_id for item in adapter.get_blocks(parent_model)]
            scores = _channel_scores(adapter, parent_model, block_ids)
            lower, upper = min(scores), max(scores)
            if upper <= lower:
                raise ValueError("threshold_validation_requires_nonuniform_channel_scores")
            threshold = lower + (upper - lower) * 0.5
            result = operator.execute(
                adapter,
                parent,
                {"block_ids": block_ids, "threshold": threshold},
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
                "threshold": float(threshold),
                "metrics": metrics,
            }
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
        return self.store.write_version(version)

    def registry_for(self, version: Optional[SearchSpaceVersion] = None, base: Optional[WanCompressionRegistry] = None) -> WanCompressionRegistry:
        target = version or self.active
        factories = {key: factory for key, factory in _TRUSTED_FACTORIES.items()}
        return (base or WanCompressionRegistry()).for_search_space(target, factories)


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
}


__all__ = [
    "OperatorSpec",
    "SearchSpaceManager",
    "SearchSpaceProposal",
    "SearchSpaceStore",
    "SearchSpaceVersion",
    "ThresholdFFNChannelOperator",
    "default_search_space",
]
