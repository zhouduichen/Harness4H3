"""Trusted contracts for a vLLM-controlled real Wan experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


class WanProposalError(ValueError):
    """Raised when a vLLM Wan proposal cannot cross the trusted boundary."""


_PROPOSAL_FIELDS_V1 = {
    "schema_version",
    "proposal_id",
    "parent_checkpoint_sha256",
    "steps",
    "teacher_reuse_steps",
    "learning_rate",
    "focus",
    "hypothesis",
}
_FOCUS = {"semantic", "temporal", "stability", "efficiency"}
_METHODS = {"progressive_distillation", "dmd2"}
_INITIALIZATION = {"inherit", "generation_head_reset", "dit_zero"}
_LOSS_REDUCTION = {"bf16", "fp32"}
_OPTIMIZER_STATE = {"bf16", "fp32"}
_PROMPT_POLICIES = {"balanced", "fixed_manifest"}
_SEED_POLICIES = {"fixed", "rotating"}
_ANCHOR_SCHEDULES = {"constant", "linear", "cosine"}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WanProposalError("%s must be an object" % name)
    return value


def _require(raw: Mapping[str, Any], names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in raw]
    if missing:
        raise WanProposalError("proposal is missing: %s" % ", ".join(missing))


def _string(value: Any, name: str, *, max_length: int | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WanProposalError("%s must be a non-empty string" % name)
    result = value.strip()
    if max_length is not None and len(result) > max_length:
        raise WanProposalError("%s is too long" % name)
    return result


def _integer(value: Any, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WanProposalError("%s must be an integer" % name)
    if value < minimum or (maximum is not None and value > maximum):
        bound = "[%d, %d]" % (minimum, maximum) if maximum is not None else ">= %d" % minimum
        raise WanProposalError("%s must be in %s" % (name, bound))
    return int(value)


def _number(value: Any, name: str, *, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WanProposalError("%s must be numeric" % name)
    result = float(value)
    if not math.isfinite(result) or result <= minimum or (maximum is not None and result > maximum):
        if maximum is None:
            raise WanProposalError("%s must be finite and > %s" % (name, minimum))
        raise WanProposalError("%s must be finite and in (%s, %s]" % (name, minimum, maximum))
    return result


def _sha256(value: Any, name: str) -> str:
    result = _string(value, name).lower()
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise WanProposalError("%s must be a lowercase SHA-256 digest" % name)
    return result


def _unknown(raw: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise WanProposalError("%s has unknown field(s): %s" % (name, ", ".join(unknown)))


def _capability_available(capabilities: Any, name: str) -> bool:
    if capabilities is None:
        return name == "progressive_distillation"
    if isinstance(capabilities, Mapping):
        entries = capabilities.get("capabilities", capabilities)
    else:
        entries = capabilities
    if isinstance(entries, Mapping):
        item = entries.get(name)
        return bool(item.get("available")) if isinstance(item, Mapping) else bool(item)
    if isinstance(entries, (list, tuple)):
        for item in entries:
            if isinstance(item, Mapping) and str(item.get("name")) == name:
                return bool(item.get("available"))
    return False


@dataclass(frozen=True)
class WanTrainingPolicy:
    method: str
    source_steps: int
    target_steps: int
    train_steps: int
    learning_rate: float
    critic_learning_rate: float
    velocity_weight: float = 1.0
    clean_anchor_weight: float = 0.5
    rollout_steps: int = 0
    rollout_every: int = 1
    rollout_endpoint_weight: float = 0.0
    rollout_clean_weight: float = 0.0
    trainable_scope: str = "all"
    fp32_optimizer_state: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, legacy: bool = False) -> "WanTrainingPolicy":
        raw = _mapping(raw, "training")
        allowed = {
            "method", "source_steps", "target_steps", "train_steps", "learning_rate", "critic_learning_rate",
            "velocity_weight", "clean_anchor_weight", "rollout_steps", "rollout_every",
            "rollout_endpoint_weight", "rollout_clean_weight", "trainable_scope", "fp32_optimizer_state",
        }
        _unknown(raw, allowed, "training")
        _require(raw, ("method", "source_steps", "target_steps", "train_steps", "learning_rate", "critic_learning_rate"))
        method = _string(raw["method"], "training.method")
        if method not in _METHODS:
            raise WanProposalError("training.method is unsupported: %s" % method)
        source_steps = _integer(raw["source_steps"], "training.source_steps", minimum=1, maximum=256)
        target_steps = _integer(raw["target_steps"], "training.target_steps", minimum=1, maximum=64)
        train_steps = _integer(raw["train_steps"], "training.train_steps", minimum=1, maximum=100000)
        if target_steps > source_steps:
            raise WanProposalError("training.target_steps must not exceed source_steps")
        if method == "progressive_distillation":
            current = source_steps
            while current > target_steps:
                if current % 2 or current // 2 < target_steps:
                    raise WanProposalError(
                        "progressive_distillation source_steps=%d cannot reach target_steps=%d by binary halving"
                        % (source_steps, target_steps)
                    )
                current //= 2
        learning_rate = _number(raw["learning_rate"], "training.learning_rate", minimum=0.0, maximum=0.01)
        critic_learning_rate = _number(raw["critic_learning_rate"], "training.critic_learning_rate", minimum=0.0, maximum=0.01)
        velocity_weight = _number(raw.get("velocity_weight", 1.0), "training.velocity_weight", minimum=-1e-12)
        clean_anchor_weight = _number(raw.get("clean_anchor_weight", 0.5), "training.clean_anchor_weight", minimum=-1e-12)
        rollout_steps = _integer(raw.get("rollout_steps", 0), "training.rollout_steps", minimum=0, maximum=4)
        if rollout_steps not in (0, 2, 4):
            raise WanProposalError("training.rollout_steps must be 0, 2, or 4")
        rollout_every = _integer(raw.get("rollout_every", 1), "training.rollout_every", minimum=1, maximum=16)
        rollout_endpoint_weight = _number(raw.get("rollout_endpoint_weight", 0.0), "training.rollout_endpoint_weight", minimum=-1e-12)
        rollout_clean_weight = _number(raw.get("rollout_clean_weight", 0.0), "training.rollout_clean_weight", minimum=-1e-12)
        if rollout_steps == 0 and (rollout_endpoint_weight != 0.0 or rollout_clean_weight != 0.0):
            raise WanProposalError("disabled rollout must have zero rollout weights")
        if rollout_steps > 0 and rollout_endpoint_weight + rollout_clean_weight <= 0:
            raise WanProposalError("active rollout requires a positive rollout weight")
        trainable_scope = _string(raw.get("trainable_scope", "all"), "training.trainable_scope")
        if trainable_scope not in {"all", "heads"} and not any(item.strip() for item in trainable_scope.split(",")):
            raise WanProposalError("training.trainable_scope must select parameters")
        fp32_optimizer_state = raw.get("fp32_optimizer_state", False)
        if not isinstance(fp32_optimizer_state, bool):
            raise WanProposalError("training.fp32_optimizer_state must be boolean")
        return cls(
            method=method,
            source_steps=source_steps,
            target_steps=target_steps,
            train_steps=train_steps,
            learning_rate=learning_rate,
            critic_learning_rate=critic_learning_rate,
            velocity_weight=velocity_weight,
            clean_anchor_weight=clean_anchor_weight,
            rollout_steps=rollout_steps,
            rollout_every=rollout_every,
            rollout_endpoint_weight=rollout_endpoint_weight,
            rollout_clean_weight=rollout_clean_weight,
            trainable_scope=trainable_scope,
            fp32_optimizer_state=fp32_optimizer_state,
        )


@dataclass(frozen=True)
class WanInitializationPolicy:
    mode: str = "inherit"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WanInitializationPolicy":
        raw = _mapping(raw, "initialization")
        _unknown(raw, {"mode"}, "initialization")
        mode = _string(raw.get("mode", "inherit"), "initialization.mode")
        if mode not in _INITIALIZATION:
            raise WanProposalError("initialization.mode is unsupported: %s" % mode)
        return cls(mode)


@dataclass(frozen=True)
class WanPrecisionPolicy:
    loss_reduction: str = "fp32"
    optimizer_state: str = "bf16"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WanPrecisionPolicy":
        raw = _mapping(raw, "precision")
        _unknown(raw, {"loss_reduction", "optimizer_state"}, "precision")
        loss_reduction = _string(raw.get("loss_reduction", "fp32"), "precision.loss_reduction")
        optimizer_state = _string(raw.get("optimizer_state", "bf16"), "precision.optimizer_state")
        if loss_reduction not in _LOSS_REDUCTION or optimizer_state not in _OPTIMIZER_STATE:
            raise WanProposalError("precision policy is unsupported")
        return cls(loss_reduction, optimizer_state)


@dataclass(frozen=True)
class WanCurriculumPolicy:
    fidelity_schedule: tuple[str, ...] = ("F1", "F2", "F3")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WanCurriculumPolicy":
        raw = _mapping(raw, "curriculum")
        _unknown(raw, {"fidelity_schedule"}, "curriculum")
        value = raw.get("fidelity_schedule", ("F1", "F2", "F3"))
        if not isinstance(value, (list, tuple)) or not value:
            raise WanProposalError("curriculum.fidelity_schedule must be a non-empty list")
        schedule = tuple(str(item) for item in value)
        if any(item not in {"F1", "F2", "F3"} for item in schedule) or schedule != tuple(sorted(schedule, key=("F1", "F2", "F3").index)):
            raise WanProposalError("curriculum.fidelity_schedule must be an ordered F1/F2/F3 prefix")
        return cls(schedule)


@dataclass(frozen=True)
class WanDataPolicy:
    manifest_digest: str = ""
    prompt_policy: str = "balanced"
    seed_policy: str = "fixed"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WanDataPolicy":
        raw = _mapping(raw, "data")
        _unknown(raw, {"manifest_digest", "prompt_policy", "seed_policy"}, "data")
        digest = _sha256(raw.get("manifest_digest", "0" * 64), "data.manifest_digest")
        prompt_policy = _string(raw.get("prompt_policy", "balanced"), "data.prompt_policy")
        seed_policy = _string(raw.get("seed_policy", "fixed"), "data.seed_policy")
        if prompt_policy not in _PROMPT_POLICIES or seed_policy not in _SEED_POLICIES:
            raise WanProposalError("data policy is unsupported")
        return cls(digest, prompt_policy, seed_policy)


@dataclass(frozen=True)
class WanContinuationPolicy:
    overlap_latent_frames: int = 7
    anchor_schedule: str = "linear"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WanContinuationPolicy":
        raw = _mapping(raw, "continuation")
        _unknown(raw, {"overlap_latent_frames", "anchor_schedule"}, "continuation")
        overlap = _integer(raw.get("overlap_latent_frames", 7), "continuation.overlap_latent_frames", minimum=6, maximum=12)
        anchor = _string(raw.get("anchor_schedule", "linear"), "continuation.anchor_schedule")
        if anchor not in _ANCHOR_SCHEDULES:
            raise WanProposalError("continuation.anchor_schedule is unsupported")
        return cls(overlap, anchor)


@dataclass(frozen=True)
class WanTrainingProposal:
    schema_version: int
    proposal_id: str
    parent_checkpoint_sha256: str
    training: WanTrainingPolicy
    initialization: WanInitializationPolicy
    precision: WanPrecisionPolicy
    curriculum: WanCurriculumPolicy
    data: WanDataPolicy
    continuation: WanContinuationPolicy
    focus: str
    hypothesis: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        expected_parent_sha256: str,
        capabilities: Any = None,
    ) -> "WanTrainingProposal":
        raw = _mapping(value, "proposal")
        schema_version = raw.get("schema_version")
        if schema_version == 1:
            _unknown(raw, _PROPOSAL_FIELDS_V1, "proposal")
            _require(raw, tuple(sorted(_PROPOSAL_FIELDS_V1)))
        elif schema_version == 2:
            _unknown(raw, {
                "schema_version", "proposal_id", "parent_checkpoint_sha256", "training", "initialization",
                "precision", "curriculum", "data", "continuation", "focus", "hypothesis",
            }, "proposal")
            _require(
                raw,
                (
                    "schema_version", "proposal_id", "parent_checkpoint_sha256", "training", "initialization",
                    "precision", "curriculum", "data", "continuation", "focus", "hypothesis",
                ),
            )
        else:
            raise WanProposalError("schema_version must be 1 or 2")
        proposal_id = _string(raw["proposal_id"], "proposal_id")
        if re.fullmatch(r"wan_[0-9]{4,}", proposal_id) is None:
            raise WanProposalError("proposal_id must match wan_[0-9]{4,}")
        parent = _sha256(raw["parent_checkpoint_sha256"], "parent_checkpoint_sha256")
        expected = str(expected_parent_sha256).strip().lower()
        if parent != expected:
            raise WanProposalError("parent_checkpoint_sha256 does not match trusted parent")
        if schema_version == 1:
            steps = raw["steps"]
            if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 32:
                raise WanProposalError("steps must be an integer in [1, 32]")
            reuse = raw["teacher_reuse_steps"]
            if isinstance(reuse, bool) or not isinstance(reuse, int) or not 1 <= reuse <= steps:
                raise WanProposalError("teacher_reuse_steps must be an integer in [1, steps]")
            legacy_learning_rate = raw["learning_rate"]
            if isinstance(legacy_learning_rate, bool) or not isinstance(legacy_learning_rate, (int, float)):
                raise WanProposalError("learning_rate must be numeric")
            if not math.isfinite(float(legacy_learning_rate)) or not 1e-8 <= float(legacy_learning_rate) <= 1e-4:
                raise WanProposalError("learning_rate must be finite and in [1e-8, 1e-4]")
            training = WanTrainingPolicy.from_mapping(
                {
                    "method": "progressive_distillation",
                    "source_steps": 32,
                    "target_steps": 8,
                    "train_steps": steps,
                    "learning_rate": float(legacy_learning_rate),
                    "critic_learning_rate": float(legacy_learning_rate),
                    "rollout_every": reuse,
                },
                legacy=True,
            )
            initialization = WanInitializationPolicy()
            precision = WanPrecisionPolicy()
            curriculum = WanCurriculumPolicy()
            data = WanDataPolicy()
            continuation = WanContinuationPolicy()
        else:
            training = WanTrainingPolicy.from_mapping(raw["training"])
            initialization = WanInitializationPolicy.from_mapping(raw.get("initialization", {}))
            precision = WanPrecisionPolicy.from_mapping(raw.get("precision", {}))
            curriculum = WanCurriculumPolicy.from_mapping(raw.get("curriculum", {}))
            data = WanDataPolicy.from_mapping(raw.get("data", {}))
            continuation = WanContinuationPolicy.from_mapping(raw.get("continuation", {}))
        if not _capability_available(capabilities, training.method):
            raise WanProposalError("capability_unavailable: %s" % training.method)
        focus = _string(raw["focus"], "focus")
        if focus not in _FOCUS:
            raise WanProposalError("focus must be one of: %s" % ", ".join(sorted(_FOCUS)))
        hypothesis = _string(raw["hypothesis"], "hypothesis", max_length=512)
        return cls(
            schema_version=int(schema_version),
            proposal_id=proposal_id,
            parent_checkpoint_sha256=parent,
            training=training,
            initialization=initialization,
            precision=precision,
            curriculum=curriculum,
            data=data,
            continuation=continuation,
            focus=focus,
            hypothesis=hypothesis,
        )

    @property
    def steps(self) -> int:
        return self.training.train_steps

    @property
    def teacher_reuse_steps(self) -> int:
        return self.training.rollout_every

    @property
    def learning_rate(self) -> float:
        return self.training.learning_rate

    def capability_requirements(self) -> dict[str, Any]:
        return {
            "method": self.training.method,
            "rollout_steps": self.training.rollout_steps,
            "trainable_scope": self.training.trainable_scope,
            "initialization": self.initialization.mode,
            "optimizer_state": self.precision.optimizer_state,
            "fidelity_schedule": list(self.curriculum.fidelity_schedule),
        }

    def fingerprint(
        self,
        *,
        manifest_digest: str = "0" * 64,
        capability_digest: str = "0" * 64,
        evaluator_digest: str = "0" * 64,
        code_digest: str = "0" * 64,
    ) -> str:
        payload = {
            "proposal": self.to_dict(),
            "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
            "manifest_digest": _sha256(manifest_digest, "manifest_digest"),
            "capability_digest": _sha256(capability_digest, "capability_digest"),
            "evaluator_digest": _sha256(evaluator_digest, "evaluator_digest"),
            "code_digest": _sha256(code_digest, "code_digest"),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        if self.schema_version == 1:
            return {
                "schema_version": 1,
                "proposal_id": self.proposal_id,
                "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
                "steps": self.training.train_steps,
                "teacher_reuse_steps": self.training.rollout_every,
                "learning_rate": self.training.learning_rate,
                "focus": self.focus,
                "hypothesis": self.hypothesis,
            }
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(evaluation: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    quality = evaluation.get("quality_metrics")
    if isinstance(quality, Mapping) and quality.get(key) is not None:
        return float(quality[key])
    direct = evaluation.get(key)
    return float(direct) if direct is not None else float(default)


def candidate_decision(
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    teacher_semantic: float,
    semantic_floor_ratio: float,
    quality_floor_ratio: float,
    min_semantic_delta: float,
) -> dict[str, Any]:
    """Return a deterministic promotion decision for one evaluated child."""

    if candidate.get("status") != "success" or not bool(candidate.get("valid", True)):
        return {"accepted": False, "reason": "evaluation_failed"}
    if candidate.get("offline_simulation") is not False:
        return {"accepted": False, "reason": "offline_simulation"}
    if not bool(candidate.get("continuation_verified")):
        return {"accepted": False, "reason": "contract_invalid"}
    semantic = _metric(candidate, "semantic")
    quality = float(candidate.get("quality_score") or _metric(candidate, "score"))
    if semantic < float(teacher_semantic) * float(semantic_floor_ratio):
        return {
            "accepted": False,
            "reason": "semantic_floor",
            "semantic": semantic,
            "semantic_floor": float(teacher_semantic) * float(semantic_floor_ratio),
        }
    parent_semantic = _metric(parent, "semantic")
    parent_quality = float(parent.get("quality_score") or _metric(parent, "score"))
    if quality < parent_quality * float(quality_floor_ratio):
        return {"accepted": False, "reason": "quality_floor", "quality": quality, "parent_quality": parent_quality}
    if semantic >= parent_semantic + float(min_semantic_delta):
        return {"accepted": True, "reason": "semantic_improved", "semantic": semantic, "quality": quality}
    if semantic >= parent_semantic and quality > parent_quality:
        return {"accepted": True, "reason": "quality_improved", "semantic": semantic, "quality": quality}
    return {"accepted": False, "reason": "no_improvement", "semantic": semantic, "quality": quality}


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_event(path: Path, event: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _json_file(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise WanProposalError("campaign artifact must be an object: %s" % path)
    return dict(value)


def load_complete_rounds(root: Path) -> tuple[list[dict[str, Any]], list[int]]:
    """Load complete round records and identify rounds interrupted mid-write."""

    root = Path(root)
    complete: list[dict[str, Any]] = []
    incomplete: list[int] = []
    for round_dir in sorted(root.glob("round-*")):
        if not round_dir.is_dir():
            continue
        try:
            round_index = int(round_dir.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        paths = {name: round_dir / name for name in ("proposal.json", "training-result.json", "evaluation-result.json", "decision.json")}
        if not all(path.is_file() for path in paths.values()):
            if any(path.exists() for path in paths.values()):
                incomplete.append(round_index)
            continue
        complete.append(
            {
                "round": round_index,
                "proposal": _json_file(paths["proposal.json"]),
                "training": _json_file(paths["training-result.json"]),
                "evaluation": _json_file(paths["evaluation-result.json"]),
                "decision": _json_file(paths["decision.json"]),
            }
        )
    complete.sort(key=lambda item: int(item["round"]))
    return complete, incomplete


def load_campaign_state(root: Path) -> dict[str, Any]:
    path = Path(root) / "campaign-state.json"
    if not path.is_file():
        return {
            "base_digest": "",
            "last_complete_round": 0,
            "next_round": 1,
            "parent_checkpoint": None,
            "parent_checkpoint_sha256": None,
            "parent_evaluation": None,
            "accepted_history": [],
            "failed_rounds": [],
        }
    state = _json_file(path)
    for name in ("base_digest", "last_complete_round", "next_round", "accepted_history", "failed_rounds"):
        if name not in state:
            raise WanProposalError("campaign state is missing %s" % name)
    return state


def write_campaign_state(root: Path, state: Mapping[str, Any]) -> None:
    atomic_write_json(Path(root) / "campaign-state.json", state)


def register_fingerprint(root: Path, fingerprint: str, metadata: Mapping[str, Any]) -> bool:
    """Append a policy fingerprint and return whether it was already present."""

    path = Path(root) / "fingerprints.jsonl"
    fingerprint = _string(fingerprint, "fingerprint")
    existing: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, Mapping) and item.get("fingerprint"):
                existing.add(str(item["fingerprint"]))
    if fingerprint in existing:
        return True
    append_event(path, {"fingerprint": fingerprint, **dict(metadata)})
    return False


__all__ = [
    "WanProposalError",
    "WanTrainingProposal",
    "append_event",
    "atomic_write_json",
    "candidate_decision",
    "load_campaign_state",
    "load_complete_rounds",
    "register_fingerprint",
    "sha256_file",
    "write_campaign_state",
]
