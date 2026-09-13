"""Atomic full-state training checkpoints."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from h3_training.algorithms.base import TrainingMethod
from .state import LoopState, TrainingFailure


CHECKPOINT_SCHEMA = 1


def canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.close(descriptor)
        torch.save(dict(payload), temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_training_checkpoint(
    path: Path,
    method: TrainingMethod,
    loop_state: LoopState,
    parent_sha256: str,
    config_digest: str,
    algorithm_generator: torch.Generator,
) -> Path:
    if loop_state.accumulation_position:
        raise TrainingFailure("invalid_training_config", "checkpoints require an accumulation boundary")
    optimizers = method.optimizer_map()
    payload: Dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "algorithm_name": method.algorithm_name,
        "method_state": method.checkpoint_state(),
        "algorithm_state": dict(method.algorithm_state()),
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "schedulers": {name: scheduler.state_dict() for name, scheduler in method.schedulers().items()},
        "loop_state": asdict(loop_state),
        "cpu_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "algorithm_rng_state": algorithm_generator.get_state(),
        "parent_sha256": parent_sha256,
        "config_digest": config_digest,
        "role_names": sorted(name for name, _ in method.named_children()),
    }
    _atomic_torch_save(payload, Path(path))
    return Path(path)


def load_training_checkpoint(
    path: Path,
    method: TrainingMethod,
    expected_parent_sha256: str,
    expected_config_digest: str,
    algorithm_generator: torch.Generator,
) -> LoopState:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise TrainingFailure("checkpoint_corrupt", str(exc)) from exc
    expected = {
        "schema_version": CHECKPOINT_SCHEMA,
        "algorithm_name": method.algorithm_name,
        "parent_sha256": expected_parent_sha256,
        "config_digest": expected_config_digest,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise TrainingFailure("resume_mismatch", f"{key} does not match")
    optimizers = method.optimizer_map()
    if set(payload.get("optimizers", {})) != set(optimizers):
        raise TrainingFailure("resume_mismatch", "optimizer roles do not match")
    if sorted(payload.get("role_names", [])) != sorted(name for name, _ in method.named_children()):
        raise TrainingFailure("resume_mismatch", "model roles do not match")
    try:
        method.load_checkpoint_state(payload["method_state"])
        method.load_algorithm_state(payload.get("algorithm_state", {}))
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][name])
        schedulers = method.schedulers()
        if set(payload.get("schedulers", {})) != set(schedulers):
            raise TrainingFailure("resume_mismatch", "scheduler roles do not match")
        for name, scheduler in schedulers.items():
            scheduler.load_state_dict(payload["schedulers"][name])
        state = LoopState(**payload["loop_state"])
        if state.accumulation_position:
            raise TrainingFailure("resume_mismatch", "checkpoint is inside accumulation")
        torch.random.set_rng_state(payload["cpu_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        algorithm_generator.set_state(payload["algorithm_rng_state"])
        return state
    except TrainingFailure:
        raise
    except Exception as exc:
        raise TrainingFailure("checkpoint_corrupt", str(exc)) from exc
