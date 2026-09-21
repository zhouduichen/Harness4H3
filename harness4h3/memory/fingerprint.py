"""Deterministic experiment identities used to block accidental repeats."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def experiment_fingerprint(
    parent_model_id: str,
    parent_system_id: str,
    operator: str,
    operator_args: Mapping[str, Any],
    target_id: str,
    device_id: Optional[str],
    benchmark_recipe: Mapping[str, Any],
) -> str:
    """Hash the complete reproducibility identity of one experiment."""

    payload = {
        "parent_model_id": str(parent_model_id),
        "parent_system_id": str(parent_system_id),
        "operator": str(operator),
        "operator_args": _normalize(operator_args),
        "target_id": str(target_id),
        "device_id": str(device_id) if device_id is not None else None,
        "benchmark_recipe": _normalize(benchmark_recipe),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["experiment_fingerprint"]
