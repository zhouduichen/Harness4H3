from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..executor.local import LocalProcessExecutor
from .policy import SourcePolicyError, validate_source


ALLOWED_BUILTINS = {
    name: __builtins__[name]  # type: ignore[index]
    for name in ("abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list", "max", "min", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip")
    if isinstance(__builtins__, dict) and name in __builtins__
}
if not isinstance(__builtins__, dict):  # pragma: no cover - interpreter detail
    ALLOWED_BUILTINS = {name: getattr(__builtins__, name) for name in ("abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list", "max", "min", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip")}


def _safe_import(name: str, globals: Optional[Mapping[str, Any]] = None, locals: Optional[Mapping[str, Any]] = None, fromlist: Sequence[str] = (), level: int = 0) -> Any:
    if level or name.split(".", 1)[0] not in {"math", "statistics", "json", "collections"}:
        raise ImportError("sandbox import denied: %s" % name)
    return __import__(name, globals, locals, fromlist, level)


ALLOWED_BUILTINS["__import__"] = _safe_import


@dataclass(frozen=True)
class CompressionPlan:
    operator: str
    args: Mapping[str, Any]

    ALLOWED_OPERATORS = frozenset({"prune_blocks", "prune_ffn_channels"})

    @classmethod
    def from_mapping(cls, raw: Any) -> "CompressionPlan":
        if not isinstance(raw, Mapping):
            raise ValueError("algorithm_output_invalid: output must be an object")
        if set(raw) != {"operator", "args"}:
            raise ValueError("algorithm_output_invalid: output keys must be operator and args")
        operator = raw["operator"]
        args = raw["args"]
        if not isinstance(operator, str) or operator not in cls.ALLOWED_OPERATORS:
            raise ValueError("algorithm_output_invalid: unsupported operator")
        if not isinstance(args, Mapping):
            raise ValueError("algorithm_output_invalid: args must be an object")
        _reject_unsafe_keys(args)
        if operator == "prune_blocks":
            if set(args) != {"block_ids"} or not isinstance(args["block_ids"], list) or not all(isinstance(item, str) for item in args["block_ids"]):
                raise ValueError("algorithm_output_invalid: prune_blocks requires block_ids")
        if operator == "prune_ffn_channels":
            required = {"block_ids", "retained_indices"}
            if set(args) - required or set(args) != required or not isinstance(args["block_ids"], list) or not isinstance(args["retained_indices"], list):
                raise ValueError("algorithm_output_invalid: prune_ffn_channels requires block_ids and retained_indices")
            if not all(isinstance(item, str) for item in args["block_ids"]):
                raise ValueError("algorithm_output_invalid: block_ids must contain strings")
            if not all(isinstance(item, int) and not isinstance(item, bool) for item in args["retained_indices"]):
                raise ValueError("algorithm_output_invalid: retained_indices must contain integers")
        return cls(operator, {str(key): value for key, value in args.items()})

    def to_dict(self) -> dict[str, Any]:
        return {"operator": self.operator, "args": dict(self.args)}


def _reject_unsafe_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in ("path", "device", "command", "shell", "evaluator", "environment")):
                raise ValueError("algorithm_output_invalid: unsafe field %s" % key)
            _reject_unsafe_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_unsafe_keys(item)


@dataclass(frozen=True)
class SandboxResult:
    status: str
    source_sha256: str
    plan: Optional[CompressionPlan] = None
    failure_type: Optional[str] = None
    message: str = ""
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


RUNNER_SOURCE = r'''
import json
import sys
import traceback

ALLOWED = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "isinstance": isinstance, "len": len, "list": list, "max": max,
    "min": min, "range": range, "round": round, "set": set, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
}

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split('.', 1)[0] not in {"math", "statistics", "json", "collections"}:
        raise ImportError("sandbox import denied: " + name)
    return __import__(name, globals, locals, fromlist, level)

ALLOWED["__import__"] = safe_import

source_path, profile_path, history_path, output_path = sys.argv[1:5]
source = open(source_path, "r", encoding="utf-8").read()
profile = json.loads(open(profile_path, "r", encoding="utf-8").read())
history = json.loads(open(history_path, "r", encoding="utf-8").read())
namespace = {"__name__": "__algorithm__", "__builtins__": ALLOWED}
try:
    exec(compile(source, source_path, "exec"), namespace, namespace)
    result = namespace["propose"](profile, history)
    json.dump(result, open(output_path, "w", encoding="utf-8"), sort_keys=True)
except Exception as exc:
    json.dump({"error": type(exc).__name__ + ": " + str(exc)}, open(output_path, "w", encoding="utf-8"), sort_keys=True)
    traceback.print_exc()
    raise
'''


class AlgorithmSandbox:
    def __init__(self, root: Path, *, timeout_s: float = 10.0, max_output_bytes: int = 64 * 1024) -> None:
        if timeout_s <= 0 or max_output_bytes <= 0:
            raise ValueError("sandbox limits must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout_s = float(timeout_s)
        self.max_output_bytes = int(max_output_bytes)
        self.executor = LocalProcessExecutor(timeout_s=self.timeout_s)

    def run(self, source: str, profile: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> SandboxResult:
        try:
            validated = validate_source(source)
        except SourcePolicyError as exc:
            return SandboxResult("failed", "", failure_type="sandbox_policy", message=str(exc))
        run_root = Path(tempfile.mkdtemp(prefix="algorithm-", dir=str(self.root)))
        source_path = run_root / "algorithm.py"
        profile_path = run_root / "profile.json"
        history_path = run_root / "history.json"
        output_path = run_root / "output.json"
        source_path.write_text(validated.source, encoding="utf-8")
        profile_path.write_text(json.dumps(profile, sort_keys=True), encoding="utf-8")
        history_path.write_text(json.dumps(list(history), sort_keys=True), encoding="utf-8")
        runner_path = run_root / "runner.py"
        runner_path.write_text(RUNNER_SOURCE, encoding="utf-8")
        result = self.executor.execute((sys.executable, "-I", str(runner_path), str(source_path), str(profile_path), str(history_path), str(output_path)), run_root)
        if result.status != "success":
            return SandboxResult("failed", validated.sha256, failure_type=result.failure_type or "sandbox_process", message=result.message, stdout_path=result.stdout_path, stderr_path=result.stderr_path)
        if not output_path.exists() or output_path.stat().st_size > self.max_output_bytes:
            return SandboxResult("failed", validated.sha256, failure_type="algorithm_output_invalid", message="output is missing or exceeds size limit", stdout_path=result.stdout_path, stderr_path=result.stderr_path)
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
            plan = CompressionPlan.from_mapping(raw)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return SandboxResult("failed", validated.sha256, failure_type="algorithm_output_invalid", message=str(exc), stdout_path=result.stdout_path, stderr_path=result.stderr_path)
        return SandboxResult("success", validated.sha256, plan=plan, stdout_path=result.stdout_path, stderr_path=result.stderr_path, metadata={"offline_simulation": True})
