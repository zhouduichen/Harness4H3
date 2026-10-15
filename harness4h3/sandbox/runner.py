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


_ALLOWED_BUILTIN_NAMES = ("abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list", "max", "min", "next", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip")

ALLOWED_BUILTINS = {
    name: __builtins__[name]  # type: ignore[index]
    for name in _ALLOWED_BUILTIN_NAMES
    if isinstance(__builtins__, dict) and name in __builtins__
}
if not isinstance(__builtins__, dict):  # pragma: no cover - interpreter detail
    ALLOWED_BUILTINS = {name: getattr(__builtins__, name) for name in _ALLOWED_BUILTIN_NAMES}


def _safe_import(name: str, globals: Optional[Mapping[str, Any]] = None, locals: Optional[Mapping[str, Any]] = None, fromlist: Sequence[str] = (), level: int = 0) -> Any:
    if level or name.split(".", 1)[0] not in {"math", "statistics", "json", "collections"}:
        raise ImportError("sandbox import denied: %s" % name)
    return __import__(name, globals, locals, fromlist, level)


ALLOWED_BUILTINS["__import__"] = _safe_import


@dataclass(frozen=True)
class CompressionPlan:
    operator: str
    args: Mapping[str, Any]

    ALLOWED_OPERATORS = frozenset({"prune_blocks", "prune_ffn_channels", "compose"})
    BASE_OPERATORS = frozenset({"prune_blocks", "prune_ffn_channels"})
    MAX_COMPOSITE_STEPS = 3

    @classmethod
    def from_mapping(cls, raw: Any, *, operator_specs: Optional[Mapping[str, Mapping[str, Any]]] = None) -> "CompressionPlan":
        if not isinstance(raw, Mapping):
            raise ValueError("algorithm_output_invalid: output must be an object")
        # Some structured providers use the repository's persisted decision
        # name (`operator_args`) when returning the sandbox plan.  Normalize
        # that single compatibility alias at the trust boundary; all later
        # validation remains identical and no extra fields are accepted.
        if set(raw) == {"operator", "operator_args"}:
            raw = {"operator": raw["operator"], "args": raw["operator_args"]}
        elif len(raw) == 1:
            # Controllers occasionally return the operator as the top-level
            # key (for example, {"prune_blocks": {"block_ids": [...]}}).
            # Normalize that harmless schema variant here, before applying
            # the same strict operator/argument validation below.
            operator, args = next(iter(raw.items()))
            if operator in cls.ALLOWED_OPERATORS:
                raw = {"operator": operator, "args": args}
        if set(raw) != {"operator", "args"}:
            raise ValueError("algorithm_output_invalid: output keys must be operator and args")
        operator = raw["operator"]
        args = raw["args"]
        extensions = dict(operator_specs or {})
        if not isinstance(operator, str) or (operator not in cls.ALLOWED_OPERATORS and operator not in extensions):
            raise ValueError("algorithm_output_invalid: unsupported operator")
        if not isinstance(args, Mapping):
            raise ValueError("algorithm_output_invalid: args must be an object")
        _reject_unsafe_keys(args)
        if operator == "compose":
            if set(args) != {"steps"} or not isinstance(args["steps"], list):
                raise ValueError("algorithm_output_invalid: compose requires steps")
            steps = args["steps"]
            if not 2 <= len(steps) <= cls.MAX_COMPOSITE_STEPS:
                raise ValueError("algorithm_output_invalid: compose requires 2 to %d steps" % cls.MAX_COMPOSITE_STEPS)
            normalized_steps = []
            for step in steps:
                if not isinstance(step, Mapping) or step.get("operator") == "compose":
                    raise ValueError("algorithm_output_invalid: nested compose is not allowed")
                normalized_steps.append(cls._from_base_mapping(step).to_dict())
            return cls("compose", {"steps": normalized_steps})
        if operator in cls.BASE_OPERATORS:
            return cls._from_base_mapping({"operator": operator, "args": args})
        return cls._from_extension_mapping(operator, args, extensions[operator])

    @classmethod
    def _from_base_mapping(cls, raw: Mapping[str, Any]) -> "CompressionPlan":
        operator = raw.get("operator")
        args = raw.get("args")
        if operator not in cls.BASE_OPERATORS or not isinstance(args, Mapping):
            raise ValueError("algorithm_output_invalid: composite steps must use base operators")
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
        return cls(str(operator), {str(key): value for key, value in args.items()})

    @classmethod
    def _from_extension_mapping(cls, operator: str, args: Mapping[str, Any], spec: Mapping[str, Any]) -> "CompressionPlan":
        schema = spec.get("argument_schema") if isinstance(spec, Mapping) else None
        if not isinstance(schema, Mapping) or set(args) != set(schema):
            raise ValueError("algorithm_output_invalid: extension arguments do not match operator schema")
        for key, type_name in schema.items():
            value = args[key]
            if type_name == "list[str]" and (not isinstance(value, list) or not all(isinstance(item, str) for item in value)):
                raise ValueError("algorithm_output_invalid: %s must be a list of strings" % key)
            if type_name == "list[int]" and (not isinstance(value, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)):
                raise ValueError("algorithm_output_invalid: %s must be a list of integers" % key)
            if type_name == "float" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                raise ValueError("algorithm_output_invalid: %s must be numeric" % key)
        return cls(str(operator), {str(key): value for key, value in args.items()})

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
import math
import sys
import traceback
try:
    import resource
except ImportError:  # pragma: no cover - Windows has no resource module
    resource = None


_ALGORITHM_PHASE = False
_ALLOWED_AUDIT_IMPORTS = {"math", "statistics", "json", "collections"}
_BLOCKED_AUDIT_PREFIXES = (
    "os.",
    "pty.",
    "socket.",
    "subprocess.",
    "ctypes.",
    "shutil.",
)


def _algorithm_audit(event, args):
    """Fail closed for operations that the source policy cannot express.

    The source runs with a reduced builtins table, but Python object graphs can
    otherwise recover the interpreter's real builtins.  Audit hooks provide a
    second, runtime boundary for file, network, process, and dynamic-code
    operations.  Trusted runner I/O is performed outside ``_ALGORITHM_PHASE``.
    """

    if not _ALGORITHM_PHASE:
        return
    if event == "import":
        name = str(args[0]) if args else ""
        if name.split(".", 1)[0] not in _ALLOWED_AUDIT_IMPORTS:
            raise PermissionError("sandbox import denied: " + name)
        return
    if event in {"open", "exec", "compile", "sys.setprofile", "sys.settrace", "sys.addaudithook"}:
        raise PermissionError("sandbox audit denied: " + event)
    if event.startswith(_BLOCKED_AUDIT_PREFIXES) or event in {
        "os.system",
        "os.exec",
        "os.fork",
        "os.posix_spawn",
        "os.spawn",
        "os.remove",
        "os.rename",
        "os.unlink",
        "os.mkdir",
        "os.rmdir",
        "os.chdir",
        "os.chmod",
        "os.chown",
        "os.link",
        "os.symlink",
    }:
        raise PermissionError("sandbox audit denied: " + event)

try:
    if resource is None:
        raise OSError("resource limits unavailable")
    _cpu_seconds = max(1, int(math.ceil(float(sys.argv[5]))))
    resource.setrlimit(resource.RLIMIT_CPU, (_cpu_seconds, _cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    if hasattr(resource, "RLIMIT_CORE"):
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except (AttributeError, OSError, ValueError) as exc:
    # A process without enforced bounds is not a safe execution environment.
    # Fail before loading untrusted algorithm code instead of silently
    # degrading to AST-only validation.
    raise RuntimeError("sandbox resource limits unavailable") from exc

# Some POSIX platforms expose RLIMIT_AS but reject lowering it from an
# unlimited process address space.  CPU, file-size, descriptor, core-dump,
# timeout, and audit boundaries remain mandatory; address-space enforcement is
# an optional stronger bound when the host kernel supports it.
if hasattr(resource, "RLIMIT_AS"):
    try:
        # The algorithm receives summaries only; it has no reason to allocate
        # hundreds of megabytes while choosing a plan.
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    except (AttributeError, OSError, ValueError):
        pass

ALLOWED = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "isinstance": isinstance, "len": len, "list": list, "max": max,
    "min": min, "next": next, "range": range, "round": round, "set": set, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
}

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split('.', 1)[0] not in {"math", "statistics", "json", "collections"}:
        raise ImportError("sandbox import denied: " + name)
    return __import__(name, globals, locals, fromlist, level)

ALLOWED["__import__"] = safe_import

source_path, profile_path, history_path, output_path, _timeout_s = sys.argv[1:6]
source = open(source_path, "r", encoding="utf-8").read()
profile = json.loads(open(profile_path, "r", encoding="utf-8").read())
history = json.loads(open(history_path, "r", encoding="utf-8").read())
namespace = {"__name__": "__algorithm__", "__builtins__": ALLOWED}
sys.addaudithook(_algorithm_audit)
try:
    _ALGORITHM_PHASE = False
    compiled = compile(source, source_path, "exec")
    exec(compiled, namespace, namespace)
    _ALGORITHM_PHASE = True
    result = namespace["propose"](profile, history)
    _ALGORITHM_PHASE = False
    json.dump(result, open(output_path, "w", encoding="utf-8"), sort_keys=True)
except Exception as exc:
    _ALGORITHM_PHASE = False
    json.dump({"error": type(exc).__name__ + ": " + str(exc)}, open(output_path, "w", encoding="utf-8"), sort_keys=True)
    traceback.print_exc()
    raise
'''


class AlgorithmSandbox:
    def __init__(self, root: Path, *, timeout_s: float = 10.0, max_output_bytes: int = 64 * 1024, operator_specs: Sequence[Any] = ()) -> None:
        if timeout_s <= 0 or max_output_bytes <= 0:
            raise ValueError("sandbox limits must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout_s = float(timeout_s)
        self.max_output_bytes = int(max_output_bytes)
        self.operator_specs = {
            str(item.name): item.to_dict() if callable(getattr(item, "to_dict", None)) else dict(item)
            for item in operator_specs
        }
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
        result = self.executor.execute((sys.executable, "-I", str(runner_path), str(source_path), str(profile_path), str(history_path), str(output_path), str(self.timeout_s)), run_root)
        if result.status != "success":
            return SandboxResult("failed", validated.sha256, failure_type=result.failure_type or "sandbox_process", message=result.message, stdout_path=result.stdout_path, stderr_path=result.stderr_path)
        if not output_path.exists() or output_path.stat().st_size > self.max_output_bytes:
            return SandboxResult("failed", validated.sha256, failure_type="algorithm_output_invalid", message="output is missing or exceeds size limit", stdout_path=result.stdout_path, stderr_path=result.stderr_path)
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
            plan = CompressionPlan.from_mapping(raw, operator_specs=self.operator_specs)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return SandboxResult("failed", validated.sha256, failure_type="algorithm_output_invalid", message=str(exc), stdout_path=result.stdout_path, stderr_path=result.stderr_path)
        return SandboxResult(
            "success",
            validated.sha256,
            plan=plan,
            stdout_path=result.stdout_path,
            stderr_path=result.stderr_path,
            metadata={
                "offline_simulation": True,
                "process_isolation": "python-isolated-process",
                "runtime_audit": "deny-file-network-process-dynamic-code",
                "resource_limits": {"timeout_s": self.timeout_s, "cpu_s": max(1, int(self.timeout_s)), "address_space_bytes": 512 * 1024 * 1024, "file_size_bytes": 1024 * 1024, "open_files": 32},
                "network_policy": "runtime-audit-deny; host network namespace not assumed",
                "search_space_operator_names": sorted(self.operator_specs),
            },
        )
