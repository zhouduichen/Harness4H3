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

    ALLOWED_OPERATORS = frozenset({
        "prune_blocks",
        "prune_layers",
        "prune_ffn_channels",
        "prune_attention_heads",
        "prune_heads",
        "prune_tokens",
        "token_merge",
        "kv_token_select",
        "sparse_runtime",
        "attention_window",
        "sampling_step_skip",
        "timestep_block_gate",
        "compose",
    })
    BASE_OPERATORS = frozenset({
        "prune_blocks",
        "prune_layers",
        "prune_ffn_channels",
        "prune_attention_heads",
        "prune_heads",
        "prune_tokens",
        "token_merge",
        "kv_token_select",
        "sparse_runtime",
        "attention_window",
        "sampling_step_skip",
        "timestep_block_gate",
    })
    MAX_COMPOSITE_STEPS = 6

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
        if operator in {"prune_blocks", "prune_layers"}:
            if set(args) != {"block_ids"} or not isinstance(args["block_ids"], list) or not all(isinstance(item, str) for item in args["block_ids"]):
                raise ValueError("algorithm_output_invalid: %s requires block_ids" % operator)
        if operator == "prune_ffn_channels":
            required = {"block_ids", "retained_indices"}
            if set(args) - required or set(args) != required or not isinstance(args["block_ids"], list) or not isinstance(args["retained_indices"], list):
                raise ValueError("algorithm_output_invalid: prune_ffn_channels requires block_ids and retained_indices")
            if not all(isinstance(item, str) for item in args["block_ids"]):
                raise ValueError("algorithm_output_invalid: block_ids must contain strings")
            if not all(isinstance(item, int) and not isinstance(item, bool) for item in args["retained_indices"]):
                raise ValueError("algorithm_output_invalid: retained_indices must contain integers")
        if operator in {"prune_attention_heads", "prune_heads"}:
            if set(args) not in ({"head_indices"}, {"head_indices", "attention_names"}):
                raise ValueError("algorithm_output_invalid: %s requires head_indices" % operator)
            if not isinstance(args["head_indices"], list) or not args["head_indices"] or not all(isinstance(item, int) and not isinstance(item, bool) for item in args["head_indices"]):
                raise ValueError("algorithm_output_invalid: head_indices must contain integers")
            if "attention_names" in args and (not isinstance(args["attention_names"], list) or not args["attention_names"] or not all(isinstance(item, str) for item in args["attention_names"])):
                raise ValueError("algorithm_output_invalid: attention_names must contain strings")
        if operator in {"prune_tokens", "sparse_runtime"}:
            if set(args) not in ({"keep_ratio"}, {"keep_ratio", "block_ids"}, {"keep_ratio", "attention"}):
                raise ValueError("algorithm_output_invalid: %s has unsupported arguments" % operator)
            if not isinstance(args["keep_ratio"], (int, float)) or isinstance(args["keep_ratio"], bool) or not 0.0 < float(args["keep_ratio"]) <= 1.0:
                raise ValueError("algorithm_output_invalid: keep_ratio must be in (0, 1]")
            if "block_ids" in args and (not isinstance(args["block_ids"], list) or not all(isinstance(item, str) for item in args["block_ids"])):
                raise ValueError("algorithm_output_invalid: block_ids must contain strings")
            if "attention" in args and not isinstance(args["attention"], str):
                raise ValueError("algorithm_output_invalid: attention must be a string")
        if operator == "token_merge":
            allowed = (
                {"keep_ratio"},
                {"keep_ratio", "block_ids"},
                {"keep_ratio", "mode"},
                {"keep_ratio", "block_ids", "mode"},
            )
            if set(args) not in allowed:
                raise ValueError("algorithm_output_invalid: token_merge has unsupported arguments")
            if not isinstance(args["keep_ratio"], (int, float)) or isinstance(args["keep_ratio"], bool) or not 0.0 < float(args["keep_ratio"]) <= 1.0:
                raise ValueError("algorithm_output_invalid: keep_ratio must be in (0, 1]")
            if "block_ids" in args and (not isinstance(args["block_ids"], list) or not all(isinstance(item, str) for item in args["block_ids"])):
                raise ValueError("algorithm_output_invalid: block_ids must contain strings")
            if "mode" in args and not isinstance(args["mode"], str):
                raise ValueError("algorithm_output_invalid: mode must be a string")
        if operator == "kv_token_select":
            allowed = (
                {"keep_ratio"},
                {"keep_ratio", "block_ids"},
                {"keep_ratio", "score"},
                {"keep_ratio", "block_ids", "score"},
            )
            if set(args) not in allowed:
                raise ValueError("algorithm_output_invalid: kv_token_select has unsupported arguments")
            if not isinstance(args["keep_ratio"], (int, float)) or isinstance(args["keep_ratio"], bool) or not 0.0 < float(args["keep_ratio"]) <= 1.0:
                raise ValueError("algorithm_output_invalid: keep_ratio must be in (0, 1]")
            if "block_ids" in args and (not isinstance(args["block_ids"], list) or not all(isinstance(item, str) for item in args["block_ids"])):
                raise ValueError("algorithm_output_invalid: block_ids must contain strings")
            if "score" in args and args["score"] not in {"l2_norm", "mean_abs"}:
                raise ValueError("algorithm_output_invalid: unsupported KV token selection score")
        if operator == "attention_window":
            if set(args) not in ({"window_size"}, {"window_size", "attention"}):
                raise ValueError("algorithm_output_invalid: attention_window has unsupported arguments")
            if not isinstance(args["window_size"], int) or isinstance(args["window_size"], bool) or not 1 <= int(args["window_size"]) <= 4096:
                raise ValueError("algorithm_output_invalid: window_size must be in [1, 4096]")
            if "attention" in args and not isinstance(args["attention"], str):
                raise ValueError("algorithm_output_invalid: attention must be a string")
        if operator == "sampling_step_skip":
            if set(args) not in ({"target_steps"}, {"target_steps", "strategy"}, {"target_steps", "strategy", "min_steps"}):
                raise ValueError("algorithm_output_invalid: sampling_step_skip has unsupported arguments")
            if not isinstance(args["target_steps"], int) or isinstance(args["target_steps"], bool) or not 1 <= int(args["target_steps"]) <= 256:
                raise ValueError("algorithm_output_invalid: target_steps must be in [1, 256]")
            if "strategy" in args and not isinstance(args["strategy"], str):
                raise ValueError("algorithm_output_invalid: strategy must be a string")
            if "min_steps" in args and (not isinstance(args["min_steps"], int) or isinstance(args["min_steps"], bool)):
                raise ValueError("algorithm_output_invalid: min_steps must be an integer")
        if operator == "timestep_block_gate":
            required = {"rules", "policy"}
            if set(args) != required or not isinstance(args["rules"], list) or not isinstance(args["policy"], str):
                raise ValueError("algorithm_output_invalid: timestep_block_gate requires rules and policy")
            if not all(isinstance(item, Mapping) for item in args["rules"]):
                raise ValueError("algorithm_output_invalid: timestep rules must contain objects")
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
            if type_name == "list[object]" and (
                not isinstance(value, list)
                or not all(isinstance(item, Mapping) for item in value)
            ):
                raise ValueError("algorithm_output_invalid: %s must be a list of objects" % key)
            if type_name == "float" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                raise ValueError("algorithm_output_invalid: %s must be numeric" % key)
            if type_name == "int" and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError("algorithm_output_invalid: %s must be an integer" % key)
            if type_name == "str" and not isinstance(value, str):
                raise ValueError("algorithm_output_invalid: %s must be a string" % key)
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


@dataclass(frozen=True)
class ProgramResult:
    """Raw JSON result from one isolated Algorithm Program invocation."""

    status: str
    source_sha256: str
    value: Optional[Any] = None
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
    _cpu_seconds = max(1, int(math.ceil(float(sys.argv[4]))))
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

source_path, payload_path, output_path, _timeout_s, entrypoint = sys.argv[1:6]
source = open(source_path, "r", encoding="utf-8").read()
payload = json.loads(open(payload_path, "r", encoding="utf-8").read())
namespace = {"__name__": "__algorithm__", "__builtins__": ALLOWED}
sys.addaudithook(_algorithm_audit)
try:
    _ALGORITHM_PHASE = False
    compiled = compile(source, source_path, "exec")
    exec(compiled, namespace, namespace)
    if entrypoint not in namespace or not callable(namespace[entrypoint]):
        raise ValueError("missing_entrypoint: " + entrypoint)
    _ALGORITHM_PHASE = True
    result = namespace[entrypoint](**payload)
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
        self.operator_specs = {}
        for item in operator_specs:
            if callable(getattr(item, "to_dict", None)):
                name = getattr(item, "name", None)
                value = item.to_dict()
            elif isinstance(item, Mapping):
                name = item.get("name")
                value = dict(item)
            else:
                raise TypeError("operator specification must be an object or mapping")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("operator specification requires a name")
            self.operator_specs[name] = value
        self.executor = LocalProcessExecutor(timeout_s=self.timeout_s)

    def run(self, source: str, profile: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> SandboxResult:
        program = self.run_program(
            source,
            entrypoint="propose",
            payload={"profile": dict(profile), "history": list(history)},
        )
        if program.status != "success":
            return SandboxResult(
                "failed",
                program.source_sha256,
                failure_type=program.failure_type or "sandbox_process",
                message=program.message,
                stdout_path=program.stdout_path,
                stderr_path=program.stderr_path,
                metadata=dict(program.metadata),
            )
        try:
            plan = CompressionPlan.from_mapping(program.value, operator_specs=self.operator_specs)
        except (TypeError, ValueError) as exc:
            return SandboxResult(
                "failed",
                program.source_sha256,
                failure_type="algorithm_output_invalid",
                message=str(exc),
                stdout_path=program.stdout_path,
                stderr_path=program.stderr_path,
                metadata=dict(program.metadata),
            )
        return SandboxResult(
            "success",
            program.source_sha256,
            plan=plan,
            stdout_path=program.stdout_path,
            stderr_path=program.stderr_path,
            metadata=dict(program.metadata),
        )

    def run_program(
        self,
        source: str,
        *,
        entrypoint: str,
        payload: Mapping[str, Any],
    ) -> ProgramResult:
        try:
            validated = validate_source(source, required_entrypoint=entrypoint)
        except SourcePolicyError as exc:
            return ProgramResult("failed", "", failure_type="sandbox_policy", message=str(exc), metadata={"entrypoint": str(entrypoint)})
        run_root = Path(tempfile.mkdtemp(prefix="algorithm-", dir=str(self.root)))
        source_path = run_root / "algorithm.py"
        payload_path = run_root / "payload.json"
        output_path = run_root / "output.json"
        source_path.write_text(validated.source, encoding="utf-8")
        try:
            payload_path.write_text(json.dumps(dict(payload), sort_keys=True), encoding="utf-8")
        except (TypeError, ValueError) as exc:
            return ProgramResult("failed", validated.sha256, failure_type="algorithm_payload_invalid", message=str(exc), metadata={"entrypoint": str(entrypoint)})
        runner_path = run_root / "runner.py"
        runner_path.write_text(RUNNER_SOURCE, encoding="utf-8")
        result = self.executor.execute(
            (
                sys.executable,
                "-I",
                str(runner_path),
                str(source_path),
                str(payload_path),
                str(output_path),
                str(self.timeout_s),
                str(entrypoint),
            ),
            run_root,
        )
        if result.status != "success":
            return ProgramResult(
                "failed",
                validated.sha256,
                failure_type=result.failure_type or "sandbox_process",
                message=result.message,
                stdout_path=result.stdout_path,
                stderr_path=result.stderr_path,
                metadata={"entrypoint": str(entrypoint), "process_isolation": "python-isolated-process"},
            )
        if not output_path.exists() or output_path.stat().st_size > self.max_output_bytes:
            return ProgramResult(
                "failed",
                validated.sha256,
                failure_type="algorithm_output_invalid",
                message="output is missing or exceeds size limit",
                stdout_path=result.stdout_path,
                stderr_path=result.stderr_path,
                metadata={"entrypoint": str(entrypoint)},
            )
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return ProgramResult(
                "failed",
                validated.sha256,
                failure_type="algorithm_output_invalid",
                message=str(exc),
                stdout_path=result.stdout_path,
                stderr_path=result.stderr_path,
                metadata={"entrypoint": str(entrypoint)},
            )
        return ProgramResult(
            "success",
            validated.sha256,
            value=raw,
            stdout_path=result.stdout_path,
            stderr_path=result.stderr_path,
            metadata={
                "entrypoint": str(entrypoint),
                "offline_simulation": True,
                "process_isolation": "python-isolated-process",
                "runtime_audit": "deny-file-network-process-dynamic-code",
                "resource_limits": {"timeout_s": self.timeout_s, "cpu_s": max(1, int(self.timeout_s)), "address_space_bytes": 512 * 1024 * 1024, "file_size_bytes": 1024 * 1024, "open_files": 32},
                "network_policy": "runtime-audit-deny; host network namespace not assumed",
                "search_space_operator_names": sorted(self.operator_specs),
            },
        )
