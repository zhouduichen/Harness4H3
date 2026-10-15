from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Iterable


ALLOWED_IMPORTS = frozenset({"math", "statistics", "json", "collections"})
FORBIDDEN_ROOTS = frozenset({"os", "sys", "subprocess", "socket", "pathlib", "shutil", "requests", "torch", "site"})
FORBIDDEN_CALLS = frozenset({"eval", "exec", "compile", "open", "input", "__import__", "breakpoint"})
FORBIDDEN_REFLECTION_NAMES = frozenset({"__builtins__", "__loader__", "__spec__", "__file__", "__package__", "__import__"})
MAX_SOURCE_BYTES = 64 * 1024


class SourcePolicyError(ValueError):
    pass


@dataclass(frozen=True)
class ValidatedSource:
    source: str
    sha256: str
    functions: tuple[str, ...]


def _root_name(node: ast.AST) -> str | None:
    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def validate_source(source: str, required_entrypoint: str = "propose") -> ValidatedSource:
    if not isinstance(source, str) or not source.strip():
        raise SourcePolicyError("source must be non-empty text")
    encoded = source.encode("utf-8")
    if len(encoded) > MAX_SOURCE_BYTES:
        raise SourcePolicyError("source exceeds sandbox size limit")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise SourcePolicyError("syntax_error: %s" % exc) from exc

    functions: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names: Iterable[str]
            if isinstance(node, ast.Import):
                names = (item.name.split(".", 1)[0] for item in node.names)
            else:
                names = (node.module.split(".", 1)[0] if node.module else "" for _ in node.names)
            for name in names:
                if name not in ALLOWED_IMPORTS:
                    raise SourcePolicyError("forbidden_import: %s" % name)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise SourcePolicyError("forbidden_call: %s" % node.func.id)
            root = _root_name(node.func)
            if root in FORBIDDEN_ROOTS:
                raise SourcePolicyError("forbidden_attribute_root: %s" % root)
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise SourcePolicyError("forbidden_dunder_attribute: %s" % node.attr)
            root = _root_name(node)
            if root in FORBIDDEN_ROOTS:
                raise SourcePolicyError("forbidden_attribute_root: %s" % root)
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_REFLECTION_NAMES:
            raise SourcePolicyError("forbidden_name: %s" % node.id)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise SourcePolicyError("forbidden_scope_declaration")
    entrypoint = str(required_entrypoint).strip()
    if not entrypoint or not entrypoint.isidentifier():
        raise SourcePolicyError("invalid_entrypoint: %s" % required_entrypoint)
    if entrypoint not in functions:
        raise SourcePolicyError("missing_entrypoint: %s" % entrypoint)
    digest = hashlib.sha256(encoded).hexdigest()
    return ValidatedSource(source, digest, tuple(functions))
