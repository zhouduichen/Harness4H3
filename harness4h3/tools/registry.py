from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    value: Any = None
    error: Optional[str] = None


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    execute: Callable[[Mapping[str, Any]], Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name or tool.name in self._tools:
            raise ValueError("tool name must be non-empty and unique: %r" % tool.name)
        self._tools[tool.name] = tool

    def visible(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(
            {"name": tool.name, "description": tool.description, "input_schema": dict(tool.input_schema)}
            for tool in self._tools.values()
        )

    def execute(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, error="unknown tool %s" % name)
        try:
            return ToolResult(True, value=tool.execute(arguments))
        except Exception as exc:
            return ToolResult(False, error=str(exc))

