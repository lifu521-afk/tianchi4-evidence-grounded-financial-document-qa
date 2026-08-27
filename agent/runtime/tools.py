from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable


ToolFunction = Callable[..., Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: ToolFunction
    required_args: tuple[str, ...] = ()


class ToolRegistry:
    """Explicit allow-list for deterministic and model-invoked tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not spec.name or spec.name in self._tools:
            raise ValueError(f"Tool name must be unique and non-empty: {spec.name!r}")
        self._tools[spec.name] = spec

    def describe(self) -> list[dict[str, Any]]:
        return [
            {"name": spec.name, "description": spec.description, "required_args": list(spec.required_args)}
            for spec in self._tools.values()
        ]

    def call(self, name: str, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        spec = self._tools[name]
        missing = [arg for arg in spec.required_args if arg not in kwargs]
        if missing:
            raise ValueError(f"Tool {name} missing required arguments: {', '.join(missing)}")
        started = perf_counter()
        result = spec.handler(**kwargs)
        return result, {"tool_name": name, "arguments": kwargs, "duration_ms": round((perf_counter() - started) * 1000, 2)}
