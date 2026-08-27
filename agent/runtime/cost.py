from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "Usage":
        raw = raw or {}
        prompt = int(raw.get("prompt_tokens", 0) or 0)
        completion = int(raw.get("completion_tokens", 0) or 0)
        total = int(raw.get("total_tokens", 0) or prompt + completion)
        return cls(prompt, completion, total)

    def as_dict(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens}


@dataclass
class CostLedger:
    """Tracks raw model usage and makes a token budget enforceable."""

    token_budget: int | None = None
    usage: Usage = field(default_factory=Usage)
    calls: int = 0

    def add(self, raw_usage: dict[str, Any] | Usage | None) -> Usage:
        incoming = raw_usage if isinstance(raw_usage, Usage) else Usage.from_dict(raw_usage)
        self.usage = Usage(
            self.usage.prompt_tokens + incoming.prompt_tokens,
            self.usage.completion_tokens + incoming.completion_tokens,
            self.usage.total_tokens + incoming.total_tokens,
        )
        self.calls += 1
        return incoming

    @property
    def remaining_tokens(self) -> int | None:
        if self.token_budget is None:
            return None
        return self.token_budget - self.usage.total_tokens

    @property
    def is_within_budget(self) -> bool:
        return self.token_budget is None or self.usage.total_tokens <= self.token_budget
