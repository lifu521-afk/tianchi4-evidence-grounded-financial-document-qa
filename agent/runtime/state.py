from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    NEEDS_REVIEW = "needs_review"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TraceEvent:
    """An auditable state transition, not a hidden chain-of-thought record."""

    step: str
    actor: str
    status: str
    started_at: str
    finished_at: str
    input_summary: str = ""
    output_summary: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    tool_name: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    task_id: str
    question: dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    plan: list[str] = field(default_factory=list)
    current_step: int = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    candidate_answer: str = ""
    final_answer: str = ""
    reasoning: str = ""
    reflection: dict[str, Any] = field(default_factory=dict)
    memory_hits: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    trace: list[TraceEvent] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_usage(self, usage: dict[str, Any] | None) -> None:
        for field_name in self.usage:
            self.usage[field_name] += int((usage or {}).get(field_name, 0) or 0)

    def record(
        self,
        *,
        step: str,
        actor: str,
        status: str,
        started_at: str,
        input_summary: str = "",
        output_summary: str = "",
        usage: dict[str, int] | None = None,
        tool_name: str = "",
        error: str = "",
    ) -> None:
        self.trace.append(
            TraceEvent(
                step=step,
                actor=actor,
                status=status,
                started_at=started_at,
                finished_at=utc_now(),
                input_summary=input_summary[:500],
                output_summary=output_summary[:1000],
                usage=dict(usage or {}),
                tool_name=tool_name,
                error=error[:500],
            )
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result
