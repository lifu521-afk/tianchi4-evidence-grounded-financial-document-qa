from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable

from .state import AgentState, TaskStatus, utc_now


Runner = Callable[[str, dict], AgentState]
Hook = Callable[[AgentState], None]


@dataclass(frozen=True)
class HarnessConfig:
    """Execution policy for one or more Agent tasks."""

    max_attempts: int = 1
    retry_failed: bool = True
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")


@dataclass(frozen=True)
class HarnessResult:
    state: AgentState
    attempts: int
    duration_ms: float

    def as_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
            "status": self.state.status.value,
        }


class AgentHarness:
    """Standard lifecycle wrapper around an evidence-grounded Agent runner.

    The harness owns operational concerns such as retries, hooks, timing, and
    batch failure policy. Domain logic stays inside the supplied runner. When
    a task is retried, all prior raw usage and trace events are merged into the
    returned state so cost accounting remains complete.
    """

    def __init__(
        self,
        runner: Runner,
        *,
        config: HarnessConfig | None = None,
        before_task: Hook | None = None,
        after_task: Hook | None = None,
    ) -> None:
        self.runner = runner
        self.config = config or HarnessConfig()
        self.before_task = before_task
        self.after_task = after_task

    def run(self, task_id: str, question: dict) -> HarnessResult:
        if not task_id:
            raise ValueError("task_id must be non-empty")
        if not isinstance(question, dict):
            raise TypeError("question must be a dict")

        started_at = perf_counter()
        attempts = 0
        history: AgentState | None = None
        final_state: AgentState | None = None
        while attempts < self.config.max_attempts:
            attempts += 1
            state = AgentState(task_id=task_id, question=dict(question))
            if self.before_task is not None:
                self.before_task(state)
            try:
                state = self.runner(task_id, question)
            except Exception as exc:
                state.status = TaskStatus.FAILED
                state.errors.append(str(exc))
                state.record(step="harness_runner", actor="harness", status="failed", started_at=utc_now(), error=str(exc))
            final_state = self._merge_history(history, state)
            history = final_state
            if final_state.status != TaskStatus.FAILED or not self.config.retry_failed:
                break

        assert final_state is not None
        final_state.record(
            step="harness",
            actor="agent_harness",
            status=final_state.status.value,
            started_at=utc_now(),
            input_summary=f"attempts={attempts}",
            output_summary=f"duration_ms={round((perf_counter() - started_at) * 1000, 2)}",
        )
        if self.after_task is not None:
            self.after_task(final_state)
        return HarnessResult(final_state, attempts, round((perf_counter() - started_at) * 1000, 2))

    def run_many(self, tasks: Iterable[tuple[str, dict]]) -> list[HarnessResult]:
        results: list[HarnessResult] = []
        for task_id, question in tasks:
            result = self.run(task_id, question)
            results.append(result)
            if self.config.fail_fast and result.state.status == TaskStatus.FAILED:
                break
        return results

    @staticmethod
    def _merge_history(previous: AgentState | None, current: AgentState) -> AgentState:
        if previous is None:
            return current
        current.trace = previous.trace + current.trace
        current.tool_calls = previous.tool_calls + current.tool_calls
        current.errors = previous.errors + current.errors
        for field_name in current.usage:
            current.usage[field_name] += previous.usage.get(field_name, 0)
        return current
