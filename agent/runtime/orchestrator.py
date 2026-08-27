from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..qwen_client import OpenAICompatibleClient
from .cost import CostLedger
from .memory import RunMemory
from .planner import build_plan
from .reflection import reflect_answer
from .state import AgentState, TaskStatus, utc_now
from .tools import ToolRegistry, ToolSpec


@dataclass(frozen=True)
class OrchestratorConfig:
    evidence_mode: str = "compact"
    token_budget_per_task: int | None = None
    enable_llm: bool = True


class EvidenceGroundedOrchestrator:
    """Plans, retrieves, solves, reflects, and records a single QA task.

    Every LLM call goes through the ledger and every deterministic action is
    written to the trace, allowing operational review without storing hidden
    reasoning traces.
    """

    def __init__(
        self,
        *,
        retrieve: Callable[[dict[str, Any], str], list[dict[str, Any]]],
        client: OpenAICompatibleClient | None = None,
        memory: RunMemory | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.retrieve = retrieve
        self.client = client
        self.memory = memory or RunMemory()
        self.config = config or OrchestratorConfig()
        self.tools = ToolRegistry()
        self.tools.register(ToolSpec("retrieve_evidence", "Retrieve traceable source chunks for a question.", self._retrieve, ("question",)))
        self.tools.register(ToolSpec("validate_output", "Validate answer shape and evidence coverage.", self._validate, ("question", "answer", "evidence")))

    def _retrieve(self, question: dict[str, Any]) -> list[dict[str, Any]]:
        return self.retrieve(question, self.config.evidence_mode)

    @staticmethod
    def _validate(question: dict[str, Any], answer: str, evidence: list[dict[str, Any]], reasoning: str = "") -> dict[str, Any]:
        return reflect_answer(question, answer, evidence, reasoning)

    def run(self, task_id: str, question: dict[str, Any]) -> AgentState:
        state = AgentState(task_id=task_id, question=dict(question), status=TaskStatus.RUNNING)
        state.plan = build_plan(question)
        state.memory_hits = self.memory.search(str(question.get("question", "")))
        ledger = CostLedger(self.config.token_budget_per_task)
        try:
            started = utc_now()
            evidence, call = self.tools.call("retrieve_evidence", question=question)
            state.evidence = evidence
            state.tool_calls.append(call)
            state.record(step="retrieve_evidence", actor="retriever", status="ok", started_at=started, output_summary=f"Retrieved {len(evidence)} evidence chunks.", tool_name="retrieve_evidence")

            if self.config.enable_llm:
                if self.client is None:
                    raise RuntimeError("LLM execution is enabled but no client was configured")
                started = utc_now()
                response = self.client.chat(self._messages(question, evidence, state.memory_hits))
                usage = ledger.add(response.usage).as_dict()
                state.add_usage(usage)
                parsed = self._parse_response(response.content)
                state.candidate_answer = str(parsed.get("answer", ""))
                state.reasoning = str(parsed.get("reasoning", ""))
                state.record(step="solve", actor="solver", status="ok", started_at=started, output_summary=state.candidate_answer, usage=usage)
                if not ledger.is_within_budget:
                    state.status = TaskStatus.BUDGET_EXCEEDED
                    state.errors.append("token_budget_exceeded")
                    return state
            else:
                state.candidate_answer = str(question.get("candidate_answer", ""))

            started = utc_now()
            reflection, call = self.tools.call("validate_output", question=question, answer=state.candidate_answer, evidence=state.evidence, reasoning=state.reasoning)
            state.reflection = reflection
            state.tool_calls.append(call)
            state.final_answer = str(reflection["normalized_answer"])
            state.status = TaskStatus.SUCCEEDED if reflection["accepted"] else TaskStatus.NEEDS_REVIEW
            state.record(step="reflect", actor="reflection_agent", status=state.status.value, started_at=started, output_summary=";".join(reflection["issues"]), tool_name="validate_output")
            self.memory.add({"task_id": task_id, "domain": question.get("domain", ""), "question": question.get("question", ""), "lesson": ";".join(reflection["issues"]), "answer": state.final_answer})
            return state
        except Exception as exc:
            state.status = TaskStatus.FAILED
            state.errors.append(str(exc))
            state.record(step="runtime", actor="orchestrator", status="failed", started_at=utc_now(), error=str(exc))
            return state

    @staticmethod
    def _messages(question: dict[str, Any], evidence: list[dict[str, Any]], memory_hits: list[dict[str, Any]]) -> list[dict[str, str]]:
        evidence_text = "\n\n".join(f"[{idx + 1}] {item.get('doc_id', '')} {item.get('text', '')}" for idx, item in enumerate(evidence))
        prompt = {
            "question": question.get("question", ""),
            "options": question.get("options", {}),
            "answer_format": question.get("answer_format", question.get("type", "")),
            "evidence": evidence_text,
            "memory_lessons": [item.get("lesson", "") for item in memory_hits],
        }
        return [
            {"role": "system", "content": "You are an evidence-grounded financial QA agent. Use only the provided evidence. Return JSON with answer and a concise auditable reasoning summary."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]

    @staticmethod
    def _parse_response(content: str) -> dict[str, Any]:
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {"answer": content}
