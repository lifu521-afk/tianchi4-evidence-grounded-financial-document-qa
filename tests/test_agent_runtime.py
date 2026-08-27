from __future__ import annotations

from agent.retrieval import LexicalIndex
from agent.runtime import EvidenceGroundedOrchestrator, OrchestratorConfig, TaskStatus, evaluate_records
from agent.runtime.cost import CostLedger
from agent.runtime.graph import GraphNode, StateGraph
from agent.runtime.memory import RunMemory
from agent.runtime.rag import EvidenceRAG
from agent.runtime.state import AgentState
from agent.runtime.tools import ToolRegistry, ToolSpec


QUESTION = {"qid": "q1", "question": "条款是否允许回售", "answer_format": "mcq", "options": {"A": "允许", "B": "不允许"}, "candidate_answer": "A"}


def test_runtime_records_plan_tools_reflection_and_memory() -> None:
    orchestrator = EvidenceGroundedOrchestrator(
        retrieve=lambda question, mode: [{"doc_id": "contract", "text": "持有人有权回售。"}],
        config=OrchestratorConfig(enable_llm=False),
    )
    state = orchestrator.run("q1", QUESTION)
    assert state.status == TaskStatus.SUCCEEDED
    assert state.final_answer == "A"
    assert state.plan[0] == "retrieve_evidence"
    assert state.tool_calls[0]["tool_name"] == "retrieve_evidence"
    assert state.trace[-1].actor == "reflection_agent"
    assert len(orchestrator.memory.records) == 1


def test_runtime_requires_evidence_before_accepting_answer() -> None:
    orchestrator = EvidenceGroundedOrchestrator(retrieve=lambda question, mode: [], config=OrchestratorConfig(enable_llm=False))
    state = orchestrator.run("q1", QUESTION)
    assert state.status == TaskStatus.NEEDS_REVIEW
    assert "missing_evidence" in state.reflection["issues"]


def test_cost_budget_and_tool_validation() -> None:
    ledger = CostLedger(token_budget=10)
    ledger.add({"prompt_tokens": 4, "completion_tokens": 7, "total_tokens": 11})
    assert not ledger.is_within_budget
    registry = ToolRegistry()
    registry.register(ToolSpec("echo", "echo", lambda value: value, ("value",)))
    assert registry.call("echo", value="ok")[0] == "ok"


def test_memory_is_scoped_and_evaluation_metrics_are_correct() -> None:
    first = RunMemory()
    second = RunMemory()
    first.add({"question": "债券期限", "lesson": "check maturity", "domain": "financial_contracts"})
    assert first.search("债券期限")
    assert not second.search("债券期限")
    report = evaluate_records([{"final_answer": "A", "expected_answer": "A", "evidence": [{}], "reflection": {}, "usage": {"total_tokens": 12}}])
    assert report.accuracy == 1.0
    assert report.evidence_coverage == 1.0
    assert report.total_tokens == 12


def test_rag_adapter_maps_context_modes_to_existing_retriever() -> None:
    chunks = [{"chunk_id": "c1", "doc_id": "contract", "domain": "financial_contracts", "text": "持有人有权按约定回售债券。"}]
    rag = EvidenceRAG(LexicalIndex(chunks))
    question = {**QUESTION, "doc_ids": ["contract"], "domain": "financial_contracts"}
    assert rag.retrieve(question, "nano")[0]["chunk_id"] == "c1"


def test_state_graph_executes_transitions_and_rejects_cycles() -> None:
    state = AgentState(task_id="q1", question=QUESTION)
    graph = StateGraph([GraphNode("plan", lambda current: "finish"), GraphNode("finish", lambda current: None)], "plan")
    assert graph.run(state).current_step == 2
    cycle = StateGraph([GraphNode("loop", lambda current: "loop")], "loop")
    cycled = cycle.run(AgentState(task_id="q2", question=QUESTION))
    assert cycled.status == TaskStatus.FAILED
    assert "graph_cycle_or_step_limit" in cycled.errors
