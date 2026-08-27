# Agent Runtime

## Purpose

`agent.runtime` upgrades the project from a collection of competition runners
to an evidence-grounded agent application architecture. It is a lightweight,
dependency-free runtime by default, so the core workflow remains executable in
restricted enterprise environments and CI. It does not claim that an external
multi-agent framework is required for correctness.

## Workflow

```text
Planner -> Retrieval Tool -> Evidence Validator -> Solver Agent
        -> Reflection Agent -> Output Validator -> Memory Commit -> Evaluation Trace
```

Each stage has a named responsibility and writes an auditable event containing
the actor, status, timestamps, input/output summaries, raw API usage where
applicable, tool name, and errors. The trace records operational decisions; it
does not store or expose hidden model chain-of-thought.

`StateGraph` is the built-in graph executor: nodes return their next state,
and cycles, unknown nodes, or an excessive number of transitions fail closed.
It makes branching and handoff boundaries explicit for multi-agent extensions
without forcing an orchestration dependency into every local installation.

## Components

| Component | Implementation | Operational role |
| --- | --- | --- |
| Planning | `planner.build_plan` | Produces a deterministic, inspectable task sequence. |
| Tool use | `ToolRegistry` | Enforces an allow-list, required arguments, and call records. |
| RAG | `EvidenceRAG` | Adapts the existing lexical index and source chunks as a traceable retrieval tool. |
| Memory | `RunMemory`, `JsonlMemory` | Separates task-scoped context from append-only reviewed lessons. |
| Reflection | `reflect_answer` | Checks answer normalization, evidence presence, and reasoning length. |
| Cost control | `CostLedger` | Aggregates raw API usage and stops a task when its budget is exceeded. |
| Evaluation | `evaluate_records` | Reports accuracy when labels exist, evidence coverage, validity, review rate, and token usage. |

## Run

The runtime is designed to be embedded by a domain runner. The following
minimal example uses the real retriever but does not call an LLM:

```python
from agent.runtime import EvidenceGroundedOrchestrator, OrchestratorConfig

orchestrator = EvidenceGroundedOrchestrator(
    retrieve=lambda question, mode: evidence_rag.retrieve(question, mode),
    config=OrchestratorConfig(enable_llm=False),
)
state = orchestrator.run("case-001", question)
print(state.final_answer, state.status)
```

For a production or competition call, pass `OpenAICompatibleClient` and set
`enable_llm=True`. Raw usage from every call contributing to a result must be
retained and submitted according to the target competition rules.

## Framework Integration

The state, tool, memory, and trace contracts are framework-neutral. A future
LangGraph adapter can map `AgentState` to graph state and `ToolRegistry` to
tool nodes without changing retrieval, evidence, or submission code. This is
deliberate: the default project install remains small and reproducible rather
than presenting an unused framework dependency as production capability.
