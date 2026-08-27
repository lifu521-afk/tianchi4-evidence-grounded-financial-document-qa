"""Optional LangGraph bridge for teams that standardize on LangGraph.

The project does not require LangGraph. This adapter deliberately raises a
clear installation message when the optional package is absent instead of
making the baseline runner fail at import time.
"""
from __future__ import annotations

from typing import Any, Callable


def build_langgraph(retrieve: Callable[[dict[str, Any], str], list[dict[str, Any]]]) -> Any:
    """Build a small retrieval/validation graph when LangGraph is installed.

    The returned graph uses public state fields only and can be expanded with
    organization-specific human-review or persistence nodes.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("Optional LangGraph support requires: pip install langgraph") from exc

    def retrieve_node(state: dict[str, Any]) -> dict[str, Any]:
        return {"evidence": retrieve(state["question"], state.get("evidence_mode", "compact"))}

    def validate_node(state: dict[str, Any]) -> dict[str, Any]:
        return {"status": "needs_solver" if state.get("evidence") else "needs_review"}

    graph = StateGraph(dict)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("validate", validate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "validate")
    graph.add_edge("validate", END)
    return graph.compile()
