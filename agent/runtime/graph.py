from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .state import AgentState, TaskStatus


Node = Callable[[AgentState], str | None]


@dataclass(frozen=True)
class GraphNode:
    name: str
    handler: Node


class StateGraph:
    """Small deterministic state-graph executor for Agent workflows.

    Nodes return their next name or ``None``. Cycles and unknown transitions
    fail closed, providing an inspectable orchestration primitive without a
    mandatory third-party dependency.
    """

    def __init__(self, nodes: list[GraphNode], start: str, max_steps: int = 32) -> None:
        self.nodes = {node.name: node for node in nodes}
        if start not in self.nodes:
            raise ValueError(f"Start node is not registered: {start}")
        self.start = start
        self.max_steps = max_steps

    def run(self, state: AgentState) -> AgentState:
        next_node: str | None = self.start
        visited: set[str] = set()
        while next_node is not None:
            if next_node not in self.nodes:
                state.status = TaskStatus.FAILED
                state.errors.append(f"unknown_graph_node:{next_node}")
                return state
            if next_node in visited or len(visited) >= self.max_steps:
                state.status = TaskStatus.FAILED
                state.errors.append("graph_cycle_or_step_limit")
                return state
            visited.add(next_node)
            state.current_step += 1
            next_node = self.nodes[next_node].handler(state)
        return state
