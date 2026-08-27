from __future__ import annotations

from typing import Any

from ..retrieval import LexicalIndex, gather_evidence


class EvidenceRAG:
    """Adapter that exposes the existing traceable lexical retriever as a RAG tool."""

    def __init__(self, index: LexicalIndex) -> None:
        self.index = index

    def retrieve(self, question: dict[str, Any], evidence_mode: str = "compact") -> list[dict[str, Any]]:
        # Keep the runtime adapter compatible with the existing retriever's
        # explicit budget arguments. The competition solvers use their own
        # richer compacting layer, while this generic runtime controls only
        # the amount of source material supplied to an agent task.
        mode_limits = {
            "full": {},
            "compact": {"max_chunks": 14, "max_chars": 12000},
            "minimal": {"max_chunks": 8, "max_chars": 7000, "neighbor_radius": 0},
            "micro": {"max_chunks": 6, "max_chars": 5000, "neighbor_radius": 0},
            "nano": {"max_chunks": 4, "max_chars": 3200, "neighbor_radius": 0},
        }
        if evidence_mode not in mode_limits:
            raise ValueError(f"Unknown evidence mode: {evidence_mode}")
        return gather_evidence(self.index, question, **mode_limits[evidence_mode])
