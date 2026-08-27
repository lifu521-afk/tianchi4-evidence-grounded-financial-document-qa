from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..tokenize import normalize_for_search, term_counts


@dataclass
class RunMemory:
    """Task-scoped memory. It never leaks records across orchestrator instances."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(dict(record))

    def search(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        query_terms = set(term_counts(normalize_for_search(query)))
        if not query_terms:
            return []
        scored: list[tuple[int, dict[str, Any]]] = []
        for record in self.records:
            searchable = " ".join(str(record.get(key, "")) for key in ("domain", "question", "lesson", "tags"))
            score = len(query_terms & set(term_counts(normalize_for_search(searchable))))
            if score:
                scored.append((score, record))
        return [record for _, record in sorted(scored, key=lambda pair: pair[0], reverse=True)[:limit]]


class JsonlMemory(RunMemory):
    """Append-only long-term lessons, intended for reviewed failures and policies."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        records: list[dict[str, Any]] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    records.append(value)
        super().__init__(records=records)

    def add(self, record: dict[str, Any]) -> None:
        super().add(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
