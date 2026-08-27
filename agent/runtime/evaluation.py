from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Iterable


@dataclass(frozen=True)
class EvaluationReport:
    tasks: int
    accuracy: float | None
    evidence_coverage: float
    answer_validity: float
    review_rate: float
    average_tokens: float
    total_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_records(records: Iterable[dict[str, Any]]) -> EvaluationReport:
    rows = list(records)
    if not rows:
        return EvaluationReport(0, None, 0.0, 0.0, 0.0, 0.0, 0)
    labels = [row.get("expected_answer") for row in rows if row.get("expected_answer") is not None]
    correct = sum(str(row.get("final_answer", "")) == str(row.get("expected_answer")) for row in rows if row.get("expected_answer") is not None)
    coverage = mean(bool(row.get("evidence")) for row in rows)
    valid = mean(bool(row.get("final_answer")) for row in rows)
    review = mean(bool((row.get("reflection") or {}).get("issues")) for row in rows)
    tokens = [int((row.get("usage") or {}).get("total_tokens", 0) or 0) for row in rows]
    return EvaluationReport(
        tasks=len(rows),
        accuracy=(correct / len(labels)) if labels else None,
        evidence_coverage=coverage,
        answer_validity=valid,
        review_rate=review,
        average_tokens=sum(tokens) / len(tokens),
        total_tokens=sum(tokens),
    )
