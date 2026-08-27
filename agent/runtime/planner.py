from __future__ import annotations

from typing import Any


DEFAULT_PLAN = ["retrieve_evidence", "validate_evidence", "solve", "reflect", "validate_output", "commit_memory"]


def build_plan(question: dict[str, Any]) -> list[str]:
    """Build a deterministic plan so execution is inspectable and reproducible."""
    question_type = str(question.get("type") or question.get("answer_format") or "").lower()
    plan = list(DEFAULT_PLAN)
    if question_type in {"calculation", "compute", "计算题"}:
        plan.insert(2, "calculate")
    return plan
