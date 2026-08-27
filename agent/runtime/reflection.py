from __future__ import annotations

from typing import Any

from ..solver import normalize_answer_for_question


def reflect_answer(question: dict[str, Any], answer: str, evidence: list[dict[str, Any]], reasoning: str = "") -> dict[str, Any]:
    """Checks public, auditable facts: answer shape, evidence, and reasoning linkage."""
    normalized = normalize_answer_for_question(answer, question)
    answer_format = str(question.get("answer_format") or question.get("type") or "").lower()
    issues: list[str] = []
    if not normalized:
        issues.append("no_valid_answer")
    elif normalized != answer:
        issues.append("answer_normalized")
    if not evidence:
        issues.append("missing_evidence")
    if answer_format == "multi" and len(normalized) > len(set(normalized)):
        issues.append("duplicate_option")
    if reasoning and len(reasoning.strip()) < 20:
        issues.append("reasoning_too_short")
    return {
        "accepted": not {"no_valid_answer", "missing_evidence"}.intersection(issues),
        "normalized_answer": normalized,
        "issues": issues,
        "evidence_count": len(evidence),
    }
