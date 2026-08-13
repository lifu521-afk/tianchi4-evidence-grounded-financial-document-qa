from __future__ import annotations

import json
from typing import Any


PROMPT_VERSION = "b-whitelist-independent-v6"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _source_blocks(evidence: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(evidence, start=1):
        location = {
            "doc_id": item.get("doc_id"),
            "source_path": item.get("source_path"),
            "page": item.get("page"),
            "chunk_id": item.get("chunk_id"),
            "table_id": item.get("table_id"),
            "row_label": item.get("row_label") or item.get("row"),
            "column_label": item.get("column_label") or item.get("column"),
        }
        blocks.append(
            f"[E{index}] location={_json(location)}\n"
            f"{str(item.get('text') or '').strip()}"
        )
    return "\n\n".join(blocks)


def choice_selection_rule(question: dict[str, Any]) -> str:
    stem = str(question.get("question") or "")
    if any(
        marker in stem
        for marker in (
            "错误的是",
            "不正确的是",
            "不准确的是",
            "不符合的是",
            "不属于的是",
            "不能成立的是",
            "错误的有",
            "不正确的有",
        )
    ):
        return "select_false_options"
    return "select_true_options"


def build_independent_messages(
    *,
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    analyst_role: str,
) -> list[dict[str, str]]:
    options = question.get("options") or {}
    option_schema = {
        letter: {
            "is_correct": True,
            "evidence_ids": ["E1"],
            "reason": "short literal comparison between this option and the cited evidence",
        }
        for letter in options
    }
    system = (
        "You are an independent evidence analyst for a Chinese financial-document "
        "benchmark. Solve the question from scratch using only the supplied source "
        "excerpts. No proposed answer, historical answer, leaderboard answer, or "
        "outside knowledge is available. Do not invent facts, values, quotations, "
        "page numbers, formulas, or interpretations. Return exactly one JSON object."
    )
    task = {
        "analyst_role": analyst_role,
        "question": {
            "qid": question.get("qid"),
            "domain": question.get("domain"),
            "type": question.get("type"),
            "question": question.get("question"),
            "options": question.get("options") or {},
            "answer_slots": question.get("answer_slots"),
            "answer_template": question.get("answer_template") or [],
            "selection_rule": choice_selection_rule(question),
        },
        "instructions": [
            "Determine the answer independently from the source excerpts.",
            "For a single-choice or true/false question, answers must contain exactly one option letter.",
            "For a multiple-choice question, answers must contain one uppercase letter string in ABCD order with no separator.",
            "Return exactly answer_slots strings. Never combine two requested answer slots into one string.",
            "For a calculation or extraction question, follow answer_template and return each requested value in its own answer slot and in the exact requested order.",
            "For every choice option, fill option_judgements separately before deciding answers. Cover every available option exactly once.",
            "Judge each choice option by literal factual entailment from the excerpts. A supported statement does not become false merely because it omits additional non-conflicting details.",
            "Never infer an expected number of correct options, use guessed exam patterns, or reject a supported option as secondary or less central.",
            "is_correct always means whether the option statement itself is factually true.",
            "Apply selection_rule exactly: select_true_options selects is_correct=true; select_false_options selects is_correct=false.",
            "Set answers to exactly all selected option letters, sorted in ABCD order.",
            "Do not select an option merely because nearby wording is similar; cite the decisive E labels for each option judgement.",
            "For calculations, identify every operand, year, entity, unit, formula, rounding rule, and requested output order.",
            "Before emitting JSON, recompute every arithmetic operation and verify that answers exactly match the final values stated in reasoning.",
            "Never leave a provisional, guessed, or superseded value in answers. If the arithmetic narrative ends at 19.75, answers must also contain 19.75.",
            "Write a concise 60-180 Chinese-character reasoning summary that cites E labels and states decisive source facts.",
            "The reasoning must directly state every returned answer value and must not contain uncertainty, correction history, alternative answers, or hidden chain-of-thought.",
            "Use plain sentences in reasoning. Do not use quotation marks, JSON-like fragments, repeated punctuation, or copied long source passages.",
            "For choice questions, summarize A/B/C/D judgements once each, then state the final option string once.",
            "Keep every option reason under 35 Chinese characters and return no keys beyond required_json_schema.",
            "Use only E labels that exist in SOURCE EXCERPTS.",
        ],
        "required_json_schema": {
            "answers": ["one to four final answer strings"],
            "evidence_sufficient": True,
            "reasoning": "concise auditable Chinese summary with E labels",
            "evidence_ids": ["E1"],
            **({"option_judgements": option_schema} if options else {}),
        },
    }
    user = f"{_json(task)}\n\nSOURCE EXCERPTS:\n{_source_blocks(evidence)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_adjudication_messages(
    *,
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
) -> list[dict[str, str]]:
    options = question.get("options") or {}
    option_schema = {
        letter: {
            "is_correct": True,
            "evidence_ids": ["E1"],
            "reason": "short literal comparison between this option and the cited evidence",
        }
        for letter in options
    }
    system = (
        "You are the final independent adjudicator for a Chinese financial-document "
        "benchmark. Both candidate analyses were produced by allowed Qwen models, but "
        "they disagree. Re-solve the question from the supplied excerpts. Do not vote "
        "or average: verify each option or calculation directly. Return one JSON object."
    )
    task = {
        "question": {
            "qid": question.get("qid"),
            "domain": question.get("domain"),
            "type": question.get("type"),
            "question": question.get("question"),
            "options": question.get("options") or {},
            "answer_slots": question.get("answer_slots"),
            "answer_template": question.get("answer_template") or [],
            "selection_rule": choice_selection_rule(question),
        },
        "candidate_a": {
            "answers": candidate_a.get("answers"),
            "reasoning": candidate_a.get("reasoning"),
            "evidence_ids": candidate_a.get("evidence_ids"),
            "option_judgements": candidate_a.get("option_judgements"),
        },
        "candidate_b": {
            "answers": candidate_b.get("answers"),
            "reasoning": candidate_b.get("reasoning"),
            "evidence_ids": candidate_b.get("evidence_ids"),
            "option_judgements": candidate_b.get("option_judgements"),
        },
        "instructions": [
            "Independently recompute the result from SOURCE EXCERPTS.",
            "Resolve the exact factual or calculation point causing the disagreement.",
            "For choice questions, independently fill option_judgements for every available option exactly once.",
            "Judge literal factual entailment. Missing non-conflicting details do not make a supported option false.",
            "Never use guessed exam patterns, an expected answer count, or the idea that a supported option is secondary.",
            "is_correct always records factual truth of the option statement.",
            "Apply selection_rule exactly: select true options for select_true_options and false options for select_false_options.",
            "Derive answers from every selected option and return the letters in ABCD order.",
            "Return exactly answer_slots strings and never combine separate requested slots.",
            "For calculations, follow answer_template, recompute the arithmetic, and return each value in the requested order and format.",
            "Before emitting JSON, verify that answers exactly match the final values stated in reasoning.",
            "Write a clean 60-180 Chinese-character final reasoning summary citing decisive E labels.",
            "Use plain sentences without quotation marks, JSON-like fragments, repeated punctuation, or copied long source passages.",
            "Do not mention candidates, disagreement, retries, corrections, or alternative results in the final reasoning.",
            "Keep every option reason under 35 Chinese characters and return no keys beyond required_json_schema.",
        ],
        "required_json_schema": {
            "answers": ["one to four final answer strings"],
            "evidence_sufficient": True,
            "reasoning": "final auditable Chinese summary with E labels",
            "evidence_ids": ["E1"],
            **({"option_judgements": option_schema} if options else {}),
        },
    }
    user = f"{_json(task)}\n\nSOURCE EXCERPTS:\n{_source_blocks(evidence)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
