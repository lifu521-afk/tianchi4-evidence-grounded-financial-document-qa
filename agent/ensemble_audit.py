from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .ensemble_prompts import (
    build_analyst_messages,
    build_judge_messages,
    build_locator_messages,
    build_skeptic_messages,
)
from .qwen_client import OpenAICompatibleClient
from .retrieval import (
    LexicalIndex,
    SearchResult,
    build_option_anchor_query,
    build_query,
    gather_evidence,
)
from .solver import (
    load_processed,
    normalize_answer_for_question,
    read_answer_rows,
    safe_cache_name,
    usage_or_estimate,
    write_answer_rows,
)


PROMPT_VERSION = "qwen-full-audit-v3"
VALID_LETTERS = "ABCD"


@dataclass(frozen=True)
class EnsembleSettings:
    initial_max_chars: int = 52000
    final_max_chars: int = 90000
    initial_max_chunks: int = 34
    final_max_chunks: int = 58
    top_global: int = 16
    top_per_option: int = 9
    top_per_doc: int = 6
    neighbor_radius: int = 2
    locator_query_top_k: int = 5
    max_locator_queries: int = 24
    confidence_threshold: float = 0.82
    question_workers: int = 1
    prompt_version: str = PROMPT_VERSION


class EnsembleStopped(Exception):
    pass


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def option_letters(question: dict[str, Any]) -> list[str]:
    letters = [
        str(letter).upper()
        for letter in (question.get("options") or {})
        if str(letter).upper() in VALID_LETTERS
    ]
    return sorted(letters) or list(VALID_LETTERS)


def normalize_truth(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "true": "true",
        "yes": "true",
        "正确": "true",
        "支持": "true",
        "成立": "true",
        "entailed": "true",
        "false": "false",
        "no": "false",
        "错误": "false",
        "不支持": "false",
        "不成立": "false",
        "contradicted": "false",
        "uncertain": "uncertain",
        "unknown": "uncertain",
        "不确定": "uncertain",
        "证据不足": "uncertain",
    }
    return aliases.get(text, "uncertain")


def normalize_relation(value: Any, truth: str) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "entailment": "entailed",
        "supported": "entailed",
        "support": "entailed",
        "contradiction": "contradicted",
        "contradictory": "contradicted",
        "uncertain": "unknown",
    }
    text = aliases.get(text, text)
    if text in {"entailed", "contradicted", "unknown"}:
        return text
    return {"true": "entailed", "false": "contradicted"}.get(truth, "unknown")


def judgement_for_letter(parsed: dict[str, Any], letter: str) -> dict[str, Any]:
    judgements = parsed.get("option_judgement")
    if isinstance(judgements, dict):
        value = judgements.get(letter) or judgements.get(letter.lower()) or {}
        return value if isinstance(value, dict) else {"judgement": value}
    if isinstance(judgements, list):
        for value in judgements:
            if isinstance(value, dict) and str(value.get("option", "")).upper() == letter:
                return value
    return {}


def judgement_truth(parsed: dict[str, Any], letter: str) -> str:
    item = judgement_for_letter(parsed, letter)
    return normalize_truth(item.get("judgement") or item.get("verdict"))


def judgement_relation(parsed: dict[str, Any], letter: str) -> str:
    item = judgement_for_letter(parsed, letter)
    truth = judgement_truth(parsed, letter)
    return normalize_relation(item.get("relation") or item.get("verdict"), truth)


def derive_answer(parsed: dict[str, Any], question: dict[str, Any]) -> str:
    true_letters = [
        letter
        for letter in option_letters(question)
        if judgement_truth(parsed, letter) == "true"
        and judgement_relation(parsed, letter) == "entailed"
    ]
    answer_format = str(question.get("answer_format", "multi"))
    if answer_format == "multi":
        return "".join(true_letters)
    if answer_format in {"mcq", "tf"} and len(true_letters) == 1:
        return true_letters[0]
    return ""


def parse_evidence_ids(value: Any) -> set[int]:
    result: set[int] = set()
    if isinstance(value, int):
        result.add(value)
    elif isinstance(value, str):
        result.update(int(match) for match in re.findall(r"\d+", value))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            result.update(parse_evidence_ids(item))
    return result


def supporting_ids(parsed: dict[str, Any], letter: str) -> set[int]:
    item = judgement_for_letter(parsed, letter)
    return parse_evidence_ids(item.get("supporting_evidence_ids") or item.get("evidence_ids"))


def contradicting_ids(parsed: dict[str, Any], letter: str) -> set[int]:
    item = judgement_for_letter(parsed, letter)
    return parse_evidence_ids(item.get("contradicting_evidence_ids") or item.get("counter_evidence_ids"))


def relevant_ids(parsed: dict[str, Any], letter: str) -> set[int]:
    item = judgement_for_letter(parsed, letter)
    return parse_evidence_ids(item.get("relevant_evidence_ids"))


def resolve_option_doc_scope(question: dict[str, Any], option_text: str) -> list[str]:
    doc_ids = [str(value) for value in question.get("doc_ids") or []]
    if not doc_ids:
        return []

    explicit = {
        int(value)
        for value in re.findall(r"[A-Za-z]+_text_?0*(\d+)", option_text, flags=re.I)
    }
    if explicit:
        matched = []
        for doc_id in doc_ids:
            suffix = re.search(r"0*(\d+)$", doc_id)
            if suffix and int(suffix.group(1)) in explicit:
                matched.append(doc_id)
        if matched:
            return matched

    ordinal_map = {
        "第一份": 0,
        "首份": 0,
        "第二份": 1,
        "第三份": 2,
        "第四份": 3,
    }
    matched = [
        doc_ids[index]
        for marker, index in ordinal_map.items()
        if marker in option_text and index < len(doc_ids)
    ]
    if matched:
        return list(dict.fromkeys(matched))
    return doc_ids


def required_docs_for_option(question: dict[str, Any], letter: str) -> list[str]:
    option_text = str((question.get("options") or {}).get(letter, ""))
    doc_ids = [str(value) for value in question.get("doc_ids") or []]
    scope = resolve_option_doc_scope(question, option_text)
    all_doc_markers = ("两份", "各文档", "均", "都", "分别", "相比", "比较")
    if len(doc_ids) > 1 and any(marker in option_text for marker in all_doc_markers):
        return doc_ids
    return scope


# Correct UTF-8 definitions. These override legacy aliases that were damaged by
# an earlier terminal encoding conversion.
def normalize_truth(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "true": "true",
        "yes": "true",
        "正确": "true",
        "支持": "true",
        "成立": "true",
        "entailed": "true",
        "false": "false",
        "no": "false",
        "错误": "false",
        "不支持": "false",
        "不成立": "false",
        "contradicted": "false",
        "uncertain": "uncertain",
        "unknown": "uncertain",
        "不确定": "uncertain",
        "证据不足": "uncertain",
    }
    return aliases.get(text, "uncertain")


def resolve_option_doc_scope(question: dict[str, Any], option_text: str) -> list[str]:
    doc_ids = [str(value) for value in question.get("doc_ids") or []]
    if not doc_ids:
        return []

    explicit = {
        int(value)
        for value in re.findall(r"[A-Za-z]+_text_?0*(\d+)", option_text, flags=re.I)
    }
    if explicit:
        matched = []
        for doc_id in doc_ids:
            suffix = re.search(r"0*(\d+)$", doc_id)
            if suffix and int(suffix.group(1)) in explicit:
                matched.append(doc_id)
        if matched:
            return matched

    ordinal_map = {
        "第一份": 0,
        "首份": 0,
        "第二份": 1,
        "第三份": 2,
        "第四份": 3,
    }
    matched = [
        doc_ids[index]
        for marker, index in ordinal_map.items()
        if marker in option_text and index < len(doc_ids)
    ]
    return list(dict.fromkeys(matched)) if matched else doc_ids


def required_docs_for_option(question: dict[str, Any], letter: str) -> list[str]:
    option_text = str((question.get("options") or {}).get(letter, ""))
    doc_ids = [str(value) for value in question.get("doc_ids") or []]
    scope = resolve_option_doc_scope(question, option_text)
    all_doc_markers = (
        "两份",
        "各文档",
        "均",
        "都",
        "分别",
        "相比",
        "比较",
    )
    if len(doc_ids) > 1 and any(marker in option_text for marker in all_doc_markers):
        return doc_ids
    return scope


def evidence_item_from_result(result: SearchResult, source: str) -> dict[str, Any]:
    item = dict(result.chunk)
    item["score"] = round(float(result.score), 4)
    item["sources"] = sorted(set(item.get("sources") or []) | {source})
    return item


def merge_evidence(
    base: list[dict[str, Any]],
    additions: Iterable[dict[str, Any]],
    *,
    max_chunks: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for item in [*base, *list(additions)]:
        chunk_id = str(item.get("chunk_id", ""))
        if not chunk_id:
            continue
        if chunk_id in positions:
            current = merged[positions[chunk_id]]
            current["sources"] = sorted(
                set(current.get("sources") or []) | set(item.get("sources") or [])
            )
            current["score"] = max(
                float(current.get("score") or 0),
                float(item.get("score") or 0),
            )
            continue
        positions[chunk_id] = len(merged)
        merged.append(dict(item))

    result: list[dict[str, Any]] = []
    total_chars = 0
    for item in merged:
        text = str(item.get("text", ""))
        if len(result) >= max_chunks or total_chars + len(text) > max_chars:
            continue
        result.append(item)
        total_chars += len(text)
    return result


def build_initial_evidence(
    index: LexicalIndex,
    question: dict[str, Any],
    settings: EnsembleSettings,
) -> list[dict[str, Any]]:
    evidence = gather_evidence(
        index,
        question,
        top_global=settings.top_global,
        top_per_option=settings.top_per_option,
        top_per_doc=settings.top_per_doc,
        neighbor_radius=settings.neighbor_radius,
        max_chunks=settings.initial_max_chunks,
        max_chars=settings.initial_max_chars,
    )
    additions: list[dict[str, Any]] = []
    domain = question.get("domain")
    for letter, option_text in sorted((question.get("options") or {}).items()):
        scope = resolve_option_doc_scope(question, str(option_text))
        query = build_option_anchor_query(str(option_text))
        for result in index.search(query, scope or None, domain, settings.top_per_option):
            additions.append(evidence_item_from_result(result, f"deep_option_{letter}"))
    return merge_evidence(
        evidence,
        additions,
        max_chunks=settings.initial_max_chunks,
        max_chars=settings.initial_max_chars,
    )


def collect_locator_queries(locator: dict[str, Any]) -> list[tuple[str | None, str]]:
    queries: list[tuple[str | None, str]] = []
    option_search = locator.get("option_search")
    if isinstance(option_search, dict):
        for letter, payload in option_search.items():
            if not isinstance(payload, dict):
                continue
            values = payload.get("search_queries") or []
            if isinstance(values, str):
                values = [values]
            for value in values:
                query = str(value).strip()
                if query:
                    queries.append((str(letter).upper(), query))
    global_values = locator.get("global_search_queries") or []
    if isinstance(global_values, str):
        global_values = [global_values]
    for value in global_values:
        query = str(value).strip()
        if query:
            queries.append((None, query))
    return list(dict.fromkeys(queries))


def augment_evidence(
    index: LexicalIndex,
    question: dict[str, Any],
    initial: list[dict[str, Any]],
    locator: dict[str, Any],
    settings: EnsembleSettings,
) -> list[dict[str, Any]]:
    additions: list[dict[str, Any]] = []
    domain = question.get("domain")
    all_doc_ids = [str(value) for value in question.get("doc_ids") or []]
    queries = collect_locator_queries(locator)[: settings.max_locator_queries]
    for query_index, (letter, query) in enumerate(queries, start=1):
        if letter and letter in (question.get("options") or {}):
            option_text = str(question["options"][letter])
            doc_ids = resolve_option_doc_scope(question, option_text)
        else:
            doc_ids = all_doc_ids
        for result in index.search(
            query,
            doc_ids or None,
            domain,
            settings.locator_query_top_k,
        ):
            source = f"locator_{letter or 'global'}_{query_index}"
            additions.append(evidence_item_from_result(result, source))
            for neighbor in index.neighbors(result.chunk["chunk_id"], radius=1):
                neighbor_result = SearchResult(neighbor, result.score * 0.68, "locator_neighbor")
                additions.append(evidence_item_from_result(neighbor_result, source + "_neighbor"))
    return merge_evidence(
        initial,
        additions,
        max_chunks=settings.final_max_chunks,
        max_chars=settings.final_max_chars,
    )


def parsed_covers_options(
    parsed: dict[str, Any],
    letters: list[str],
    schema_key: str,
) -> bool:
    values = parsed.get(schema_key)
    if not isinstance(values, dict):
        return False
    return all(
        isinstance(values.get(letter) or values.get(letter.lower()), dict)
        for letter in letters
    )


def call_stage(
    client: OpenAICompatibleClient,
    messages: list[dict[str, str]],
    *,
    letters: list[str],
    schema_key: str,
    max_parse_attempts: int = 2,
) -> dict[str, Any]:
    working_messages = list(messages)
    attempts: list[dict[str, Any]] = []
    combined_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    parsed: dict[str, Any] = {}
    content = ""
    for attempt_index in range(1, max_parse_attempts + 1):
        result = client.chat(working_messages)
        content = result.content
        usage = usage_or_estimate(working_messages, content, result.usage)
        for key in combined_usage:
            combined_usage[key] += int(usage.get(key, 0) or 0)
        parsed = extract_json_object(content)
        valid = parsed_covers_options(parsed, letters, schema_key)
        attempts.append(
            {
                "attempt": attempt_index,
                "raw_output": content,
                "parsed_valid": valid,
                "usage": usage,
            }
        )
        if valid:
            break
        if attempt_index < max_parse_attempts:
            working_messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "上一个输出不是完整合法的 JSON，或没有覆盖全部选项。"
                            f"请只重新输出一个 JSON 对象，并在 {schema_key} 中完整包含 "
                            f"{'/'.join(letters)}。不要输出 Markdown 代码块或额外说明。"
                        ),
                    },
                ]
            )
    return {
        "raw_output": content,
        "parsed": parsed,
        "parsed_valid": parsed_covers_options(parsed, letters, schema_key),
        "attempts": attempts,
        "usage": combined_usage,
        "prompt_chars": sum(len(message["content"]) for message in messages),
    }


def stage_usage(stages: dict[str, Any]) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for stage in stages.values():
        if not isinstance(stage, dict):
            continue
        usage = stage.get("usage")
        if not isinstance(usage, dict):
            continue
        for key in total:
            total[key] += int(usage.get(key, 0) or 0)
    return total


def confidence_for(parsed: dict[str, Any], letter: str) -> float:
    item = judgement_for_letter(parsed, letter)
    value = item.get("confidence", parsed.get("overall_confidence", 0))
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if result > 1:
        result /= 100.0
    return max(0.0, min(result, 1.0))


def normalized_quote(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def quoted_clauses_for(parsed: dict[str, Any], letter: str) -> list[dict[str, Any]]:
    item = judgement_for_letter(parsed, letter)
    values = item.get("quoted_clauses") or item.get("quotes") or []
    if isinstance(values, dict):
        values = [values]
    if isinstance(values, str):
        values = [{"quote": values}]
    return [value for value in values if isinstance(value, dict)]


def exact_excerpt(text: str, option_text: str, max_chars: int = 320) -> str:
    if not text:
        return ""
    keywords = re.findall(r"[\u4e00-\u9fff]{2,}|\d+(?:\.\d+)?%?|[A-Za-z]{2,}", option_text)
    candidates: list[str] = []
    for keyword in keywords:
        if re.fullmatch(r"[\u4e00-\u9fff]+", keyword) and len(keyword) > 8:
            candidates.extend(keyword[index : index + 6] for index in range(0, len(keyword) - 5, 3))
        else:
            candidates.append(keyword)
    start = -1
    for keyword in sorted(set(candidates), key=len, reverse=True):
        start = text.find(keyword)
        if start >= 0:
            break
    if start < 0:
        start = 0
    left = max(0, start - max_chars // 3)
    right = min(len(text), left + max_chars)
    return text[left:right].strip()


def materialize_option_citations(
    question: dict[str, Any],
    letter: str,
    parsed: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    citation_origin: str = "judge",
) -> list[dict[str, Any]]:
    ids = (
        supporting_ids(parsed, letter)
        | contradicting_ids(parsed, letter)
        | relevant_ids(parsed, letter)
    )
    quoted_by_id: dict[int, list[str]] = {}
    for row in quoted_clauses_for(parsed, letter):
        evidence_id_values = parse_evidence_ids(row.get("evidence_id"))
        quote = str(row.get("quote") or row.get("quoted_clause") or "").strip()
        for evidence_id in evidence_id_values:
            if quote:
                quoted_by_id.setdefault(evidence_id, []).append(quote)

    option_text = str((question.get("options") or {}).get(letter, ""))
    citations: list[dict[str, Any]] = []
    for evidence_id in sorted(ids):
        if evidence_id < 1 or evidence_id > len(evidence):
            continue
        source = evidence[evidence_id - 1]
        source_text = str(source.get("text", ""))
        model_quotes = quoted_by_id.get(evidence_id, [])
        verified_quote = next(
            (
                quote
                for quote in model_quotes
                if normalized_quote(quote)
                and normalized_quote(quote) in normalized_quote(source_text)
            ),
            "",
        )
        excerpt = verified_quote or exact_excerpt(source_text, option_text)
        citations.append(
            {
                "evidence_id": evidence_id,
                "doc_id": source.get("doc_id"),
                "chunk_id": source.get("chunk_id"),
                "page": source.get("page"),
                "page_char_start": source.get("page_char_start"),
                "page_char_end": source.get("page_char_end"),
                "source_path": source.get("source_path"),
                "quote": excerpt,
                "model_quote_verified": bool(verified_quote),
                "source_excerpt_verified": bool(
                    excerpt
                    and normalized_quote(excerpt) in normalized_quote(source_text)
                ),
                "citation_origin": citation_origin,
            }
        )
    return citations


def fallback_option_evidence_id(
    question: dict[str, Any],
    letter: str,
    evidence: list[dict[str, Any]],
) -> int | None:
    if not evidence:
        return None
    option_text = str((question.get("options") or {}).get(letter, ""))
    question_text = str(question.get("question") or "")
    required_docs = set(required_docs_for_option(question, letter))
    raw_terms = re.findall(
        r"[\u4e00-\u9fff]{2,}|\d+(?:\.\d+)?%?|[A-Za-z][A-Za-z0-9._-]+",
        f"{question_text} {option_text}",
    )
    terms: set[str] = set()
    for term in raw_terms:
        if re.fullmatch(r"[\u4e00-\u9fff]+", term) and len(term) > 6:
            terms.update(term[index : index + 4] for index in range(len(term) - 3))
        else:
            terms.add(term)

    candidates = [
        (index, item)
        for index, item in enumerate(evidence, start=1)
        if not required_docs or str(item.get("doc_id")) in required_docs
    ]
    if not candidates:
        candidates = list(enumerate(evidence, start=1))

    def score(item: dict[str, Any]) -> tuple[int, int]:
        text = normalized_quote(str(item.get("text") or ""))
        overlap = sum(
            max(len(normalized_quote(term)), 1) ** 2
            for term in terms
            if normalized_quote(term) in text
        )
        source_bonus = len(item.get("sources") or [])
        return overlap, source_bonus

    return max(candidates, key=lambda row: score(row[1]))[0]


def audit_option_citations(
    question: dict[str, Any],
    letter: str,
    analyst: dict[str, Any],
    skeptic: dict[str, Any],
    judge: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for origin, parsed in (
        ("judge", judge),
        ("analyst", analyst),
        ("skeptic", skeptic),
    ):
        citations = materialize_option_citations(
            question,
            letter,
            parsed,
            evidence,
            citation_origin=origin,
        )
        if citations:
            return citations

    fallback_id = fallback_option_evidence_id(question, letter, evidence)
    if fallback_id is None:
        return []
    fallback = {
        "option_judgement": {
            letter: {
                "relevant_evidence_ids": [fallback_id],
            }
        }
    }
    return materialize_option_citations(
        question,
        letter,
        fallback,
        evidence,
        citation_origin="lexical_fallback",
    )


def option_vote_summary(
    letter: str,
    analyst: dict[str, Any],
    skeptic: dict[str, Any],
    judge: dict[str, Any],
) -> dict[str, str]:
    return {
        "analyst": judgement_truth(analyst, letter),
        "skeptic": judgement_truth(skeptic, letter),
        "judge": judgement_truth(judge, letter),
    }


def conservative_decision(
    question: dict[str, Any],
    baseline_answer: str,
    ensemble_answer: str,
    analyst: dict[str, Any],
    skeptic: dict[str, Any],
    judge: dict[str, Any],
    evidence: list[dict[str, Any]],
    settings: EnsembleSettings,
) -> tuple[str, bool, list[str], dict[str, Any]]:
    if not ensemble_answer:
        return baseline_answer, False, ["judge_did_not_produce_a_valid_answer"], {}
    if ensemble_answer == baseline_answer:
        return baseline_answer, True, ["unchanged"], {}

    baseline_action = str(judge.get("baseline_action") or "").strip().lower()
    change_classification = str(
        judge.get("change_classification") or ""
    ).strip().lower()
    if baseline_action in {"keep", "unresolved_keep"}:
        return (
            baseline_answer,
            False,
            [f"judge_baseline_action_{baseline_action}"],
            {
                "baseline_action": baseline_action,
                "change_classification": change_classification,
            },
        )
    if change_classification == "semantic_dispute":
        return (
            baseline_answer,
            False,
            ["semantic_dispute_must_keep_baseline"],
            {
                "baseline_action": baseline_action,
                "change_classification": change_classification,
            },
        )

    baseline_set = set(baseline_answer)
    ensemble_set = set(ensemble_answer)
    changed_letters = sorted(baseline_set ^ ensemble_set)
    failures: list[str] = []
    details: dict[str, Any] = {}

    for letter in changed_letters:
        target_truth = "true" if letter in ensemble_set else "false"
        target_relation = "entailed" if target_truth == "true" else "contradicted"
        votes = option_vote_summary(letter, analyst, skeptic, judge)
        matching_votes = sum(value == target_truth for value in votes.values())
        relation = judgement_relation(judge, letter)
        confidence = confidence_for(judge, letter)
        ids = (
            supporting_ids(judge, letter)
            if target_truth == "true"
            else contradicting_ids(judge, letter)
        )
        valid_ids = {value for value in ids if 1 <= value <= len(evidence)}
        cited_docs = {
            str(evidence[value - 1].get("doc_id"))
            for value in valid_ids
            if evidence[value - 1].get("doc_id")
        }
        required_docs = set(required_docs_for_option(question, letter))
        citations = materialize_option_citations(question, letter, judge, evidence)
        judgement_item = judgement_for_letter(judge, letter)
        semantic_equivalence = str(
            judgement_item.get("semantic_equivalence") or ""
        ).strip().lower()

        letter_failures: list[str] = []
        if votes["judge"] != target_truth or relation != target_relation:
            letter_failures.append("judge_relation_mismatch")
        if matching_votes < 2:
            letter_failures.append("fewer_than_two_agents_agree")
        if confidence < settings.confidence_threshold:
            letter_failures.append("confidence_below_threshold")
        if not valid_ids:
            letter_failures.append("no_valid_source_citation")
        if target_truth == "true" and required_docs and not required_docs.issubset(cited_docs):
            letter_failures.append("required_document_coverage_missing")
        if not citations:
            letter_failures.append("no_materialized_source_location")
        if citations and not any(row["model_quote_verified"] for row in citations):
            letter_failures.append("no_verified_verbatim_quote")
        if (
            str(judgement_item.get("error_type") or "").strip().lower()
            == "semantic_mismatch"
            and semantic_equivalence in {"equivalent", "uncertain"}
        ):
            letter_failures.append("semantic_equivalence_not_materially_resolved")

        details[letter] = {
            "target_truth": target_truth,
            "votes": votes,
            "relation": relation,
            "confidence": confidence,
            "valid_evidence_ids": sorted(valid_ids),
            "required_doc_ids": sorted(required_docs),
            "cited_doc_ids": sorted(cited_docs),
            "citations": citations,
            "semantic_equivalence": semantic_equivalence,
            "failures": letter_failures,
        }
        failures.extend(f"{letter}:{reason}" for reason in letter_failures)

    passed = not failures
    return (ensemble_answer if passed else baseline_answer), passed, failures or ["passed"], details


def option_audit_rows(
    question: dict[str, Any],
    baseline_answer: str,
    ensemble_answer: str,
    recommended_answer: str,
    analyst: dict[str, Any],
    skeptic: dict[str, Any],
    judge: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for letter in option_letters(question):
        citations = audit_option_citations(
            question,
            letter,
            analyst,
            skeptic,
            judge,
            evidence,
        )
        citation_text = " || ".join(
            f"{row['doc_id']}#{row['chunk_id']} p{row['page']} "
            f"chars={row['page_char_start']}:{row['page_char_end']} "
            f"origin={row['citation_origin']} "
            f"model_quote_verified={row['model_quote_verified']} "
            f"source_excerpt_verified={row['source_excerpt_verified']} :: {row['quote']}"
            for row in citations
        )
        rows.append(
            {
                "qid": question["qid"],
                "domain": question.get("domain"),
                "answer_format": question.get("answer_format"),
                "option": letter,
                "option_text": (question.get("options") or {}).get(letter, ""),
                "baseline_selected": letter in baseline_answer,
                "ensemble_selected": letter in ensemble_answer,
                "recommended_selected": letter in recommended_answer,
                "analyst_judgement": judgement_truth(analyst, letter),
                "skeptic_judgement": judgement_truth(skeptic, letter),
                "judge_judgement": judgement_truth(judge, letter),
                "judge_relation": judgement_relation(judge, letter),
                "judge_confidence": confidence_for(judge, letter),
                "citation_count": len(citations),
                "verified_quote_count": sum(
                    bool(row["model_quote_verified"]) for row in citations
                ),
                "source_excerpt_count": sum(
                    bool(row["source_excerpt_verified"]) for row in citations
                ),
                "citation_origin": ",".join(
                    sorted({str(row["citation_origin"]) for row in citations})
                ),
                "source_locations": citation_text,
                "judge_reasoning": judgement_for_letter(judge, letter).get("reasoning", ""),
            }
        )
    return rows


def load_cache(path: Path, prompt_version: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if value.get("prompt_version") != prompt_version:
        return {}
    return value if isinstance(value, dict) else {}


def run_question(
    question: dict[str, Any],
    baseline_answer: str,
    index: LexicalIndex,
    client: OpenAICompatibleClient,
    cache_path: Path,
    settings: EnsembleSettings,
    stop_file: Path | None,
) -> dict[str, Any]:
    payload = load_cache(cache_path, settings.prompt_version)
    if not payload:
        payload = {
            "schema_version": 1,
            "prompt_version": settings.prompt_version,
            "qid": question["qid"],
            "baseline_answer": baseline_answer,
            "stages": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def save() -> None:
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(cache_path, payload)

    def check_stop() -> None:
        if stop_file and stop_file.exists():
            save()
            raise EnsembleStopped(f"Stop requested by {stop_file}")

    if not payload.get("initial_evidence"):
        payload["initial_evidence"] = build_initial_evidence(index, question, settings)
        save()
    check_stop()

    stages = payload.setdefault("stages", {})
    letters = option_letters(question)
    if "locator" not in stages:
        stages["locator"] = call_stage(
            client,
            build_locator_messages(question, payload["initial_evidence"]),
            letters=letters,
            schema_key="option_search",
        )
        save()
    check_stop()

    if not payload.get("retrieved_evidence"):
        locator = stages["locator"].get("parsed") or {}
        payload["retrieved_evidence"] = augment_evidence(
            index,
            question,
            payload["initial_evidence"],
            locator,
            settings,
        )
        save()
    evidence = payload["retrieved_evidence"]

    missing_reviews: dict[str, list[dict[str, str]]] = {}
    if "analyst" not in stages:
        missing_reviews["analyst"] = build_analyst_messages(question, evidence)
    if "skeptic" not in stages:
        missing_reviews["skeptic"] = build_skeptic_messages(question, evidence)
    if len(missing_reviews) == 2:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_name = {
                executor.submit(
                    call_stage,
                    client,
                    messages,
                    letters=letters,
                    schema_key="option_judgement",
                ): name
                for name, messages in missing_reviews.items()
            }
            for future in as_completed(future_to_name):
                stages[future_to_name[future]] = future.result()
                save()
    else:
        for name, messages in missing_reviews.items():
            stages[name] = call_stage(
                client,
                messages,
                letters=letters,
                schema_key="option_judgement",
            )
            save()
    check_stop()

    locator = stages["locator"].get("parsed") or {}
    analyst = stages["analyst"].get("parsed") or {}
    skeptic = stages["skeptic"].get("parsed") or {}
    if "judge" not in stages:
        stages["judge"] = call_stage(
            client,
            build_judge_messages(
                question,
                evidence,
                baseline_answer,
                locator,
                analyst,
                skeptic,
            ),
            letters=letters,
            schema_key="option_judgement",
        )
        save()
    judge = stages["judge"].get("parsed") or {}

    ensemble_answer = derive_answer(judge, question)
    if not ensemble_answer:
        ensemble_answer = normalize_answer_for_question(str(judge.get("answer", "")), question)
    recommended_answer, gate_passed, gate_reasons, gate_details = conservative_decision(
        question,
        baseline_answer,
        ensemble_answer,
        analyst,
        skeptic,
        judge,
        evidence,
        settings,
    )
    usage = stage_usage(stages)
    payload["result"] = {
        "ensemble_answer": ensemble_answer or baseline_answer,
        "recommended_answer": recommended_answer,
        "changed_raw": (ensemble_answer or baseline_answer) != baseline_answer,
        "changed_recommended": recommended_answer != baseline_answer,
        "evidence_gate_passed": gate_passed,
        "evidence_gate_reasons": gate_reasons,
        "evidence_gate_details": gate_details,
        "overall_confidence": judge.get("overall_confidence", 0),
        "change_reason": judge.get("change_reason", ""),
        "unresolved": judge.get("unresolved", []),
        "usage": usage,
    }
    payload["option_audit"] = option_audit_rows(
        question,
        baseline_answer,
        payload["result"]["ensemble_answer"],
        recommended_answer,
        analyst,
        skeptic,
        judge,
        evidence,
    )
    save()
    return payload


def output_rows_from_baseline(
    questions: list[dict[str, Any]],
    baseline_rows: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    answer_kind: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question in questions:
        qid = question["qid"]
        base = dict(baseline_rows[qid])
        result_payload = results.get(qid)
        if result_payload:
            result = result_payload["result"]
            usage = result["usage"]
            base["answer"] = result[answer_kind]
            base["prompt_tokens"] = usage["prompt_tokens"]
            base["completion_tokens"] = usage["completion_tokens"]
            base["total_tokens"] = usage["total_tokens"]
        rows.append(
            {
                "qid": qid,
                "answer": base["answer"],
                "prompt_tokens": int(base.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(base.get("completion_tokens", 0) or 0),
                "total_tokens": int(base.get("total_tokens", 0) or 0),
            }
        )
    return rows


def write_run_outputs(
    output_dir: Path,
    questions: list[dict[str, Any]],
    baseline_rows: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    settings: EnsembleSettings,
) -> dict[str, Any]:
    raw_rows = output_rows_from_baseline(
        questions,
        baseline_rows,
        results,
        "ensemble_answer",
    )
    recommended_rows = output_rows_from_baseline(
        questions,
        baseline_rows,
        results,
        "recommended_answer",
    )
    write_answer_rows(output_dir / "answer.raw_ensemble.csv", raw_rows)
    write_answer_rows(output_dir / "answer.csv", recommended_rows)
    write_answer_rows(output_dir / "answer.checkpoint.csv", recommended_rows)

    audit_rows = [results[q["qid"]] for q in questions if q["qid"] in results]
    atomic_write_json(output_dir / "audit.json", audit_rows)

    difference_rows: list[dict[str, Any]] = []
    option_rows: list[dict[str, Any]] = []
    for question in questions:
        qid = question["qid"]
        payload = results.get(qid)
        if not payload:
            continue
        result = payload["result"]
        difference_rows.append(
            {
                "qid": qid,
                "domain": question.get("domain"),
                "baseline_answer": payload["baseline_answer"],
                "ensemble_answer": result["ensemble_answer"],
                "recommended_answer": result["recommended_answer"],
                "changed_raw": result["changed_raw"],
                "changed_recommended": result["changed_recommended"],
                "evidence_gate_passed": result["evidence_gate_passed"],
                "overall_confidence": result["overall_confidence"],
                "gate_reasons": ";".join(result["evidence_gate_reasons"]),
                "change_reason": result["change_reason"],
            }
        )
        option_rows.extend(payload.get("option_audit") or [])

    write_csv(
        output_dir / "differences.csv",
        [
            "qid",
            "domain",
            "baseline_answer",
            "ensemble_answer",
            "recommended_answer",
            "changed_raw",
            "changed_recommended",
            "evidence_gate_passed",
            "overall_confidence",
            "gate_reasons",
            "change_reason",
        ],
        difference_rows,
    )
    option_fieldnames = [
        "qid",
        "domain",
        "answer_format",
        "option",
        "option_text",
        "baseline_selected",
        "ensemble_selected",
        "recommended_selected",
        "analyst_judgement",
        "skeptic_judgement",
        "judge_judgement",
        "judge_relation",
        "judge_confidence",
        "citation_count",
        "verified_quote_count",
        "source_excerpt_count",
        "citation_origin",
        "source_locations",
        "judge_reasoning",
    ]
    write_csv(output_dir / "option_audit.csv", option_fieldnames, option_rows)
    # Backward-compatible filename used by earlier experiments.
    write_csv(output_dir / "option_audit_400.csv", option_fieldnames, option_rows)

    summary = {
        "questions_total": len(questions),
        "questions_audited": len(results),
        "raw_changes": sum(bool(row["changed_raw"]) for row in difference_rows),
        "recommended_changes": sum(bool(row["changed_recommended"]) for row in difference_rows),
        "gate_rejections": sum(
            bool(row["changed_raw"]) and not bool(row["evidence_gate_passed"])
            for row in difference_rows
        ),
        "options_total": len(option_rows),
        "options_with_source_location": sum(
            int(row.get("citation_count", 0) or 0) > 0 for row in option_rows
        ),
        "options_with_verified_model_quote": sum(
            int(row.get("verified_quote_count", 0) or 0) > 0 for row in option_rows
        ),
        "options_with_exact_source_excerpt": sum(
            int(row.get("source_excerpt_count", 0) or 0) > 0 for row in option_rows
        ),
        "settings": asdict(settings),
        "output_dir": str(output_dir),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def run_ensemble_audit(
    *,
    processed_dir: Path,
    baseline_csv: Path,
    output_dir: Path,
    client: OpenAICompatibleClient,
    settings: EnsembleSettings,
    limit: int | None = None,
    qid_filter: set[str] | None = None,
    stop_file: Path | None = None,
) -> dict[str, Any]:
    questions, chunks = load_processed(processed_dir)
    baseline_list = read_answer_rows(baseline_csv)
    baseline_rows = {str(row.get("qid")): row for row in baseline_list}
    missing_baseline = [q["qid"] for q in questions if q["qid"] not in baseline_rows]
    if missing_baseline:
        raise RuntimeError(f"Baseline is missing {len(missing_baseline)} qids: {missing_baseline[:8]}")

    targets = [
        question
        for question in questions
        if not qid_filter or question["qid"] in qid_filter
    ]
    if limit is not None:
        targets = targets[:limit]
    if not targets:
        raise RuntimeError("No questions matched the requested qid filter/limit")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache" / "questions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    index = LexicalIndex(chunks)
    results: dict[str, dict[str, Any]] = {}

    def audit_target(idx: int, question: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any]]:
        qid = question["qid"]
        baseline_answer = normalize_answer_for_question(
            str(baseline_rows[qid].get("answer", "")),
            question,
        )
        cache_path = cache_dir / f"{safe_cache_name(qid)}.json"
        print(f"[{idx}/{len(targets)}] {qid} baseline={baseline_answer} start", flush=True)
        payload = run_question(
            question,
            baseline_answer,
            index,
            client,
            cache_path,
            settings,
            stop_file,
        )
        return idx, question, payload

    workers = max(1, min(int(settings.question_workers), len(targets)))
    if workers == 1:
        completed_payloads = (
            audit_target(idx, question)
            for idx, question in enumerate(targets, start=1)
        )
        for idx, question, payload in completed_payloads:
            qid = question["qid"]
            results[qid] = payload
            result = payload["result"]
            print(
                f"[{idx}/{len(targets)}] {qid} "
                f"raw={result['ensemble_answer']} recommended={result['recommended_answer']} "
                f"tokens={result['usage']['total_tokens']} gate={result['evidence_gate_passed']}",
                flush=True,
            )
            write_run_outputs(output_dir, questions, baseline_rows, results, settings)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(audit_target, idx, question)
                for idx, question in enumerate(targets, start=1)
            ]
            try:
                for future in as_completed(futures):
                    idx, question, payload = future.result()
                    qid = question["qid"]
                    results[qid] = payload
                    result = payload["result"]
                    print(
                        f"[{idx}/{len(targets)}] {qid} "
                        f"raw={result['ensemble_answer']} recommended={result['recommended_answer']} "
                        f"tokens={result['usage']['total_tokens']} gate={result['evidence_gate_passed']}",
                        flush=True,
                    )
                    write_run_outputs(output_dir, questions, baseline_rows, results, settings)
            except Exception:
                for future in futures:
                    future.cancel()
                if results:
                    write_run_outputs(output_dir, questions, baseline_rows, results, settings)
                raise

    return write_run_outputs(output_dir, questions, baseline_rows, results, settings)
