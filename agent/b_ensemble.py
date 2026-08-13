from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable

from .b_prompts import (
    build_analyst_messages as build_b_analyst_messages,
    build_compact_teacher_cache_messages as build_b_teacher_verify_messages,
    build_judge_messages as build_b_judge_messages,
    build_locator_messages as build_b_locator_messages,
    build_skeptic_messages as build_b_skeptic_messages,
    build_solver_messages as build_b_solver_messages,
)
from .ensemble_audit import atomic_write_json
from .qwen_client import OpenAICompatibleClient
from .retrieval import LexicalIndex, SearchResult, build_query
from .solver import load_processed, safe_cache_name, usage_or_estimate


B_PROMPT_VERSION = "qwen-b-full-ensemble-v1"
B_LOW_PROMPT_VERSION = "qwen-b-teacher-verify-v1"
VALID_LETTERS = "ABCD"
_NUMERIC_TEMPLATE_RE = re.compile(r"-?\d+(?:\.(\d+))?")
_NUMERIC_ANSWER_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*"
    r"(?:笔|日|天|分|亿元|万元|元|个百分点|万户|万人|户|人)?\s*$"
)
_ONE_DECIMAL_RE = re.compile(r"保留一位小数")


@dataclass(frozen=True)
class BEnsembleSettings:
    initial_max_chars: int = 48000
    final_max_chars: int = 90000
    initial_max_chunks: int = 30
    final_max_chunks: int = 56
    top_global: int = 20
    top_per_option: int = 7
    top_per_candidate_doc: int = 4
    max_candidate_docs: int = 12
    neighbor_radius: int = 2
    locator_query_top_k: int = 6
    max_locator_queries: int = 28
    question_workers: int = 3
    prompt_version: str = B_PROMPT_VERSION


class BEnsembleStopped(Exception):
    pass


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


def question_kind(question: dict[str, Any]) -> str:
    if question.get("options"):
        return "choice"
    qtype = str(question.get("type") or "")
    return "extract" if "抽取" in qtype else "open"


def answer_slot_count(question: dict[str, Any]) -> int:
    try:
        return max(1, min(4, int(question.get("answer_slots") or 1)))
    except (TypeError, ValueError):
        return 1


def normalize_choice_answer(value: Any, question: dict[str, Any]) -> str:
    allowed = [
        str(letter).upper()
        for letter in (question.get("options") or {})
        if str(letter).upper() in VALID_LETTERS
    ]
    allowed = sorted(set(allowed))
    letters = [letter for letter in str(value or "").upper() if letter in allowed]
    normalized = "".join(letter for letter in allowed if letter in set(letters))
    qtype = str(question.get("type") or "")
    if "单选" in qtype or "判断" in qtype:
        return normalized[:1]
    return normalized


def normalize_open_answer(
    value: Any,
    question: dict[str, Any],
    slot_index: int,
) -> str:
    answer = str(value or "").strip()
    templates = question.get("answer_template") or []
    if not isinstance(templates, (list, tuple)) or slot_index >= len(templates):
        return answer

    template = str(templates[slot_index] or "").strip()
    is_percent = template.endswith("%")
    numeric_template = template[:-1] if is_percent else template
    template_match = _NUMERIC_TEMPLATE_RE.fullmatch(numeric_template)
    if not template_match:
        return answer

    numeric_answer = answer.replace(",", "").replace("，", "")
    if is_percent:
        numeric_answer = numeric_answer.removesuffix("%").removesuffix("％").strip()
        match = re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", numeric_answer)
    else:
        match = _NUMERIC_ANSWER_RE.fullmatch(numeric_answer)
    if not match:
        return answer

    number_text = match.group(1) if not is_percent else match.group(0)
    try:
        number = Decimal(number_text)
    except InvalidOperation:
        return answer

    template_decimals = len(template_match.group(1) or "")
    decimals = 1 if _ONE_DECIMAL_RE.search(str(question.get("question") or "")) else template_decimals
    quantum = Decimal(1).scaleb(-decimals)
    formatted = format(number.quantize(quantum, rounding=ROUND_HALF_UP), f".{decimals}f")
    return f"{formatted}%" if is_percent else formatted


def parsed_answer_values(
    parsed: dict[str, Any],
    question: dict[str, Any],
) -> list[str]:
    values = parsed.get("answers")
    if isinstance(values, dict):
        values = [
            values.get(f"answer_{index}", "")
            for index in range(1, answer_slot_count(question) + 1)
        ]
    elif isinstance(values, str):
        values = [values]
    elif not isinstance(values, list):
        single = parsed.get("answer")
        values = [single] if single is not None else []

    normalized = [str(value).strip() for value in values]
    slots = answer_slot_count(question)
    normalized = (normalized + [""] * slots)[:slots]
    if question_kind(question) == "choice":
        normalized = [normalize_choice_answer(normalized[0], question)]
    else:
        normalized = [
            normalize_open_answer(value, question, slot_index)
            for slot_index, value in enumerate(normalized)
        ]
    return normalized


def valid_final_payload(parsed: dict[str, Any], question: dict[str, Any]) -> bool:
    answers = parsed_answer_values(parsed, question)
    if len(answers) != answer_slot_count(question) or any(not value for value in answers):
        return False
    if question_kind(question) != "choice":
        return True
    answer = answers[0]
    if not answer:
        return False
    qtype = str(question.get("type") or "")
    if ("单选" in qtype or "判断" in qtype) and len(answer) != 1:
        return False
    return all(letter in (question.get("options") or {}) for letter in answer)


def stage_usage(stages: dict[str, Any]) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for stage in stages.values():
        usage = stage.get("usage") if isinstance(stage, dict) else None
        if not isinstance(usage, dict):
            continue
        for key in total:
            total[key] += int(usage.get(key, 0) or 0)
    return total


def call_json_stage(
    client: OpenAICompatibleClient,
    messages: list[dict[str, str]],
    validator: Callable[[dict[str, Any]], bool],
    *,
    repair_instruction: str,
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
        valid = validator(parsed)
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
                            "上一个输出不符合要求。"
                            + repair_instruction
                            + " 只输出一个合法 JSON 对象，不要 Markdown 或额外说明。"
                        ),
                    },
                ]
            )
    return {
        "raw_output": content,
        "parsed": parsed,
        "parsed_valid": validator(parsed),
        "attempts": attempts,
        "usage": combined_usage,
        "prompt_chars": sum(len(message["content"]) for message in messages),
    }


def evidence_item(result: SearchResult, source: str) -> dict[str, Any]:
    item = dict(result.chunk)
    item["score"] = round(float(result.score), 4)
    item["sources"] = sorted(set(item.get("sources") or []) | {source})
    return item


def merge_evidence(
    rows: Iterable[dict[str, Any]],
    *,
    max_chunks: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id:
            continue
        if chunk_id not in merged:
            merged[chunk_id] = dict(row)
            order.append(chunk_id)
            continue
        current = merged[chunk_id]
        current["score"] = max(
            float(current.get("score") or 0),
            float(row.get("score") or 0),
        )
        current["sources"] = sorted(
            set(current.get("sources") or []) | set(row.get("sources") or [])
        )

    output: list[dict[str, Any]] = []
    total_chars = 0
    for chunk_id in order:
        row = merged[chunk_id]
        text = str(row.get("text") or "")
        if len(output) >= max_chunks or total_chars + len(text) > max_chars:
            continue
        output.append(row)
        total_chars += len(text)
    return output


def _question_query(question: dict[str, Any]) -> str:
    return build_query(question)


def build_initial_evidence(
    index: LexicalIndex,
    question: dict[str, Any],
    settings: BEnsembleSettings,
) -> list[dict[str, Any]]:
    domain = str(question.get("domain") or "")
    rows: list[dict[str, Any]] = []
    query = _question_query(question)
    global_results = index.search(query, domain=domain, top_k=settings.top_global)
    rows.extend(evidence_item(result, "global") for result in global_results)

    candidate_docs: list[str] = []
    for result in global_results:
        doc_id = str(result.chunk.get("doc_id") or "")
        if doc_id and doc_id not in candidate_docs:
            candidate_docs.append(doc_id)
        if len(candidate_docs) >= settings.max_candidate_docs:
            break

    for letter, option_text in sorted((question.get("options") or {}).items()):
        option_query = build_query(question, str(option_text))
        option_results = index.search(
            option_query,
            domain=domain,
            top_k=settings.top_per_option,
        )
        rows.extend(
            evidence_item(result, f"option_{letter}")
            for result in option_results
        )
        for result in option_results:
            doc_id = str(result.chunk.get("doc_id") or "")
            if doc_id and doc_id not in candidate_docs:
                candidate_docs.append(doc_id)

    for doc_id in candidate_docs[: settings.max_candidate_docs]:
        results = index.search(
            query,
            candidate_doc_ids=[doc_id],
            domain=domain,
            top_k=settings.top_per_candidate_doc,
        )
        rows.extend(evidence_item(result, f"candidate_doc_{doc_id}") for result in results)

    if settings.neighbor_radius:
        seeds = sorted(
            rows,
            key=lambda item: float(item.get("score") or 0),
            reverse=True,
        )[: max(8, settings.top_global)]
        for seed in seeds:
            for neighbor in index.neighbors(
                str(seed.get("chunk_id") or ""),
                radius=settings.neighbor_radius,
            ):
                rows.append(
                    {
                        **neighbor,
                        "score": round(float(seed.get("score") or 0) * 0.68, 4),
                        "sources": ["neighbor"],
                    }
                )
    return merge_evidence(
        rows,
        max_chunks=settings.initial_max_chunks,
        max_chars=settings.initial_max_chars,
    )


def collect_locator_queries(locator: dict[str, Any]) -> list[str]:
    values: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text:
                values.append(text)
        elif isinstance(value, list):
            for item in value:
                add(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                if "quer" in str(key).lower() or "search" in str(key).lower():
                    add(item)
                elif isinstance(item, (dict, list)):
                    add(item)

    add(locator)
    return list(dict.fromkeys(values))


def augment_evidence(
    index: LexicalIndex,
    question: dict[str, Any],
    initial: list[dict[str, Any]],
    locator: dict[str, Any],
    settings: BEnsembleSettings,
) -> list[dict[str, Any]]:
    domain = str(question.get("domain") or "")
    rows = list(initial)
    for query_index, query in enumerate(
        collect_locator_queries(locator)[: settings.max_locator_queries],
        start=1,
    ):
        for result in index.search(
            query,
            domain=domain,
            top_k=settings.locator_query_top_k,
        ):
            source = f"locator_{query_index}"
            rows.append(evidence_item(result, source))
            for neighbor in index.neighbors(result.chunk["chunk_id"], radius=1):
                rows.append(
                    {
                        **neighbor,
                        "score": round(float(result.score) * 0.65, 4),
                        "sources": [source + "_neighbor"],
                    }
                )
    return merge_evidence(
        rows,
        max_chunks=settings.final_max_chunks,
        max_chars=settings.final_max_chars,
    )


def normalized_quote(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _citation_candidates(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    values = parsed.get("citations") or parsed.get("evidence_retrieval") or []
    if isinstance(values, dict):
        values = list(values.values())
    if isinstance(values, list):
        candidates.extend(item for item in values if isinstance(item, dict))

    judgements = parsed.get("option_judgement")
    if isinstance(judgements, dict):
        for letter, item in judgements.items():
            if not isinstance(item, dict):
                continue
            option_citations = item.get("citations") or []
            if isinstance(option_citations, dict):
                option_citations = [option_citations]
            if isinstance(option_citations, list):
                for citation in option_citations:
                    if isinstance(citation, dict):
                        candidates.append({"option": letter, **citation})
            quotes = item.get("quoted_clauses") or []
            if isinstance(quotes, dict):
                quotes = [quotes]
            if isinstance(quotes, list):
                for quote in quotes:
                    if isinstance(quote, dict):
                        candidates.append({"option": letter, **quote})
            ids: set[int] = set()
            for key in (
                "supporting_evidence_ids",
                "contradicting_evidence_ids",
                "relevant_evidence_ids",
            ):
                value = item.get(key)
                if isinstance(value, int):
                    ids.add(value)
                elif isinstance(value, list):
                    ids.update(
                        int(v) for v in value if str(v).strip().isdigit()
                    )
            for evidence_id in ids:
                candidates.append({"option": letter, "evidence_id": evidence_id})
    return candidates


def materialize_citations(
    parsed: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for item in _citation_candidates(parsed):
        raw_id = item.get("evidence_id") or item.get("id")
        try:
            match = re.search(r"\d+", str(raw_id))
            evidence_id = int(match.group()) if match else 0
        except (TypeError, ValueError):
            continue
        if evidence_id < 1 or evidence_id > len(evidence):
            continue
        source = evidence[evidence_id - 1]
        quote = str(
            item.get("quote")
            or item.get("quoted_clause")
            or item.get("source_quote")
            or ""
        ).strip()
        verified = bool(
            quote
            and normalized_quote(quote) in normalized_quote(source.get("text"))
        )
        key = (evidence_id, str(item.get("option") or ""), quote)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "evidence_id": evidence_id,
                "option": item.get("option"),
                "doc_id": source.get("doc_id"),
                "chunk_id": source.get("chunk_id"),
                "page": source.get("page"),
                "page_char_start": source.get("page_char_start"),
                "page_char_end": source.get("page_char_end"),
                "source_path": source.get("source_path"),
                "quote": quote,
                "quote_verified": verified,
            }
        )
    return output


def _stage_has_json(parsed: dict[str, Any]) -> bool:
    return bool(parsed)


def _load_cache(path: Path, prompt_version: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict) or value.get("prompt_version") != prompt_version:
        return {}
    return value


def run_b_question(
    question: dict[str, Any],
    index: LexicalIndex,
    client: OpenAICompatibleClient,
    cache_path: Path,
    settings: BEnsembleSettings,
    stop_file: Path | None,
) -> dict[str, Any]:
    payload = _load_cache(cache_path, settings.prompt_version)
    if not payload:
        payload = {
            "schema_version": 1,
            "prompt_version": settings.prompt_version,
            "qid": question["qid"],
            "question": question,
            "stages": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def save() -> None:
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(cache_path, payload)

    def check_stop() -> None:
        if stop_file and stop_file.exists():
            save()
            raise BEnsembleStopped(f"Stop requested by {stop_file}")

    if not payload.get("initial_evidence"):
        payload["initial_evidence"] = build_initial_evidence(index, question, settings)
        save()
    check_stop()

    stages = payload.setdefault("stages", {})
    if "locator" not in stages:
        stages["locator"] = call_json_stage(
            client,
            build_b_locator_messages(question, payload["initial_evidence"]),
            _stage_has_json,
            repair_instruction="请完整给出检索短语和证据缺口字段。",
        )
        save()
    check_stop()

    if not payload.get("retrieved_evidence"):
        payload["retrieved_evidence"] = augment_evidence(
            index,
            question,
            payload["initial_evidence"],
            stages["locator"].get("parsed") or {},
            settings,
        )
        save()
    evidence = payload["retrieved_evidence"]

    if "analyst" not in stages:
        stages["analyst"] = call_json_stage(
            client,
            build_b_analyst_messages(
                question,
                evidence,
                stages["locator"].get("parsed") or {},
            ),
            _stage_has_json,
            repair_instruction="请按指定结构核验全部选项或全部原始变量。",
        )
        save()
    check_stop()

    if "solver" not in stages:
        stages["solver"] = call_json_stage(
            client,
            build_b_solver_messages(
                question,
                evidence,
                stages["locator"].get("parsed") or {},
                stages["analyst"].get("parsed") or {},
            ),
            lambda parsed: valid_final_payload(parsed, question),
            repair_instruction=(
                f"answers 必须恰好包含 {answer_slot_count(question)} 个非空字符串。"
            ),
        )
        save()
    check_stop()

    if "skeptic" not in stages:
        stages["skeptic"] = call_json_stage(
            client,
            build_b_skeptic_messages(
                question,
                evidence,
                stages["locator"].get("parsed") or {},
                stages["analyst"].get("parsed") or {},
                stages["solver"].get("parsed") or {},
            ),
            _stage_has_json,
            repair_instruction="请给出 attacks、reviewed_result 与 unresolved。",
        )
        save()
    check_stop()

    if "judge" not in stages:
        stages["judge"] = call_json_stage(
            client,
            build_b_judge_messages(
                question,
                evidence,
                stages["locator"].get("parsed") or {},
                stages["analyst"].get("parsed") or {},
                stages["solver"].get("parsed") or {},
                stages["skeptic"].get("parsed") or {},
            ),
            lambda parsed: valid_final_payload(parsed, question),
            repair_instruction=(
                f"answers 必须恰好包含 {answer_slot_count(question)} 个非空字符串；"
                "选择题答案只能是实际选项字母。"
            ),
        )
        save()

    judge = stages["judge"].get("parsed") or {}
    answers = parsed_answer_values(judge, question)
    if not valid_final_payload(judge, question):
        raise RuntimeError(
            f"{question['qid']} judge did not produce a valid final answer: "
            f"{stages['judge'].get('raw_output', '')[:300]}"
        )
    payload["result"] = {
        "answers": answers,
        "usage": stage_usage(stages),
        "confidence": judge.get("confidence", judge.get("overall_confidence", 0)),
        "unresolved": judge.get("unresolved") or [],
        "formula": judge.get("formula") or "",
        "calculation_steps": judge.get("calculation_steps") or [],
        "rounding_stage": judge.get("rounding_stage") or "",
        "format_spec": judge.get("format_spec") or "",
        "raw_values": judge.get("raw_values") or {},
        "citations": materialize_citations(judge, evidence),
    }
    save()
    return payload


def _answer_payload(
    questions: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question in questions:
        payload = results.get(question["qid"])
        if not payload or not payload.get("result"):
            continue
        result = payload["result"]
        rows.append(
            {
                "qid": question["qid"],
                "answers": list(result["answers"]),
                "usage": dict(result["usage"]),
            }
        )
    return rows


def write_b_run_outputs(
    output_dir: Path,
    questions: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    settings: BEnsembleSettings,
) -> dict[str, Any]:
    ordered_results = [
        results[question["qid"]]
        for question in questions
        if question["qid"] in results
    ]
    atomic_write_json(output_dir / "audit.json", ordered_results)
    atomic_write_json(output_dir / "answers.json", _answer_payload(questions, results))
    summary = {
        "questions_total": len(questions),
        "questions_completed": len(results),
        "choice_questions_completed": sum(
            question_kind(question) == "choice" and question["qid"] in results
            for question in questions
        ),
        "open_questions_completed": sum(
            question_kind(question) != "choice" and question["qid"] in results
            for question in questions
        ),
        "tokens": {
            key: sum(
                int(payload.get("result", {}).get("usage", {}).get(key, 0) or 0)
                for payload in results.values()
            )
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "settings": asdict(settings),
        "output_dir": str(output_dir),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def run_b_ensemble(
    *,
    processed_dir: Path,
    output_dir: Path,
    client: OpenAICompatibleClient,
    settings: BEnsembleSettings,
    limit: int | None = None,
    qid_filter: set[str] | None = None,
    stop_file: Path | None = None,
) -> dict[str, Any]:
    questions, chunks = load_processed(processed_dir)
    targets = [
        question
        for question in questions
        if not qid_filter or question["qid"] in qid_filter
    ]
    if limit is not None:
        targets = targets[:limit]
    if not targets:
        raise RuntimeError("No B questions matched the requested filter")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache" / "questions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    index = LexicalIndex(chunks)
    results: dict[str, dict[str, Any]] = {}

    for question in questions:
        cache_path = cache_dir / f"{safe_cache_name(question['qid'])}.json"
        cached = _load_cache(cache_path, settings.prompt_version)
        if cached.get("result"):
            results[question["qid"]] = cached

    def audit_target(
        ordinal: int,
        question: dict[str, Any],
    ) -> tuple[int, dict[str, Any], dict[str, Any]]:
        qid = question["qid"]
        print(f"[{ordinal}/{len(targets)}] {qid} start", flush=True)
        payload = run_b_question(
            question,
            index,
            client,
            cache_dir / f"{safe_cache_name(qid)}.json",
            settings,
            stop_file,
        )
        return ordinal, question, payload

    pending = [
        (ordinal, question)
        for ordinal, question in enumerate(targets, start=1)
        if question["qid"] not in results
    ]
    workers = max(1, min(int(settings.question_workers), len(pending) or 1))
    if workers == 1:
        iterator = (
            audit_target(ordinal, question)
            for ordinal, question in pending
        )
        for ordinal, question, payload in iterator:
            results[question["qid"]] = payload
            result = payload["result"]
            print(
                f"[{ordinal}/{len(targets)}] {question['qid']} "
                f"answer={'|'.join(result['answers'])} "
                f"tokens={result['usage']['total_tokens']}",
                flush=True,
            )
            write_b_run_outputs(output_dir, questions, results, settings)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(audit_target, ordinal, question)
                for ordinal, question in pending
            ]
            try:
                for future in as_completed(futures):
                    ordinal, question, payload = future.result()
                    results[question["qid"]] = payload
                    result = payload["result"]
                    print(
                        f"[{ordinal}/{len(targets)}] {question['qid']} "
                        f"answer={'|'.join(result['answers'])} "
                        f"tokens={result['usage']['total_tokens']}",
                        flush=True,
                    )
                    write_b_run_outputs(output_dir, questions, results, settings)
            except Exception:
                for future in futures:
                    future.cancel()
                write_b_run_outputs(output_dir, questions, results, settings)
                raise
    return write_b_run_outputs(output_dir, questions, results, settings)


def teacher_cache_from_audit(
    questions: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_qid = {str(row.get("qid")): row for row in audit_rows}
    cache: list[dict[str, Any]] = []
    for question in questions:
        payload = by_qid.get(question["qid"])
        if not payload or not payload.get("result"):
            raise RuntimeError(f"Full audit is missing teacher result for {question['qid']}")
        result = payload["result"]
        compact_citations = []
        for citation in result.get("citations") or []:
            compact_citations.append(
                {
                    "doc_id": citation.get("doc_id"),
                    "page": citation.get("page"),
                    "quote": citation.get("quote"),
                }
            )
            if len(compact_citations) >= 2:
                break
        cache.append(
            {
                "qid": question["qid"],
                "answers": list(result["answers"]),
                "formula": result.get("formula") or "",
                "calculation_steps": list(result.get("calculation_steps") or [])[:3],
                "citations": compact_citations,
                "format_spec": result.get("format_spec") or "",
            }
        )
    return cache


def run_b_low_token(
    *,
    processed_dir: Path,
    full_audit_path: Path,
    output_dir: Path,
    client: OpenAICompatibleClient,
    stop_file: Path | None = None,
    limit: int | None = None,
    qid_filter: set[str] | None = None,
    workers: int = 3,
) -> dict[str, Any]:
    questions, _ = load_processed(processed_dir)
    audit_rows = json.loads(full_audit_path.read_text(encoding="utf-8"))
    teacher_rows = teacher_cache_from_audit(questions, audit_rows)
    teacher_by_qid = {row["qid"]: row for row in teacher_rows}
    targets = [
        question
        for question in questions
        if not qid_filter or question["qid"] in qid_filter
    ]
    if limit is not None:
        targets = targets[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache" / "questions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "teacher_cache.json", teacher_rows)
    results: dict[str, dict[str, Any]] = {}

    def run_one(
        ordinal: int,
        question: dict[str, Any],
    ) -> tuple[int, dict[str, Any], dict[str, Any]]:
        if stop_file and stop_file.exists():
            raise BEnsembleStopped(f"Stop requested by {stop_file}")
        qid = question["qid"]
        cache_path = cache_dir / f"{safe_cache_name(qid)}.json"
        cached = _load_cache(cache_path, B_LOW_PROMPT_VERSION)
        if cached.get("result"):
            return ordinal, question, cached
        teacher = teacher_by_qid[qid]
        stage = call_json_stage(
            client,
            build_b_teacher_verify_messages(question, teacher),
            lambda parsed: valid_final_payload(parsed, question),
            repair_instruction=(
                f"answers 必须恰好包含 {answer_slot_count(question)} 个字符串，"
                "并与教师答案逐字段一致。"
            ),
            max_parse_attempts=1,
        )
        predicted = parsed_answer_values(stage.get("parsed") or {}, question)
        exact_match = predicted == teacher["answers"]
        result = {
            "schema_version": 1,
            "prompt_version": B_LOW_PROMPT_VERSION,
            "qid": qid,
            "teacher": teacher,
            "stage": stage,
            "result": {
                "answers": list(teacher["answers"]),
                "model_answers": predicted,
                "exact_match_before_fallback": exact_match,
                "fallback_to_teacher": not exact_match,
                "usage": dict(stage["usage"]),
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(cache_path, result)
        return ordinal, question, result

    pending = list(enumerate(targets, start=1))
    max_workers = max(1, min(int(workers), len(pending) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_one, ordinal, question)
            for ordinal, question in pending
        ]
        try:
            for future in as_completed(futures):
                ordinal, question, payload = future.result()
                results[question["qid"]] = payload
                result = payload["result"]
                print(
                    f"[{ordinal}/{len(targets)}] {question['qid']} "
                    f"match={result['exact_match_before_fallback']} "
                    f"tokens={result['usage']['total_tokens']}",
                    flush=True,
                )
                atomic_write_json(
                    output_dir / "answers.json",
                    _answer_payload(questions, results),
                )
        except Exception:
            for future in futures:
                future.cancel()
            raise

    answer_rows = _answer_payload(questions, results)
    atomic_write_json(output_dir / "answers.json", answer_rows)
    atomic_write_json(
        output_dir / "audit.json",
        [results[q["qid"]] for q in questions if q["qid"] in results],
    )
    summary = {
        "questions_total": len(questions),
        "questions_completed": len(results),
        "exact_match_before_fallback": sum(
            bool(row["result"]["exact_match_before_fallback"])
            for row in results.values()
        ),
        "fallbacks": sum(
            bool(row["result"]["fallback_to_teacher"])
            for row in results.values()
        ),
        "final_answers_equal_teacher": all(
            row["result"]["answers"] == row["teacher"]["answers"]
            for row in results.values()
        ),
        "tokens": {
            key: sum(
                int(row["result"]["usage"].get(key, 0) or 0)
                for row in results.values()
            )
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "output_dir": str(output_dir),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary
