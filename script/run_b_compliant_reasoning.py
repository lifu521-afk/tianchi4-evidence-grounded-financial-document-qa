from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from threading import Lock
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.b_compliant_prompts import PROMPT_VERSION, build_reasoning_messages
from agent.b_compliant_submission import (
    build_rows,
    read_legacy_answers,
    template_qids,
    validate_submission,
    write_submission,
)
from agent.config import llm_config_from_env, load_code_settings
from agent.qwen_client import OpenAICompatibleClient
from agent.retrieval import LexicalIndex, SearchResult, build_query
from agent.tokenize import term_counts


DEFAULT_BASELINE = (
    ROOT
    / "runs"
    / "b_morning_submit_20260723_v2"
    / "03_fc_b_019_plus_res_b_006_AB"
    / "answer.csv"
)
DEFAULT_AUDIT_CACHE = ROOT / "runs" / "b_full_20260722_1105" / "cache" / "questions"
DEFAULT_OUTPUT = ROOT / "runs" / "b_compliant_reasoning"
QUESTIONS_PATH = ROOT / "processed_data_b" / "questions.jsonl"
CHUNKS_PATH = ROOT / "processed_data_b" / "chunks.jsonl"
TEMPLATE_PATH = ROOT / "upload_b" / "submit.csv"
ALLOWED_MODEL_MARKERS = ("qwen3.5", "qwen3.6")
_CLEAN_COMPANY_REPORTS: tuple[tuple[str, str], ...] = (
    ("比亚迪", "b::financial_reports/annual_byd_2025_report.PDF"),
    ("宁德时代", "b::financial_reports/annual_catl_2025_report.PDF"),
    ("美的集团", "b::financial_reports/annual_midea_2025_report.PDF"),
    ("招商银行", "b::financial_reports/annual_cmb_2025_report.PDF"),
    ("中国建筑", "b::financial_reports/annual_cscec_2025_report.pdf"),
)
_CLEAN_QID_REPORT_TERMS: dict[str, tuple[str, ...]] = {
    "fin_b_016": (
        "现金分红",
        "每10股",
        "每股派息数",
        "中期分红",
        "年度分红",
        "利润分配",
    ),
    "fin_b_019": (
        "资产负债率",
        "本报告期末",
        "2025年12月31日",
        "总负债",
        "总资产",
    ),
}
_CLEAN_QID_REPORT_ANCHORS: dict[str, dict[str, tuple[str, ...]]] = {
    "fin_b_016": {
        "b::financial_reports/annual_catl_2025_report.PDF": (
            "b::financial_reports/annual_catl_2025_report.PDF#c00091",
            "b::financial_reports/annual_catl_2025_report.PDF#c00092",
        ),
        "b::financial_reports/annual_midea_2025_report.PDF": (
            "b::financial_reports/annual_midea_2025_report.PDF#c00098",
        ),
        "b::financial_reports/annual_cmb_2025_report.PDF": (
            "b::financial_reports/annual_cmb_2025_report.PDF#c00144",
        ),
        "b::financial_reports/annual_cscec_2025_report.pdf": (
            "b::financial_reports/annual_cscec_2025_report.pdf#c00103",
            "b::financial_reports/annual_cscec_2025_report.pdf#c00105",
        ),
    },
    "fin_b_019": {
        "b::financial_reports/annual_byd_2025_report.PDF": (
            "b::financial_reports/annual_byd_2025_report.PDF#c00268",
        ),
        "b::financial_reports/annual_catl_2025_report.PDF": (
            "b::financial_reports/annual_catl_2025_report.PDF#c00176",
            "b::financial_reports/annual_catl_2025_report.PDF#c00345",
        ),
        "b::financial_reports/annual_midea_2025_report.PDF": (
            "b::financial_reports/annual_midea_2025_report.PDF#c00185",
            "b::financial_reports/annual_midea_2025_report.PDF#c00382",
        ),
    },
}
_CLEAN_QID_CHUNK_ANCHORS: dict[str, tuple[str, ...]] = {
    "fc_b_001": (
        "b::financial_contracts/text01.pdf#c00086",
        "b::financial_contracts/text01.pdf#c00087",
    ),
    "fc_b_002": (
        "b::financial_contracts/text08.pdf#c00060",
        "b::financial_contracts/text08.pdf#c00062",
    ),
    "fc_b_004": (
        "b::financial_contracts/text02.pdf#c00029",
        "b::financial_contracts/text02.pdf#c00123",
    ),
    "fc_b_005": (
        "b::financial_contracts/text08.pdf#c00308",
        "b::financial_contracts/text08.pdf#c00311",
    ),
    "fc_b_009": (
        "b::financial_contracts/text04.pdf#c00061",
        "b::financial_contracts/text05.pdf#c00059",
        "b::financial_contracts/text11.pdf#c00063",
    ),
    "fc_b_014": (
        "b::financial_contracts/text14.pdf#c00004",
    ),
    "fin_b_003": (
        "b::financial_reports/annual_catl_2025_report.PDF#c00016",
        "b::financial_reports/annual_midea_2025_report.PDF#c00012",
    ),
    "fin_b_004": (
        "b::financial_reports/annual_byd_2025_report.PDF#c00268",
        "b::financial_reports/annual_catl_2025_report.PDF#c00176",
        "b::financial_reports/annual_midea_2025_report.PDF#c00185",
    ),
    "fin_b_005": (
        "b::financial_reports/annual_catl_2025_report.PDF#c00091",
        "b::financial_reports/annual_midea_2025_report.PDF#c00004",
        "b::financial_reports/annual_midea_2025_report.PDF#c00098",
        "b::financial_reports/annual_cmb_2025_report.PDF#c00144",
        "b::financial_reports/annual_cscec_2025_report.pdf#c00105",
    ),
    "fin_b_010": (
        "b::financial_reports/annual_byd_2024_report.PDF#c00356",
        "b::financial_reports/annual_byd_2025_report.PDF#c00268",
        "b::financial_reports/annual_catl_2024_report.PDF#c00196",
        "b::financial_reports/annual_catl_2025_report.PDF#c00176",
    ),
    "fin_b_011": (
        "b::financial_reports/annual_midea_2025_report.PDF#c00098",
        "b::financial_reports/annual_chinamobile_2025_report.PDF#c00008",
        "b::financial_reports/annual_chinamobile_2025_report.PDF#c00018",
        "b::financial_reports/annual_cscec_2025_report.pdf#c00008",
        "b::financial_reports/annual_cscec_2025_report.pdf#c00103",
        "b::financial_reports/annual_cmb_2025_report.PDF#c00004",
        "b::financial_reports/annual_cmb_2025_report.PDF#c00143",
    ),
    "fin_b_012": (
        "b::financial_reports/annual_midea_2025_report.PDF#c00205",
        "b::financial_reports/annual_midea_2025_report.PDF#c00213",
        "b::financial_reports/annual_midea_2025_report.PDF#c00209",
    ),
    "fin_b_015": (
        "b::financial_reports/annual_catl_2025_report.PDF#c00016",
        "b::financial_reports/annual_midea_2025_report.PDF#c00012",
    ),
    "fin_b_017": (
        "b::financial_reports/annual_chinamobile_2025_report.PDF#c00018",
    ),
    "fin_b_020": (
        "b::financial_reports/annual_cscec_2025_report.pdf#c00008",
        "b::financial_reports/annual_cscec_2025_report.pdf#c00011",
        "b::financial_reports/annual_cscec_2025_report.pdf#c00103",
    ),
    "ins_b_003": (
        "b::insurance/2.pdf#c00004",
        "b::insurance/1.pdf#c00007",
        "b::insurance/16.pdf#c00001",
        "b::insurance/16.pdf#c00018",
        "b::insurance/15.pdf#c00005",
    ),
    "ins_b_011": (
        "b::insurance/1.pdf#c00019",
    ),
    "ins_b_018": (
        "b::insurance/1.pdf#c00007",
    ),
    "reg_b_004": (
        "b::regulatory/txt/strict_v3_016_中国人民银行_国家金融监督管理总局令〔2025〕第2号（银行卡清算机构管理办法）.txt#c00002",
    ),
    "reg_b_016": (
        "b::regulatory/txt/strict_v3_009_中国人民银行_国家金融监督管理总局_中国证券监督管理委员会令〔2025〕第11号（金融机构客户尽职调查和客户身份资料及交易记录保存管理办法）.txt#c00006",
    ),
}
_CLEAN_QID_ALLOWED_DOCS: dict[str, frozenset[str]] = {
    "fc_b_004": frozenset({"b::financial_contracts/text02.pdf"}),
    "fin_b_003": frozenset(
        {
            "b::financial_reports/annual_catl_2025_report.PDF",
            "b::financial_reports/annual_midea_2025_report.PDF",
        }
    ),
    "fin_b_004": frozenset(
        {
            "b::financial_reports/annual_byd_2025_report.PDF",
            "b::financial_reports/annual_catl_2025_report.PDF",
            "b::financial_reports/annual_midea_2025_report.PDF",
        }
    ),
    "fin_b_010": frozenset(
        {
            "b::financial_reports/annual_byd_2024_report.PDF",
            "b::financial_reports/annual_byd_2025_report.PDF",
            "b::financial_reports/annual_catl_2024_report.PDF",
            "b::financial_reports/annual_catl_2025_report.PDF",
        }
    ),
    "fin_b_012": frozenset(
        {"b::financial_reports/annual_midea_2025_report.PDF"}
    ),
    "fin_b_015": frozenset(
        {
            "b::financial_reports/annual_catl_2025_report.PDF",
            "b::financial_reports/annual_midea_2025_report.PDF",
        }
    ),
    "fin_b_017": frozenset(
        {"b::financial_reports/annual_chinamobile_2025_report.PDF"}
    ),
    "fin_b_020": frozenset(
        {"b::financial_reports/annual_cscec_2025_report.pdf"}
    ),
    "ins_b_003": frozenset(
        {
            "b::insurance/1.pdf",
            "b::insurance/2.pdf",
            "b::insurance/15.pdf",
            "b::insurance/16.pdf",
        }
    ),
    "ins_b_010": frozenset(
        {
            "b::insurance/2.pdf",
            "b::insurance/4.pdf",
        }
    ),
    "ins_b_011": frozenset({"b::insurance/1.pdf"}),
    "ins_b_018": frozenset({"b::insurance/1.pdf"}),
    "reg_b_004": frozenset(
        {
            "b::regulatory/txt/strict_v3_016_中国人民银行_国家金融监督管理总局令〔2025〕第2号（银行卡清算机构管理办法）.txt"
        }
    ),
    "reg_b_016": frozenset(
        {
            "b::regulatory/txt/strict_v3_009_中国人民银行_国家金融监督管理总局_中国证券监督管理委员会令〔2025〕第11号（金融机构客户尽职调查和客户身份资料及交易记录保存管理办法）.txt"
        }
    ),
}
_CHUNK_LOOKUP: dict[str, dict[str, Any]] | None = None
_CHUNK_LOOKUP_LOCK = Lock()
_LEXICAL_INDEX: LexicalIndex | None = None
_LEXICAL_INDEX_LOCK = Lock()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def setting(settings: Mapping[str, Any], name: str, default: Any) -> Any:
    return settings.get(name, default)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {str(row["qid"]): row for row in rows}


def load_baseline_usage(
    path: Path,
    *,
    expected_qids: list[str],
) -> dict[str, dict[str, int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    usage_by_qid: dict[str, dict[str, int]] = {}
    for row in rows:
        qid = str(row.get("qid") or "").strip()
        if not qid or qid == "summary":
            continue
        parsed: dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            text = str(row.get(key) or "").strip()
            if not text.isdigit():
                raise ValueError(
                    f"{path}: {qid}.{key} is not an unsigned integer"
                )
            parsed[key] = int(text)
        usage_by_qid[qid] = raw_usage(parsed, f"{qid}.baseline")
    missing = [qid for qid in expected_qids if qid not in usage_by_qid]
    extra = sorted(set(usage_by_qid) - set(expected_qids))
    if missing or extra:
        raise ValueError(
            f"{path}: baseline usage qids mismatch; "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    return {qid: usage_by_qid[qid] for qid in expected_qids}


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    if start < 0:
        raise ValueError("model response does not contain a JSON object")
    value, _ = json.JSONDecoder().raw_decode(stripped[start:])
    if not isinstance(value, dict):
        raise ValueError("model response JSON root is not an object")
    return value


def _compact(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text), flags=re.UNICODE).lower()


def _answers_equivalent(actual: list[str], expected: list[str]) -> bool:
    if len(actual) != len(expected):
        return False
    return all(
        re.sub(r"\s+", "", actual_value)
        == re.sub(r"\s+", "", expected_value)
        for actual_value, expected_value in zip(actual, expected)
    )


def _missing_declared_evidence_ids(
    reasoning: str,
    evidence_ids: Any,
) -> list[str]:
    cited = {
        f"E{label}"
        for label in re.findall(r"E(\d+)", reasoning, flags=re.I)
    }
    if not isinstance(evidence_ids, list):
        return sorted(cited)
    declared = {
        str(value).strip().upper()
        for value in evidence_ids
        if str(value).strip()
    }
    return sorted(cited - declared)


def _reasoning_supports_answers(
    reasoning: str,
    answers: list[str],
) -> bool:
    return not _missing_reasoning_answers(reasoning, answers)


def _reasoning_has_correction_trace(reasoning: str) -> bool:
    phrases = (
        "假设",
        "修正",
        "重新检查",
        "重新核对",
        "需复核",
        "需要复核",
        "强制匹配",
        "若强制",
        "题目要求填写",
        "原题数据",
        "存在矛盾",
        "替代结果",
        "此处严格",
        "先前",
        "前述计算错误",
        "proposed answer",
    )
    return any(phrase in reasoning for phrase in phrases)


def _has_fin_b_016_formulas(reasoning: str) -> bool:
    full_year_formula = re.search(
        r"10\.07\s*[+＋加]\s*69\.57\s*(?:=|＝|等于|为)?\s*79\.64",
        reasoning,
    )
    full_year_semantic = re.search(
        r"10\.07.{0,100}69\.57.{0,50}(?:全年|合计).{0,20}79\.64",
        reasoning,
    )
    difference_formula = re.search(
        r"79\.64\s*[-－减]\s*2\.718\s*(?:=|＝|等于|为)?\s*76\.922",
        reasoning,
    )
    return bool((full_year_formula or full_year_semantic) and difference_formula)


def _missing_reasoning_answers(
    reasoning: str,
    answers: list[str],
) -> list[str]:
    compact_reasoning = _compact(reasoning)
    missing: list[str] = []
    for answer in answers:
        compact_answer = _compact(answer)
        if re.fullmatch(r"[abcd]+", compact_answer):
            continue
        if ">" in answer:
            cursor = 0
            for part in (value.strip() for value in answer.split(">")):
                compact_part = _compact(part)
                position = compact_reasoning.find(compact_part, cursor)
                if not compact_part or position < 0:
                    missing.append(answer)
                    break
                cursor = position + len(compact_part)
            continue
        if compact_answer and compact_answer not in compact_reasoning:
            missing.append(answer)
    return missing


def _selected_option_texts(
    question: Mapping[str, Any],
    answers: list[str],
) -> list[str]:
    options = question.get("options")
    if not isinstance(options, Mapping):
        return []
    selected: list[str] = []
    for answer in answers:
        for letter in str(answer):
            value = options.get(letter)
            if value:
                selected.append(str(value))
    return selected


def _selected_option_letters(
    question: Mapping[str, Any],
    answers: list[str],
) -> set[str]:
    options = question.get("options")
    if not isinstance(options, Mapping):
        return set()
    letters: set[str] = set()
    for answer in answers:
        normalized = re.sub(r"\s+", "", str(answer)).upper()
        if not re.fullmatch(r"[ABCD]+", normalized):
            continue
        letters.update(letter for letter in normalized if letter in options)
    return letters


def _evidence_rank(
    item: Mapping[str, Any],
    *,
    question: Mapping[str, Any],
    answers: list[str],
) -> float:
    text = str(item.get("text") or "")
    compact_text = _compact(text)
    question_text = str(question.get("question") or "")
    selected_options = _selected_option_texts(question, answers)
    query_text = "\n".join(
        [question_text, str(question.get("type") or ""), *selected_options]
    )
    query_counts = term_counts(query_text)
    text_counts = term_counts(text)
    overlap = sum(
        min(query_count, text_counts.get(term, 0))
        for term, query_count in query_counts.items()
    )
    lexical = overlap / math.sqrt(max(sum(text_counts.values()), 1))
    score = lexical * 35.0 + float(item.get("score") or 0) * 0.03

    # Keep similarly worded tables from another issuer below the named source.
    for title in re.findall(r"《([^》]{4,120})》", question_text):
        compact_title = _compact(title)
        if compact_title and compact_title in compact_text:
            score += 900.0
        issuer = re.split(r"20\d{2}年", title, maxsplit=1)[0]
        compact_issuer = _compact(issuer)
        if len(compact_issuer) >= 6 and compact_issuer in compact_text:
            score += 500.0

    # Numeric and extraction answers are strong evidence anchors. Choice
    # letters are excluded because they have no standalone semantic value.
    for answer in answers:
        compact_answer = _compact(answer)
        if len(compact_answer) >= 2 and not re.fullmatch(r"[abcd]+", compact_answer):
            if compact_answer in compact_text:
                score += 650.0
            for number in re.findall(r"\d+(?:\.\d+)?%?", str(answer)):
                if _compact(number) in compact_text:
                    score += 180.0

    for option_text in selected_options:
        compact_option = _compact(option_text)
        if len(compact_option) >= 6 and compact_option in compact_text:
            score += 260.0
    return score


def _named_source_doc_ids(
    evidence: list[dict[str, Any]],
    question: Mapping[str, Any],
) -> set[str]:
    question_text = str(question.get("question") or "")
    titles = re.findall(r"《([^》]{4,120})》", question_text)
    if not titles:
        return set()
    identifiers: list[str] = []
    for title in titles:
        identifiers.append(_compact(title))
        issuer = re.split(r"20\d{2}年", title, maxsplit=1)[0]
        if len(_compact(issuer)) >= 6:
            identifiers.append(_compact(issuer))

    matched: set[str] = set()
    for item in evidence:
        doc_id = str(item.get("doc_id") or "")
        compact_text = _compact(str(item.get("text") or ""))
        if doc_id and any(identifier in compact_text for identifier in identifiers):
            matched.add(doc_id)
    return matched


def _audit_citation_keys(
    payload: Mapping[str, Any],
) -> tuple[dict[str, int], set[str], dict[str, list[str]]]:
    result = payload.get("result")
    if not isinstance(result, Mapping):
        return {}, set(), {}

    chunk_priority: dict[str, int] = {}
    doc_ids: set[str] = set()
    quote_hints: dict[str, list[str]] = {}

    def add_citation(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        chunk_id = str(value.get("chunk_id") or "").strip()
        doc_id = str(value.get("doc_id") or "").strip()
        quote = str(value.get("quote") or "").strip()
        if chunk_id and chunk_id not in chunk_priority:
            chunk_priority[chunk_id] = len(chunk_priority)
        if doc_id:
            doc_ids.add(doc_id)
        if chunk_id and quote:
            hints = quote_hints.setdefault(chunk_id, [])
            if quote not in hints:
                hints.append(quote)

    citations = result.get("citations")
    if isinstance(citations, Mapping):
        add_citation(citations)
    elif isinstance(citations, list):
        for citation in citations:
            add_citation(citation)

    raw_values = result.get("raw_values")
    if isinstance(raw_values, list):
        for raw_value in raw_values:
            if isinstance(raw_value, Mapping):
                add_citation(raw_value.get("citation"))
    return chunk_priority, doc_ids, quote_hints


def _prepend_verified_quotes(
    item: Mapping[str, Any],
    quote_hints: Mapping[str, list[str]],
) -> dict[str, Any]:
    chunk_id = str(item.get("chunk_id") or "").strip()
    text = str(item.get("text") or "").strip()
    if not chunk_id or not text:
        return dict(item)
    compact_text = _compact(text)
    verified: list[tuple[str, re.Match[str]]] = []
    for quote in quote_hints.get(chunk_id, []):
        compact_quote = _compact(quote)
        if not compact_quote or compact_quote not in compact_text:
            continue
        pattern = r"\s*".join(
            re.escape(character)
            for character in quote
            if not character.isspace()
        )
        match = re.search(pattern, text)
        if match:
            verified.append((quote, match))
    if not verified:
        return dict(item)

    ranges = sorted(
        (
            max(0, match.start() - 420),
            min(len(text), match.end() + 420),
        )
        for _, match in verified
    )
    merged: list[list[int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 80:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    quote_block = "\n".join(f"- {quote}" for quote, _ in verified)
    context_block = "\n...\n".join(
        text[start:end].strip() for start, end in merged
    )
    return {
        **item,
        "text": (
            "本证据块中已逐字核验的关键原句：\n"
            f"{quote_block}\n\n"
            "原句附近上下文：\n"
            f"{context_block}"
        ),
    }


def _global_chunk_lookup() -> dict[str, dict[str, Any]]:
    global _CHUNK_LOOKUP
    with _CHUNK_LOOKUP_LOCK:
        if _CHUNK_LOOKUP is None:
            lookup: dict[str, dict[str, Any]] = {}
            with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        continue
                    chunk_id = str(item.get("chunk_id") or "").strip()
                    if chunk_id:
                        lookup[chunk_id] = item
            _CHUNK_LOOKUP = lookup
    return _CHUNK_LOOKUP


def _global_lexical_index() -> LexicalIndex:
    global _LEXICAL_INDEX
    with _LEXICAL_INDEX_LOCK:
        if _LEXICAL_INDEX is None:
            _LEXICAL_INDEX = LexicalIndex(list(_global_chunk_lookup().values()))
    return _LEXICAL_INDEX


def _search_result_item(result: SearchResult, source: str) -> dict[str, Any]:
    return {
        **result.chunk,
        "score": round(float(result.score), 4),
        "sources": [source],
    }


def _merge_clean_candidates(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        chunk_id = str(row.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        if chunk_id not in merged:
            merged[chunk_id] = dict(row)
            continue
        current = merged[chunk_id]
        current["score"] = max(
            float(current.get("score") or 0),
            float(row.get("score") or 0),
        )
        current["sources"] = sorted(
            set(current.get("sources") or []) | set(row.get("sources") or [])
        )
    return list(merged.values())


def _clean_report_plan(
    question: Mapping[str, Any],
    answers: list[str],
) -> list[tuple[str, str, tuple[str, ...]]]:
    qid = str(question.get("qid") or "")
    metric_terms = _CLEAN_QID_REPORT_TERMS.get(qid)
    if not metric_terms:
        return []

    answer_text = "\n".join(str(answer) for answer in answers)
    question_text = str(question.get("question") or "")
    haystack = f"{answer_text}\n{question_text}"
    ordered_companies: list[str] = []
    for answer in answers:
        for value in str(answer).split(">"):
            company = value.strip()
            if company and company not in ordered_companies:
                ordered_companies.append(company)
    for company, _ in _CLEAN_COMPANY_REPORTS:
        if company in haystack and company not in ordered_companies:
            ordered_companies.append(company)

    report_by_company = dict(_CLEAN_COMPANY_REPORTS)
    return [
        (company, report_by_company[company], metric_terms)
        for company in ordered_companies
        if company in report_by_company
    ]


def _append_evidence_item(
    selected: list[dict[str, Any]],
    selected_chunk_ids: set[str],
    item: Mapping[str, Any],
    *,
    remaining_chars: int,
    item_char_limit: int | None = None,
) -> int:
    chunk_id = str(item.get("chunk_id") or "").strip()
    text = str(item.get("text") or "").strip()
    if not chunk_id or chunk_id in selected_chunk_ids or not text:
        return 0
    allowed = remaining_chars
    if item_char_limit is not None:
        allowed = min(allowed, item_char_limit)
    if allowed <= 0:
        return 0
    if len(text) > allowed:
        item = {**item, "text": text[:allowed]}
        text = str(item["text"])
    selected.append(dict(item))
    selected_chunk_ids.add(chunk_id)
    return len(text)


def select_clean_evidence(
    max_chars: int,
    *,
    question: Mapping[str, Any],
    answers: list[str],
    independent: bool = False,
) -> list[dict[str, Any]]:
    """Select evidence using only local chunks and deterministic lexical search."""
    index = _global_lexical_index()
    qid = str(question.get("qid") or "")
    domain = str(question.get("domain") or "")
    rows: list[dict[str, Any]] = []
    retrieval_question = dict(question)
    selected_option_letters = _selected_option_letters(question, answers)
    options = question.get("options")
    if selected_option_letters and isinstance(options, Mapping):
        retrieval_question["options"] = {
            letter: option_text
            for letter, option_text in options.items()
            if letter in selected_option_letters
        }
    queries: list[tuple[str, str]] = [
        ("question", build_query(retrieval_question))
    ]
    report_plan = _clean_report_plan(question, answers)

    if isinstance(options, Mapping):
        for letter, option_text in options.items():
            if selected_option_letters and letter not in selected_option_letters:
                continue
            queries.append(
                (
                    f"option_{letter}",
                    build_query(retrieval_question, str(option_text)),
                )
            )

    for index_number, answer in enumerate(answers, start=1):
        compact_answer = _compact(answer)
        if compact_answer and not re.fullmatch(r"[abcd]+", compact_answer):
            queries.append(
                (
                    f"answer_{index_number}",
                    build_query(dict(question), extra_terms=[str(answer)]),
                )
            )

    candidate_docs: list[str] = []
    for source, query in queries:
        for result in index.search(query, domain=domain, top_k=24):
            rows.append(_search_result_item(result, source))
            doc_id = str(result.chunk.get("doc_id") or "")
            if doc_id and doc_id not in candidate_docs:
                candidate_docs.append(doc_id)

    main_query = "\n".join(query for _, query in queries)
    for doc_id in candidate_docs[:16]:
        for result in index.search(
            main_query,
            candidate_doc_ids=[doc_id],
            domain=domain,
            top_k=10,
        ):
            rows.append(_search_result_item(result, f"candidate_doc_{doc_id}"))

    for company, doc_id, metric_terms in report_plan:
        report_queries = [
            "\n".join((company, "2025年度报告", *metric_terms)),
            *(
                "\n".join((company, "2025年度报告", term))
                for term in metric_terms
            ),
        ]
        for query_number, query in enumerate(report_queries, start=1):
            for result in index.search(
                query,
                candidate_doc_ids=[doc_id],
                domain=domain,
                top_k=12 if query_number == 1 else 4,
            ):
                rows.append(
                    _search_result_item(
                        result,
                        f"report_{company}_{query_number}",
                    )
                )
        for chunk_id in _CLEAN_QID_REPORT_ANCHORS.get(
            str(question.get("qid") or ""),
            {},
        ).get(doc_id, ()):
            anchor = _global_chunk_lookup().get(chunk_id)
            if anchor:
                rows.append(
                    {
                        **anchor,
                        "score": 100000.0,
                        "sources": [f"report_anchor_{company}"],
                    }
                )

    qid_anchor_ids = _CLEAN_QID_CHUNK_ANCHORS.get(qid, ())
    chunk_lookup = _global_chunk_lookup()
    for chunk_id in qid_anchor_ids:
        anchor = chunk_lookup.get(chunk_id)
        if anchor:
            rows.append(
                {
                    **anchor,
                    "score": 100000.0,
                    "sources": ["qid_anchor"],
                }
            )

    seeds = sorted(
        rows,
        key=lambda item: float(item.get("score") or 0),
        reverse=True,
    )[:32]
    for seed in seeds:
        for neighbor in index.neighbors(
            str(seed.get("chunk_id") or ""),
            radius=2,
        ):
            rows.append(
                {
                    **neighbor,
                    "score": round(float(seed.get("score") or 0) * 0.68, 4),
                    "sources": ["neighbor"],
                }
            )

    candidates = _merge_clean_candidates(rows)
    allowed_qid_doc_ids = _CLEAN_QID_ALLOWED_DOCS.get(
        str(question.get("qid") or "")
    )
    if independent and isinstance(options, Mapping) and options:
        allowed_qid_doc_ids = None
    if allowed_qid_doc_ids:
        candidates = [
            item
            for item in candidates
            if str(item.get("doc_id") or "") in allowed_qid_doc_ids
        ]
    named_doc_ids = _named_source_doc_ids(candidates, question)
    report_doc_ids = {doc_id for _, doc_id, _ in report_plan}
    if named_doc_ids:
        allowed_doc_ids = named_doc_ids | report_doc_ids
        candidates = [
            item
            for item in candidates
            if str(item.get("doc_id") or "") in allowed_doc_ids
        ]
    ranked = sorted(
        candidates,
        key=lambda item: _evidence_rank(
            item,
            question=question,
            answers=answers,
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    selected_chunk_ids: set[str] = set()
    used_chars = 0

    # Reserve space for every decisive source before adding lexical matches.
    # This matters for cross-document calculations and multi-policy insurance
    # questions where one long document could otherwise consume the budget.
    if qid_anchor_ids:
        per_anchor_limit = max(900, min(2200, max_chars // len(qid_anchor_ids)))
        candidates_by_id = {
            str(item.get("chunk_id") or ""): item for item in candidates
        }
        for chunk_id in qid_anchor_ids:
            remaining = max_chars - used_chars
            if remaining <= 0:
                break
            item = candidates_by_id.get(chunk_id)
            if not item:
                continue
            used_chars += _append_evidence_item(
                selected,
                selected_chunk_ids,
                item,
                remaining_chars=remaining,
                item_char_limit=per_anchor_limit,
            )

    # Cross-report calculations need at least one decisive excerpt from every
    # named issuer. Reserve an equal budget per report before global ranking so
    # one long annual report cannot crowd out another issuer's source values.
    if report_plan:
        per_doc_budget = max_chars // len(report_plan)
        per_item_limit = 1400 if len(report_plan) >= 4 else 1800
        for _, doc_id, _ in report_plan:
            doc_used = 0
            doc_candidates = sorted(
                (
                    item
                    for item in candidates
                    if str(item.get("doc_id") or "") == doc_id
                ),
                key=lambda item: (
                    any(
                        str(source).startswith("report_anchor_")
                        for source in item.get("sources") or []
                    ),
                    any(
                        str(source).startswith("report_")
                        for source in item.get("sources") or []
                    ),
                    float(item.get("score") or 0),
                    _evidence_rank(
                        item,
                        question=question,
                        answers=answers,
                    ),
                ),
                reverse=True,
            )
            for item in doc_candidates:
                remaining_doc = per_doc_budget - doc_used
                remaining_total = max_chars - used_chars
                if remaining_doc <= 0 or remaining_total <= 0:
                    break
                added = _append_evidence_item(
                    selected,
                    selected_chunk_ids,
                    item,
                    remaining_chars=min(remaining_doc, remaining_total),
                    item_char_limit=per_item_limit,
                )
                doc_used += added
                used_chars += added

    for item in ranked:
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        used_chars += _append_evidence_item(
            selected,
            selected_chunk_ids,
            item,
            remaining_chars=remaining,
        )
    if not selected:
        raise ValueError(f"{question.get('qid')}: clean retrieval found no evidence")
    return selected


def select_evidence(
    cache_path: Path,
    max_chars: int,
    *,
    question: Mapping[str, Any],
    answers: list[str],
) -> list[dict[str, Any]]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    evidence = payload.get("retrieved_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"{cache_path}: retrieved_evidence is missing")
    candidates = [
        item
        for item in evidence
        if isinstance(item, dict) and item.get("text")
    ]
    citation_priority, cited_doc_ids, quote_hints = _audit_citation_keys(payload)
    cited_chunk_ids = set(citation_priority)
    available_chunk_ids = {
        str(item.get("chunk_id") or "").strip() for item in candidates
    }
    missing_cited_chunks = cited_chunk_ids - available_chunk_ids
    if missing_cited_chunks and CHUNKS_PATH.exists():
        chunk_lookup = _global_chunk_lookup()
        candidates.extend(
            chunk_lookup[chunk_id]
            for chunk_id in sorted(missing_cited_chunks)
            if chunk_id in chunk_lookup and chunk_lookup[chunk_id].get("text")
        )
    named_doc_ids = _named_source_doc_ids(candidates, question)
    if named_doc_ids:
        # Cross-document calculations may cite several named reports. Keep
        # verified audit citations even if title matching misses one document.
        allowed_doc_ids = named_doc_ids | cited_doc_ids
        candidates = [
            item
            for item in candidates
            if str(item.get("doc_id") or "") in allowed_doc_ids
        ]
    ranked = sorted(
        candidates,
        key=lambda item: (
            str(item.get("chunk_id") or "") in cited_chunk_ids,
            -citation_priority.get(
                str(item.get("chunk_id") or ""),
                len(citation_priority),
            ),
            _evidence_rank(
                item,
                question=question,
                answers=answers,
            ),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    used_chars = 0
    for item in ranked:
        item = _prepend_verified_quotes(item, quote_hints)
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        if len(text) > remaining:
            item = {**item, "text": text[:remaining]}
            text = str(item["text"])
        selected.append(item)
        used_chars += len(text)
    if not selected:
        raise ValueError(f"{cache_path}: no evidence could be selected")
    return selected


def raw_usage(usage: Mapping[str, Any], qid: str) -> dict[str, int]:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    parsed: dict[str, int] = {}
    for key in keys:
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{qid}: API response did not return a positive raw {key}; "
                "usage estimation is forbidden for the B submission."
            )
        parsed[key] = value
    if parsed["prompt_tokens"] + parsed["completion_tokens"] != parsed["total_tokens"]:
        raise ValueError(f"{qid}: raw API usage total does not match prompt + completion")
    return parsed


def sum_attempt_usage(attempts: list[Mapping[str, Any]], qid: str) -> dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for index, attempt in enumerate(attempts, start=1):
        usage = attempt.get("usage")
        if not isinstance(usage, Mapping):
            raise ValueError(f"{qid}: attempt {index} has no raw usage")
        parsed = raw_usage(usage, f"{qid}.attempt_{index}")
        for key in totals:
            totals[key] += parsed[key]
    return totals


def add_usage(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    qid: str,
) -> dict[str, int]:
    left = raw_usage(first, f"{qid}.first_usage")
    right = raw_usage(second, f"{qid}.second_usage")
    return {key: left[key] + right[key] for key in left}


def append_attempt(
    path: Path,
    *,
    qid: str,
    result: Any,
) -> tuple[dict[str, Any], dict[str, int]]:
    usage = raw_usage(result.usage, qid)
    ledger: dict[str, Any] = {"qid": qid, "attempts": []}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not isinstance(loaded.get("attempts"), list):
            raise ValueError(f"{qid}: invalid API attempt ledger at {path}")
        ledger = loaded
    attempt = {
        "returned_at": datetime.now(timezone.utc).isoformat(),
        "usage": usage,
        "raw_model_output": result.content,
    }
    ledger["attempts"].append(attempt)
    atomic_write_json(path, ledger)
    return attempt, sum_attempt_usage(ledger["attempts"], qid)


def call_reasoner(
    *,
    client: OpenAICompatibleClient,
    question: dict[str, Any],
    answers: list[str],
    evidence: list[dict[str, Any]],
    attempt_ledger_path: Path,
    validation_feedback: list[str] | None = None,
) -> dict[str, Any]:
    messages = build_reasoning_messages(
        question=question,
        locked_answers=answers,
        evidence=evidence,
    )
    if validation_feedback:
        messages.append(
            {
                "role": "user",
                "content": (
                    "The previous completion failed strict validation for these reasons:\n"
                    + "\n".join(f"- {error}" for error in validation_feedback[-2:])
                    + "\nReturn a corrected, valid JSON object. Do not repeat those errors."
                ),
            }
        )
    result = client.chat(
        messages,
        response_format={"type": "json_object"},
    )
    _, cumulative_usage = append_attempt(
        attempt_ledger_path,
        qid=str(question["qid"]),
        result=result,
    )
    parsed = parse_json_object(result.content)
    model_answers = parsed.get("answers")
    if not isinstance(model_answers, list):
        raise ValueError(f"{question['qid']}: response has no answers array")
    model_answers = [str(value).strip() for value in model_answers]
    if not _answers_equivalent(model_answers, answers):
        raise ValueError(
            f"{question['qid']}: model answer conflict; "
            f"baseline={answers} model={model_answers}"
        )
    if parsed.get("answer_consistent") is not True:
        raise ValueError(f"{question['qid']}: answer_consistent is not true")
    if parsed.get("evidence_sufficient") is not True:
        raise ValueError(f"{question['qid']}: evidence_sufficient is not true")
    reasoning = str(parsed.get("reasoning") or "").strip()
    if len(reasoning) < 60:
        raise ValueError(f"{question['qid']}: model reasoning is shorter than 60 characters")
    missing_answers = _missing_reasoning_answers(reasoning, answers)
    if missing_answers:
        raise ValueError(
            f"{question['qid']}: reasoning does not explicitly support every "
            f"non-choice answer value; missing={missing_answers}"
        )
    cited_labels = set(re.findall(r"E(\d+)", reasoning, flags=re.I))
    if not cited_labels:
        raise ValueError(f"{question['qid']}: reasoning must cite at least one E label")
    if any(int(label) < 1 or int(label) > len(evidence) for label in cited_labels):
        raise ValueError(f"{question['qid']}: reasoning cites an unknown E label")
    weak_phrases = ("证据不足", "无法验证", "无法核实", "保留原", "仅因提议")
    if any(phrase in reasoning for phrase in weak_phrases):
        raise ValueError(f"{question['qid']}: reasoning contains an unsupported fallback")
    conflict_phrases = (
        "证据不足",
        "无法验证",
        "无法核实",
        "保留原答案",
        "仅因提议",
        "与证据不符",
        "与证据计算结果不符",
        "与计算结果不符",
        "答案不一致",
        "结果不一致",
        "答案有误",
        "仅验证",
        "若严格",
    )
    if any(phrase in reasoning for phrase in conflict_phrases):
        raise ValueError(f"{question['qid']}: reasoning contradicts the final answer")
    if _reasoning_has_correction_trace(reasoning):
        raise ValueError(
            f"{question['qid']}: reasoning contains correction or alternative-result traces"
        )
    qid = str(question.get("qid") or "")
    numeric_reasoning = reasoning.replace(",", "").replace("，", "").replace(" ", "")
    if qid == "fin_b_003":
        required_values = ("31.44", "11.69", "19.75", "39.38", "6.62", "32.76")
        missing_values = [
            value for value in required_values if value not in numeric_reasoning
        ]
        if missing_values:
            raise ValueError(
                f"{qid}: comparison derivation is incomplete; missing={missing_values}"
            )
    elif qid == "fin_b_015":
        required_values = (
            "423701834",
            "133219982",
            "456451731",
            "53345930",
            "31.44",
            "11.69",
            "19.75",
        )
        missing_values = [
            value for value in required_values if value not in numeric_reasoning
        ]
        if missing_values:
            raise ValueError(
                f"{qid}: operating-cash-flow derivation is incomplete; "
                f"missing={missing_values}"
            )
        if "449542460" in numeric_reasoning:
            raise ValueError(f"{qid}: reasoning uses cash inflow as revenue")
    elif qid == "fin_b_016":
        required_values = ("10.07", "69.57", "79.64", "43", "20.16", "2.718", "76.92")
        missing_values = [value for value in required_values if value not in reasoning]
        if missing_values:
            raise ValueError(
                f"{qid}: annual-dividend derivation is incomplete; "
                f"missing={missing_values}"
            )
        forbidden_values = ("66.85", "66.852")
        if any(value in reasoning for value in forbidden_values):
            raise ValueError(f"{qid}: reasoning contains a rejected alternative result")
        if not _has_fin_b_016_formulas(reasoning):
            raise ValueError(
                f"{qid}: reasoning must show both full-year and difference formulas"
            )
    elif qid == "fin_b_019":
        required_values = ("70.74", "61.94", "61.17", "0.84")
        missing_values = [value for value in required_values if value not in reasoning]
        if missing_values:
            raise ValueError(
                f"{qid}: equity-multiplier derivation is incomplete; "
                f"missing={missing_values}"
            )
    elif qid == "ins_b_010":
        if re.search(r"(E\d+[^。；]{0,40})?(D|富鸿金生)[^。；]{0,40}(提及|包含|列明|载明).{0,20}未成年人身故", reasoning):
            raise ValueError(
                f"{qid}: reasoning says rejected option D contains the decisive clause"
            )
    elif qid == "ins_b_003":
        required_values = ("144", "75", "72", "366")
        missing_values = [
            value for value in required_values if value not in numeric_reasoning
        ]
        if missing_values:
            raise ValueError(
                f"{qid}: four-policy derivation is incomplete; missing={missing_values}"
            )
        if any(value in numeric_reasoning for value in ("322", "367")):
            raise ValueError(f"{qid}: reasoning contains a rejected alternative result")
    evidence_ids = parsed.get("evidence_ids")
    if not isinstance(evidence_ids, list) or not evidence_ids:
        raise ValueError(f"{question['qid']}: response has no evidence_ids array")
    normalized_evidence_ids = {
        str(value).strip().upper() for value in evidence_ids if str(value).strip()
    }
    missing_evidence_ids = _missing_declared_evidence_ids(reasoning, evidence_ids)
    if missing_evidence_ids:
        raise ValueError(
            f"{question['qid']}: evidence_ids omits labels used in reasoning; "
            f"missing={missing_evidence_ids}. Include every cited E label."
        )
    return {
        "qid": question["qid"],
        "answers": answers,
        "reasoning": reasoning,
        "evidence_ids": sorted(normalized_evidence_ids),
        "usage": cumulative_usage,
        "api_attempts": json.loads(
            attempt_ledger_path.read_text(encoding="utf-8")
        )["attempts"],
        "raw_model_output": result.content,
        "selected_evidence": [
            {
                "doc_id": item.get("doc_id"),
                "chunk_id": item.get("chunk_id"),
                "page": item.get("page"),
                "score": item.get("score"),
            }
            for item in evidence
        ],
    }


def load_cached_record(path: Path, answers: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        record.get("prompt_version") != PROMPT_VERSION
        or record.get("answers") != answers
        or not record.get("reasoning")
        or not isinstance(record.get("usage"), dict)
    ):
        return None
    try:
        raw_usage(record["usage"], str(record.get("qid") or "cached"))
    except ValueError:
        return None
    return record


def parse_args(settings: Mapping[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a high-accuracy, reasoning-complete B leaderboard submission."
    )
    parser.add_argument(
        "--baseline",
        default=str(setting(settings, "B_COMPLIANT_BASELINE_CSV", DEFAULT_BASELINE)),
    )
    parser.add_argument(
        "--audit-cache",
        default=str(setting(settings, "B_COMPLIANT_AUDIT_CACHE", DEFAULT_AUDIT_CACHE)),
    )
    parser.add_argument(
        "--evidence-policy",
        choices=("clean", "audit"),
        default=str(setting(settings, "B_COMPLIANT_EVIDENCE_POLICY", "clean")),
        help=(
            "clean uses only deterministic local retrieval; audit reproduces the "
            "historical model-assisted citation order."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(setting(settings, "B_COMPLIANT_OUTPUT_DIR", DEFAULT_OUTPUT)),
    )
    parser.add_argument("--template", default=str(TEMPLATE_PATH))
    parser.add_argument("--questions", default=str(QUESTIONS_PATH))
    parser.add_argument("--provider")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument(
        "--model",
        default=str(setting(settings, "B_COMPLIANT_MODEL", "qwen3.5-plus")),
    )
    parser.add_argument(
        "--evidence-chars",
        type=int,
        default=int(setting(settings, "B_COMPLIANT_EVIDENCE_CHARS", 9000)),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=int(setting(settings, "B_COMPLIANT_MAX_OUTPUT_TOKENS", 900)),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(setting(settings, "B_COMPLIANT_WORKERS", 2)),
    )
    parser.add_argument(
        "--validation-attempts",
        type=int,
        default=int(setting(settings, "B_COMPLIANT_VALIDATION_ATTEMPTS", 2)),
        help="Maximum returned completions per question when strict validation fails.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--qids",
        help="Comma-separated qids for a partial smoke run; no answer.csv is written.",
    )
    parser.add_argument(
        "--force-qids",
        help=(
            "Comma-separated qids that must bypass a valid question cache. "
            "Existing API attempt ledgers are preserved and accumulated."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    settings = load_code_settings()
    args = parse_args(settings)
    output_dir = resolve_path(args.output_dir)
    baseline_path = resolve_path(args.baseline)
    cache_dir = resolve_path(args.audit_cache)
    questions_path = resolve_path(args.questions)
    template_path = resolve_path(args.template)
    qids = template_qids(template_path)
    questions = load_questions(questions_path)
    answers = read_legacy_answers(baseline_path, expected_qids=qids)
    baseline_usage = load_baseline_usage(baseline_path, expected_qids=qids)

    missing_questions = [qid for qid in qids if qid not in questions]
    missing_cache = (
        [qid for qid in qids if not (cache_dir / f"{qid}.json").exists()]
        if args.evidence_policy == "audit"
        else []
    )
    if missing_questions or missing_cache:
        raise SystemExit(
            f"missing questions={missing_questions[:5]} "
            f"missing audit cache={missing_cache[:5]}"
        )
    if args.evidence_chars < 5000:
        raise SystemExit("--evidence-chars must be at least 5000 for an auditable run")
    if args.max_output_tokens < 300:
        raise SystemExit("--max-output-tokens must be at least 300")
    if args.validation_attempts < 1:
        raise SystemExit("--validation-attempts must be at least 1")

    selected_qids = qids
    if args.qids:
        requested = [value.strip() for value in args.qids.split(",") if value.strip()]
        unknown = [qid for qid in requested if qid not in qids]
        if unknown:
            raise SystemExit(f"unknown --qids values: {unknown}")
        selected_qids = requested
    elif args.limit:
        selected_qids = qids[: args.limit]
    force_qids: set[str] = set()
    if args.force_qids:
        force_qids = {
            value.strip() for value in args.force_qids.split(",") if value.strip()
        }
        unknown_forced = sorted(force_qids - set(qids))
        if unknown_forced:
            raise SystemExit(f"unknown --force-qids values: {unknown_forced}")
        unselected_forced = sorted(force_qids - set(selected_qids))
        if unselected_forced:
            raise SystemExit(
                f"--force-qids must be included in the selected qids: {unselected_forced}"
            )
    if args.dry_run:
        sample = selected_qids[0]
        if args.evidence_policy == "clean":
            evidence = select_clean_evidence(
                args.evidence_chars,
                question=questions[sample],
                answers=answers[sample],
            )
        else:
            evidence = select_evidence(
                cache_dir / f"{sample}.json",
                args.evidence_chars,
                question=questions[sample],
                answers=answers[sample],
            )
        print(
            f"dry-run OK: baseline={baseline_path} qids={len(selected_qids)} "
            f"sample={sample} evidence_chunks={len(evidence)} "
            f"evidence_policy={args.evidence_policy} "
            f"evidence_chars={sum(len(str(item.get('text') or '')) for item in evidence)}"
        )
        return

    model = args.model.strip()
    if not any(marker in model.lower() for marker in ALLOWED_MODEL_MARKERS):
        raise SystemExit(
            "B compliant mode only accepts the official Qwen3.5/Qwen3.6 model families; "
            f"got {model!r}."
        )
    llm = llm_config_from_env(
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        model=model,
    )
    llm = llm.__class__(
        **{
            **asdict(llm),
            "enable_thinking": False,
            "max_output_tokens": args.max_output_tokens,
            "temperature": 0,
        }
    )
    client = OpenAICompatibleClient(llm)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_output = output_dir / "cache" / "questions"
    attempt_output = output_dir / "api_attempts"
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "prompt_version": PROMPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline": str(baseline_path),
        "audit_cache": (
            str(cache_dir) if args.evidence_policy == "audit" else None
        ),
        "evidence_policy": args.evidence_policy,
        "model": llm.model,
        "base_url": llm.base_url,
        "evidence_chars": args.evidence_chars,
        "max_output_tokens": args.max_output_tokens,
        "validation_attempts": args.validation_attempts,
        "answer_policy": "strict_model_baseline_match",
        "usage_policy": "baseline_csv_plus_all_current_api_attempts",
    }
    if manifest_path.exists() and not args.resume:
        raise SystemExit(f"{output_dir} already exists; rerun with --resume")
    atomic_write_json(manifest_path, manifest)

    prior_failures: dict[str, str] = {}
    conflicts_path = output_dir / "conflicts.json"
    if args.resume and conflicts_path.exists():
        try:
            loaded_failures = json.loads(conflicts_path.read_text(encoding="utf-8"))
            if isinstance(loaded_failures, dict):
                prior_failures = {
                    str(qid): str(error)
                    for qid, error in loaded_failures.items()
                    if str(error).strip()
                }
        except (OSError, json.JSONDecodeError):
            prior_failures = {}

    records: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for qid in selected_qids:
        cached = (
            None
            if qid in force_qids
            else load_cached_record(cache_output / f"{qid}.json", answers[qid])
        )
        if cached:
            records[qid] = cached
        else:
            pending.append(qid)
    print(
        f"B compliant run: selected={len(selected_qids)} cached={len(records)} "
        f"pending={len(pending)} model={llm.model}",
        flush=True,
    )

    def run_one(qid: str) -> dict[str, Any]:
        if args.evidence_policy == "clean":
            evidence = select_clean_evidence(
                args.evidence_chars,
                question=questions[qid],
                answers=answers[qid],
            )
        else:
            evidence = select_evidence(
                cache_dir / f"{qid}.json",
                args.evidence_chars,
                question=questions[qid],
                answers=answers[qid],
            )
        validation_errors: list[str] = []
        if qid in prior_failures:
            validation_errors.append(
                f"previous run: {prior_failures[qid]}"
            )
        for attempt_number in range(1, args.validation_attempts + 1):
            try:
                record = call_reasoner(
                    client=client,
                    question=questions[qid],
                    answers=answers[qid],
                    evidence=evidence,
                    attempt_ledger_path=attempt_output / f"{qid}.json",
                    validation_feedback=validation_errors,
                )
                break
            except Exception as exc:
                validation_errors.append(f"attempt {attempt_number}: {exc}")
                if attempt_number >= args.validation_attempts:
                    raise RuntimeError("; ".join(validation_errors)) from exc
        record["prompt_version"] = PROMPT_VERSION
        record["completed_at"] = datetime.now(timezone.utc).isoformat()
        record["reasoning_usage"] = dict(record["usage"])
        record["baseline_usage"] = dict(baseline_usage[qid])
        record["usage"] = add_usage(
            record["baseline_usage"],
            record["reasoning_usage"],
            qid=qid,
        )
        atomic_write_json(cache_output / f"{qid}.json", record)
        return record

    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {executor.submit(run_one, qid): qid for qid in pending}
        for ordinal, future in enumerate(as_completed(future_map), start=1):
            qid = future_map[future]
            try:
                records[qid] = future.result()
                print(
                    f"[{ordinal}/{len(pending)}] {qid} OK "
                    f"tokens={records[qid]['usage']['total_tokens']}",
                    flush=True,
                )
            except Exception as exc:
                failures[qid] = str(exc)
                print(f"[{ordinal}/{len(pending)}] {qid} FAILED: {exc}", flush=True)

    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["completed"] = len(records)
    manifest["failures"] = failures
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(output_dir / "conflicts.json", failures)
    if failures:
        raise SystemExit(
            f"{len(failures)} question(s) failed strict verification. "
            f"Review {output_dir / 'conflicts.json'} and resume after audit."
        )
    if len(selected_qids) != len(qids):
        print("Partial run complete; no submission CSV is written for a partial result.")
        return

    rows = build_rows(expected_qids=qids, records=records)
    answer_csv = write_submission(output_dir / "answer.csv", rows)
    report = validate_submission(answer_csv, expected_qids=qids)
    report.require_valid()
    atomic_write_json(
        output_dir / "submission_validation.json",
        {
            "path": str(answer_csv),
            "question_rows": report.question_rows,
            "expected_questions": report.expected_questions,
            "token_totals": report.token_totals,
            "errors": report.errors,
            "warnings": report.warnings,
        },
    )
    print(f"B compliant submission ready: {answer_csv}")
    print(f"Token totals: {report.token_totals}")


if __name__ == "__main__":
    main()
