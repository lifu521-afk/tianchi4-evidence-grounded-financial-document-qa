from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .domain import get_profile
from .io_utils import read_jsonl, write_json
from .qwen_client import QwenClient, approx_token_count
from .retrieval import LexicalIndex, gather_evidence


VALID_LETTERS = "ABCD"
ANSWER_FIELDS = ["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"]
QUESTION_CACHE_VERSION = 1
BROAD_MULTI_REVIEW_REASON = "multi_requires_skeptical_review"
TRUE_JUDGEMENT_VALUES = {
    "true", "yes", "正确", "支持", "supported", "成立", "符合", "准确", "对",
}
FALSE_JUDGEMENT_VALUES = {
    "false", "no", "错误", "不支持", "unsupported", "不成立", "不符合", "不准确", "错",
}
UNCERTAIN_JUDGEMENT_VALUES = {
    "uncertain", "unknown", "不确定", "无法判断", "证据不足", "无法确认", "未提及",
}
MISSING_EVIDENCE_MARKERS = (
    "证据不足", "缺乏证据", "未提供", "未包含", "未提及", "未明确提及", "未给出",
    "无法判断", "无法确认", "无法得出", "无法直接判断", "无法进行对比", "无法计算",
)
INFERENTIAL_REASONING_MARKERS = (
    "必然", "推定", "推断", "通常", "一般而言", "作为标准", "可判定", "结合常识",
)
UNKNOWN_ERROR_TYPES = {"missing_evidence", "retrieval_gap", "unknown"}
FOCUS_TERMS = (
    "营业收入", "营业总收入", "归属于上市公司股东", "归属于母公司股东", "归母", "净利润",
    "经营活动产生的现金流量净额", "经营活动", "现金流量净额", "研发投入", "研发费用",
    "研发投入占营业收入", "分红", "现金分红", "回购", "每10股", "同比", "增长", "下降",
    "保险责任", "身故保险金", "现金价值", "账户价值", "退保", "已交保险费", "领取", "免责",
    "第", "条", "应当", "不得", "可以", "必须", "工作日", "报告", "处罚", "股东会", "股东大会",
    "普通决议", "特别决议", "三分之二", "过半数", "本章程的修改", "章程修改", "对外担保",
    "资产负债率", "变更募集资金用途", "独立董事", "董事候选人", "利害关系", "法律顾问",
    "票面利率", "付息", "兑付", "评级", "担保", "回售", "赎回", "募集资金",
    "行业", "公司", "市场", "增速", "毛利率", "份额", "预测", "风险", "结论", "观点",
)
NUMERIC_REVIEW_HINTS = (
    "同比", "增长", "下降", "高于", "低于", "超过", "不超过", "不少于", "不高于", "不低于",
    "收入", "利润", "现金流", "研发", "分红", "回购", "亿元", "万元", "占比", "比例", "%",
    "毛利率", "增速", "每10股", "年度", "年",
)
CROSS_DOC_HINTS = (
    "比较", "相比", "分别", "各", "两", "多个", "均", "同时", "高于", "低于", "大于", "小于",
    "多于", "少于", "是否都", "是否均",
)


class RunStopped(Exception):
    def __init__(self, completed: int, total: int, stop_file: Path) -> None:
        super().__init__(f"Stop requested by {stop_file}")
        self.completed = completed
        self.total = total
        self.stop_file = stop_file


def normalize_answer(answer: str, answer_format: str) -> str:
    letters = [ch for ch in answer.upper() if ch in VALID_LETTERS]
    if answer_format == "multi":
        return "".join(sorted(set(letters)))
    if answer_format in {"mcq", "tf"}:
        return letters[0] if letters else ""
    return "".join(sorted(set(letters)))


def allowed_answer_letters(question: dict) -> list[str]:
    letters = option_letters_for_question(question)
    if question.get("answer_format") == "tf":
        letters = [letter for letter in letters if letter in {"A", "B"}]
    return letters or list(VALID_LETTERS)


def normalize_answer_for_question(answer: str, question: dict) -> str:
    answer_format = str(question.get("answer_format", "multi"))
    allowed = allowed_answer_letters(question)
    filtered = [letter for letter in str(answer).upper() if letter in allowed]
    if answer_format == "multi":
        return "".join(letter for letter in allowed if letter in set(filtered))
    return filtered[0] if filtered else ""


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def safe_cache_name(qid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", qid)


def cache_manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def read_cache_manifest(cache_dir: Path) -> dict[str, Any]:
    path = cache_manifest_path(cache_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def prepare_question_cache(cache_dir: Path | None, resume: bool) -> str | None:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_cache_manifest(cache_dir)
    if resume:
        return str(manifest["run_id"]) if manifest.get("run_id") else None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    atomic_write_json(
        cache_manifest_path(cache_dir),
        {
            "version": QUESTION_CACHE_VERSION,
            "run_id": run_id,
            "created_at": run_id,
            "note": "Per-question answer cache. Use --resume to continue this run.",
        },
    )
    return run_id


def write_question_cache(
    cache_dir: Path | None,
    qid: str,
    row: dict[str, Any],
    evidence_row: dict[str, Any],
    run_id: str | None,
) -> None:
    if cache_dir is None:
        return
    payload = {
        "version": QUESTION_CACHE_VERSION,
        "run_id": run_id,
        "qid": qid,
        "answer_row": row,
        "evidence_row": evidence_row,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(cache_dir / f"{safe_cache_name(qid)}.json", payload)


def read_question_cache(
    cache_dir: Path | None,
    run_id: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    rows: dict[str, dict[str, Any]] = {}
    evidence_rows: dict[str, dict[str, Any]] = {}
    bad_files: list[str] = []
    if cache_dir is None or not cache_dir.exists():
        return rows, evidence_rows, bad_files

    if run_id is None:
        manifest = read_cache_manifest(cache_dir)
        run_id = str(manifest.get("run_id") or "") or None

    for path in sorted(cache_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            bad_files.append(str(path))
            continue
        if run_id and payload.get("run_id") not in {run_id, None}:
            continue
        row = payload.get("answer_row")
        evidence_row = payload.get("evidence_row")
        qid = str(payload.get("qid") or (row or {}).get("qid") or "")
        if not qid or not isinstance(row, dict):
            bad_files.append(str(path))
            continue
        row["qid"] = qid
        rows[qid] = row
        if isinstance(evidence_row, dict):
            evidence_row["qid"] = qid
            evidence_rows[qid] = evidence_row
    return rows, evidence_rows, bad_files


def format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{key}. {value}" for key, value in sorted(options.items()))



def option_letters_for_question(question: dict) -> list[str]:
    letters = sorted(str(key).upper() for key in question.get("options", {}) if str(key).upper() in VALID_LETTERS)
    if letters:
        return letters
    return list(VALID_LETTERS)

def option_tags(item: dict) -> str:
    letters = []
    for source in item.get("sources", []):
        if source.startswith("option_") and len(source) == len("option_A"):
            letters.append(source[-1])
    return "".join(sorted(set(letters))) or "-"


def focus_terms_for_question(question: dict) -> list[str]:
    question_text = question.get("question", "")
    text = "\n".join([question_text, question.get("type", ""), *question.get("options", {}).values()])
    terms = {term for term in FOCUS_TERMS if term in text}
    terms.update(
        re.findall(
            r"第\s*[一二三四五六七八九十百千万0-9]+\s*条|"
            r"\d+(?:\.\d+)?%?|\d{4}\s*年|每\s*10\s*股|\d+\s*亿元|\d+\s*万元",
            text,
        )
    )

    # Quoted terms are often the exact clause keyword. They must survive compact
    # prompt extraction even when the source chunk is a very long PDF page.
    for quoted in re.findall(r"[“\"「『]([^”\"」』]{2,32})[”\"」』]", question_text):
        terms.add(quoted.strip())
    for part in re.split(r"[，。；;、（）()\s“”\"「」『』：:？?]+", question_text):
        part = re.sub(r"^(?:关于|根据|结合|针对|请问)", "", part.strip())
        if 2 <= len(part) <= 16:
            terms.add(part)

    for option_text in question.get("options", {}).values():
        for part in re.split(r"[，。；;、（）()\s]+", option_text):
            part = part.strip()
            if not part or re.fullmatch(r"[ABCD]", part):
                continue
            is_named_entity = bool(re.search(r"公司|集团|证券|银行|保险|股份|有限公司|管理人|发行人", part))
            if 2 <= len(part) <= 14 or (is_named_entity and len(part) <= 36):
                terms.add(part)
    return sorted(terms, key=len, reverse=True)[:100]


def segment_score(segment: str, terms: list[str]) -> int:
    score = 0
    squashed = segment.replace(" ", "")
    for term in terms:
        clean = term.replace(" ", "")
        if len(clean) < 2 and not re.search(r"\d", clean):
            continue
        if clean and clean in squashed:
            score += 4 if re.search(r"\d", clean) else 2
    if "|" in segment:
        score += 2
    if "[TABLE" in segment:
        score += 2
    return score


def focus_positions(segment: str, terms: list[str]) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    seen: set[tuple[int, str]] = set()
    normalized_terms = [re.sub(r"\s+", " ", term).strip() for term in terms]
    for term in sorted(set(normalized_terms), key=len, reverse=True):
        if len(term) < 2 and not re.search(r"\d", term):
            continue
        weight = 5 if re.search(r"\d", term) else min(4, max(2, len(term) // 4))
        for match in re.finditer(re.escape(term), segment):
            key = (match.start(), term)
            if key not in seen:
                seen.add(key)
                positions.append((match.start(), weight))
    return positions


def focused_windows(segment: str, terms: list[str], max_chars: int = 480, limit: int = 2) -> list[tuple[int, int, str]]:
    base_score = segment_score(segment, terms)
    if len(segment) <= max_chars:
        return [(base_score, 0, segment)] if base_score else []
    positions = focus_positions(segment, terms)
    if not positions:
        return []

    candidates: list[tuple[int, int, int, str]] = []
    window_len = max(160, max_chars - 6)
    for pos, _ in positions:
        start = max(0, min(len(segment) - window_len, pos - window_len // 3))
        end = min(len(segment), start + window_len)
        window = segment[start:end]
        score = sum(weight for p, weight in positions if start <= p < end)
        if "|" in window:
            score += 2
        if "[TABLE" in window:
            score += 2
        candidates.append((score, start, end, window))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected: list[tuple[int, int, int, str]] = []
    for candidate in candidates:
        _, start, end, _ = candidate
        too_close = False
        for _, old_start, old_end, _ in selected:
            overlap = max(0, min(end, old_end) - max(start, old_start))
            if overlap / max(1, end - start) > 0.55:
                too_close = True
                break
        if not too_close:
            selected.append(candidate)
        if len(selected) >= limit:
            break

    result: list[tuple[int, int, str]] = []
    for score, start, end, window in sorted(selected, key=lambda item: item[1]):
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(segment) else ""
        result.append((score, start, f"{prefix}{window.strip()}{suffix}"))
    return result


def compact_text(text: str, terms: list[str], max_chars: int = 900) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # PDF extraction inserts hard line wraps inside a sentence. Split only on
    # blank lines or sentence punctuation so a matched clause keeps its tail.
    raw_segments = re.split(r"\n\s*\n|(?<=[。；;])", text)
    scored: list[tuple[int, int, str]] = []
    for idx, segment in enumerate(raw_segments):
        segment = re.sub(r"\s+", " ", segment).strip()
        if not segment:
            continue
        score = segment_score(segment, terms)
        if not score:
            continue
        if idx + 1 < len(raw_segments):
            continuation = re.sub(r"\s+", " ", raw_segments[idx + 1]).strip()
            if continuation and len(continuation) <= 280 and len(segment) + len(continuation) <= 760:
                segment = f"{segment} {continuation}"
        if len(segment) > 520:
            windows = focused_windows(segment, terms, max_chars=480, limit=2)
            if windows:
                for win_order, (win_score, _start, window) in enumerate(windows):
                    scored.append((max(score, win_score), idx * 10 + win_order, window))
                continue
            segment = segment[:480] + "..."
        scored.append((score, idx * 10, segment))
    if not scored:
        compacted = re.sub(r"\s+", " ", text).strip()
        windows = focused_windows(compacted, terms, max_chars=max_chars, limit=1)
        if windows:
            return windows[0][2]
        return compacted[:max_chars].strip() + "..."
    picked = sorted(sorted(scored, key=lambda item: item[0], reverse=True)[:6], key=lambda item: item[1])
    result = " / ".join(segment for _, _, segment in picked)
    if len(result) > max_chars:
        result = result[:max_chars] + "..."
    return result

def format_evidence(evidence: list[dict], question: dict | None = None, mode: str = "compact") -> str:
    terms = focus_terms_for_question(question or {}) if question else []
    domain = (question or {}).get("domain") if question else ""
    blocks: list[str] = []
    if mode == "nano":
        prompt_evidence = evidence[:4]
    elif mode == "micro":
        prompt_evidence = evidence[:6]
    elif mode == "minimal":
        prompt_evidence = evidence[:10]
    else:
        prompt_evidence = evidence
    for idx, item in enumerate(prompt_evidence, start=1):
        page = item.get("page")
        location = f"page={page}" if page else "page=N/A"
        sources = ",".join(item.get("sources", []))
        text = item.get("text", "")
        if mode in {"compact", "minimal", "micro", "nano"}:
            max_chars_by_mode = {"nano": 180, "micro": 260, "minimal": 420, "compact": 900}
            compact_max_chars = max_chars_by_mode[mode]
            if mode == "compact" and domain in {"financial_reports", "research"}:
                compact_max_chars = 720
            elif mode == "compact" and domain == "financial_contracts":
                compact_max_chars = 800
            text = compact_text(text, terms, max_chars=compact_max_chars)
        if mode in {"micro", "nano"}:
            blocks.append(
                f"[{idx}] doc={item['doc_id']} chunk={item['chunk_id']} {location} "
                f"opt={option_tags(item)} src={sources}\n{text}"
            )
        else:
            blocks.append(
                f"[{idx}] doc_id={item['doc_id']} chunk_id={item['chunk_id']} {location} "
                f"score={item.get('score')} options={option_tags(item)} sources={sources}\n{text}"
            )
    return "\n\n".join(blocks)



def evidence_index_hint(evidence: list[dict]) -> str:
    if not evidence:
        return "无证据"
    doc_map: dict[str, list[int]] = {}
    option_map: dict[str, list[int]] = {letter: [] for letter in VALID_LETTERS}
    for idx, item in enumerate(evidence, start=1):
        doc_id = str(item.get("doc_id", ""))
        if doc_id:
            doc_map.setdefault(doc_id, []).append(idx)
        for source in item.get("sources", []):
            if source.startswith("option_") and len(source) == len("option_A"):
                letter = source[-1]
                if letter in option_map:
                    option_map[letter].append(idx)
    doc_part = "；".join(f"{doc}:[{','.join(map(str, ids[:8]))}]" for doc, ids in doc_map.items())
    option_part = "；".join(
        f"{letter}:[{','.join(map(str, ids[:8])) or '-'}]" for letter, ids in option_map.items()
    )
    return f"文档覆盖 {doc_part or '-'}\n选项命中 {option_part}"

def build_low_token_messages(
    question: dict,
    evidence: list[dict],
    reference_answer: str = "",
) -> list[dict[str, str]]:
    option_letters = "/".join(option_letters_for_question(question))
    system = "只按给定原文核验金融题。禁止常识补全。只输出紧凑JSON。"
    user = f"""
ID:{question['qid']} 类型:{question.get('answer_format')}
题干:{question['question']}
选项:
{format_options(question.get('options', {}))}
已有答案:{reference_answer or '-'}（仅作待核验基线，不得盲从）
证据:
{format_evidence(evidence, question, 'minimal')}

逐项核验 {option_letters}。原文直接支持=entailed；原文明示相反或对象/指标/单位/年份/条件不符=contradicted；没找到充分原文=unknown。
只输出JSON:{{"answer":"按字母排序","option_judgement":{{"A":{{"judgement":"true/false/uncertain","relation":"entailed/contradicted/unknown","error_type":"none/missing_evidence/entity_mismatch/metric_mismatch/unit_mismatch/time_mismatch/condition_mismatch/scope_mismatch/negation_mismatch","supporting_evidence_ids":[1]}}}}}}
answer必须等于relation=entailed的选项；判断题A=正确、B=错误；不要输出解释或额外字段。
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def build_micro_messages(
    question: dict,
    evidence: list[dict],
    reference_answer: str = "",
) -> list[dict[str, str]]:
    option_letters = "/".join(option_letters_for_question(question))
    system = "Only use the supplied evidence. Return compact JSON only."
    user = f"""
QID:{question['qid']} FORMAT:{question.get('answer_format')}
Q:{question['question']}
OPTIONS:
{format_options(question.get('options', {}))}
BASE:{reference_answer or '-'} (for comparison only)
EVIDENCE:
{format_evidence(evidence, question, 'micro')}

Decide each option {option_letters}. If evidence is insufficient, use uncertain.
For tf: A=true, B=false.
Return exactly:
{{"answer":"LETTERS","knowledge_point":"one short finance/legal point","error_summary":"main trap or none","option_judgement":{{"A":{{"judgement":"true/false/uncertain","relation":"entailed/contradicted/unknown","supporting_evidence_ids":[1]}}}}}}
answer must equal the true options, sorted. For mcq/tf, answer is one letter. No explanation.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def build_nano_messages(
    question: dict,
    evidence: list[dict],
    reference_answer: str = "",
) -> list[dict[str, str]]:
    system = "Use only the evidence. Return one minified JSON object."
    user = f"""
{question.get('answer_format')}|{question['question']}
{format_options(question.get('options', {}))}
E:
{format_evidence(evidence, question, 'nano')}
Return {{"answer":"LETTERS"}} only. Sort multi-select letters. For tf, A=true and B=false.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def build_messages(
    question: dict,
    evidence: list[dict],
    evidence_mode: str = "compact",
    reference_answer: str = "",
) -> list[dict[str, str]]:
    if evidence_mode == "nano":
        return build_nano_messages(question, evidence, reference_answer)
    if evidence_mode == "micro":
        return build_micro_messages(question, evidence, reference_answer)
    if evidence_mode == "minimal":
        return build_low_token_messages(question, evidence, reference_answer)
    profile = get_profile(question.get("domain"))
    option_letters = "/".join(option_letters_for_question(question))
    system = "你是金融长文档问答专家。只依据证据判断选项，不要用外部常识。先逐项判定，再给最终答案。只输出 JSON。"
    user = f"""
题目ID: {question['qid']}
领域: {question.get('domain')}；题型: {question.get('answer_format')}（mcq单选，multi多选，tf判断）。
能力标签: {question.get('type', '')}

题干:
{question['question']}

选项:
{format_options(question.get('options', {}))}

证据摘录（options 表示该证据由哪些选项检索命中，- 表示全局/文档覆盖命中）:
{format_evidence(evidence, question, evidence_mode)}

证据索引辅助（只用于定位，最终仍以原文为准）:
{evidence_index_hint(evidence)}

领域核对清单:
{profile.prompt_checklist}

输出 JSON，字段:
- answer: 大写字母；multi 按字母升序无分隔符，例如 AC；mcq/tf 只能一个字母。
- knowledge_point: 本题考查的一个核心金融/监管/保险/财报/研报知识点，限 40 字。
- error_summary: 本题最容易错读的点，例如对象混淆、年份单位不符、条款例外、跨文档比较遗漏；没有则写 none。
- option_judgement: {option_letters} 对象，每项含 judgement(true/false/uncertain)、relation(entailed/contradicted/unknown)、error_type(none/missing_evidence/entity_mismatch/metric_mismatch/unit_mismatch/time_mismatch/condition_mismatch/scope_mismatch/negation_mismatch)、supporting_evidence_ids、reasoning。
- evidence_retrieval: 数组，每项含 doc_id、chunk_id、quoted_clause、reasoning。
规则: 证据不足不要选；多选题允许只有一个正确项，不要因为题型是 multi 就凑多个选项；最终 answer 必须等于 option_judgement 中 judgement=true 的选项集合，不得选择 false/uncertain 项；supporting_evidence_ids 必须引用上方证据编号；多文档比较题必须分别核对相关 doc_id；涉及数字先核对年份、单位、同比方向。
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def build_review_messages(
    question: dict,
    evidence: list[dict],
    first_output: str,
    evidence_mode: str = "compact",
    risk_reasons: list[str] | None = None,
) -> list[dict[str, str]]:
    profile = get_profile(question.get("domain"))
    option_letters = "/".join(option_letters_for_question(question))
    system = "你是严格的金融长文档答案审稿人。只依据证据独立复判，优先发现错选、漏选、年份/单位/否定词/并列条件误读。只输出 JSON。"
    risk_text = "；".join(risk_reasons or []) or "常规复核"
    user = f"""
题目ID: {question['qid']}
领域: {question.get('domain')}；题型: {question.get('answer_format')}
题干: {question['question']}
选项:
{format_options(question.get('options', {}))}

证据摘录:
{format_evidence(evidence, question, evidence_mode)}

证据索引辅助:
{evidence_index_hint(evidence)}

初判输出（只能作为待审草稿，不要默认相信）:
{first_output}

触发复核原因:
{risk_text}

领域核对清单:
{profile.prompt_checklist}

请独立重做判断：
1. 逐项寻找支持证据和反证；年份/对象/单位不一致或有明确反证时判 false；仅仅没有召回相关条款时必须判 uncertain，不能把缺证据当成错误。
2. 题干或选项含“且、均、都、同时、连续、两份、两年、分别”等复合条件时，所有条件都被证据支持才可判 true。
3. 多选题允许只有一个正确项；不要为了凑答案而多选；证据不足判 uncertain，只有明确反证才判 false。
4. 判断题 A=正确、B=错误；只有整句话所有条件都正确才选 A。
5. 题目只有这些选项：{option_letters}；最终 answer 必须等于 judgement=true 的选项集合。

每个 option_judgement 还必须给出 relation(entailed/contradicted/unknown) 和 error_type：
- 直接支持为 entailed；原文明确相反或条件/对象/指标/单位/年份/范围不一致为 contradicted；没有召回充分证据为 unknown。
- error_type 只能是 none、missing_evidence、entity_mismatch、metric_mismatch、unit_mismatch、time_mismatch、condition_mismatch、scope_mismatch、negation_mismatch。
- unknown 必须配 missing_evidence；不得把 unknown 写成 false。

只输出 JSON，字段：answer、changed(boolean)、knowledge_point、error_summary、option_judgement、evidence_retrieval、review_reason。
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def load_processed(processed_dir: Path) -> tuple[list[dict], list[dict]]:
    questions = read_jsonl(processed_dir / "questions.jsonl")
    chunks = read_jsonl(processed_dir / "chunks.jsonl")
    if not questions:
        raise RuntimeError(f"No questions found in {processed_dir / 'questions.jsonl'}")
    if not chunks:
        raise RuntimeError(f"No chunks found in {processed_dir / 'chunks.jsonl'}; run script/run_preprocess.py first.")
    return questions, chunks


def usage_or_estimate(messages: list[dict[str, str]], content: str, usage: dict[str, int]) -> dict[str, int]:
    if usage.get("total_tokens"):
        return usage
    prompt_text = "\n".join(message["content"] for message in messages)
    prompt_tokens = approx_token_count(prompt_text)
    completion_tokens = approx_token_count(content)
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}


def totals_from_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in rows:
        for key in totals:
            totals[key] += int(row.get(key, 0) or 0)
    return totals


def write_answer_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANSWER_FIELDS)
        writer.writeheader()
        writer.writerow({"qid": "summary", "answer": "", **totals_from_rows(rows)})
        writer.writerows(rows)
    tmp.replace(path)


def read_answer_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row.get("qid") and row.get("qid") != "summary"]


def read_evidence_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_evidence_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def write_evidence_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def ordered_values_by_questions(mapping: dict[str, Any], questions: list[dict]) -> list[Any]:
    return [mapping[q["qid"]] for q in questions if q["qid"] in mapping]


def normalize_judgement_text(value: Any) -> str:
    text = str(value).strip().lower()
    if not text:
        return ""
    compact = re.sub(r"[\s:：，。；;、,.]+", "", text)
    if compact in FALSE_JUDGEMENT_VALUES:
        return "false"
    if compact in UNCERTAIN_JUDGEMENT_VALUES:
        return "uncertain"
    if compact in TRUE_JUDGEMENT_VALUES:
        return "true"
    if any(marker in compact for marker in ("不支持", "不成立", "不符合", "不准确", "错误", "证据不足", "无法判断", "无法确认")):
        return "uncertain" if any(marker in compact for marker in ("证据不足", "无法判断", "无法确认")) else "false"
    if any(marker in compact for marker in ("支持", "成立", "符合", "准确", "正确")):
        return "true"
    return compact


def true_letters_from_option_judgement(parsed: dict[str, Any]) -> list[str]:
    judgements = parsed.get("option_judgement") or {}
    true_letters: list[str] = []
    if isinstance(judgements, dict):
        for letter in VALID_LETTERS:
            item = judgements.get(letter) or judgements.get(letter.lower()) or {}
            judgement = normalize_judgement_text(item.get("judgement", "")) if isinstance(item, dict) else normalize_judgement_text(item)
            if judgement == "true":
                true_letters.append(letter)
    return true_letters


def answer_from_option_judgement(parsed: dict[str, Any], answer_format: str) -> str:
    true_letters = true_letters_from_option_judgement(parsed)
    if answer_format == "multi":
        return "".join(true_letters)
    if answer_format in {"mcq", "tf"} and len(true_letters) == 1:
        return true_letters[0]
    return ""


def question_search_text(question: dict) -> str:
    return "\n".join([question.get("question", ""), question.get("type", ""), *question.get("options", {}).values()])


def judgement_for_letter(parsed: dict[str, Any], letter: str) -> dict[str, Any]:
    judgements = parsed.get("option_judgement")
    if not isinstance(judgements, dict):
        return {}
    item = judgements.get(letter) or judgements.get(letter.lower()) or {}
    return item if isinstance(item, dict) else {"judgement": item}


def judgement_value(item: dict[str, Any]) -> str:
    return normalize_judgement_text(item.get("judgement", ""))


def is_true_judgement(item: dict[str, Any]) -> bool:
    return judgement_value(item) == "true"


def is_false_judgement(item: dict[str, Any]) -> bool:
    return judgement_value(item) == "false"


def judgement_reasoning(item: dict[str, Any]) -> str:
    return str(item.get("reasoning", "") or item.get("reason", ""))


def judgement_relation(item: dict[str, Any]) -> str:
    relation = str(item.get("relation", "")).strip().lower()
    aliases = {
        "entailment": "entailed",
        "support": "entailed",
        "supported": "entailed",
        "contradiction": "contradicted",
        "contradictory": "contradicted",
        "uncertain": "unknown",
    }
    relation = aliases.get(relation, relation)
    if relation in {"entailed", "contradicted", "unknown"}:
        return relation
    value = judgement_value(item)
    return {"true": "entailed", "false": "contradicted", "uncertain": "unknown"}.get(value, "unknown")


def judgement_error_type(item: dict[str, Any]) -> str:
    return str(item.get("error_type", "")).strip().lower()


def judgement_relies_on_missing_evidence(item: dict[str, Any]) -> bool:
    reasoning = judgement_reasoning(item)
    return (
        judgement_relation(item) == "unknown"
        or judgement_error_type(item) in UNKNOWN_ERROR_TYPES
        or any(marker in reasoning for marker in MISSING_EVIDENCE_MARKERS)
    )


def judgement_is_inferential(item: dict[str, Any]) -> bool:
    reasoning = judgement_reasoning(item)
    return any(marker in reasoning for marker in INFERENTIAL_REASONING_MARKERS)


def judgement_has_measure_mismatch(item: dict[str, Any], option_text: str) -> bool:
    reasoning = judgement_reasoning(item)
    if "每股" in option_text and "总额" in reasoning:
        return True
    if "比例" in option_text and "总额" in reasoning and "比例" not in reasoning:
        return True
    return False


def parse_evidence_ids(value: Any) -> set[int]:
    ids: set[int] = set()
    if isinstance(value, int):
        ids.add(value)
    elif isinstance(value, str):
        ids.update(int(match) for match in re.findall(r"\d+", value))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            ids.update(parse_evidence_ids(item))
    return ids


def evidence_ids_for_letter(parsed: dict[str, Any], letter: str) -> set[int]:
    item = judgement_for_letter(parsed, letter)
    return parse_evidence_ids(item.get("supporting_evidence_ids"))


def valid_evidence_ids_for_letter(parsed: dict[str, Any], letter: str, evidence: list[dict]) -> set[int]:
    return {
        evidence_id
        for evidence_id in evidence_ids_for_letter(parsed, letter)
        if 1 <= evidence_id <= len(evidence)
    }


def cited_evidence_text(parsed: dict[str, Any], letter: str, evidence: list[dict]) -> str:
    texts: list[str] = []
    for evidence_id in sorted(valid_evidence_ids_for_letter(parsed, letter, evidence)):
        texts.append(str(evidence[evidence_id - 1].get("text", "")))
    return "\n".join(texts)


def extract_claim_numbers(text: str) -> list[str]:
    """Extract claim values without treating document identifiers as numbers."""
    return re.findall(
        r"(?<![A-Za-z0-9_])\d+(?:,\d{3})*(?:\.\d+)?%?(?![A-Za-z0-9_])",
        text,
    )


def cited_evidence_matches_option_document(question: dict, letter: str, parsed: dict[str, Any], evidence: list[dict]) -> bool:
    option_text = str(question.get("options", {}).get(letter, ""))
    references = {int(value) for value in re.findall(r"[A-Za-z]+_text_?0*(\d+)", option_text)}
    if not references:
        return True
    cited_doc_ids = {
        str(evidence[evidence_id - 1].get("doc_id", ""))
        for evidence_id in valid_evidence_ids_for_letter(parsed, letter, evidence)
    }
    cited_suffixes = {
        int(match.group(1))
        for doc_id in cited_doc_ids
        if (match := re.search(r"0*(\d+)$", doc_id))
    }
    return references.issubset(cited_suffixes)


def option_semantics_are_supported(question: dict, letter: str, parsed: dict[str, Any], evidence: list[dict]) -> bool:
    option_text = str(question.get("options", {}).get(letter, ""))
    cited_text = cited_evidence_text(parsed, letter, evidence)
    reasoning = judgement_reasoning(judgement_for_letter(parsed, letter))

    if not cited_evidence_matches_option_document(question, letter, parsed, evidence):
        return False
    if "违约利息" in option_text and "违约利息" not in cited_text:
        return False
    if "每股" in option_text and not any(term in cited_text for term in ("每股", "每10股", "每 10 股")):
        return False
    if "主体信用评级" in option_text and not any(term in cited_text for term in ("主体信用评级", "主体信用等级")):
        return False
    if "债项信用评级" in option_text and not any(term in cited_text for term in ("债项信用评级", "债项信用等级", "债项评级")):
        return False
    if "研发投入占营业收入" in option_text and "研发费用占营业收入" in reasoning and "研发投入占营业收入" not in reasoning:
        return False
    return True


def option_numbers_are_supported(question: dict, letter: str, parsed: dict[str, Any], evidence: list[dict]) -> bool:
    option_text = str(question.get("options", {}).get(letter, ""))
    claim_text = (
        f"{question.get('question', '')} {option_text}"
        if question.get("answer_format") == "tf"
        else option_text
    )
    numbers = extract_claim_numbers(claim_text)
    if not numbers:
        return True
    cited_text = re.sub(r"\s+", "", cited_evidence_text(parsed, letter, evidence))
    return all(re.sub(r"\s+", "", number) in cited_text for number in numbers)


def merge_answer_with_evidence_gate(
    reference_answer: str,
    parsed: dict[str, Any],
    question: dict,
    evidence: list[dict],
) -> str:
    """Apply review changes only when they are backed by direct evidence.

    Missing retrieval evidence is uncertainty, not contradiction. This keeps a
    review pass from deleting a base option merely because the compact context
    omitted its clause.
    """
    answer_format = str(question.get("answer_format", "multi"))
    reference = normalize_answer(reference_answer, answer_format)
    if not isinstance(parsed.get("option_judgement"), dict):
        return reference

    letters = option_letters_for_question(question)
    if answer_format == "multi":
        selected = set(reference)
        for letter in letters:
            item = judgement_for_letter(parsed, letter)
            has_evidence = bool(valid_evidence_ids_for_letter(parsed, letter, evidence))
            if is_true_judgement(item) and judgement_relation(item) == "entailed":
                option_text = " ".join(
                    [str(question.get("question", "")), str(question.get("options", {}).get(letter, ""))]
                )
                if (
                    has_evidence
                    and not judgement_is_inferential(item)
                    and not judgement_has_measure_mismatch(item, option_text)
                    and option_semantics_are_supported(question, letter, parsed, evidence)
                    and option_numbers_are_supported(question, letter, parsed, evidence)
                ):
                    selected.add(letter)
            elif is_false_judgement(item) and judgement_relation(item) == "contradicted":
                if (
                    has_evidence
                    and not judgement_relies_on_missing_evidence(item)
                    and cited_evidence_matches_option_document(question, letter, parsed, evidence)
                ):
                    selected.discard(letter)
        return "".join(letter for letter in letters if letter in selected)

    supported_true = [
        letter
        for letter in letters
        if is_true_judgement(judgement_for_letter(parsed, letter))
        and judgement_relation(judgement_for_letter(parsed, letter)) == "entailed"
        and bool(valid_evidence_ids_for_letter(parsed, letter, evidence))
        and not judgement_is_inferential(judgement_for_letter(parsed, letter))
        and not judgement_has_measure_mismatch(
            judgement_for_letter(parsed, letter),
            " ".join(
                [str(question.get("question", "")), str(question.get("options", {}).get(letter, ""))]
            ),
        )
        and option_semantics_are_supported(question, letter, parsed, evidence)
        and option_numbers_are_supported(question, letter, parsed, evidence)
    ]
    if len(supported_true) != 1:
        return reference
    candidate = supported_true[0]
    if not reference or candidate == reference:
        return candidate
    current_item = judgement_for_letter(parsed, reference[0])
    current_is_directly_false = (
        is_false_judgement(current_item)
        and judgement_relation(current_item) == "contradicted"
        and bool(valid_evidence_ids_for_letter(parsed, reference[0], evidence))
        and not judgement_relies_on_missing_evidence(current_item)
    )
    return candidate if current_is_directly_false else reference


def selected_supporting_ids(parsed: dict[str, Any], answer: str) -> set[int]:
    ids: set[int] = set()
    for letter in answer:
        ids.update(evidence_ids_for_letter(parsed, letter))
    return ids


def doc_ids_from_evidence_ids(evidence: list[dict], evidence_ids: set[int]) -> set[str]:
    doc_ids: set[str] = set()
    for evidence_id in evidence_ids:
        idx = evidence_id - 1
        if 0 <= idx < len(evidence):
            doc_id = evidence[idx].get("doc_id")
            if doc_id:
                doc_ids.add(str(doc_id))
    return doc_ids


def parsed_model_doc_ids(parsed: dict[str, Any]) -> set[str]:
    doc_ids: set[str] = set()
    rows = parsed.get("evidence_retrieval")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("doc_id"):
                doc_ids.add(str(row["doc_id"]))
    return doc_ids


def output_text_for_review(parsed: dict[str, Any]) -> str:
    if not parsed:
        return ""
    return json.dumps(parsed, ensure_ascii=False)


def answer_disagrees_with_judgement(parsed: dict[str, Any], answer: str, answer_format: str) -> bool:
    if not answer or not isinstance(parsed.get("option_judgement"), dict):
        return False
    true_letters = {letter for letter in VALID_LETTERS if is_true_judgement(judgement_for_letter(parsed, letter))}
    answer_letters = set(answer)
    if answer_format == "multi":
        return bool(true_letters) and true_letters != answer_letters
    if answer_format in {"mcq", "tf"}:
        selected = judgement_for_letter(parsed, answer[0])
        return (true_letters and answer[0] not in true_letters) or judgement_value(selected) in {"false", "no", "错误", "不支持", "unsupported", "uncertain"}
    return False


def review_risk_reasons(
    question: dict,
    parsed: dict[str, Any],
    answer: str,
    evidence: list[dict],
    fallback_used: bool = False,
    broad_multi_review: bool = False,
) -> list[str]:
    answer_format = question.get("answer_format", "multi")
    reasons: list[str] = []
    if not parsed:
        reasons.append("json_missing")
    if not answer:
        reasons.append("answer_missing")
    if answer_format == "multi" and answer and not re.fullmatch(r"[ABCD]+", answer):
        reasons.append("invalid_multi_answer")
    if answer_format == "mcq" and answer and not re.fullmatch(r"[ABCD]", answer):
        reasons.append("invalid_mcq_answer")
    if answer_format == "tf" and answer and not re.fullmatch(r"[AB]", answer):
        reasons.append("invalid_tf_answer")

    true_letters: list[str] = []
    judgements = parsed.get("option_judgement")
    if not isinstance(judgements, dict):
        reasons.append("option_judgement_missing")
    else:
        true_letters = true_letters_from_option_judgement(parsed)
        uncertain = sum(
            1
            for item in judgements.values()
            if isinstance(item, dict) and judgement_value(item) == "uncertain"
        )
        if uncertain >= 2:
            reasons.append("too_many_uncertain_options")
        if answer_format in {"mcq", "tf"} and len(true_letters) > 1:
            reasons.append("single_choice_has_multiple_true_judgements")

    if fallback_used:
        reasons.append("answer_derived_from_option_judgement")
    if answer_disagrees_with_judgement(parsed, answer, answer_format):
        reasons.append("answer_conflicts_with_option_judgement")

    support_ids = selected_supporting_ids(parsed, answer) if answer else set()
    if answer and not support_ids:
        reasons.append("selected_options_without_supporting_evidence_ids")
    invalid_ids = sorted(eid for eid in support_ids if eid < 1 or eid > len(evidence))
    if invalid_ids:
        reasons.append(f"supporting_evidence_ids_out_of_range:{invalid_ids[:5]}")

    options_count = len(question.get("options", {})) or len(VALID_LETTERS)
    selected_count = len(set(answer)) if answer else 0
    if answer_format == "multi":
        if broad_multi_review:
            reasons.append(BROAD_MULTI_REVIEW_REASON)
        if selected_count >= 3:
            reasons.append("multi_answer_selects_three_or_more_options")
        if answer and selected_count >= min(options_count, 4):
            reasons.append("multi_answer_selects_all_options")

    qtext = question_search_text(question)
    compound_markers = ("且", "并且", "同时", "均", "都", "两份", "两年", "连续", "分别", "以及")
    if answer_format == "tf" and (len(question.get("doc_ids") or []) > 1 or any(marker in qtext for marker in compound_markers)):
        reasons.append("tf_compound_or_cross_doc_statement")
    parsed_text = output_text_for_review(parsed)
    numeric_like = question.get("domain") == "financial_reports" or any(hint in qtext for hint in NUMERIC_REVIEW_HINTS)
    if numeric_like and re.search(r"\d|同比|增长|下降|高于|低于", qtext) and not re.search(r"\d", parsed_text):
        reasons.append("numeric_or_comparison_question_without_numbers_in_reasoning")

    if question.get("domain") == "regulatory":
        has_clause = re.search(r"第\s*[一二三四五六七八九十百千万0-9]+\s*条", parsed_text)
        if not has_clause and not support_ids:
            reasons.append("regulatory_answer_without_clause_or_evidence_ids")

    candidate_doc_ids = [str(doc_id) for doc_id in question.get("doc_ids") or []]
    if len(candidate_doc_ids) > 1 and any(hint in qtext for hint in CROSS_DOC_HINTS):
        cited_doc_ids = parsed_model_doc_ids(parsed) | doc_ids_from_evidence_ids(evidence, support_ids)
        if len(cited_doc_ids) <= 1:
            reasons.append("cross_document_question_cites_one_or_zero_docs")

    return sorted(set(reasons))


def needs_review(parsed: dict[str, Any], answer: str, answer_format: str) -> bool:
    if not parsed or not answer:
        return True
    if answer_format == "multi" and not re.fullmatch(r"[ABCD]+", answer):
        return True
    if answer_format == "mcq" and not re.fullmatch(r"[ABCD]", answer):
        return True
    if answer_format == "tf" and not re.fullmatch(r"[AB]", answer):
        return True
    judgements = parsed.get("option_judgement")
    if not isinstance(judgements, dict):
        return True
    uncertain = 0
    for item in judgements.values():
        if isinstance(item, dict) and str(item.get("judgement", "")).lower() == "uncertain":
            uncertain += 1
    return uncertain >= 2


def save_checkpoints(
    checkpoint_csv: Path | None,
    checkpoint_evidence_jsonl: Path | None,
    row_by_qid: dict[str, dict[str, Any]],
    evidence_by_qid: dict[str, dict[str, Any]],
    questions: list[dict],
) -> None:
    ordered_rows = ordered_values_by_questions(row_by_qid, questions)
    ordered_evidence = ordered_values_by_questions(evidence_by_qid, questions)
    if checkpoint_csv:
        write_answer_rows(checkpoint_csv, ordered_rows)
    if checkpoint_evidence_jsonl:
        write_evidence_jsonl(checkpoint_evidence_jsonl, ordered_evidence)


def solve_questions(
    processed_dir: Path,
    answer_csv: Path,
    evidence_json: Path,
    client: QwenClient,
    limit: int | None = None,
    max_context_chars: int | None = None,
    review: bool = False,
    review_mode: str = "off",
    review_policy: str = "replace",
    initial_policy: str = "replace",
    evidence_mode: str = "compact",
    checkpoint_csv: Path | None = None,
    checkpoint_evidence_jsonl: Path | None = None,
    cache_dir: Path | None = None,
    stop_file: Path | None = None,
    resume: bool = False,
    qid_filter: set[str] | None = None,
    base_answer_csv: Path | None = None,
    base_evidence_json: Path | None = None,
) -> dict[str, int]:
    questions, chunks = load_processed(processed_dir)
    if limit is not None:
        questions = questions[:limit]
    qids = {q["qid"] for q in questions}
    target_qids = {qid for qid in (qid_filter or qids) if qid in qids}
    unknown_qids = sorted((qid_filter or set()) - qids)
    if unknown_qids:
        print(f"Warning: ignored {len(unknown_qids)} qids not found in processed questions: {unknown_qids[:8]}")
    if qid_filter and not target_qids:
        raise RuntimeError("qid_filter did not match any processed questions")

    cache_run_id = prepare_question_cache(cache_dir, resume)
    row_by_qid: dict[str, dict[str, Any]] = {}
    evidence_by_qid: dict[str, dict[str, Any]] = {}
    base_row_by_qid: dict[str, dict[str, Any]] = {}
    if base_answer_csv:
        for row in read_answer_rows(base_answer_csv):
            qid = row.get("qid")
            if qid in qids:
                base_row_by_qid[qid] = row
                if qid not in target_qids:
                    row_by_qid[qid] = row
    if base_evidence_json:
        for row in read_evidence_json(base_evidence_json):
            qid = row.get("qid")
            if qid in qids and qid not in target_qids:
                evidence_by_qid[qid] = row
    if qid_filter:
        print(
            f"Refine mode: rerun {len(target_qids)} qids, keep {len(row_by_qid)} base rows. "
            f"Final output remains ordered by all {len(questions)} questions."
        )

    if resume:
        if checkpoint_csv:
            for row in read_answer_rows(checkpoint_csv):
                if row.get("qid") in qids:
                    row_by_qid[row["qid"]] = row
        if checkpoint_evidence_jsonl:
            for row in read_evidence_jsonl(checkpoint_evidence_jsonl):
                if row.get("qid") in qids:
                    evidence_by_qid[row["qid"]] = row
        cache_rows, cache_evidence_rows, bad_cache_files = read_question_cache(cache_dir, cache_run_id)
        for qid, row in cache_rows.items():
            if qid in qids:
                row_by_qid[qid] = row
        for qid, row in cache_evidence_rows.items():
            if qid in qids:
                evidence_by_qid[qid] = row
        if bad_cache_files:
            print(f"Warning: ignored {len(bad_cache_files)} broken cache files")
        print(
            f"Resume loaded {len(row_by_qid)}/{len(questions)} completed questions "
            f"from checkpoint/cache"
        )
        save_checkpoints(checkpoint_csv, checkpoint_evidence_jsonl, row_by_qid, evidence_by_qid, questions)
    else:
        save_checkpoints(checkpoint_csv, checkpoint_evidence_jsonl, row_by_qid, evidence_by_qid, questions)

    index = LexicalIndex(chunks)
    if review:
        review_mode = "always"

    try:
        for idx, question in enumerate(questions, start=1):
            qid = question["qid"]
            if stop_file and stop_file.exists():
                save_checkpoints(checkpoint_csv, checkpoint_evidence_jsonl, row_by_qid, evidence_by_qid, questions)
                raise RunStopped(len(row_by_qid), len(questions), stop_file)
            if qid not in target_qids:
                if qid in row_by_qid:
                    print(f"[{idx}/{len(questions)}] {qid} kept (base)")
                else:
                    print(f"[{idx}/{len(questions)}] {qid} skipped (outside qid filter)")
                continue
            if qid in row_by_qid:
                print(f"[{idx}/{len(questions)}] {qid} skipped (checkpoint/cache)")
                continue

            evidence = gather_evidence(index, question, max_chars=max_context_chars)
            base_answer = normalize_answer_for_question(
                str(base_row_by_qid.get(qid, {}).get("answer", "")),
                question,
            )
            messages = build_messages(question, evidence, evidence_mode, base_answer)
            result = client.chat(messages)
            parsed = extract_json_object(result.content)
            answer_format = question.get("answer_format", "multi")
            answer = normalize_answer_for_question(str(parsed.get("answer", "")), question)
            fallback_used = False
            answer_aligned_to_judgement = False
            judgement_answer = normalize_answer_for_question(answer_from_option_judgement(parsed, answer_format), question)
            if not answer and judgement_answer:
                answer = judgement_answer
                fallback_used = True
            elif judgement_answer and answer_disagrees_with_judgement(parsed, answer, answer_format):
                answer = judgement_answer
                answer_aligned_to_judgement = True

            if initial_policy == "preserve" and base_answer:
                answer = base_answer
            elif initial_policy == "evidence_gate" and base_answer:
                answer = merge_answer_with_evidence_gate(
                    base_answer,
                    parsed,
                    question,
                    evidence,
                ) or base_answer

            usage = usage_or_estimate(messages, result.content, result.usage)
            row_usage = dict(usage)

            review_output = None
            review_usage = None
            review_parsed: dict[str, Any] = {}
            review_answer_aligned_to_judgement = False
            base_answer_fallback_used = False
            broad_multi_review = review_mode in {"broad", "always"}
            risk_reasons = review_risk_reasons(question, parsed, answer, evidence, fallback_used, broad_multi_review=broad_multi_review)
            do_review = review_mode == "always" or (review_mode in {"auto", "broad"} and bool(risk_reasons))
            if do_review:
                review_messages = build_review_messages(question, evidence, result.content, evidence_mode, risk_reasons)
                review_result = client.chat(review_messages)
                review_parsed = extract_json_object(review_result.content)
                reviewed_answer = normalize_answer_for_question(str(review_parsed.get("answer", "")), question)
                review_judgement_answer = normalize_answer_for_question(answer_from_option_judgement(review_parsed, answer_format), question)
                if not reviewed_answer:
                    reviewed_answer = review_judgement_answer
                elif review_judgement_answer and answer_disagrees_with_judgement(review_parsed, reviewed_answer, answer_format):
                    reviewed_answer = review_judgement_answer
                    review_answer_aligned_to_judgement = True
                if reviewed_answer:
                    if review_policy == "evidence_gate":
                        reference_answer = normalize_answer_for_question(
                            str(base_row_by_qid.get(qid, {}).get("answer", "")),
                            question,
                        ) or answer
                        answer = merge_answer_with_evidence_gate(
                            reference_answer,
                            review_parsed,
                            question,
                            evidence,
                        ) or reference_answer
                    else:
                        answer = reviewed_answer
                review_usage = usage_or_estimate(review_messages, review_result.content, review_result.usage)
                for key in row_usage:
                    row_usage[key] += int(review_usage.get(key, 0))
                review_output = review_result.content

            if not answer:
                base_answer = normalize_answer_for_question(
                    str(base_row_by_qid.get(qid, {}).get("answer", "")),
                    question,
                )
                if base_answer:
                    answer = base_answer
                    base_answer_fallback_used = True
                    risk_reasons.append("empty_answer_fell_back_to_base")
                else:
                    raise RuntimeError(
                        f"No valid answer produced for {qid}, and no valid base answer is available. "
                        "The run stopped before writing an invalid submission."
                    )

            row = {
                "qid": qid,
                "answer": answer,
                "prompt_tokens": row_usage["prompt_tokens"],
                "completion_tokens": row_usage["completion_tokens"],
                "total_tokens": row_usage["total_tokens"],
            }
            prompt_chars = sum(len(message["content"]) for message in messages)
            evidence_row = {
                "qid": qid,
                "answer": answer,
                "retrieved_evidence": evidence,
                "evidence_mode": evidence_mode,
                "prompt_chars": prompt_chars,
                "review_mode": review_mode,
                "review_policy": review_policy,
                "initial_policy": initial_policy,
                "reviewed": do_review,
                "review_risk_reasons": risk_reasons,
                "fallback_used": fallback_used,
                "base_answer_fallback_used": base_answer_fallback_used,
                "answer_aligned_to_judgement": answer_aligned_to_judgement,
                "review_answer_aligned_to_judgement": review_answer_aligned_to_judgement,
                "knowledge_point": review_parsed.get("knowledge_point") or parsed.get("knowledge_point", ""),
                "error_summary": review_parsed.get("error_summary") or parsed.get("error_summary", ""),
                "option_judgement": parsed.get("option_judgement", {}),
                "review_option_judgement": review_parsed.get("option_judgement", {}) if review_parsed else {},
                "final_option_judgement": (review_parsed.get("option_judgement") if review_parsed.get("option_judgement") else parsed.get("option_judgement", {})),
                "model_evidence": parsed.get("evidence_retrieval", []),
                "review_model_evidence": review_parsed.get("evidence_retrieval", []) if review_parsed else [],
                "raw_model_output": result.content,
                "raw_review_output": review_output,
                "usage": usage,
                "review_usage": review_usage,
            }
            row_by_qid[qid] = row
            evidence_by_qid[qid] = evidence_row

            write_question_cache(cache_dir, qid, row, evidence_row, cache_run_id)
            save_checkpoints(checkpoint_csv, checkpoint_evidence_jsonl, row_by_qid, evidence_by_qid, questions)

            review_mark = " reviewed" if do_review else ""
            risk_mark = f" risk={','.join(risk_reasons[:3])}" if risk_reasons and review_mode == "auto" else ""
            print(f"[{idx}/{len(questions)}] {qid} -> {answer} tokens={row_usage['total_tokens']}{review_mark}{risk_mark}")
    except KeyboardInterrupt:
        save_checkpoints(checkpoint_csv, checkpoint_evidence_jsonl, row_by_qid, evidence_by_qid, questions)
        print("\nStopped by Ctrl+C. Completed questions have been saved to checkpoint/cache files.")
        raise

    rows = ordered_values_by_questions(row_by_qid, questions)
    evidence_rows = ordered_values_by_questions(evidence_by_qid, questions)
    write_answer_rows(answer_csv, rows)
    write_json(evidence_json, evidence_rows)
    return {"questions": len(rows), **totals_from_rows(rows)}


def write_retrieval_preview(processed_dir: Path, output_json: Path, limit: int | None = 5) -> dict[str, int]:
    questions, chunks = load_processed(processed_dir)
    if limit is not None:
        questions = questions[:limit]
    index = LexicalIndex(chunks)
    preview = []
    for question in questions:
        evidence = gather_evidence(index, question)
        preview.append({"qid": question["qid"], "doc_ids": question.get("doc_ids", []), "evidence": evidence})
    write_json(output_json, preview)
    return {"questions": len(preview), "chunks": len(chunks)}
