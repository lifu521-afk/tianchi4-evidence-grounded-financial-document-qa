from __future__ import annotations

import json
from typing import Any


PROMPT_VERSION = "b-ensemble-v1"
CHOICE_TYPES = ("单选", "多选", "判断", "single", "multiple", "choice", "true_false")
OPEN_TYPES = ("计算", "抽取", "calculation", "extract", "open")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def question_kind(question: dict[str, Any]) -> str:
    """Return the B-list prompt family used by a question."""
    options = question.get("options")
    if isinstance(options, dict) and options:
        return "choice"

    type_text = str(question.get("type", "")).strip().lower()
    if any(marker in type_text for marker in OPEN_TYPES):
        return "open"
    if any(marker in type_text for marker in CHOICE_TYPES):
        return "choice"
    return "open"


def _answer_slots(question: dict[str, Any], answer_slots: int | None) -> int:
    value = answer_slots
    if value is None:
        value = question.get("answer_slots") or question.get("answer_count") or 1
    try:
        slots = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("answer_slots must be an integer from 1 to 4") from exc
    if slots not in {1, 2, 3, 4}:
        raise ValueError("answer_slots must be from 1 to 4")
    return slots


def format_question(question: dict[str, Any], answer_slots: int | None = None) -> str:
    slots = _answer_slots(question, answer_slots)
    options = question.get("options") or {}
    option_lines = "\n".join(
        f"{str(letter).upper()}. {text}"
        for letter, text in sorted(options.items(), key=lambda item: str(item[0]))
    )
    return (
        f"qid: {question.get('qid', '')}\n"
        f"domain: {question.get('domain', '')}\n"
        f"split: {question.get('split', 'B')}\n"
        f"question_kind: {question_kind(question)}\n"
        f"type: {question.get('type', '')}\n"
        f"answer_slots: {slots}\n"
        f"question: {question.get('question', '')}\n"
        f"options:\n{option_lines or '(empty: calculation/extraction question)'}"
    )


def format_evidence(evidence: list[dict[str, Any]]) -> str:
    """Render evidence with stable IDs and auditable source positions."""
    rendered: list[str] = []
    for index, item in enumerate(evidence, start=1):
        evidence_id = item.get("evidence_id") or f"E{index}"
        location = {
            "doc_id": item.get("doc_id"),
            "source_path": item.get("source_path") or item.get("path"),
            "page": item.get("page"),
            "chunk_id": item.get("chunk_id"),
            "page_char_start": item.get("page_char_start"),
            "page_char_end": item.get("page_char_end"),
            "table_id": item.get("table_id"),
            "row_label": item.get("row_label") or item.get("row"),
            "column_label": item.get("column_label") or item.get("column"),
            "year": item.get("year"),
            "unit": item.get("unit"),
        }
        rendered.append(
            f"[{evidence_id}] location={_json(location)}\n"
            f"{str(item.get('text', '')).strip()}"
        )
    return "\n\n".join(rendered) or "(no evidence supplied)"


def _citation_schema() -> dict[str, Any]:
    return {
        "evidence_id": "E1",
        "quote": "逐字复制的原文，不得改写或拼接",
        "doc_id": "document id",
        "source_path": "relative/original source path",
        "page": 1,
        "chunk_id": "doc#chunk",
        "page_char_start": 0,
        "page_char_end": 20,
        "table_id": "table id or empty string",
        "row_label": "table row label or empty string",
        "column_label": "table column label or empty string",
        "year": "对应年份或期间；不适用为空字符串",
        "unit": "元/万元/% 等；不适用为空字符串",
    }


def _choice_option_schema() -> dict[str, Any]:
    return {
        "atomic_claims": ["拆分后的最小事实条件"],
        "relation": "entailed/contradicted/unknown",
        "judgement": "true/false/uncertain",
        "citations": [_citation_schema()],
        "reasoning": "逐项核对主体、指标、年份、单位、范围、条件和否定词",
        "confidence": 0.0,
    }


def _open_final_schema(slots: int) -> dict[str, Any]:
    return {
        "answers": [f"answer_{index}" for index in range(1, slots + 1)],
        "raw_values": [
            {
                "name": "原始变量名",
                "value": "保持原文精度的值",
                "year": "年份或期间",
                "unit": "原始单位",
                "citation": _citation_schema(),
            }
        ],
        "formula": "完整公式；抽取题写 direct_extraction",
        "calculation_steps": [
            "先统一年份、口径与单位",
            "代入未提前舍入的原始值",
            "按运算顺序计算",
        ],
        "rounding_stage": "仅在最后一步按题目/模板要求舍入；说明位数和规则",
        "format_spec": {
            "answer_slots": slots,
            "slot_rules": [
                "每个答案必须是字符串",
                "百分数必须带 %",
                "日期使用 YYYY年M月D日",
                "排序使用半角 > 且两侧无空格",
                "数值单位类结果只填数字；除题目另有明确要求外保留两位小数",
                "不要保留笔、天、日、分、元、万元、亿元等单位文字",
            ],
        },
        "citations": [_citation_schema()],
        "confidence": 0.0,
        "unresolved": [],
    }


def _choice_final_schema(question: dict[str, Any]) -> dict[str, Any]:
    option_schema = {
        str(letter).upper(): _choice_option_schema()
        for letter in sorted((question.get("options") or {}).keys())
    }
    return {
        "option_judgement": option_schema,
        "answers": ["按字母升序拼接的最终选项，如 AC；判断题也只填一个字母"],
        "confidence": 0.0,
        "unresolved": [],
    }


def _system(role: str) -> str:
    return (
        f"你是 B 榜金融长文档多智能体系统中的 {role}。"
        "只能依据消息内提供的题目、候选原文和上游 JSON 工作，不得用外部常识补足缺失证据。"
        "你的完整回复必须是一个合法 JSON 对象，禁止 Markdown、代码围栏、解释性前后缀和 JSON 之外的文字。"
        "所有 quote 必须从对应证据逐字复制；不能把推理、改写或多个不连续片段冒充原文。"
    )


def _audit_rules(kind: str, slots: int) -> str:
    common = """
通用硬规则：
1. 证据位置至少核对 doc_id/source_path、page 或 chunk_id；若来源提供字符区间，也要保留 page_char_start/page_char_end。
2. 表格证据必须同时核对表名或 table_id、行名、列名、列对应年份/期间和单位，禁止跨列、跨年或跨单位抄数。
3. 区分报告期、同比期、期末、期初、预测值和实际值；区分元、万元、亿元、百分数和百分点。
4. 主体、指标定义、适用范围、条件、例外、否定词和比较方向任一不匹配，都不能判定为直接支持。
5. 没找到证据只能标记 unknown/uncertain，不能据此判 false；原文明确相反或关键条件不匹配才可判 false。
6. 引用必须逐字、短而完整，并能在给定 evidence_id 的 text 中直接查到。
""".strip()
    if kind == "choice":
        return (
            common
            + """

选择题规则：
1. 必须覆盖题目中每一个实际选项，先拆成 atomic_claims，再分别给 relation、judgement、citations。
2. entailed 必须对应 true，contradicted 必须对应 false，unknown 必须对应 uncertain。
3. 多选题不预设正确选项数量；单选和判断题最终只能有一个字母。
4. final answers 是长度为 1 的字符串数组，元素为按字母升序拼接的最终答案，例如 ["AC"]。
""".rstrip()
        )
    return (
        common
        + f"""

开放题规则：
1. 最终 answers 必须恰有 {slots} 个字符串，并严格对应 answer_1 到 answer_{slots} 的模板顺序。
2. 先记录 raw_values 及其逐字证据位置，再统一口径和单位，再代入公式；中间步骤不得提前四舍五入。
3. 只在最后一步执行舍入，并在 rounding_stage 说明规则。数值单位类结果只填数字，除题目另有明确要求外保留两位小数；
   百分数带 % 且保留两位小数；日期为 YYYY年M月D日；排序用半角 > 且无空格。
4. 计算题必须写公式和逐步计算；抽取题 formula 写 direct_extraction，但仍要核对列、年份、单位和最终格式。
5. 禁止把千分号、百分号、负号、括号表示的负数或表格单位遗漏在转换过程。
6. 题目要求“全年、累计、合计”时，必须主动检索并枚举同一报告期内的中期、年度、特别分红或其他组成项，
   判断各项是已实施、拟实施还是仅重复披露；只有确认口径不重叠后才能求和，禁止把单个年末方案直接当作全年总额。
""".rstrip()
    )


def _messages(role: str, body: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system(role)},
        {"role": "user", "content": body.strip()},
    ]


def build_locator_messages(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    answer_slots: int | None = None,
) -> list[dict[str, str]]:
    slots = _answer_slots(question, answer_slots)
    kind = question_kind(question)
    if kind == "choice":
        schema = {
            "option_targets": {
                str(letter).upper(): {
                    "atomic_claims": ["最小事实条件"],
                    "supporting_evidence_ids": ["E1"],
                    "contradicting_evidence_ids": ["E2"],
                    "relevant_evidence_ids": ["E1", "E2"],
                    "quoted_clauses": [
                        {"evidence_id": "E1", "quote": "逐字原文"}
                    ],
                    "missing_search_queries": ["仍缺证据时的精确检索短语"],
                    "coverage": "complete/partial/missing",
                }
                for letter in sorted((question.get("options") or {}).keys())
            },
            "global_search_queries": [],
            "unresolved": [],
        }
        task = (
            "定位员不作最终选择。逐选项定位直接支持、直接反证和必要上下文，"
            "优先找定义、表头、年份、单位、公式、脚注和例外条款。"
        )
    else:
        schema = {
            "required_facts": [
                {
                    "name": "待抽取变量/条件",
                    "purpose": "用于哪个答案槽或公式",
                    "evidence_ids": ["E1"],
                    "quoted_clauses": [
                        {"evidence_id": "E1", "quote": "逐字原文"}
                    ],
                    "year": "年份或期间",
                    "unit": "原始单位",
                    "table_coordinates": {
                        "table_id": "",
                        "row_label": "",
                        "column_label": "",
                    },
                    "coverage": "complete/partial/missing",
                }
            ],
            "candidate_formula": "候选公式或 direct_extraction",
            "missing_search_queries": [],
            "unresolved": [],
        }
        task = (
            "定位员不计算最终答案。识别每个答案槽需要的原始变量、定义、公式、"
            "表格行列、年份、单位、脚注和格式条件，并给出逐字证据。"
            "若题目要求全年、累计或合计值，还要分别定位中期、年度、特别项目及其是否已实施，"
            "明确列出可能需要相加或排除重复计算的组成项。"
        )
    body = f"""
{format_question(question, slots)}

候选原文：
{format_evidence(evidence)}

{_audit_rules(kind, slots)}

任务：{task}
只输出以下 JSON 结构：
{_json(schema)}
"""
    return _messages("文档定位员 locator", body)


def build_evidence_analyst_messages(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    locator: dict[str, Any] | None = None,
    *,
    answer_slots: int | None = None,
) -> list[dict[str, str]]:
    slots = _answer_slots(question, answer_slots)
    kind = question_kind(question)
    if kind == "choice":
        schema = {
            "option_judgement": {
                str(letter).upper(): _choice_option_schema()
                for letter in sorted((question.get("options") or {}).keys())
            },
            "evidence_conflicts": [],
            "unresolved": [],
        }
        task = (
            "逐选项独立审计。每个 atomic claim 都要落到逐字证据位置，"
            "再给出 relation/judgement；此阶段不得靠选项数量分布猜答案。"
        )
    else:
        schema = {
            "verified_values": [
                {
                    "name": "变量名",
                    "raw_value": "原文值",
                    "normalized_value": "仅统一单位后的值，不提前舍入",
                    "year": "年份或期间",
                    "source_unit": "原始单位",
                    "target_unit": "目标单位",
                    "citations": [_citation_schema()],
                    "status": "verified/conflicted/missing",
                }
            ],
            "verified_formula": "文档公式、题意公式或 direct_extraction",
            "answer_slot_mapping": [
                {"slot": 1, "meaning": "answer_1 对应含义"}
            ],
            "format_requirements": [],
            "evidence_conflicts": [],
            "unresolved": [],
        }
        task = (
            "核验所有原始值、定义、公式、答案槽顺序与格式要求。"
            "必须核对表格行列、年份和单位；不要计算或猜最终答案。"
        )
    body = f"""
{format_question(question, slots)}

候选原文：
{format_evidence(evidence)}

定位员 JSON：
{_json(locator or {})}

{_audit_rules(kind, slots)}

任务：{task}
只输出以下 JSON 结构：
{_json(schema)}
"""
    return _messages("证据分析员 evidence analyst", body)


def build_independent_solver_messages(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    locator: dict[str, Any] | None = None,
    analyst: dict[str, Any] | None = None,
    *,
    answer_slots: int | None = None,
) -> list[dict[str, str]]:
    slots = _answer_slots(question, answer_slots)
    kind = question_kind(question)
    schema = (
        {
            "option_judgement": {
                str(letter).upper(): _choice_option_schema()
                for letter in sorted((question.get("options") or {}).keys())
            },
            "answers": ["按字母升序拼接的候选答案"],
            "confidence": 0.0,
            "unresolved": [],
        }
        if kind == "choice"
        else _open_final_schema(slots)
    )
    task = (
        "从原文独立完成题目。可以参考上游定位与核验结果，但必须自行复查逐字引用。"
        "选择题逐项判断后生成 answers；开放题按原始值、单位转换、公式、运算顺序、"
        "最终舍入和格式化的顺序生成 answers。"
    )
    body = f"""
{format_question(question, slots)}

候选原文：
{format_evidence(evidence)}

定位员 JSON：
{_json(locator or {})}

证据分析员 JSON：
{_json(analyst or {})}

{_audit_rules(kind, slots)}

任务：{task}
只输出以下 JSON 结构：
{_json(schema)}
"""
    return _messages("独立解题员 independent solver", body)


def build_skeptic_messages(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    locator: dict[str, Any] | None = None,
    analyst: dict[str, Any] | None = None,
    solver: dict[str, Any] | None = None,
    *,
    answer_slots: int | None = None,
) -> list[dict[str, str]]:
    slots = _answer_slots(question, answer_slots)
    kind = question_kind(question)
    reviewed = (
        {
            "option_judgement": {
                str(letter).upper(): _choice_option_schema()
                for letter in sorted((question.get("options") or {}).keys())
            },
            "answers": ["审查后的候选答案"],
        }
        if kind == "choice"
        else _open_final_schema(slots)
    )
    schema = {
        "attacks": [
            {
                "target": "solver/analyst/locator",
                "issue": (
                    "主体/指标/年份/单位/表格列/范围/否定/公式/"
                    "运算顺序/提前舍入/最终格式错误"
                ),
                "citations": [_citation_schema()],
                "material": True,
            }
        ],
        "reviewed_result": reviewed,
        "confidence": 0.0,
        "unresolved": [],
    }
    body = f"""
{format_question(question, slots)}

候选原文：
{format_evidence(evidence)}

定位员 JSON：
{_json(locator or {})}

证据分析员 JSON：
{_json(analyst or {})}

独立解题员 JSON：
{_json(solver or {})}

{_audit_rules(kind, slots)}

任务：从零对抗复核，不按多数投票。主动寻找错误的主体、指标口径、年份、单位、
表格行列、条件例外、否定词、公式、运算顺序、中间提前舍入和最终格式。
每个实质性攻击必须有逐字证据位置；若攻击成立，在 reviewed_result 中给出纠正结果。
只输出以下 JSON 结构：
{_json(schema)}
"""
    return _messages("对抗复核员 skeptic", body)


def build_final_judge_messages(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    locator: dict[str, Any] | None = None,
    analyst: dict[str, Any] | None = None,
    solver: dict[str, Any] | None = None,
    skeptic: dict[str, Any] | None = None,
    *,
    answer_slots: int | None = None,
) -> list[dict[str, str]]:
    slots = _answer_slots(question, answer_slots)
    kind = question_kind(question)
    schema = (
        _choice_final_schema(question)
        if kind == "choice"
        else _open_final_schema(slots)
    )
    body = f"""
{format_question(question, slots)}

候选原文：
{format_evidence(evidence)}

定位员 JSON：
{_json(locator or {})}

证据分析员 JSON：
{_json(analyst or {})}

独立解题员 JSON：
{_json(solver or {})}

对抗复核员 JSON：
{_json(skeptic or {})}

{_audit_rules(kind, slots)}

最终裁决要求：
1. 回到原文逐项裁决，不按智能体多数投票；上游结论不是证据。
2. 校验每条 quote 确实逐字存在于对应 evidence_id，并校验文档、页码/chunk、表格行列、年份和单位。
3. 选择题必须先完成每个选项的 relation/judgement/citations，再生成唯一 final answers。
4. 开放题必须重新核算 raw_values、formula、calculation_steps 和 rounding_stage，最后才应用 format_spec。
5. answers 中每个元素必须是可直接写入 answer_1..answer_4 的字符串，不得包含解释。
6. 证据不足或角色分歧未解决时写入 unresolved，不得伪造确定性。

只输出以下最终 JSON 结构：
{_json(schema)}
"""
    return _messages("最终裁决员 final judge", body)


def build_compact_teacher_review_messages(
    question: dict[str, Any],
    teacher_cache: dict[str, Any],
    *,
    answer_slots: int | None = None,
) -> list[dict[str, str]]:
    """Build the low-token exact-answer review prompt.

    The caller should exact-match this output against the teacher answers and
    fall back to the teacher cache on any mismatch.
    """
    slots = _answer_slots(question, answer_slots)
    compact_question = {
        "qid": question.get("qid", ""),
        "type": question.get("type", ""),
        "question": question.get("question", ""),
        "options": question.get("options") or {},
        "answer_slots": slots,
    }
    compact_cache = {
        "answers": teacher_cache.get("answers") or [],
        "key_evidence": teacher_cache.get("key_evidence")
        or teacher_cache.get("citations")
        or [],
        "formula": teacher_cache.get("formula", ""),
        "format_spec": teacher_cache.get("format_spec", {}),
    }
    system = (
        "你是 teacher-cache 的低 token 一致性复核器。"
        "只检查 teacher answers 的槽数和明显格式，不重新解题、不扩写证据、不改变答案。"
        "完整回复必须是一个合法 JSON 对象，且只能包含 answers 这一个键；"
        "禁止 Markdown、解释、confidence、reasoning、citations 或任何额外字段。"
    )
    body = f"""
题目：
{_json(compact_question)}

teacher cache：
{_json(compact_cache)}

原样返回 teacher cache 中的 answers。必须恰有 {slots} 个字符串，顺序、字符、百分号、
日期、排序符号和小数位均不得改变。只输出：
{_json({"answers": [f"answer_{index}" for index in range(1, slots + 1)]})}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": body.strip()},
    ]


# Short aliases keep the API convenient and mirror the existing A-list module.
build_analyst_messages = build_evidence_analyst_messages
build_solver_messages = build_independent_solver_messages
build_judge_messages = build_final_judge_messages
build_compact_teacher_cache_messages = build_compact_teacher_review_messages


__all__ = [
    "PROMPT_VERSION",
    "question_kind",
    "format_question",
    "format_evidence",
    "build_locator_messages",
    "build_evidence_analyst_messages",
    "build_analyst_messages",
    "build_independent_solver_messages",
    "build_solver_messages",
    "build_skeptic_messages",
    "build_final_judge_messages",
    "build_judge_messages",
    "build_compact_teacher_review_messages",
    "build_compact_teacher_cache_messages",
]
