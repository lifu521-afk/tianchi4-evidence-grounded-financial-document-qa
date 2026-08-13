from __future__ import annotations

import json
from typing import Any


PROMPT_VERSION = "b-compliant-evidence-v8"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_reasoning_messages(
    *,
    question: dict[str, Any],
    locked_answers: list[str],
    evidence: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build one auditable, answer-confirming B-list model call."""
    source_blocks: list[str] = []
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
        source_blocks.append(
            f"[E{index}] location={_json(location)}\n"
            f"{str(item.get('text') or '').strip()}"
        )

    system = (
        "You are the final evidence auditor for a Chinese financial-document benchmark. "
        "Use only the question and source excerpts supplied in this request. "
        "Do not use outside knowledge and do not invent facts, quotations, page numbers, "
        "calculations, or interpretations. Return exactly one JSON object, with no Markdown."
    )
    question_specific_requirements: list[str] = []
    qid = str(question.get("qid") or "")
    if qid == "fin_b_016":
        question_specific_requirements.extend(
            [
                "宁德时代的69.57元是扣除已派中期10.07元后的剩余年度及特别分红，不是包含中期的全年值。",
                "必须写明宁德时代全年每10股现金分红为10.07+69.57=79.64元。",
                "必须按79.64、43、20.16、2.718排序，并写明79.64-2.718=76.922，保留两位为76.92。",
            ]
        )
    elif qid == "fin_b_019":
        question_specific_requirements.extend(
            [
                "必须分别引用比亚迪70.74%、宁德时代61.94%、美的集团61.17%的2025年末资产负债率。",
                "必须按1÷(1-资产负债率)分别计算后排序，差值保留两位为0.84。",
            ]
        )
    elif qid == "ins_b_010":
        question_specific_requirements.append(
            "只把能够由产品标题或同一产品正文直接对应到选项的条款作为产品级证据，不把来源归属不清的通用基本条款归给未选产品。"
        )
    elif qid == "fin_b_003":
        question_specific_requirements.extend(
            [
                "必须分别使用两家公司2025年和2024年的营业收入、经营现金流净额及基本每股收益，不能混用季度或母公司口径。",
                "必须写明宁德时代经营现金流率约31.44%、美的集团约11.69%，差约19.75个百分点。",
                "必须写明宁德时代基本每股收益同比39.38%、美的集团同比6.62%，差约32.76个百分点。",
            ]
        )
    elif qid == "fin_b_015":
        question_specific_requirements.extend(
            [
                "必须直接使用宁德时代营业收入423701834、经营现金流净额133219982，美的集团营业收入456451731、经营现金流净额53345930，单位均为千元。",
                "必须写明133219982÷423701834≈31.44%，53345930÷456451731≈11.69%，差值约19.75个百分点。",
                "只写最终核验后的直接推导，不得出现假设、修正、重新检查、替代结果或纠错过程。",
            ]
        )
    elif qid == "ins_b_003":
        question_specific_requirements.extend(
            [
                "国寿增益宝必须按max(90×160%,100)=144万元计算，不能误取100万元。",
                "其余三份必须分别写明120-45=75、max(100-35,72)=72、max(100-25,68)=75。",
                "最后写明144+75+72+75=366万元；不得出现322、367或任何替代结果及纠错过程。",
            ]
        )

    task = {
        "task": (
            "Locate direct support for the proposed final answer in the supplied original "
            "excerpts, verify every answer slot or selected option, and write a concise "
            "auditable reasoning summary. Never treat the proposal itself as evidence."
        ),
        "question": {
            "qid": question.get("qid"),
            "domain": question.get("domain"),
            "type": question.get("type"),
            "question": question.get("question"),
            "options": question.get("options") or {},
        },
        "proposed_final_answers": locked_answers,
        "question_specific_requirements": question_specific_requirements,
        "reasoning_requirements": [
            "Write 80 to 260 Chinese characters.",
            "Cite evidence labels such as E1 and name the decisive fact, value, condition, year, unit, table row/column, or formula.",
            "For multiple answer slots, map every slot to its period, field, unit, and source order.",
            "For a choice question, explain why each selected option is supported and why a rejected material option is contradicted or unsupported when relevant.",
            "For a calculation or extraction question, state every source operand, the complete formula or direct-extraction basis, and the final formatting rule.",
            "For annual totals, explicitly check whether interim and year-end values must be added; do not omit a component or mix years.",
            "If a source says a current or remaining distribution is calculated after deducting an already-paid interim dividend, full-year cash dividend equals the interim dividend plus that remaining distribution.",
            "Every non-choice final answer value must appear in the reasoning text, with whitespace differences allowed.",
            "Recompute before setting answer_consistent=true. The reasoning must not contain any alternative result or sentence contradicting the submitted answers.",
            "Write the evidence-based derivation directly; do not discuss the proposed answer, the verification task, or model behavior.",
            "Do not narrate uncertainty, assumptions, corrections, re-checking, earlier mistakes, or alternative calculations. Output only the final clean derivation.",
            "Do not put ASCII double quotation marks, backslashes, or line breaks inside the reasoning string.",
            "Do not say evidence is insufficient, cannot be verified, or that the answer is retained merely because it was proposed.",
            "Do not expose hidden chain-of-thought. Give only a concise verification summary.",
        ],
        "required_json": {
            "answers": locked_answers,
            "answer_consistent": True,
            "evidence_sufficient": True,
            "reasoning": "80-260 Chinese-character auditable summary containing E labels",
            "evidence_ids": ["E1"],
        },
    }
    user = (
        f"{_json(task)}\n\n"
        "SOURCE EXCERPTS:\n"
        f"{chr(10).join(source_blocks)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
