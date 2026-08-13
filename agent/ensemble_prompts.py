from __future__ import annotations

import json
from typing import Any


ERROR_TYPES = (
    "none/missing_evidence/entity_mismatch/metric_mismatch/unit_mismatch/"
    "time_mismatch/condition_mismatch/scope_mismatch/negation_mismatch/"
    "calculation_mismatch/category_mismatch/semantic_mismatch"
)


def format_question(question: dict[str, Any]) -> str:
    options = "\n".join(
        f"{letter}. {text}"
        for letter, text in sorted((question.get("options") or {}).items())
    )
    doc_ids = [str(doc_id) for doc_id in question.get("doc_ids") or []]
    doc_order = "\n".join(
        f"- 第{index}份文档 = {doc_id}"
        for index, doc_id in enumerate(doc_ids, start=1)
    )
    return (
        f"题目ID: {question['qid']}\n"
        f"领域: {question.get('domain', '')}\n"
        f"题型: {question.get('answer_format', '')}\n"
        f"能力标签: {question.get('type', '')}\n"
        f"指定文档: {', '.join(doc_ids)}\n"
        f"文档顺序映射:\n{doc_order or '-'}\n"
        f"题干: {question.get('question', '')}\n"
        f"选项:\n{options}"
    )


def format_evidence(evidence: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for evidence_id, item in enumerate(evidence, start=1):
        page = item.get("page")
        start = item.get("page_char_start")
        end = item.get("page_char_end")
        sources = ",".join(str(value) for value in item.get("sources") or [])
        blocks.append(
            f"[证据{evidence_id}] "
            f"doc_id={item.get('doc_id')} "
            f"chunk_id={item.get('chunk_id')} "
            f"page={page if page is not None else 'N/A'} "
            f"page_chars={start if start is not None else 'N/A'}:"
            f"{end if end is not None else 'N/A'} "
            f"sources={sources or '-'}\n"
            f"{item.get('text', '')}"
        )
    return "\n\n".join(blocks)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def common_option_rules() -> str:
    return """
- 对 A/B/C/D 每个实际存在的选项分别判断，不得因题型或常见答案分布凑选项数量。
- 先拆分主体、客体、指标、单位、年份、范围、否定词、例外、并列条件和计算关系；全部成立才可判 true。
- “均、都、同时、且、分别、连续、至少、不超过、原则上、除外”等限定必须逐一核对。
- 先恢复题干、同段落和文档标题已经唯一确定的上下文主体。不得仅因选项省略该主体就判错；
  只有省略后产生两个以上合理主体或实质扩大适用范围时，才可判 semantic_mismatch 或 scope_mismatch。
- 同义概括不要求逐字一致。只有概括改变法律效果、经济含义、计算口径或责任范围时才判错；
  对“欠款/借款及借款利息”等术语，必须先核对文档定义和计算公式，不得机械逐字苛判。
- 分类或限定类别题必须分两步：先判断选项陈述本身是否真实，再判断它是否属于题干要求的类别；
  陈述真实但不属于限定类别时，使用 category_mismatch 并不选。
- 跨文档比较必须覆盖选项涉及的每一份文档；“第一份/第二份文档”严格按题目中的文档顺序映射。
- 数字题必须核对指标名称、报告期、单位、同比方向、分子分母和计算口径。
- 判断题中，A 表示题干整句正确，B 表示题干整句错误。
- judgement=true 必须对应 relation=entailed；judgement=false 必须对应 relation=contradicted；
  证据不足必须用 judgement=uncertain、relation=unknown。
- quote 必须逐字复制自对应证据，不得改写、拼接或把推理文字冒充原文。
- supporting_evidence_ids、contradicting_evidence_ids、relevant_evidence_ids 只能填写当前证据区中的编号。
- 仅存在语义概括争议、上下文主体争议或轻微措辞差异时，必须标记 unresolved；
  最终裁决员不得据此修改已有基线，除非原文能证明含义发生实质变化。
""".strip()


def domain_specific_rules(question: dict[str, Any]) -> str:
    domain = str(question.get("domain", ""))
    if domain == "insurance":
        return """
- 明确每份保单的被保险人范围，不能把投保人、配偶、家庭成员或附加险责任相互替换。
- 多保险赔付必须按顺序列式计算：医保补偿、先赔商业险、其他商业保险补偿扣除、剩余可赔费用。
- 核对费用补偿原则和禁止重复补偿规则；同一笔费用已被其他商业保险补偿后不得再次全额计算。
- 区分家庭共享免赔额、个人免赔额、年度免赔额、单次免赔额，并核对是否达到门槛。
- 涉及金额的选项必须输出逐步算式；无法确定赔付顺序或扣除项时判 uncertain。
""".strip()
    if domain == "regulatory":
        return """
- 法规题严格核对“以上/超过”“以内/少于”“应当/可以”“原则上/必须”和例外条款。
- 高风险、较高风险、高风险以上等等级不得互换。
- 先判断法条陈述真假，再判断是否属于题干限定的审批程序、金额门槛、保存期限等类别。
- 调任“其他高级管理人员职位”不得扩大为任意“其他职位”。
""".strip()
    if domain in {"research", "financial_reports"}:
        return """
- 指标名称中的修饰词是指标的一部分，例如“除客户资金杠杆”不得改成“客户资金杠杆”。
- 报告标题、图表标题和紧邻段落能够唯一限定地区、行业或主体时，先恢复该上下文再判断选项。
- 预测值、实际值、同比、环比、复合增速和时间区间必须分别核对。
""".strip()
    if domain == "financial_contracts":
        return """
- 债券名称、发行规模、注册额度、本期规模、募集资金总额和余额不是当然同一指标。
- 核对“含本数”的边界、转股价格、评级对象、担保主体和中介机构角色。
- 合同术语可按文档定义作同义概括，但不得改变金额口径、权利义务或适用对象。
""".strip()
    return "- 使用题目所属领域的原文定义，不用外部常识替代文档证据。"


def build_locator_messages(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system = (
        "你是金融长文档证据定位员。你的任务不是猜最终答案，而是为每个选项寻找直接支持、"
        "直接反证和仍缺失的检索线索。只能使用给定证据，不得用外部常识补全。"
        "现有最佳答案不会提供给你，以避免锚定。只输出一个合法 JSON 对象。"
    )
    user = f"""
{format_question(question)}

候选原文:
{format_evidence(evidence)}

通用语义规则:
{common_option_rules()}

领域专项规则:
{domain_specific_rules(question)}

逐个选项执行：
1. 把选项拆成最小事实条件，特别检查主体、指标、单位、年份、范围、否定词、例外和并列条件。
2. 列出可能支持该选项、可能反驳该选项以及仅与该选项相关的证据编号。
3. 从证据中复制最关键的短句。quote 必须逐字来自对应证据，不得改写。
4. 如果证据不足，给出 1-4 条适合在指定文档中继续检索的短语，优先使用原文可能出现的术语。
5. 即使当前证据只能证明“未覆盖”，也要为该选项列出最接近的 relevant_evidence_ids。
6. 额外寻找能够恢复上下文主体的标题/同段落、术语定义、题干限定类别和保险计算条款。

只输出以下 JSON 结构：
{{
  "option_search": {{
    "A": {{
      "atomic_claims": ["条件1"],
      "supporting_evidence_ids": [1],
      "contradicting_evidence_ids": [2],
      "relevant_evidence_ids": [1, 2],
      "quoted_clauses": [{{"evidence_id": 1, "quote": "逐字原文"}}],
      "search_queries": ["精确检索短语"],
      "coverage": "complete/partial/missing"
    }}
  }},
  "global_search_queries": ["跨选项补充检索短语"],
  "retrieval_notes": "一句话说明主要证据缺口"
}}

必须覆盖题目中实际存在的每一个选项字母。
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_analyst_messages(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system = (
        "你是独立的金融法规、合同、财报、保险与研报分析员。你看不到任何既有答案，"
        "必须完全依据原文逐项证明。只有原文完整支持选项全部条件时才判 true；"
        "原文明示相反或关键条件不匹配时判 false；检索不足判 uncertain。"
        "只输出一个合法 JSON 对象。"
    )
    user = f"""
{format_question(question)}

完整证据区:
{format_evidence(evidence)}

审查规则:
{common_option_rules()}

领域专项规则:
{domain_specific_rules(question)}

只输出以下 JSON：
{{
  "answer": "按字母升序",
  "option_judgement": {{
    "A": {{
      "atomic_claims": [
        {{
          "claim": "最小事实条件",
          "status": "supported/contradicted/unknown",
          "evidence_ids": [1]
        }}
      ],
      "judgement": "true/false/uncertain",
      "relation": "entailed/contradicted/unknown",
      "error_type": "{ERROR_TYPES}",
      "supporting_evidence_ids": [1],
      "contradicting_evidence_ids": [2],
      "relevant_evidence_ids": [1, 2],
      "quoted_clauses": [{{"evidence_id": 1, "quote": "逐字原文"}}],
      "context_subject": "由题干、标题或段落恢复的主体；不适用填空字符串",
      "semantic_equivalence": "equivalent/material_change/not_applicable/uncertain",
      "category_match": "yes/no/not_applicable/uncertain",
      "calculation_steps": ["涉及数字或赔付时逐步列式；不适用为空数组"],
      "reasoning": "说明所有条件如何被支持、反驳或为何仍无法确认",
      "confidence": 0.0
    }}
  }},
  "overall_confidence": 0.0,
  "unresolved": ["仍缺少什么证据"]
}}

answer 必须等于 judgement=true 且 relation=entailed 的选项集合。
单选题和判断题只能有一个答案字母。必须覆盖每个实际存在的选项。
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_skeptic_messages(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system = (
        "你是独立的对抗性证据审稿人。你看不到既有答案或其他分析员结论。"
        "请从头完成题目，重点寻找看似正确选项中的范围扩大、对象偷换、指标口径、年份、"
        "单位、条件、否定和例外错误，同时寻找可能被漏选的直接支持项。"
        "只输出一个合法 JSON 对象。"
    )
    user = f"""
{format_question(question)}

完整证据区:
{format_evidence(evidence)}

审查规则:
{common_option_rules()}

领域专项规则:
{domain_specific_rules(question)}

对每个选项同时尝试“证成”和“证伪”：
- 有直接支持且找不到破坏性限定时才判 true。
- 有明确相反条款，或主体/指标/单位/年份/范围/条件不一致时判 false。
- 仅仅没有检索到不能判 false，必须判 uncertain。
- 每个选项至少给出 relevant_evidence_ids 和对应原文位置；有直接支持或反证时还要给逐字 quote。

只输出以下 JSON：
{{
  "answer": "按字母升序",
  "option_judgement": {{
    "A": {{
      "atomic_claims": [
        {{
          "claim": "最小事实条件",
          "status": "supported/contradicted/unknown",
          "evidence_ids": [1]
        }}
      ],
      "judgement": "true/false/uncertain",
      "relation": "entailed/contradicted/unknown",
      "error_type": "{ERROR_TYPES}",
      "supporting_evidence_ids": [1],
      "contradicting_evidence_ids": [2],
      "relevant_evidence_ids": [1, 2],
      "quoted_clauses": [{{"evidence_id": 1, "quote": "逐字原文"}}],
      "context_subject": "由题干、标题或段落恢复的主体；不适用填空字符串",
      "semantic_equivalence": "equivalent/material_change/not_applicable/uncertain",
      "category_match": "yes/no/not_applicable/uncertain",
      "calculation_steps": ["涉及数字或赔付时逐步列式；不适用为空数组"],
      "strongest_counterargument": "最强反方理由",
      "reasoning": "最终判断依据",
      "confidence": 0.0
    }}
  }},
  "overall_confidence": 0.0,
  "unresolved": ["仍存在的歧义"]
}}

answer 必须等于 judgement=true 且 relation=entailed 的选项集合。
必须覆盖每个实际存在的选项。
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_judge_messages(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    baseline_answer: str,
    locator: dict[str, Any],
    analyst: dict[str, Any],
    skeptic: dict[str, Any],
) -> list[dict[str, str]]:
    system = (
        "你是最终证据裁决员。你必须回到原文证据裁决，不得机械按多数投票。"
        "既有答案只是待审基线，不是证据；只有直接原文足以证明变化时才修改。"
        "你必须校验其他角色所引用的证据编号和逐字引文。只输出一个合法 JSON 对象。"
    )
    user = f"""
{format_question(question)}

完整证据区:
{format_evidence(evidence)}

待审基线答案: {baseline_answer or '-'}

证据定位员输出:
{json_text(locator)}

独立分析员输出:
{json_text(analyst)}

对抗审稿人输出:
{json_text(skeptic)}

裁决规则:
{common_option_rules()}

领域专项规则:
{domain_specific_rules(question)}

额外要求：
1. 对每个选项重新核对证据编号和逐字 quote，说明采纳或否决两位审稿人的原因。
2. true 必须有直接支持；false 必须有直接反证或明确的主体、指标、年份、单位、条件、范围不匹配。
3. 证据不足时用 uncertain。若因此无法形成合法单选/判断答案，可保留基线，但必须标记 unresolved。
4. 修改基线时，change_reason 必须指出具体选项、doc_id、页码或 chunk_id 和原文依据。
5. 不论最终真假，每个选项都必须给出 relevant_evidence_ids；true/false 还必须给出可验证逐字 quote。
6. 最终 answer 与 option_judgement 中 true + entailed 的集合保持一致。
7. 对省略上下文主体的选项，先写出恢复后的主体；若题干、标题或同段落只能指向一个主体，不得因省略而判错。
8. 对术语概括先判断是否改变法律或经济含义。仅非逐字一致不能作为修改基线的依据。
9. 对分类题分别填写“事实真假”和“类别是否匹配”；真实但不属于题干限定类别时才以 category_mismatch 排除。
10. 多保险计算必须列出被保险人、赔付顺序、其他商业保险扣除、费用补偿原则及免赔额类型。
11. 若变化仅依赖语义争议，baseline_action 必须为 unresolved_keep，answer 保留基线，并在 unresolved 中说明。

只输出以下 JSON：
{{
  "answer": "按字母升序",
  "option_judgement": {{
    "A": {{
      "atomic_claims": [
        {{
          "claim": "最小事实条件",
          "status": "supported/contradicted/unknown",
          "evidence_ids": [1]
        }}
      ],
      "judgement": "true/false/uncertain",
      "relation": "entailed/contradicted/unknown",
      "error_type": "{ERROR_TYPES}",
      "supporting_evidence_ids": [1],
      "contradicting_evidence_ids": [2],
      "relevant_evidence_ids": [1, 2],
      "quoted_clauses": [{{"evidence_id": 1, "quote": "逐字原文"}}],
      "context_subject": "由题干、标题或段落恢复的主体；不适用填空字符串",
      "semantic_equivalence": "equivalent/material_change/not_applicable/uncertain",
      "category_match": "yes/no/not_applicable/uncertain",
      "calculation_steps": ["涉及数字或赔付时逐步列式；不适用为空数组"],
      "reasoning": "最终裁决依据",
      "confidence": 0.0
    }}
  }},
  "overall_confidence": 0.0,
  "changed_from_baseline": true,
  "baseline_action": "change/keep/unresolved_keep",
  "change_classification": "direct_contradiction/context_error/semantic_dispute/category_mismatch/calculation_correction/no_change",
  "change_reason": "逐项说明修改依据；不修改则说明保留依据",
  "agent_disagreements": ["分歧及裁决"],
  "unresolved": ["仍无法完全确认的点"]
}}

必须覆盖每个实际存在的选项。
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
