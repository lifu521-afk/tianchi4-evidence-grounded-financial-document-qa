from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Iterable

from .domain import get_profile
from .tokenize import bm25_score, normalize_for_search, term_counts


@dataclass(frozen=True)
class SearchResult:
    chunk: dict
    score: float
    source: str = "search"


QUERY_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("营收", "营业收入", "营业总收入"), ("营业收入", "营业总收入", "收入合计")),
    (("归母", "归属于上市公司股东", "归属于母公司股东"), ("归属于上市公司股东的净利润", "归属于母公司股东的净利润", "归母净利润")),
    (("净利润", "利润"), ("净利润", "归属于上市公司股东的净利润", "归属于母公司股东的净利润")),
    (("现金流", "现金流量", "经营活动"), ("经营活动产生的现金流量净额", "经营现金流", "现金流量净额")),
    (("研发", "研发投入", "研发费用", "研发强度"), ("研发投入", "研发费用", "研发投入占营业收入比例", "研发费用占营业收入比例")),
    (
        ("分红", "股息", "派息", "利润分配"),
        (
            "现金分红",
            "每10股派发",
            "利润分配预案",
            "股息",
            "派息",
            "中期分红",
            "年度分红",
            "特别分红",
            "全年现金分红",
        ),
    ),
    (("回购", "股份回购"), ("股份回购", "回购金额", "回购方案")),
    (("同比", "增长", "下降", "增速", "降幅"), ("同比增长", "同比下降", "较上年", "上年同期", "增长率")),
    (("保险责任", "身故", "退保", "现金价值"), ("保险责任", "身故保险金", "现金价值", "退保金", "保单账户价值")),
    (("已交保费", "已交保险费", "保费"), ("已交保险费", "累计已交保险费", "基本保险金额", "现金价值")),
    (("股东大会", "股东会"), ("股东大会", "股东会", "普通决议", "特别决议")),
    (("应当", "不得", "可以", "必须"), ("应当", "不得", "可以", "必须", "须", "应")),
    (("工作日", "日内", "期限", "时限"), ("工作日", "日内", "期限", "报告", "备案")),
    (("评级", "信用等级", "主体信用"), ("信用等级", "主体信用等级", "债项信用等级", "评级展望", "资信评级")),
    (("票面利率", "利率", "付息", "兑付"), ("票面利率", "付息日", "兑付日", "到期一次还本", "按年付息")),
    (("募集资金", "用途"), ("募集资金用途", "募集资金拟用于", "偿还有息债务", "补充流动资金")),
    (("发行主体", "发行人名称"), ("发行人", "发行主体", "公司名称")),
    (("发行规模", "发行金额", "金额上限"), ("本期债券发行规模", "发行金额", "不超过")),
    (("受托管理人", "主承销商", "中介机构", "簿记管理人"), ("受托管理人", "牵头主承销商", "联席主承销商", "簿记管理人")),
    (("增信", "担保", "无增信"), ("增信措施", "无增信", "担保", "保证担保")),
    (("除外", "例外", "另有规定", "不适用"), ("除外", "但书", "例外", "另有规定", "不适用")),
    (("普通决议", "特别决议", "表决权", "三分之二"), ("普通决议", "特别决议", "表决权", "过半数", "三分之二")),
    (("股东大会", "股东会", "职权", "会议规则"), ("股东会", "股东大会", "董事会的工作报告", "变更募集资金用途", "担保事项", "普通决议", "特别决议")),
    (("对外担保", "担保对象", "资产负债率", "70%", "百分之七十"), ("对外担保", "担保对象", "资产负债率", "百分之七十", "单笔担保额", "净资产百分之十", "须经股东会审议通过")),
    (("募集资金用途", "变更募集资金", "募投项目", "变更投向"), ("变更募集资金用途", "审议批准变更募集资金用途事项", "普通决议", "特别决议", "除法律、行政法规规定")),
    (("章程修改", "修改公司章程", "本章程的修改", "特别决议事项"), ("本章程的修改", "特别决议", "三分之二以上", "第八十二条", "公司章程可以")),
    (("独立董事", "法律顾问", "法律服务", "独立性", "利害关系"), ("独立董事", "任职条件", "选举更换程序", "直接或者间接利害关系", "影响其进行独立客观判断", "董事候选人")),
    (("等待期", "保险期间", "保单年度", "给付比例"), ("等待期", "保险期间", "保单年度", "给付比例", "基本保险金额")),
    (("目标价", "投资评级", "推荐", "买入", "盈利预测"), ("目标价", "投资评级", "推荐评级", "买入评级", "盈利预测", "风险提示")),
    (("资产负债率", "负债率", "杠杆"), ("资产负债率", "负债合计", "资产总计", "杠杆率", "有息负债")),
    (("毛利率", "净利率", "ROE", "净资产收益率"), ("毛利率", "销售毛利率", "净利率", "加权平均净资产收益率", "ROE")),
    (("转股价格", "转股价", "向下修正"), ("转股价格", "初始转股价格", "转股价格向下修正", "修正后的转股价格")),
    (("违约", "赔偿", "违约金", "逾期利息"), ("违约", "违约责任", "违约金", "逾期利息", "赔偿", "150%")),
    (("犹豫期", "冷静期", "退还保险费"), ("犹豫期", "冷静期", "无息退还", "已交保险费", "保险合同终止")),
    (("免赔", "赔付比例", "报销比例"), ("免赔额", "赔付比例", "给付比例", "报销比例", "扣除免赔额")),
    (("受益所有人", "尽职调查", "身份资料", "交易记录"), ("受益所有人", "尽职调查", "客户身份资料", "交易记录", "保存期限", "重大差异")),
)


STRUCTURED_TERM_PATTERNS: tuple[str, ...] = (
    r"第\s*[一二三四五六七八九十百千万0-9]+\s*条(?:之[一二三四五六七八九十0-9]+)?",
    r"\d{4}\s*年(?:\s*\d{1,2}\s*月\s*\d{1,2}\s*日)?",
    r"\d+(?:\.\d+)?\s*(?:亿元|万元|元|%|个工作日|日|年|个月|倍)",
    r"每\s*10\s*股",
    r"[A-Za-z]+_text_?0*\d+",
    r"(?:AAA|AA\+?|A\+?)\s*级?",
)


FINANCIAL_METRIC_TERMS = (
    "营业收入", "营业总收入", "归母净利润", "归属于上市公司股东的净利润", "净利润",
    "经营活动产生的现金流量净额", "现金流量净额", "研发投入", "研发费用", "研发强度",
    "资产负债率", "流动比率", "速动比率", "毛利率", "净利率", "ROE", "净资产收益率",
    "每股收益", "每10股", "现金分红", "回购", "票面利率", "主体信用评级", "债项信用评级",
    "转股价格", "回售", "赎回", "担保", "增信", "违约金", "逾期利息", "受托管理人",
    "保险责任", "身故保险金", "现金价值", "犹豫期", "等待期", "免赔额", "赔付比例",
    "受益所有人", "尽职调查", "保存期限", "工作日", "普通决议", "特别决议",
    "股东会", "股东大会", "董事会的工作报告", "变更募集资金用途", "对外担保",
    "担保对象", "资产负债率", "本章程的修改", "章程修改", "三分之二", "过半数",
    "独立董事", "董事候选人", "法律顾问", "法律服务", "利害关系", "客观判断",
    "投资评级", "目标价", "盈利预测", "风险提示", "市场规模", "同比增长", "CAGR",
)


class LexicalIndex:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.chunk_terms: list[Counter[str]] = []
        self.chunk_lens: list[int] = []
        doc_freq: Counter[str] = Counter()
        for chunk in chunks:
            terms = term_counts(chunk.get("text", ""))
            self.chunk_terms.append(terms)
            length = sum(terms.values())
            self.chunk_lens.append(length)
            doc_freq.update(terms.keys())
        n_docs = max(len(chunks), 1)
        self.avg_len = sum(self.chunk_lens) / n_docs
        self.idf = {
            term: math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in doc_freq.items()
        }
        self.by_doc: dict[str, list[int]] = defaultdict(list)
        self.by_domain: dict[str, list[int]] = defaultdict(list)
        self.by_chunk_id: dict[str, int] = {}
        for idx, chunk in enumerate(chunks):
            self.by_doc[chunk["doc_id"]].append(idx)
            self.by_domain[chunk["domain"]].append(idx)
            self.by_chunk_id[chunk["chunk_id"]] = idx

    def search(
        self,
        query: str,
        candidate_doc_ids: Iterable[str] | None = None,
        domain: str | None = None,
        top_k: int = 8,
    ) -> list[SearchResult]:
        query_terms = term_counts(query)
        if candidate_doc_ids:
            indices: list[int] = []
            for doc_id in candidate_doc_ids:
                indices.extend(self.by_doc.get(doc_id, []))
        elif domain:
            indices = list(self.by_domain.get(domain, []))
        else:
            indices = list(range(len(self.chunks)))

        query_norm = normalize_for_search(query)
        results: list[SearchResult] = []
        for idx in indices:
            text = self.chunks[idx].get("text", "")
            score = bm25_score(query_terms, self.chunk_terms[idx], self.chunk_lens[idx], self.avg_len, self.idf)
            score += exact_bonus(query_norm, normalize_for_search(text))
            if score > 0:
                results.append(SearchResult(self.chunks[idx], score))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def neighbors(self, chunk_id: str, radius: int = 1) -> list[dict]:
        idx = self.by_chunk_id.get(chunk_id)
        if idx is None:
            return []
        doc_id = self.chunks[idx]["doc_id"]
        doc_indices = self.by_doc.get(doc_id, [])
        try:
            pos = doc_indices.index(idx)
        except ValueError:
            return []
        start = max(0, pos - radius)
        end = min(len(doc_indices), pos + radius + 1)
        return [self.chunks[i] for i in doc_indices[start:end] if i != idx]


def exact_bonus(query: str, text: str) -> float:
    bonus = 0.0
    text_compact = text.replace(" ", "")
    for phrase in important_phrases(query):
        if phrase and phrase in text:
            bonus += min(10.0, 1.5 + len(phrase) / 8)
    for number in extract_numbers(query):
        if number and number in text:
            bonus += 2.5
    for term in extract_structured_terms(query):
        compact = term.replace(" ", "")
        if compact and compact in text_compact:
            if re.search(r"^第.+条", compact):
                bonus += 7.0
            elif re.search(r"\d", compact):
                bonus += 4.0
            else:
                bonus += 3.0
    return bonus


def important_phrases(query: str) -> list[str]:
    pieces = []
    for sep in "，。；;:：、()（）[]【】\n":
        query = query.replace(sep, "|")
    for piece in query.split("|"):
        piece = piece.strip()
        if 3 <= len(piece) <= 48:
            pieces.append(piece)
    return pieces[:60]


def build_query(question: dict, option_text: str | None = None, extra_terms: Iterable[str] = ()) -> str:
    base_texts = [question.get("question", ""), question.get("type", "")]
    if option_text:
        base_texts.append(option_text)
    else:
        base_texts.extend(question.get("options", {}).values())
    expanded = expand_query_terms(*base_texts)
    parts = [*base_texts, *extra_terms, *expanded]
    parts.extend(extract_numbers(question.get("question", "")))
    parts.extend(extract_structured_terms(question.get("question", "")))
    if option_text:
        parts.extend(extract_numbers(option_text))
        parts.extend(extract_structured_terms(option_text))
    return "\n".join(part for part in parts if part)


def option_document_ids(question: dict, option_text: str) -> list[str]:
    """Resolve references such as fc_text_003 to the question's text03 doc id."""
    references = {int(value) for value in re.findall(r"[A-Za-z]+_text_?0*(\d+)", option_text)}
    if not references:
        return []
    matched: list[str] = []
    for doc_id in question.get("doc_ids") or []:
        suffix = re.search(r"0*(\d+)$", str(doc_id))
        if suffix and int(suffix.group(1)) in references:
            matched.append(str(doc_id))
    return matched


def build_option_anchor_query(option_text: str) -> str:
    """Build a low-noise query from one option without unrelated question terms."""
    cleaned = re.sub(r"[A-Za-z]+_text_?\d+", " ", option_text)
    parts = [cleaned, *expand_query_terms(cleaned), *extract_numbers(cleaned), *extract_structured_terms(cleaned)]
    return "\n".join(part for part in parts if part)


def expand_query_terms(*texts: str) -> list[str]:
    haystack = normalize_for_search("\n".join(texts))
    terms: list[str] = []
    for triggers, expansions in QUERY_EXPANSIONS:
        if any(trigger.lower() in haystack for trigger in triggers):
            terms.extend(expansions)
    return sorted(set(terms))


def extract_structured_terms(text: str) -> list[str]:
    terms: set[str] = set()
    for pattern in STRUCTURED_TERM_PATTERNS:
        terms.update(re.sub(r"\s+", "", match) for match in re.findall(pattern, text, flags=re.I))
    compact_text = text.replace(" ", "")
    normalized_text = normalize_for_search(text)
    for term in FINANCIAL_METRIC_TERMS:
        if term.lower() in normalized_text or term.replace(" ", "") in compact_text:
            terms.add(term)
    for quoted in re.findall(r"[“\"「『]([^”\"」』]{2,32})[”\"」』]", text):
        terms.add(quoted.strip())
    return sorted(terms, key=lambda item: (not bool(re.search(r"\d", item)), -len(item), item))[:80]


def gather_evidence(
    index: LexicalIndex,
    question: dict,
    top_global: int | None = None,
    top_per_option: int | None = None,
    top_per_doc: int | None = None,
    neighbor_radius: int | None = None,
    max_chunks: int | None = None,
    max_chars: int | None = None,
) -> list[dict]:
    profile = get_profile(question.get("domain"))
    top_global = top_global if top_global is not None else profile.top_global
    top_per_option = top_per_option if top_per_option is not None else profile.top_per_option
    top_per_doc = top_per_doc if top_per_doc is not None else profile.top_per_doc
    neighbor_radius = neighbor_radius if neighbor_radius is not None else profile.neighbor_radius
    max_chunks = max_chunks if max_chunks is not None else profile.max_chunks
    max_chars = max_chars if max_chars is not None else profile.max_chars

    candidate_doc_ids = question.get("doc_ids") or None
    domain = question.get("domain")
    selected: dict[str, SearchResult] = {}
    tags: dict[str, set[str]] = defaultdict(set)

    def add(result: SearchResult, tag: str) -> None:
        chunk_id = result.chunk["chunk_id"]
        old = selected.get(chunk_id)
        if old is None or result.score > old.score:
            selected[chunk_id] = SearchResult(result.chunk, result.score, tag)
        tags[chunk_id].add(tag)

    global_query = build_query(question, extra_terms=profile.query_terms)
    for result in index.search(global_query, candidate_doc_ids, domain, top_global):
        add(result, "global")

    for letter, option_text in sorted(question.get("options", {}).items()):
        option_query = build_query(question, option_text, profile.query_terms)
        for result in index.search(option_query, candidate_doc_ids, domain, top_per_option):
            add(result, f"option_{letter}")

        # A second low-noise search prevents terms from the question or another
        # option from overwhelming the clause actually asserted by this option.
        anchor_query = build_option_anchor_query(option_text)
        anchor_doc_ids = option_document_ids(question, option_text) or candidate_doc_ids
        for result in index.search(anchor_query, anchor_doc_ids, domain, min(3, top_per_option)):
            add(result, f"option_{letter}")

    for doc_id in candidate_doc_ids or []:
        for result in index.search(global_query, [doc_id], domain, top_per_doc):
            add(result, f"doc_{doc_id}")

    if neighbor_radius:
        seeds = sorted(selected.values(), key=lambda item: item.score, reverse=True)[: max(6, top_global)]
        for seed in seeds:
            for chunk in index.neighbors(seed.chunk["chunk_id"], neighbor_radius):
                add(SearchResult(chunk, seed.score * 0.72, "neighbor"), "neighbor")

    ordered = prioritize_doc_coverage(selected.values(), candidate_doc_ids, tags)
    evidence: list[dict] = []
    total_chars = 0
    for result in ordered:
        text = result.chunk.get("text", "")
        if len(evidence) >= max_chunks or total_chars + len(text) > max_chars:
            continue
        item = dict(result.chunk)
        item["score"] = round(result.score, 4)
        item["sources"] = sorted(tags.get(result.chunk["chunk_id"], {result.source}))
        evidence.append(item)
        total_chars += len(text)
    return evidence


def prioritize_doc_coverage(
    results: Iterable[SearchResult],
    candidate_doc_ids: Iterable[str] | None,
    tags: dict[str, set[str]] | None = None,
) -> list[SearchResult]:
    all_results = sorted(results, key=lambda item: item.score, reverse=True)
    picked_ids: set[str] = set()
    priority: list[SearchResult] = []

    def pick_first(candidates: list[SearchResult]) -> None:
        for item in candidates:
            chunk_id = item.chunk["chunk_id"]
            if chunk_id not in picked_ids:
                priority.append(item)
                picked_ids.add(chunk_id)
                return

    for doc_id in candidate_doc_ids or []:
        pick_first([item for item in all_results if item.chunk.get("doc_id") == doc_id])

    if tags:
        for letter in "ABCD":
            option_tag = f"option_{letter}"
            pick_first([
                item
                for item in all_results
                if option_tag in tags.get(item.chunk["chunk_id"], set())
            ])

    priority.extend(item for item in all_results if item.chunk["chunk_id"] not in picked_ids)
    return priority


def extract_numbers(text: str) -> list[str]:
    numbers = re.findall(r"\d+(?:\.\d+)?%?|\d{4}\s*年|\d+\s*个工作日|\d+\s*日|每\s*10\s*股|\d+\s*亿元|\d+\s*万元", text)
    return [item.replace(" ", "") for item in numbers]
