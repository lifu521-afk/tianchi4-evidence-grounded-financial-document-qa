from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainProfile:
    top_global: int
    top_per_option: int
    top_per_doc: int
    neighbor_radius: int
    max_chunks: int
    max_chars: int
    query_terms: tuple[str, ...]
    prompt_checklist: str


DEFAULT_PROFILE = DomainProfile(
    top_global=8,
    top_per_option=4,
    top_per_doc=2,
    neighbor_radius=1,
    max_chunks=20,
    max_chars=20000,
    query_terms=(),
    prompt_checklist="逐项核对题干条件、选项断言和证据原文，证据不足时不要选择该项。",
)


DOMAIN_PROFILES: dict[str, DomainProfile] = {
    "financial_contracts": DomainProfile(
        top_global=8,
        top_per_option=5,
        top_per_doc=3,
        neighbor_radius=1,
        max_chunks=22,
        max_chars=18500,
        query_terms=("发行人", "债券", "募集说明书", "票面利率", "期限", "付息", "兑付", "评级", "担保", "回售", "赎回"),
        prompt_checklist=(
            "金融合同题优先核对发行主体、债券期限、利率、付息兑付、评级、担保、回售赎回、募集资金用途。"
            "同一题含多只债券时必须分别比较，不要把一个文档的条款套到另一个文档。"
        ),
    ),
    "financial_reports": DomainProfile(
        top_global=10,
        top_per_option=5,
        top_per_doc=3,
        neighbor_radius=1,
        max_chunks=24,
        max_chars=20500,
        query_terms=("营业收入", "归属于上市公司股东", "净利润", "经营活动", "现金流量净额", "研发投入", "分红", "回购", "同比", "占营业收入"),
        prompt_checklist=(
            "财报题要核对年份、单位、同比口径、归母净利润、经营现金流、研发投入比例、分红和回购金额。"
            "比较题必须逐年列出数值后再判断选项。"
        ),
    ),
    "insurance": DomainProfile(
        top_global=8,
        top_per_option=5,
        top_per_doc=3,
        neighbor_radius=1,
        max_chunks=22,
        max_chars=22000,
        query_terms=("保险责任", "身故保险金", "现金价值", "退保", "领取", "保单账户", "已交保险费", "等待期", "免责", "合同终止"),
        prompt_checklist=(
            "保险题要核对保险责任触发条件、年龄/期间限制、已交保费/现金价值/账户价值公式、领取或退保状态、免责条款。"
            "涉及多产品比较时逐产品判断。"
        ),
    ),
    "regulatory": DomainProfile(
        top_global=9,
        top_per_option=5,
        top_per_doc=3,
        neighbor_radius=1,
        max_chunks=22,
        max_chars=22000,
        query_terms=("第", "条", "规定", "应当", "不得", "可以", "工作日", "报告", "处罚", "股东会", "特别决议", "普通决议", "三分之二", "本章程的修改", "对外担保", "资产负债率", "募集资金用途", "独立董事"),
        prompt_checklist=(
            "监管题必须定位具体条款，区分应当/不得/可以、普通决议/特别决议、期限和例外条件。"
            "不要用常识替代法条；选项中的绝对化表述要特别核对。"
            "涉及担保、募集资金用途、独立董事和章程修改时，分别核对股东会审议触发条件、是否属于特别决议列举事项、独立性利害关系、以及章程修改须特别决议。"
        ),
    ),
    "research": DomainProfile(
        top_global=10,
        top_per_option=5,
        top_per_doc=3,
        neighbor_radius=1,
        max_chunks=24,
        max_chars=20500,
        query_terms=("行业", "公司", "市场", "增速", "同比", "毛利率", "份额", "预测", "风险", "推荐", "结论", "观点"),
        prompt_checklist=(
            "研报题要核对研究结论、行业趋势、公司比较、指标口径、预测年份和风险提示。"
            "选项若改写研究观点，必须确认方向和限定条件一致。"
        ),
    ),
}


def get_profile(domain: str | None) -> DomainProfile:
    return DOMAIN_PROFILES.get(domain or "", DEFAULT_PROFILE)
