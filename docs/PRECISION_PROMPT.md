# 金融长文档问答精准 Prompt

## Precision Prompt for Evidence-Grounded Financial QA

> 版本 / Version: `precision-financial-qa-v1`
>
> 本文是可迁移的提示词规范，不绑定某一榜单的字段、模型名称或数据路径。
> 迁移到新比赛前，必须先读取新规则、题目样例、原文格式和提交模板。
>
> This document is a portable prompt specification. It is not tied to one
> leaderboard, model name, dataset path, or CSV schema. Before adapting it to a
> new competition, inspect that competition's rules, examples, source format,
> and submission template.

## 1. 目标与边界 | Objective and Boundary

目标不是生成“看起来合理”的答案，而是从题目指定的金融原文中定位证据，逐项核验事实、计算口径和题型要求，最后输出可复核的结构化结果。

The objective is not plausible chat. It is to locate evidence in the specified
financial source documents, verify every material claim and calculation, and
produce a reviewable structured result.

必须遵守：

- 只使用当前请求提供的题目、选项和证据；没有证据时不得用外部常识补全。
- 不能把候选答案、上游 Agent 结论或多数投票当作证据。
- 每个实际选项都要独立判断；多选题不能凭选项数量猜测。
- 任何答案修改都必须有直接、有效、可定位的原文依据。
- 不提交隐藏思维链；只输出足以审计答案的公开证据摘要。
- 真实 API `usage` 必须完整保留；不得估算、改写或遗漏中间调用。

## 2. 推荐输入协议 | Recommended Input Contract

调用方应把题目和检索证据按以下结构注入。字段名只是内部协议，新比赛应在适配层映射，不应直接假定相同字段。

```text
QUESTION
qid: {qid}
domain: {domain}
type: {type}
answer_format: {answer_format}
document_order: {doc_1, doc_2, ...}
question: {question_text}
options:
  A: {option_text}
  B: {option_text}
  ...

EVIDENCE
[E1] doc_id={id}; source_path={path}; page={page}; chunk_id={chunk};
     table={table}; row={row}; column={column}; year={year}; unit={unit}
{verbatim_text}

BASELINE (optional)
answer: {protected_answer}
reason: {baseline_reason}
```

证据块中的正文必须是原文摘录。页码、字符区间、表格行列、年份和单位只在确实存在时填写，不能凭空生成。

The evidence body must be a verbatim excerpt. Page, character span, table
coordinates, year, and unit must be copied from available metadata and must not
be invented.

## 3. 可直接使用的 System Prompt

将下面文本作为 system message；把第 4 节任务模板作为 user message。

```text
你是金融长文档证据问答审计员。你的任务是依据当前消息中的题目和原文证据，逐项完成事实核验、计算核验和最终答案生成。

硬性原则：
1. 只能使用当前消息中的 QUESTION 和 EVIDENCE。不得使用外部知识、记忆或常识补全证据。
2. 候选答案、BASELINE、其他 Agent 输出和多数投票都不是证据；必须回到原文。
3. 每个实际选项都拆成最小事实条件，逐一核对主体、对象、指标、数值、单位、期间、范围、方向、否定词、例外和并列关系。
4. 原文完整支持全部关键条件才是 supported/true；原文明确相反或关键条件不匹配才是 contradicted/false；证据不足是 unknown/uncertain，不得把未知当错误。
5. 引用必须逐字复制对应证据块中的连续原文，不得拼接不连续文字、改写原文或伪造位置。
6. 数字必须核对指标口径、年份、实际/预测、同比/环比、分子分母、单位和舍入时点。先统一口径和单位，使用未舍入原始值计算，只在最终一步按题目要求舍入。
7. 跨文档题必须按题目给定的文档顺序处理，并覆盖所有涉及文档。不得把第一份文档的事实错配给第二份。
8. 语义近似只有在法律效果、经济含义、责任范围、金额口径和适用对象均不变时才可接受；仅有同义改写、主体省略或上下文可唯一恢复时，不得擅自改答案。
9. 分类题分两步：先判断选项事实是否成立，再判断是否属于题干要求的类别。事实正确但类别不符，标记 category_mismatch，不选择。
10. 最终答案必须与逐项判断、计算结果和 reasoning 完全一致。
11. reasoning 是公开审计摘要，不是隐藏思维链。它必须包含证据位置、关键事实、必要计算或排除依据以及与答案一致的结论；不能写空泛模板、纠错过程、替代答案或未申报事实。
12. 若输出用于比赛提交，prompt_tokens、completion_tokens、total_tokens 只能填写允许模型 API 原始 usage 的逐项合计。没有原始 usage 时必须报错或标记为不可提交。

先核验，再裁决，最后格式化。输出必须严格符合 user message 要求的 JSON schema；不要输出 Markdown 或 JSON 之外的文字。
```

## 4. 高精度 User Prompt 模板

```text
请完成这道题的证据审计和最终作答。

QUESTION
{question_block}

EVIDENCE
{evidence_blocks}

BASELINE（可选，仅用于判断是否发生变化，不是证据）
{baseline_block}

执行顺序：
1. 识别题型、答案槽数量、文档顺序和问题要求。
2. 将每个选项或每个答案槽拆成最小事实条件。
3. 为每个条件寻找直接支持、直接反证或最相关证据；记录证据标签和精确位置。
4. 检查主体、指标、期间、单位、范围、模态词、否定、例外、类别和计算口径。
5. 对计算题列出原始值、公式、代入、运算顺序和最终舍入。
6. 对多选题完成全部选项判断后再生成完整答案；对单选/判断题只保留一个合法字母。
7. 只有在原文直接证明且证据门控通过时，才允许改变 BASELINE；语义争议或证据不足时保持 BASELINE 并列入 unresolved。
8. 生成不含隐藏思维链的公开 reasoning 摘要，并检查其与最终答案逐字一致。

输出严格为一个 JSON 对象：
{
  "answers": ["answer_1", "answer_2", "answer_3", "answer_4"],
  "option_judgement": {
    "A": {
      "atomic_claims": [
        {"claim": "最小事实条件", "status": "supported|contradicted|unknown", "evidence_ids": ["E1"]}
      ],
      "judgement": "true|false|uncertain",
      "relation": "entailed|contradicted|unknown",
      "error_type": "none|missing_evidence|entity_mismatch|metric_mismatch|unit_mismatch|time_mismatch|condition_mismatch|scope_mismatch|negation_mismatch|calculation_mismatch|category_mismatch|semantic_mismatch",
      "citations": [
        {"evidence_id": "E1", "quote": "从证据中逐字复制的连续原文", "location": "doc/page/chunk/table/row/column"}
      ],
      "reasoning": "不超过必要长度的选项级依据",
      "confidence": 0.0
    }
  },
  "calculation": {
    "raw_values": [],
    "formula": "公式或 direct_extraction",
    "steps": [],
    "rounding": "最终舍入规则"
  },
  "reasoning": "包含证据位置、关键事实、必要计算或排除依据，并与 answers 一致",
  "evidence_ids": ["E1"],
  "overall_confidence": 0.0,
  "changed_from_baseline": false,
  "change_reason": "只有直接证据支持时填写",
  "unresolved": []
}
```

对于开放题，`answers` 必须严格对应题目定义的槽位；没有的槽位填空字符串。对于选择题，通常用一个字符串保存按字母升序、去重后的答案，例如 `['AC']`，但最终写入 CSV 前必须由适配器按官方模板拆分。

For open questions, `answers` must follow the question's declared slot order;
unused slots are empty strings. For choice questions, an answer is commonly a
deduplicated ascending string such as `['AC']`, but the competition adapter must
split it according to the official submission schema.

## 5. 领域核验规则 | Domain Checklists

### 5.1 金融合同与债券 | Contracts and Bonds

- 区分发行人、发行主体、担保人、受托管理人、主承销商和承销机构。
- 区分发行规模、注册金额、本期规模、募集资金、余额和偿债资金。
- 核对期限、票面利率、付息/兑付、回售/赎回/转股、评级和募集资金用途。
- “含本数”“不含税”“余额”“本期”“累计”等口径必须保持原文含义。

### 5.2 财务报告与研究报告 | Financial Reports and Research

- 区分实际值、预测值、目标值、期末值、平均值、同比、环比和复合增长。
- 指标修饰语属于指标定义的一部分，例如“扣除客户资金杠杆”不能省略。
- 表格必须同时核对表名、行名、列名、年份、单位和脚注。
- 标题或相邻段落唯一确定的主体、地区或行业可以恢复；存在多个可能主体时标为 uncertain。

### 5.3 保险 | Insurance

- 明确每份保单的被保险人、赔付顺序、费用补偿范围和免赔额类型。
- 区分医保、先前商业保险和后续商业保险，逐步计算并防止重复补偿。
- 核对家庭共享、个人、年度、单次或每次事故免赔额及其触发门槛。
- 缺少赔付顺序或扣除规则时不得自行补算。

### 5.4 监管与公司治理 | Regulation and Governance

- 逐字核对“以上/超过”“以内/少于”“应当/可以”“原则上/无条件”和例外条款。
- 区分“高风险”“较高风险”“高风险以上”等风险等级。
- 角色范围必须按原文限定，不把“其他高级管理人员职务”扩大为“其他职务”。
- 核对生效日期、适用主体、报告/留存期限、普通/特别决议和表决比例。

## 6. Evidence Gate：答案修改门控

当存在基线答案时，使用以下保守规则：

1. 对每个拟修改选项给出直接支持新结论的证据，而非只给出原答案缺少证据。
2. 引用标签必须存在，引用文字必须在对应证据块中逐字匹配。
3. 证据必须覆盖题目要求的全部文档、年份、表格列或计算组成项。
4. 变化必须属于直接矛盾、主体/口径错误、明确计算错误、类别不匹配或明确条件遗漏。
5. 仅有轻微措辞差异、同义改写或模型置信度变化，不得改答案。
6. 证据不足、角色分歧未解决或位置不稳定时保留基线，并记录 `unresolved`。
7. 代码层面将“候选修改”和“受保护基线”分离保存，禁止覆盖原始最优结果。

建议的修改记录：

```json
{
  "qid": "...",
  "old_answer": "...",
  "new_answer": "...",
  "change_class": "direct_contradiction|context_error|calculation_correction|category_mismatch",
  "evidence_ids": ["E2"],
  "location": "doc/page/chunk/table/row/column",
  "verbatim_basis": "逐字引用",
  "confidence": 0.0
}
```

## 7. 两种运行模式 | Two Execution Modes

### 7.1 高精度全量审查 | Full Precision Audit

适用于首次建立答案、关键版本冻结前或高风险题复核：

```text
1. Retriever：按题干和每个选项检索，补充相邻上下文和跨文档片段。
2. Locator：逐选项/答案槽定位原文、表格坐标、年份、单位和缺口。
3. Analyst：独立核验事实关系、分类关系和计算口径。
4. Solver：独立生成答案，不读取基线。
5. Skeptic：从反方向寻找主体、范围、数字、边界和舍入错误。
6. Final Judge：回到原文裁决，不按多数投票；通过 Evidence Gate 才改变基线。
7. Formatter/Validator：确定性规范化答案、reasoning、token 和 CSV。
```

这种模式可以使用多次 Qwen 调用，但每题所有相关调用的原始 usage 必须逐项相加并写入该题。

### 7.2 低 Token 单次调用 | Compact Single Call

适用于证据已经稳定、需要重新生成一份合规提交时：

- 由代码先完成文档解析、检索、证据去重和上下文压缩。
- 每题只发送必要的题干、选项、证据位置和短原文片段。
- 使用一次 Qwen 调用同时生成 `answers` 与公开 `reasoning`。
- `temperature=0` 或平台允许的最低随机性；禁止多轮改写和额外摘要调用。
- 返回 JSON 后只做确定性的 schema、答案槽、引用和一致性检查。
- 任一检查失败就保留缓存/基线或标记失败，不用第二次模型调用偷偷修复。

低 Token 不是把 token 字段改小，而是减少真实上下文和模型调用次数。提交用的 token 必须来自这一次真实 Qwen API 返回的原始 `usage`。

## 8. B 榜提交适配注意事项 | B-Track Adapter Notes

当前项目曾使用的 B 榜要求包括 `qid`、`answer_1` 到 `answer_4`、`reasoning` 以及原始 token 字段和 `summary` 行。但新项目可能不同，适配前必须读取官方 `submit.csv` 或等价模板。

适配器至少应完成：

- 将内部答案映射到官方答案列，不能凭字段名猜测。
- 保证 qid 唯一、完整、顺序符合平台要求。
- 将每题所有相关模型调用的 `prompt_tokens`、`completion_tokens`、`total_tokens` 分别求和。
- 验证每题 `total_tokens = prompt_tokens + completion_tokens`。
- 验证 summary 行等于普通题行求和；没有官方 summary 要求时不要擅自添加。
- 验证 reasoning 长度、证据位置、关键事实、公式和最终答案一致。
- 在提交前生成独立校验报告，并保存本次模型、Prompt 版本、规则版本和运行目录。

## 9. 迁移到 tianchi3 的步骤 | Migration to tianchi3

不要直接复制当前项目的 B 榜代码。按以下顺序适配：

1. 读取 `tianchi3` 的 README、规则文件、`submit.csv`、题目 JSON 和文档样例。
2. 确认题型、答案列、reasoning 要求、token 规则、允许模型和是否需要 summary。
3. 把新数据字段映射到内部协议：question、options、type、source documents、answer slots。
4. 复用本 Prompt 的证据门控、领域核验和计算核验；删除不适用的 A/B 榜特殊分支。
5. 先做 dry-run：检查文档解析、题目覆盖、证据位置和输出 schema，不调用模型。
6. 运行小样本并人工核对原文位置，再运行全量。
7. 生成候选版本、校验报告和可恢复 checkpoint；保护原始结果不被覆盖。
8. 仅将脱敏代码、Prompt、配置模板、测试和文档上传 GitHub；不上传数据集、答案、证据、缓存、运行结果和 API key。

## 10. 可迁移变量 | Parameters to Configure

```text
MODEL_NAME              # 必须符合比赛白名单
BASE_URL                # 官方 API 或允许的兼容端点
API_KEY                 # 仅放本地环境变量或 ignored local_config.py
QUESTION_SCHEMA         # 新比赛题目字段
SUBMISSION_SCHEMA       # 新比赛提交字段
ANSWER_NORMALIZER       # 单选/多选/判断/计算/抽取的确定性规范化
TOKEN_POLICY            # 原始 usage 字段和 summary 规则
RETRIEVAL_POLICY        # top-k、上下文窗口、跨文档覆盖
PROMPT_VERSION          # 用于结果审计和复现
```

### 不应做的事情 | Do Not

- 不把 `train.py` 的运行模式描述成对 Qwen 参数的微调训练，除非实际接入了训练框架。
- 不使用手工答案覆盖模型结果后，再声称答案来自一次模型调用。
- 不手工修改 token 统计以追求分数。
- 不把比赛得分当成企业生产准确率证明。
- 不在 GitHub 提交真实 key、数据集、标准答案、证据、缓存或完整运行结果。

