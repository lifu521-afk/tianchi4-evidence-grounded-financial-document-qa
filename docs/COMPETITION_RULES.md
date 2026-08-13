# 赛题规则与工程合规说明 | Competition Rules and Engineering Compliance

> 本文档是对比赛规则的工程化整理，不替代赛事平台最新公告。提交前必须以官方模板和最新规则为准。
>
> This is an engineering-oriented summary and does not replace the latest official announcement. Always use the official template and current rules before submission.

## 1. 任务目标 | Task Objective

比赛要求参赛者根据给定金融原始文档回答问题。系统不仅要输出答案，还要保留能够支持答案的证据和符合平台要求的 token 记录。

The task is to answer questions using the supplied financial source documents. A robust system must produce the answer, retain supporting evidence, and report token usage in the format required by the platform.

核心原则 | Core principles:

- 先定位原文，再判断答案；先 retrieve, then decide。
- 每个选项都要单独检查，尤其是多选题；inspect every option, especially for multi-choice questions。
- 数字、年份、单位、比例、条款义务必须与原文口径一致；preserve the source's numbers, periods, units, percentages, and obligations。
- 证据不足时不能把模型猜测写成确定事实；do not convert insufficient evidence into a confident answer。

## 2. A 榜规则摘要 | A-Track Summary

- 题目按保险、监管、金融合同、财务报告、研究报告等领域组织。
- 常见题型包括单选、多选和判断题。
- 多选题通常要求完整匹配，漏选、错选和多选都可能计错。
- 判断题必须使用平台规定的判断答案编码，不能自行输出自然语言替代值。
- 每个 `qid` 只能出现一次，题目行数必须与官方题目数一致。

- Questions are grouped into domains such as insurance, regulation, financial contracts, financial reports, and research reports.
- Common types include single-choice, multiple-choice, and true/false.
- Multiple-choice answers generally require an exact set match; missing, extra, or incorrect options may all be wrong.
- True/false answers must use the platform's required encoding.
- Each `qid` must appear exactly once and the number of question rows must match the official question set.

典型 A 榜字段 | Typical A-track fields:

```text
qid,answer,prompt_tokens,completion_tokens,total_tokens
```

第一行通常是 `summary` 汇总行。每题的 `total_tokens` 应满足：

The first row is commonly a `summary` row. For every question:

```text
total_tokens = prompt_tokens + completion_tokens
```

summary 中的三类 token 还应等于所有普通题目行对应字段的总和。

The three summary token values must also equal the sums of the corresponding values across all question rows.

## 3. B 榜提交格式 | B-Track Submission Format

B 榜提交文件至少包含以下表头：

The B-track submission must contain at least the following header:

```text
qid,answer_1,answer_2,answer_3,answer_4,prompt_tokens,completion_tokens,total_tokens,reasoning
```

### 普通题目行 | Question rows

- 每道题一行，`qid` 必须与官方题目编号对应。
- 单答案填入 `answer_1`，其余答案列留空。
- 多答案按题目要求的顺序放入 `answer_1` 至 `answer_4`。
- 无对应答案的槽位必须为空，不能填 `None`、`null` 或额外文字。
- `reasoning` 必须与本题答案和证据一致。

- One row is required per question and `qid` must match the official identifier.
- Put a single answer in `answer_1` and leave the remaining slots empty.
- Put multiple answers in `answer_1` through `answer_4` in the required order.
- Unused slots must be empty, not `None`, `null`, or explanatory text.
- `reasoning` must agree with the answer and the cited evidence for that question.

### summary 行 | Summary row

建议第一行使用 `qid=summary`，答案列和 reasoning 留空：

The recommended first row is `qid=summary` with empty answer and reasoning fields:

```text
summary,,,,,prompt_tokens_sum,completion_tokens_sum,total_tokens_sum,
```

必须满足：

The following must hold:

```text
summary.prompt_tokens = sum(question.prompt_tokens)
summary.completion_tokens = sum(question.completion_tokens)
summary.total_tokens = sum(question.total_tokens)
summary.total_tokens = summary.prompt_tokens + summary.completion_tokens
```

## 4. Token 统计 | Token Accounting

token 统计必须来源于允许模型 API 返回的原始 `usage` 字段。不能使用其他 tokenizer 重新估算，也不能为了提高 TokenScore 手动改小数值。

Token counts must come from the raw `usage` fields returned by an allowed model API. Do not replace them with another tokenizer or manually lower the values to improve the token score.

同一道题的以下调用都应计入该题：

The following calls must be included when they are made for a question:

- 证据摘要 / evidence summarization
- 上下文压缩 / context compression
- 证据判断 / evidence verification
- 答案生成 / answer generation
- 自检和复核 / self-check and review
- reasoning 生成或改写 / reasoning generation or rewriting
- 失败重试 / retry calls

如果发生多次调用：

For multiple calls:

```text
prompt_tokens     = sum(prompt_tokens_k)
completion_tokens = sum(completion_tokens_k)
total_tokens      = sum(total_tokens_k)
```

不能只记录最后一次调用。任何用于生成最终提交内容的模型调用都必须进入题目 token 账本。

Do not record only the last call. Every model call used to produce the final submission must be included in the per-question ledger.

## 5. reasoning 要求 | Reasoning Requirements

reasoning 是可审计的推理摘要，不要求提交完整思维链。建议采用“证据位置 + 关键事实 + 判断/计算 + 结论”的结构。

`reasoning` is an auditable reasoning summary, not a request for hidden chain-of-thought. A useful structure is “source location + key fact + decision/calculation + conclusion”.

合格摘要应：

A valid summary should:

1. 指出使用的文档、页码、表格或条款位置；identify the document, page, table, or clause location;
2. 提取与答案直接相关的事实；state the facts directly relevant to the answer;
3. 对多选题逐项说明支持或排除原因；explain support or exclusion for each option in multi-choice questions;
4. 对计算题展示必要的公式和代入值；show the necessary formula and values for calculations;
5. 与最终答案一致且长度足以支持其内容；agree with the final answer and be sufficiently detailed.

不合格摘要包括空文本、只重复答案、与题目无关、和答案矛盾、引用题目没有提供的信息，或用极短 token 声称完成了复杂审查。

Invalid summaries include empty text, answer-only repetition, irrelevant text, contradictions, unsupported facts, or a very short output claiming a complex audit.

## 6. 评分与策略 | Scoring and Strategy

B 榜曾采用准确率、推理过程和 Token 效率的综合评分；不同阶段可能使用不同权重和预算。工程策略应按当前公告调整，不能把旧阶段的权重当作永久规则。

The B track has used combined scoring over answer accuracy, reasoning quality, and token efficiency. Weights and budgets may change between stages, so the current announcement takes precedence over historical settings.

无论评分公式如何变化，优先级都应是：

Regardless of the formula, the priority order should be:

1. 原文证据支持的答案 / source-supported answers
2. 一致且具体的 reasoning / consistent, specific reasoning
3. 真实完整的 token 统计 / complete raw usage accounting
4. 严格符合模板的 CSV / exact CSV compliance

## 7. 合规检查清单 | Compliance Checklist

提交前逐项检查：

Before submission, verify:

- [ ] 使用赛事允许的 Qwen 模型和 API / an allowed Qwen model and API is used。
- [ ] API key 未写入仓库或提交文件 / no API key is committed or submitted。
- [ ] 题目数量、qid 唯一性和顺序正确 / question count, qid uniqueness, and ordering are correct。
- [ ] 表头与官方模板完全一致 / the header exactly matches the official template。
- [ ] 所有答案槽位格式正确 / all answer slots are valid。
- [ ] 每题 token 非负且为整数 / per-question tokens are non-negative integers。
- [ ] 每题和 summary 的 token 加和一致 / per-question and summary sums agree。
- [ ] reasoning 与答案和证据一致 / reasoning agrees with answers and evidence。
- [ ] 已运行对应校验脚本 / the appropriate validator has been run。

```powershell
python script\check_submission.py --file answer.csv
python script\check_b_submission.py --file runs\b_run\answer.csv
python script\check_b_compliant_submission.py --file runs\b_run\answer.csv
```
