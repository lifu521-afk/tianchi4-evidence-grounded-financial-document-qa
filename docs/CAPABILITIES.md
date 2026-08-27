# 项目能力与已完成内容 | Capabilities and Completed Work

本文档详细说明该项目**能够完成什么、已经实现什么、如何完成、会生成什么结果，以及当前不能保证什么**。它描述的是当前公开代码的真实能力，不包含已从公开版移除的排行榜探针、手工答案修补和历史临时脚本。

This document explains **what the project can do, what has been implemented,
how each capability works, what artifacts it produces, and what it does not
guarantee**. It describes the current public codebase and excludes historical
leaderboard probes, manual answer patches, and temporary experiment scripts.

## 1. 项目解决的问题 | Problem Addressed

金融长文档问答并不只是把一份 PDF 发送给大模型。真实难点包括：

Financial long-document QA is not simply a matter of sending a PDF to a large
language model. The practical difficulties include:

- 原始材料可能是 PDF、HTML 或 TXT，并包含跨页表格、脚注和扫描布局；
- 一道题可能要求同时比较多份文档；
- 多选题的每个选项可能对应不同证据位置；
- 年份、单位、百分比、条款编号和否定词容易被误读；
- API 上下文有限，不能把全部原文无差别放进 prompt；
- 长任务容易因网络、超时或终端中断而丢失结果；
- 最终得分可能同时受到答案、reasoning、token 和 CSV 格式影响。

- Source material may be PDF, HTML, or TXT and may contain cross-page tables,
  footnotes, and complex layouts.
- A question may require comparison across several documents.
- Each option in a multiple-choice question may require different evidence.
- Reporting periods, units, percentages, clause identifiers, and negations are
  easy to misread.
- API context is limited, so the full corpus cannot be inserted blindly.
- Long-running jobs may be interrupted by network failures, timeouts, or the
  terminal.
- Final evaluation may depend on answers, reasoning, tokens, and CSV format.

本项目将这些问题拆分成确定性预处理、分层检索、证据约束作答、风险复核、运行恢复和提交校验六个阶段。

The project separates these concerns into deterministic preprocessing,
layered retrieval, evidence-constrained generation, risk review, recoverable
execution, and submission validation.

## 2. 能力总览 | Capability Matrix

| 能力 / Capability | 状态 / Status | 说明 / Description |
| --- | --- | --- |
| PDF/HTML/TXT 解析 | 已实现并测试 / Implemented and tested | 提取正文、页码信息和表格文本。Extracts text, page metadata, and table text. |
| A 榜题目预处理 | 已实现 / Implemented | 标准化问题、选项、领域、题型和文档映射。Normalizes questions, options, domains, types, and document mappings. |
| B 榜题目预处理 | 已实现并测试 / Implemented and tested | 支持单选、多选、判断、计算、抽取及多答案槽位。Supports choice, true/false, calculation, extraction, and answer slots. |
| 词法证据检索 | 已实现并测试 / Implemented and tested | 使用词项、数字、条款号和中文 n-gram，不需要 embedding。Uses terms, numbers, clauses, and Chinese n-grams without embeddings. |
| 选项级证据召回 | 已实现并测试 / Implemented and tested | A/B/C/D 独立检索，改善多选题覆盖。Retrieves evidence independently for each option. |
| 跨文档覆盖 | 已实现 / Implemented | 为目标文档保留候选证据。Retains evidence candidates per target document. |
| Qwen/OpenAI-compatible API | 已实现并测试 / Implemented and tested | 支持 DashScope 和兼容中转接口。Supports DashScope and compatible relay endpoints. |
| A 榜全量答题 | 已实现 / Implemented | 生成完整答案、证据和 token 统计。Produces full answers, evidence, and token usage. |
| A 榜风险题重跑 | 已实现 / Implemented | targeted、broad、precision、micro。Provides several risk-based rerun scopes. |
| B 榜多阶段审查 | 已实现并测试 / Implemented and tested | 支持全量 ensemble、低 token 和紧凑 Qwen3.6 流程。Supports full ensemble, low-token, and compact Qwen3.6 workflows. |
| reasoning 生成与校验 | 已实现并测试 / Implemented and tested | 生成可审计摘要并检查答案一致性。Produces auditable summaries and validates consistency. |
| 原始 API usage 汇总 | 已实现并测试 / Implemented and tested | 按题累加所有相关调用。Aggregates all relevant calls per question. |
| checkpoint 与断点续跑 | 已实现并测试 / Implemented and tested | 单题原子缓存、停止信号、重建 checkpoint。Atomic question caches, stop signals, and checkpoint rebuilding. |
| A/B CSV 格式校验 | 已实现并测试 / Implemented and tested | 检查字段、qid、答案、token 和 summary。Checks fields, qids, answers, usage, and summary. |
| GitHub CI | 已实现 / Implemented | Windows 上运行 Python 3.10/3.12 编译和测试。Runs compile and tests on Python 3.10/3.12. |
| Agent 状态图与规划 | 已实现并测试 / Implemented and tested | 显式任务计划、状态、步骤轨迹和失败状态。Explicit plans, state, step traces, and failure states. |
| Tool Use 注册表 | 已实现并测试 / Implemented and tested | 检索和校验工具使用 allow-list、参数检查与调用记录。Retrieval and validation tools use an allow-list, argument checks, and call records. |
| 运行/长期记忆接口 | 已实现并测试 / Implemented and tested | 任务隔离短期记忆和 JSONL 审核经验存储。Task-isolated short-term memory and JSONL reviewed-lesson storage. |
| Reflection 与预算治理 | 已实现并测试 / Implemented and tested | 检查答案形状/证据覆盖并累计原始 usage 与任务预算。Checks answer shape/evidence coverage and aggregates raw usage against a task budget. |
| 离线 Agent 评测 | 已实现并测试 / Implemented and tested | 准确率、证据覆盖、有效答案、复核率和 token 指标。Accuracy, evidence coverage, valid-answer, review-rate, and token metrics. |
| 数据集自动下载 | 未实现 / Not implemented | 数据受比赛授权约束，必须由用户本地准备。Users must provide authorized data locally. |
| 自动获得标准答案 | 未实现 / Not implemented | 系统不包含标准答案或绕过评测的方法。No answer key or evaluation bypass is included. |
| 准确率保证 | 不保证 / Not guaranteed | 无标签数据不能证明每题正确。Unlabeled data cannot prove every answer correct. |

## 3. 已完成的文档处理 | Completed Document Processing

### 3.1 输入发现和映射 | Input discovery and mapping

`agent/preprocess.py` 和 `agent/b_preprocess.py` 负责发现原始文档和题目文件，将官方 `doc_id`、文件路径和问题引用关系转换为统一内部结构。

`agent/preprocess.py` and `agent/b_preprocess.py` discover source documents and
question files and convert official document IDs, file paths, and question
references into normalized internal structures.

已完成内容：

Completed work:

- 识别支持的 PDF、HTML、HTM 和 TXT 文件；
- 使用多种文本编码读取文本文件；
- 记录文档 ID、源路径、页码和抽取统计；
- 对照问题中的文档引用检查缺失 `doc_id`；
- 生成预处理报告，便于先发现数据问题再调用模型。

- Recognizes supported PDF, HTML, HTM, and TXT files.
- Reads text files with encoding fallbacks.
- Records document IDs, source paths, page numbers, and extraction statistics.
- Checks missing document IDs against question references.
- Produces a preprocessing report before any model call is made.

### 3.2 PDF 与表格抽取 | PDF and table extraction

项目使用 `pdfplumber` 和 `pypdf`。PDF 页面正文按页提取，`pdfplumber.extract_tables()` 得到的表格行会转换为可检索文本，降低财报和保险表格中的数字丢失概率。

The project uses `pdfplumber` and `pypdf`. PDF text is extracted page by page,
and rows returned by `pdfplumber.extract_tables()` are converted into
retrievable text to reduce numeric loss in financial and insurance tables.

该能力特别适用于：

This is particularly useful for:

- 财务报告中的年度指标和单位；
- 债券、合同中的利率和期限；
- 保险条款中的给付比例和责任表；
- 研报中的公司对比和预测表格。

- Annual metrics and units in financial reports.
- Rates and terms in bonds and contracts.
- Benefit ratios and coverage tables in insurance documents.
- Company comparisons and forecasts in research reports.

### 3.3 可追踪切片 | Traceable chunking

长文本按照段落和字符 span 切片，每个 chunk 保存文档、页码、顺序和源位置。检索证据因此能够回指原文，而不是只保留一段失去来源的模型上下文。

Long text is chunked by paragraphs and character spans. Each chunk retains its
document, page, order, and source location, allowing retrieved evidence to be
traced back instead of becoming an anonymous prompt fragment.

## 4. 已完成的检索系统 | Completed Retrieval System

### 4.1 无 embedding 词法索引 | Embedding-free lexical index

`agent/retrieval.py` 建立轻量词法索引，综合中文字符、二元片段、英文/数字词项、条款编号和问题中的关键实体进行评分。它不下载向量模型，不产生 embedding API 成本。

`agent/retrieval.py` builds a lightweight lexical index over Chinese
characters, bigrams, English and numeric terms, clause identifiers, and key
entities. It does not download a vector model or incur embedding API cost.

优势 | Advantages:

- 能稳定匹配精确金额、年份、百分比和条款号；
- 本地运行、结果可解释；
- 索引构建快，适合反复调试；
- 不引入比赛可能禁止的 embedding 服务。

- Reliable exact matching for amounts, years, percentages, and clauses.
- Local and interpretable execution.
- Fast rebuilding for repeated experiments.
- Avoids introducing potentially disallowed embedding services.

限制 | Limitation:

语义改写和同义表达的召回能力弱于高质量向量检索，因此项目还加入领域同义词、选项级查询和邻近扩展进行补偿。

Semantic paraphrases are harder to retrieve than with a strong vector model,
so domain synonyms, option-level queries, and neighbor expansion compensate for
this limitation.

### 4.2 四层召回 | Four retrieval layers

1. **题干级召回**：识别主题和主要结论。
2. **选项级召回**：分别搜索每个选项中的数字、限制和否定表达。
3. **文档覆盖召回**：跨文档题为每份目标文档保留证据。
4. **邻近上下文扩展**：补充标题、表头、脚注和后续限定语。

1. **Question-level recall** captures the topic and main conclusion.
2. **Option-level recall** searches numbers, constraints, and negations for
   each option separately.
3. **Document-coverage recall** retains evidence from every requested source.
4. **Neighbor expansion** recovers headings, headers, footnotes, and adjacent
   qualifications.

### 4.3 检索诊断 | Retrieval diagnostics

已经提供以下不调用模型的工具：

The following API-free tools are available:

```powershell
python train.py --mode dry-run
python script\analyze_retrieval_coverage.py
python script\diagnose_question.py --qid fin_a_001
python script\search_chunks.py "关键词"
```

它们可以检查目标文档是否被召回、各选项是否有证据、上下文长度是否合理，以及某个关键词出现在什么位置。

They inspect target-document coverage, option evidence, context length, and the
source locations of a search term.

## 5. 已完成的模型调用层 | Completed Model Integration

### 5.1 OpenAI-compatible 客户端 | OpenAI-compatible client

`agent/qwen_client.py` 使用最小化 HTTP 客户端调用 `/chat/completions`，支持：

`agent/qwen_client.py` calls `/chat/completions` through a minimal HTTP client
and supports:

- DashScope compatible mode；
- 符合规则的兼容 base URL；
- 自定义模型名、温度、超时和重试；
- Qwen `enable_thinking`；
- `max_tokens` 输出限制；
- HTTP 401、404、429 和服务器错误提示；
- API 原始 `usage` 字段读取。

- DashScope compatible mode.
- A rule-compliant compatible base URL.
- Configurable model, temperature, timeout, and retries.
- Qwen `enable_thinking`.
- `max_tokens` output limits.
- Diagnostics for HTTP 401, 404, 429, and server failures.
- Raw API `usage` capture.

配置优先级为：命令行参数 > `local_config.py` > 环境变量 > 默认值。公开仓库只包含 `local_config.example.py`，真实 key 不会提交。

Configuration precedence is CLI arguments, `local_config.py`, environment
variables, and defaults. Only `local_config.example.py` is public; real keys
remain local.

### 5.2 领域化作答 | Domain-aware answering

已经针对五类金融材料设置核查重点：

Implemented checks cover five financial domains:

- **保险**：保险责任、免责、等待期、现金价值、给付条件；
- **监管**：应当/不得/可以、时限、表决比例、条款适用对象；
- **金融合同**：期限、利率、付息、兑付、担保、回售和赎回；
- **财务报告**：报告年份、单位、营业收入、利润、现金流和研发投入；
- **研究报告**：预测年份、指标口径、公司比较、趋势和风险提示。

- **Insurance**: coverage, exclusions, waiting periods, cash value, and benefit
  conditions.
- **Regulation**: obligations, prohibitions, permissions, deadlines, voting
  ratios, and applicable entities.
- **Financial contracts**: maturity, rates, payments, guarantees, put, and call
  terms.
- **Financial reports**: reporting periods, units, revenue, profit, cash flow,
  and R&D.
- **Research reports**: forecast years, metric definitions, comparisons,
  trends, and risks.

### 5.3 结构化输出 | Structured output

模型被要求返回结构化字段，包括最终答案、选项判断、证据 ID、置信度和简要依据。代码会解析 JSON、规范答案并检查最终答案和逐项判断是否一致。

The model is asked to return structured fields such as final answer, per-option
judgments, evidence IDs, confidence, and concise justification. The code parses
JSON, normalizes answers, and checks consistency between option judgments and
the final answer.

## 6. A 榜已完成功能 | Completed A-Track Features

### 6.1 完整生成 | Full generation

```powershell
python train.py --mode full
```

对全部题目进行检索和模型作答，生成：

Retrieves and answers all questions and produces:

- `runs/full/answer.csv`：完整答案和逐题 token；
- `runs/full/evidence.json`：检索证据、模型输出和审计信息；
- checkpoint 和逐题缓存：用于恢复任务。

- `runs/full/answer.csv`: complete answers and per-question usage.
- `runs/full/evidence.json`: evidence, raw outputs, and audit information.
- Checkpoints and per-question caches for recovery.

### 6.2 风险题重跑 | Risk-based reruns

```powershell
python train.py --mode targeted
python train.py --mode broad
python train.py --mode precision
```

- `targeted` 选择结构风险较高的少量题；
- `broad` 扩大到更多多选、跨文档和复杂判断题；
- `precision` 对最终仍有风险的题使用更强复核，并默认不覆盖根目录答案。

- `targeted` selects a smaller set of structurally risky questions.
- `broad` expands review to more multi-choice, cross-document, and compound
  questions.
- `precision` applies stronger review to remaining risks and does not overwrite
  the root answer by default.

风险规则包括：答案与选项判断冲突、选择多个选项但证据不足、跨文档题覆盖不足、财报数字推理缺少数字、监管题缺少条款证据等。

Risk signals include conflicts between final answers and option judgments,
selected options without evidence, insufficient cross-document coverage,
numeric questions without numeric reasoning, and regulatory answers without
clause evidence.

### 6.3 低 token 模式 | Low-token modes

```powershell
python train.py --mode low-token
python train.py --mode micro
python train.py --mode super-low
```

这些模式通过缩短证据、限制输出、关闭 thinking 或只处理风险题降低 token。它们用于成本实验，不保证总能保持完整模式的准确率，因此结果默认隔离保存。

These modes reduce token use by shortening evidence, limiting output,
disabling thinking, or processing only risky questions. They are cost
experiments and do not guarantee full-mode accuracy, so outputs are isolated by
default.

## 7. B 榜已完成功能 | Completed B-Track Features

### 7.1 题型和答案槽位 | Types and answer slots

B 榜预处理支持官方字段 `qid`、`domain`、`split`、`question`、`type` 和 `options`，并处理：

B-track preprocessing supports the official `qid`, `domain`, `split`,
`question`, `type`, and `options` fields and handles:

- 单选题 / single-choice
- 多选题 / multiple-choice
- 判断题 / true/false
- 计算题 / calculation
- 抽取题 / extraction
- `answer_1` 至 `answer_4` 多答案槽位 / multiple answer slots

### 7.2 全量多阶段审查 | Full multi-stage audit

```powershell
python script\run_b_full_ensemble.py
```

该流程对每题执行证据定位、初始判断和最终裁决等阶段，适合精度优先实验。所有阶段 usage 会按题累加，任务支持缓存和恢复。

This workflow performs evidence location, initial judgment, and final
adjudication stages per question. It targets accuracy-first experiments,
aggregates usage by question, and supports caching and resume.

### 7.3 低 token 与紧凑流程 | Low-token and compact workflows

```powershell
python script\run_b_low_token.py
python script\run_b_qwen36_compact.py
```

`run_b_low_token.py` 减少阶段和上下文；`run_b_qwen36_compact.py` 使用单题紧凑证据、答案和 reasoning 调用，适合在真实 usage 完整的前提下降低成本。

`run_b_low_token.py` reduces stages and context. `run_b_qwen36_compact.py` uses
a compact per-question evidence, answer, and reasoning call to reduce cost
while preserving complete raw usage.

### 7.4 reasoning 与 usage 合规 | Reasoning and usage compliance

`agent/b_compliant_submission.py` 和 `script/run_b_compliant_reasoning.py` 已完成：

`agent/b_compliant_submission.py` and
`script/run_b_compliant_reasoning.py` implement:

- reasoning 非空和基本长度检查；
- reasoning 与答案字段一致性检查；
- 原始 API usage 必填；
- 同题多次调用逐项求和；
- `total_tokens = prompt_tokens + completion_tokens`；
- 普通题目行和 summary 行汇总一致；
- qid 缺失、重复和多余检查。

- Non-empty and minimum-length reasoning checks.
- Reasoning-to-answer consistency checks.
- Mandatory raw API usage.
- Aggregation of every call associated with a question.
- `total_tokens = prompt_tokens + completion_tokens`.
- Agreement between question rows and summary totals.
- Missing, duplicate, and extra qid checks.

## 8. 已完成的运行恢复 | Completed Recovery Features

### 8.1 每题原子缓存 | Atomic question cache

每完成一题，系统先写独立 JSON，再刷新 checkpoint。即使主进程中断，已完成题仍可恢复。

After each question, the system writes an independent JSON record before
refreshing the checkpoint. Completed questions survive process interruption.

### 8.2 安全停止 | Safe stop

```powershell
python train.py --mode stop
python train.py --mode clear-stop
python train.py --mode resume
```

停止信号让主任务在当前题完成并保存后退出，而不是在写文件过程中强制终止。也可以在当前终端使用 `Ctrl+C`。

The stop signal asks the runner to exit after the current question is safely
saved instead of terminating during a write. `Ctrl+C` is also supported.

### 8.3 checkpoint 重建 | Checkpoint rebuilding

```powershell
python script\check_progress.py
python script\rebuild_checkpoint.py
```

当汇总 checkpoint 损坏但单题缓存仍存在时，可以重建答案和证据 checkpoint，避免重新调用已完成题。

If an aggregate checkpoint is damaged while per-question caches remain, answer
and evidence checkpoints can be rebuilt without repeating completed API calls.

## 9. 已完成的提交校验 | Completed Submission Validation

### 9.1 A 榜校验 | A-track validation

`script/check_submission.py` 检查：

`script/check_submission.py` checks:

- 表头是否完全匹配；
- summary 是否位于第一行；
- qid 是否缺失、重复或未知；
- 单选、多选和判断答案是否合法；
- 多选字母是否去重并排序；
- token 是否为非负整数；
- 每题和 summary 的 token 加和是否一致。

- Exact headers.
- Summary placement.
- Missing, duplicate, or unknown qids.
- Valid single-choice, multiple-choice, and true/false answers.
- Unique and sorted multi-choice letters.
- Non-negative integer usage.
- Per-question and summary token equations.

### 9.2 B 榜校验 | B-track validation

```powershell
python script\check_b_submission.py --file path\to\answer.csv
python script\check_b_compliant_submission.py --file path\to\answer.csv
```

除答案槽位外，还检查 reasoning、官方模板字段和原始 usage 合规。规则详情见 `docs/COMPETITION_RULES.md`。

In addition to answer slots, these tools validate reasoning, official schema,
and raw usage compliance. See `docs/COMPETITION_RULES.md` for details.

## 10. 已完成的工程化内容 | Completed Engineering Work

### 10.1 VS Code 入口 | VS Code integration

`.vscode/tasks.json` 和 `.vscode/launch.json` 提供 targeted、broad、precision、full、low-token、resume、preprocess、dry-run、stop 和 check 等常用任务。

`.vscode/tasks.json` and `.vscode/launch.json` provide common tasks for
targeted, broad, precision, full, low-token, resume, preprocessing, dry-run,
stop, and validation.

### 10.2 自动化测试 | Automated tests

公开版包含测试，覆盖：

The public release includes tests for:

- 答案约束和 evidence gate；
- PDF 表格文本转换；
- 检索锚点；
- Qwen 客户端请求和错误处理；
- A/B 榜提示词；
- B 榜预处理、ensemble、reasoning 和提交格式；
- token 汇总与恢复逻辑。

- Answer constraints and evidence gating.
- PDF table conversion.
- Retrieval anchors.
- Qwen client requests and error handling.
- A/B prompt behavior.
- B preprocessing, ensemble, reasoning, and submission formats.
- Token aggregation and recovery logic.

### 10.3 GitHub CI | GitHub CI

`.github/workflows/ci.yml` 在 Windows 上使用 Python 3.10 和 3.12：

`.github/workflows/ci.yml` uses Python 3.10 and 3.12 on Windows to:

1. 安装开发依赖；
2. 编译 `train.py`、`agent/` 和 `script/`；
3. 运行 pytest；
4. 检查数据集、答案、证据、缓存和本地配置没有被提交。

1. Install development dependencies.
2. Compile `train.py`, `agent/`, and `script/`.
3. Run pytest.
4. Verify datasets, answers, evidence, caches, and local credentials are absent.

### 10.4 安全和公开发布 | Security and public release

已完成的安全整理：

Completed security work:

- `local_config.py` 被忽略，只公开空 key 示例；
- `.env`、数据集、运行目录和提交结果被忽略；
- `SECURITY.md` 说明 key 暴露后的撤销要求；
- GitHub 发布历史重建为单一干净 commit，不包含早期临时文件；
- CI 再次检查本地专用文件未进入仓库。

- `local_config.py` is ignored and only an empty-key example is public.
- `.env`, datasets, runs, and submissions are ignored.
- `SECURITY.md` explains credential revocation after exposure.
- The GitHub release history was rebuilt as one clean commit without temporary
  historical files.
- CI rechecks that local-only files are absent.

## 11. 项目会生成什么 | Generated Artifacts

运行过程中可能在本地生成：

The project may generate the following local artifacts:

| 文件 / Artifact | 用途 / Purpose | 是否上传 / Public? |
| --- | --- | --- |
| `processed_data*/questions.jsonl` | 标准化问题 / normalized questions | 否 / No |
| `processed_data*/documents.jsonl` | 文档映射 / document map | 否 / No |
| `processed_data*/chunks.jsonl` | 检索片段 / retrieval chunks | 否 / No |
| `preprocess_report.json` | 数据完整性报告 / data integrity report | 否 / No |
| `runs/<mode>/answer.csv` | 候选提交 / candidate submission | 否 / No |
| `runs/<mode>/evidence.json` | 证据审计 / evidence audit | 否 / No |
| `answer.checkpoint.csv` | 增量答案 / incremental answers | 否 / No |
| `evidence.checkpoint.jsonl` | 增量证据 / incremental evidence | 否 / No |
| `cache/questions/<qid>.json` | 单题原子缓存 / atomic cache | 否 / No |

这些文件包含比赛数据、模型输出或提交信息，只能保存在授权环境中。

These files contain competition data, model output, or submission information
and must remain in an authorized local environment.

## 12. 已取得的实验进展 | Experimental Progress

项目记录显示，A 榜从早期约 58 分逐步提升到 89 分以上；B 榜低 token 版本约 86.7 分，证据复核和合规版本曾达到约 91.7 分。

Project records show an A-track progression from roughly 58 to above 89, a
B-track low-token result around 86.7, and an evidence-reviewed compliant result
around 91.7.

最有价值的改进是：

The most valuable improvements were:

1. 选项级独立检索；
2. 跨文档覆盖；
3. 表格文本抽取；
4. 领域化核查；
5. evidence gate 基线保护；
6. token 与提交格式前置设计；
7. 风险题局部复核而不是无条件全量重跑。

1. Independent option-level retrieval.
2. Cross-document coverage.
3. Table text extraction.
4. Domain-aware checks.
5. Baseline protection through evidence gating.
6. Token and submission format as first-class design constraints.
7. Localized risk review instead of unconditional full reruns.

这些分数仅是实验记录，不是代码对未来数据或模型版本的准确率保证。

These scores are experimental records, not guarantees for future data or model
versions.

## 13. 当前没有完成或不能保证的内容 | Not Implemented or Not Guaranteed

### 没有上传数据集 | No dataset distribution

由于数据许可和比赛合规要求，公开仓库没有数据集、标准答案、历史提交和 evidence。用户必须自行准备官方授权数据。

For licensing and competition compliance, the repository contains no dataset,
answer key, historical submissions, or evidence. Users must provide authorized
official data.

### 不保证全部题目正确 | No perfect-accuracy guarantee

系统能提高证据覆盖和减少结构错误，但没有标准答案时无法证明每题正确。模型仍可能误读复杂表格、隐含条件和语义改写。

The system improves evidence coverage and reduces structural errors, but cannot
prove every answer without labels. Models may still misread complex tables,
implicit conditions, and semantic paraphrases.

### 不包含训练大模型 | No foundation-model training

项目中的“训练模式”指比赛推理、答案生成和候选实验，不是对 Qwen 参数进行梯度训练或微调。

The term “training mode” in this project refers to competition inference,
answer generation, and candidate experiments. It does not train or fine-tune
Qwen model parameters.

### 不自动判断最新比赛规则 | No automatic rule updates

比赛字段、白名单模型和评分公式可能变化。`docs/COMPETITION_RULES.md` 是工程摘要，提交前仍需核对官方最新公告和模板。

Competition fields, allowed models, and scoring formulas may change.
`docs/COMPETITION_RULES.md` is an engineering summary; the latest official
announcement and template remain authoritative.

## 14. 推荐使用顺序 | Recommended Workflow

```powershell
# 1. 安装并配置 / install and configure
python -m pip install -r requirements-dev.txt
Copy-Item local_config.example.py local_config.py

# 2. 检查数据 / inspect data
python script\inspect_dataset.py

# 3. 预处理 / preprocess
python train.py --mode preprocess

# 4. 不调用 API 检查检索 / retrieval dry run
python train.py --mode dry-run

# 5. 少量题验证配置 / small API smoke test
python script\run_answer.py --limit 3 --no-sync

# 6. 完整运行 / full run
python train.py --mode full --no-sync

# 7. 检查风险并局部复核 / targeted review
python train.py --mode targeted --no-sync

# 8. 校验候选文件 / validate candidate
python script\check_submission.py --file runs\targeted\answer.csv
```

B 榜应使用 B 榜专用入口和校验器，不能直接把 A 榜 CSV 当作 B 榜提交。

The B track must use B-specific runners and validators. An A-track CSV must not
be submitted as a B-track file.

## 15. 公开版交付清单 | Public Deliverables

当前公开项目已经完成以下交付：

The current public release includes:

- 核心 Agent 代码 / core agent code
- A/B 榜运行脚本 / A/B runners
- A/B 提交校验器 / A/B validators
- 测试套件 / test suite
- VS Code 任务 / VS Code tasks
- GitHub Actions CI / GitHub Actions CI
- API key 示例配置 / credential-free configuration example
- 中英文 README / bilingual README
- 中英文赛题规则 / bilingual competition rules
- 中英文数据布局 / bilingual data layout
- 中英文系统架构 / bilingual architecture
- 中英文项目总结 / bilingual project summary
- 本文能力与完成内容清单 / this detailed capability inventory

项目源代码已经整理为不包含数据和凭据的干净 Git 发布版本。GitHub 目标仓库需要先在账号下创建，之后才能完成远程推送。

The source has been prepared as a clean Git release without data or
credentials. The target GitHub repository must exist under the account before
the final remote push can be completed.
