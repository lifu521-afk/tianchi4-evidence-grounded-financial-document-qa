# 天池金融长文档问答 Agent | Tianchi Long-Document QA Agent

一个面向金融长文档问答比赛的证据增强型 Qwen Agent，覆盖 A 榜和 B 榜。项目将原始文档解析、结构化切分、关键词检索、模型作答、token 统计、断点续跑和提交校验组织成一条可审计、可复现的流程。

An evidence-grounded Qwen agent for financial long-document question answering. It supports both the A and B tracks and turns document parsing, structured chunking, lexical retrieval, answer generation, token accounting, checkpointing, and submission validation into one auditable and reproducible pipeline.

## 项目定位 | Project Scope

本仓库保存**方法、代码和文档**，不保存比赛数据、模型输出或 API key。数据集必须由使用者根据比赛授权放在本地目录，目录结构见 [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md)；真实 key 通过 `local_config.py` 或环境变量配置。

This repository contains the **method, source code, and documentation**. It intentionally excludes competition data, model outputs, caches, submission files, and API keys. Users must provide authorized datasets locally according to [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md) and configure credentials through `local_config.py` or environment variables.

完整功能清单、已完成模块、输入输出、运行模式和当前局限见 [docs/CAPABILITIES.md](docs/CAPABILITIES.md)。

See [docs/CAPABILITIES.md](docs/CAPABILITIES.md) for the complete capability
inventory, implemented modules, inputs and outputs, run modes, and current
limitations.

## 项目主要针对的问题（结合赛题规则） | Rule-Driven Problems Addressed

本项目主要针对的不是普通的开放式聊天，而是“**必须依据指定金融原文作答，并同时满足准确率、推理摘要、Token 统计和 CSV 提交格式要求**”的长文档问答任务。赛题规则决定了系统必须同时解决以下问题：

This project targets more than ordinary open-ended chat. It addresses a
long-document QA task in which answers must be grounded in the supplied
financial sources while satisfying **accuracy, reasoning-summary, token
accounting, and CSV submission requirements**. The rules create the following
engineering problems:

### 1. 长文档中快速找到正确原文 | Finding the Right Evidence in Long Documents

赛题答案必须来自给定文档，而不是来自模型的常识或猜测。金融原文通常很长，关键信息可能位于不同页、表格、脚注或多个文档中。如果直接把整份文档放进 prompt，容易超出上下文限制，也会使模型忽略真正相关的段落。

The answer must come from the supplied documents rather than model memory or
guesswork. Financial sources are long, and the decisive fact may be in a
different page, table, footnote, or document. Sending an entire document to the
prompt can exceed the context budget and dilute the relevant evidence.

因此项目实现文档解析、表格保留、可追踪切片和分层检索，优先把与题目相关的原文位置交给模型。

The project therefore implements document parsing, table preservation,
traceable chunking, and layered retrieval to provide the model with the most
relevant source locations first.

### 2. 多选题不能漏选、错选或多选 | Exact Multiple-Choice Matching

按照赛题规则，多选题通常需要与标准答案完整匹配，漏选、错选和多选都可能判错。只让模型直接输出一个整体答案，容易出现“选项判断正确但最终组合错误”，或者某个选项没有被单独核实。

Under the rules, multiple-choice answers generally require an exact match.
Missing, incorrect, or extra options may all be scored wrong. Asking the model
for only one final string can produce an incorrect combination even when some
individual judgments are correct.

项目让 A/B/C/D 选项分别参与检索和判断，再进行去重、排序和最终答案组合，从流程上降低漏选、误选和字母顺序错误。

The project retrieves and judges A/B/C/D options independently, then performs
deduplication, sorting, and final answer assembly to reduce omissions, false
selections, and ordering errors.

### 3. 金融数字和条款容易被误读 | Financial Numbers and Clauses Are Easy to Misread

赛题中的错误经常来自年份、报告期、单位、百分比、金额、指标口径、条款编号、时限或“应当/不得/可以”等义务强度，而不是来自普通语言理解。一个数字看似相同，可能属于不同年份或不同单位。

Many errors come from reporting periods, units, percentages, amounts, metric
definitions, clause identifiers, deadlines, or obligation strength such as
“must”, “must not”, and “may”. A matching number may still belong to a
different year or unit.

项目针对保险、监管、金融合同、财务报告和研究报告设置领域化核查规则，要求模型同时核对数值、口径、时间和适用范围。

The project adds domain-aware checks for insurance, regulation, financial
contracts, financial reports, and research reports. The model must verify the
value, definition, period, and scope together.

### 4. B 榜 reasoning 不能空泛 | B-Track Reasoning Must Be Auditable

B 榜规则要求提交 `reasoning`，并可能根据逻辑连贯性、论证完整性和表达清晰度评分。空文本、只重复答案、与题目无关或与答案矛盾的摘要可能失分。

The B-track requires a `reasoning` field and may score logical coherence,
completeness, and clarity. Empty, answer-only, irrelevant, or contradictory
summaries can receive no credit.

项目生成“证据位置 + 关键事实 + 必要计算或判断 + 最终结论”的简短摘要，不要求提交完整思维链，但要求摘要能够支持最终答案并适合审计。

The project produces a concise summary in the form “evidence location + key
fact + necessary calculation or judgment + conclusion”. It does not require a
hidden chain of thought, but the summary must support the submitted answer and
remain auditable.

### 5. Token 必须真实、完整且逐题统计 | Token Usage Must Be Real and Complete

B 榜规则要求 `prompt_tokens`、`completion_tokens` 和 `total_tokens` 直接来自允许模型 API 的原始 `usage` 字段。同一道题发生多次调用时，证据摘要、答案生成、复核、reasoning 和重试等相关调用都必须计入，不能只记录最后一次调用或手动估算。

The B-track requires `prompt_tokens`, `completion_tokens`, and `total_tokens`
to come directly from the raw `usage` fields returned by an allowed model API.
When a question uses multiple calls, related evidence, answer, review,
reasoning, and retry calls must all be included. Recording only the last call or
manually estimating usage is not compliant.

项目为每题建立 usage 账本，按题汇总所有相关调用，并检查：

The project keeps a per-question usage ledger and validates:

```text
total_tokens = prompt_tokens + completion_tokens
summary.total_tokens = sum(question.total_tokens)
```

### 6. 答案正确也可能因 CSV 格式失分 | Correct Answers Can Still Fail Validation

赛题要求固定表头、逐题一行、合法 `qid`、正确答案槽位和 summary 汇总。A 榜与 B 榜字段不同，不能把 A 榜的 `answer` 列直接当作 B 榜的 `answer_1` 至 `answer_4`，也不能遗漏 B 榜 `reasoning`。

The rules require an exact header, one row per question, valid qids, correct
answer slots, and consistent summary totals. A- and B-track schemas differ: an
A-track `answer` column cannot be submitted directly as B-track
`answer_1`-`answer_4`, and B-track `reasoning` cannot be omitted.

项目提供独立的 A 榜和 B 榜校验器，在调用模型和提交平台之前检查字段、答案编码、qid、token、reasoning 和汇总行。

Separate A- and B-track validators check fields, answer encoding, qids, token
usage, reasoning, and summary rows before a file is sent to the platform.

### 7. 长任务中断会造成重复消耗 | Long Jobs Can Be Interrupted

100 道长文档题目的完整运行可能耗时较长，网络错误、API 超时、电脑休眠或终端关闭都会导致任务中断。如果没有缓存，续跑时可能重新调用已经完成的题目，浪费 token 并增加结果不一致的风险。

Running a full set of long-document questions can take time. Network errors,
API timeouts, sleep, or terminal closure may interrupt the process. Without
caching, resuming may repeat completed calls, wasting tokens and increasing
result variance.

项目提供单题缓存、checkpoint、停止信号和 resume 模式，使任务可以在当前题完成保存后安全停止，并从未完成题继续。

The project provides per-question caches, checkpoints, stop signals, and a
resume mode. A run can stop after safely saving the current question and
continue with unfinished questions later.

### 8. 精度提升不能靠无依据地整体改答案 | Accuracy Improvements Must Be Evidence-Based

多轮复核不一定提升准确率，模型有可能把已经正确的基线答案改错。赛题又是逐题精确匹配，因此“所有题重新生成一遍”并不等于更高分。

More review calls do not automatically improve accuracy. A model may change a
correct baseline answer into a wrong one, and exact per-question scoring makes
such regressions costly. Rerunning every question is not the same as improving
accuracy.

项目使用风险题筛选、证据门控和基线保护：只有新答案得到直接证据支持，或旧答案被直接证据否定时，才允许修改基线；低风险题保留已验证答案，高风险题才进行额外复核。

The project uses risk filtering, evidence gating, and baseline protection. A
new answer is accepted only when directly supported by evidence or when the
baseline is directly contradicted. Low-risk answers are preserved while
high-risk questions receive additional review.

### 9. 公开项目必须兼顾复现和数据安全 | Reproducibility Must Coexist with Data Safety

项目需要展示方法、代码和实验结论，但比赛数据、标准答案、evidence、提交结果和 API key 不能直接上传公开仓库。因此仓库提供数据目录说明和空 key 配置模板，使用者在本地准备授权数据后即可复现实验流程。

The project must expose its method, code, and experimental conclusions without
publishing competition data, answer keys, evidence, submission files, or API
keys. The repository therefore provides a local data-layout guide and a
credential-free configuration template so users can reproduce the workflow
with authorized local inputs.

## 这个项目做了什么 | What This Project Does

### 1. 文档结构化 | Document Structuring

解析 PDF、HTML 和 TXT，尽可能保留文档 ID、页码、段落、表格文本和字符位置，再按可追踪的 span 切分成检索片段。每个片段都能回指原始文档位置，便于审计和定位证据。

The pipeline parses PDF, HTML, and TXT files while preserving document IDs, page numbers, paragraphs, table text, and character spans where possible. Text is split into traceable retrieval chunks so each piece of evidence can be linked back to its source location.

### 2. 分层证据检索 | Layered Evidence Retrieval

项目不依赖 embedding 模型，而是使用关键词、数字、条款编号、中文字符 n-gram 和 BM25 风格评分。检索包括四层：

1. 题干级召回：找到与整道题最相关的文本。
2. 选项级召回：A/B/C/D 每个选项独立检索，减少细节被高频题干词淹没。
3. 文档覆盖召回：跨文档题尽量覆盖题目要求比较的每个文档。
4. 邻近上下文扩展：补充命中片段的前后片段，减少证据落在切分边界外。

The project does not require an embedding model. It uses keywords, numbers, clause identifiers, Chinese character n-grams, and BM25-style scoring. Retrieval has four layers: question-level recall, option-level recall, document-coverage recall, and neighboring-context expansion.

### 3. 领域化提示词 | Domain-Aware Prompts

针对不同金融文本类型强化不同的核查重点：保险责任与免责、监管条款与时限、合同期限与利率、财报年份与单位、研报指标口径与预测年份。模型必须区分“原文明确支持”“原文明确否定”和“证据不足”。

Prompts emphasize domain-specific checks: insurance coverage and exclusions, regulatory clauses and deadlines, contract terms and rates, reporting periods and units, and research-report metrics and forecast years. The model is instructed to distinguish direct support, direct contradiction, and insufficient evidence.

### 4. 答案与提交约束 | Answer and Submission Constraints

系统按题型处理单选、多选、判断、计算和抽取题，统一答案格式，校验多选去重排序、答案槽位、CSV 表头、逐题 token 以及 summary 汇总，避免“答案本身正确但提交格式不合规”。

The system handles single-choice, multiple-choice, true/false, calculation, and extraction questions. It normalizes answers and validates multiple-choice ordering, answer slots, CSV headers, per-question usage, and summary totals.

### 5. 精度与成本控制 | Accuracy and Cost Control

支持以下运行方式：

| 模式 | 作用 |
| --- | --- |
| `targeted` | 只重跑结构风险较高的题，保留基础答案，节省 token。 |
| `broad` | 重跑更大范围的风险题，适合扩大复核范围。 |
| `full` | 从头生成完整答案。 |
| `precision` | 对剩余结构风险做更强复核，结果默认隔离保存。 |
| `low-token` | 缩短证据上下文并关闭额外思考，降低成本。 |
| `resume` | 从 checkpoint 和单题缓存继续运行。 |
| `dry-run` | 只检查检索，不调用模型。 |

Available modes include targeted risk reruns, broad review, full generation, precision review, low-token generation, resume from checkpoints, and retrieval-only dry runs.

### 6. 可恢复和可审计运行 | Recoverable and Auditable Runs

每道题完成后写入单题缓存和 checkpoint；任务可通过 `Ctrl+C` 或停止信号安全暂停，后续跳过已完成题目。证据、原始模型输出和 usage 记录用于复盘，不应被手工伪造或覆盖。

Each completed question is written to an atomic cache and checkpoint. A run can be safely interrupted with `Ctrl+C` or a stop signal and resumed without repeating completed questions. Evidence, raw model output, and usage records are retained for review and must not be manually fabricated.

## 最有价值的提升 | Most Valuable Improvements

本地实验表明，最有效的提升来自结构化和风险控制，而不是无条件增加模型轮数：

Local experiments suggest that the largest gains came from structure and risk control rather than blindly adding more model calls:

1. **选项级召回 | Option-level retrieval**：显著改善多选题和细粒度判断题的证据覆盖。
2. **基线保护 | Baseline protection**：只有新答案获得直接证据支持时才接受修改，避免一次复核破坏已验证答案。
3. **领域核查 | Domain-specific checks**：减少年份、单位、百分比、条款义务和责任范围误读。
4. **格式优先 | Format-first validation**：在提交前发现字段、排序、summary 和 token 求和问题。
5. **真实 usage 账本 | Raw usage ledger**：记录每次 API 调用并按题合计，支持 B 榜 token 审计。

## 快速开始 | Quick Start

### 安装 | Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item local_config.example.py local_config.py
```

在 `local_config.py` 中填写自己的配置：

Fill in the local configuration:

```python
PROVIDER = "qwen"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "your-own-api-key"
MODEL = "qwen3.6-plus"
```

也可以使用环境变量。命令行参数优先级最高，其次是 `local_config.py`，最后是环境变量和默认值。

Environment variables are also supported. The precedence is command-line arguments, `local_config.py`, environment variables, and defaults.

### A 榜 | A Track

将授权的数据放入本地目录后运行：

```powershell
python train.py --mode preprocess
python train.py --mode dry-run
python train.py --mode targeted
python train.py --mode check
```

完整运行或继续中断任务：

```powershell
python train.py --mode full
python train.py --mode precision
python train.py --mode resume
```

### B 榜 | B Track

将 B 榜题目和原始文档放在本地 `upload_b/`，再运行：

```powershell
python script\run_b_full_ensemble.py
python script\run_b_low_token.py
python script\run_b_qwen36_compact.py
python script\check_b_submission.py
python script\check_b_compliant_submission.py
```

B 榜字段、reasoning、token 和评分规则见 [docs/COMPETITION_RULES.md](docs/COMPETITION_RULES.md)。系统结构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，实验总结见 [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)。

See [docs/COMPETITION_RULES.md](docs/COMPETITION_RULES.md) for B-track fields, reasoning, token accounting, and scoring requirements. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the system design and [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) for experiments and transferable lessons.

## VS Code | VS Code

`.vscode/tasks.json` 提供预处理、检索检查、训练、续跑和提交校验任务。也可以直接运行：

The VS Code task file provides preprocessing, retrieval checks, training, resume, and submission-validation tasks. You can also run:

```powershell
python train.py
```

## 测试 | Tests

```powershell
python -m pytest
python -m compileall -q train.py agent script
```

## 目录 | Layout

```text
agent/                    核心模块 | Core modules
script/                   运行与校验入口 | Runners and validators
tests/                    自动化测试 | Automated tests
docs/                     双语能力、规则、数据布局、架构与总结 | Bilingual capabilities, rules, data layout, architecture, and summaries
skills/                   全量审查提示词 | Full-audit prompt skill
train.py                  统一入口 | Unified CLI entry point
```

数据集、`runs/`、缓存、答案、证据、虚拟环境和 API key 都在 `.gitignore` 中排除。安全说明见 [SECURITY.md](SECURITY.md)。

Datasets, `runs/`, caches, answers, evidence, virtual environments, and API keys are excluded by `.gitignore`. See [SECURITY.md](SECURITY.md) for security guidance.
