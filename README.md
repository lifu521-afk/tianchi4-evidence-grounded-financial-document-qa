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
