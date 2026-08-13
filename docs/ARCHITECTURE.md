# 系统架构 | System Architecture

## 1. 数据流 | Data Flow

```text
本地原始文档 + 题目
Local documents + questions
          |
          v
预处理与结构化 / Preprocessing
PDF, HTML, TXT -> documents -> chunks -> questions
          |
          v
分层检索 / Layered retrieval
question query + option queries + document coverage + neighbors
          |
          v
紧凑证据上下文 / Compact evidence context
          |
          v
Qwen API / OpenAI-compatible endpoint
          |
          v
结构化答案 + reasoning + raw usage
Structured answer + reasoning + raw usage
          |
          v
checkpoint/cache -> evidence audit -> CSV validator -> submission file
```

The pipeline is deliberately split into deterministic preparation, evidence
retrieval, model generation, and submission validation. This makes it possible
to diagnose whether an error came from parsing, retrieval, prompting, model
choice, or CSV formatting.

## 2. 模块职责 | Module Responsibilities

| 模块 | 职责 | Module | Responsibility |
| --- | --- | --- | --- |
| `agent/preprocess.py` | A 榜文档解析、表格保留、切片和题目标准化 | `agent/preprocess.py` | A-track parsing, table preservation, chunking, and question normalization |
| `agent/b_preprocess.py` | B 榜题目类型、答案槽位和 B 榜文档预处理 | `agent/b_preprocess.py` | B-track types, answer slots, and document preprocessing |
| `agent/retrieval.py` | 词法索引、数字/条款锚点和候选片段召回 | `agent/retrieval.py` | Lexical index, numeric/clause anchors, and candidate retrieval |
| `agent/solver.py` | A 榜提示词、答案解析、复核、缓存和 checkpoint | `agent/solver.py` | A-track prompts, parsing, review, caching, and checkpoints |
| `agent/b_ensemble.py` | B 榜多阶段证据和答案生成 | `agent/b_ensemble.py` | B-track multi-stage evidence and answer generation |
| `agent/b_compliant_submission.py` | B 榜 reasoning、usage 和 CSV 合规 | `agent/b_compliant_submission.py` | B-track reasoning, usage, and CSV compliance |
| `agent/qwen_client.py` | OpenAI-compatible Qwen 请求和原始 usage 接收 | `agent/qwen_client.py` | OpenAI-compatible Qwen requests and raw usage capture |
| `script/run_*.py` | 面向终端和 VS Code 的运行入口 | `script/run_*.py` | Terminal and VS Code runners |
| `script/check_*.py` | A/B 榜提交文件验证 | `script/check_*.py` | A/B submission validators |
| `tests/` | 无 API 的逻辑、格式和提示词测试 | `tests/` | API-free logic, format, and prompt tests |

## 3. 证据检索策略 | Evidence Retrieval Strategy

### 题干级召回 | Question-level recall

先使用题目文本、领域关键词、数字和文档约束召回全局候选。它适合找到主题和总体结论，但可能遗漏某个选项中的局部限制。

The question text, domain terms, numbers, and document constraints are used to
retrieve global candidates. This captures the topic and overall conclusion but
may miss a local constraint appearing only in one option.

### 选项级召回 | Option-level recall

每个选项单独生成查询，强化否定词、金额、百分比、日期、条款和指标名称。对于多选题，系统要求模型逐项给出支持、反驳或证据不足的判断。

Each option is queried separately with emphasis on negations, amounts,
percentages, dates, clauses, and metric names. For multiple-choice questions,
the model is asked to classify each option as supported, contradicted, or
insufficiently evidenced.

### 文档覆盖 | Document coverage

跨文档题不能只因为某个文档得分高就忽略其他文档。系统为目标文档保留候选片段，再将它们合并到紧凑上下文中。

Cross-document questions must not discard requested documents simply because
one document received a higher lexical score. Candidate passages are retained
for each target document before compact context construction.

### 邻近扩展 | Neighbor expansion

命中的片段会根据配置补充前后邻居，以恢复标题、表头、脚注或紧接着的限定条件。扩展过大则增加 token，因此由 `compact`、`minimal` 和 `nano` 模式控制。

Neighboring chunks are added to recover headings, table headers, footnotes, and
immediately adjacent conditions. Larger neighborhoods improve context but cost
tokens, so `compact`, `minimal`, and `nano` modes control the trade-off.

## 4. 精度决策 | Accuracy Decisions

模型输出不是直接覆盖基线，而要经过：

Model output does not automatically overwrite the baseline. It passes through:

1. JSON/字段解析 / JSON and field parsing
2. 题型答案规范化 / question-type normalization
3. 选项判断与最终答案一致性检查 / option-to-answer consistency check
4. 证据 ID 和文档覆盖检查 / evidence ID and document coverage check
5. 风险规则筛选 / structural risk rules
6. 需要时进行复核 / optional review call
7. `evidence_gate` 决定是否接受答案变化 / evidence gate decides whether a change is accepted

The evidence gate is intentionally conservative. A review proposal should be
accepted only when it contains direct support for the new answer or direct
contradiction of the baseline. This protects a validated baseline from broad
review regressions.

## 5. Token 与缓存 | Token and Caching

- `agent/qwen_client.py` preserves raw API usage fields.
- A-track solver usage is aggregated per question across initial and review calls.
- B-track compliant runners reject missing or estimated raw usage for submission output.
- Per-question caches allow resume without repeating completed calls.
- `evidence.json` and checkpoint JSONL are audit artifacts, not public source files.

`usage_or_estimate` exists for local debugging when a provider omits usage, but
estimated values must not be used for a compliant competition submission. A
submission runner must fail closed when raw usage is required and unavailable.

## 6. 可扩展点 | Extension Points

建议按以下顺序扩展，避免同时改动多个变量：

Recommended extension order:

1. 先增加离线检索测试 / add offline retrieval tests first
2. 再增加一个领域或题型的提示词测试 / add a prompt test for one domain or type
3. 对比 evidence coverage 和 answer consistency / compare evidence coverage and answer consistency
4. 只对高风险题做小范围 API 实验 / run a small API experiment on high-risk questions
5. 最后才调整上下文长度、复核轮数或模型参数 / adjust context length, review count, or model parameters last

This order keeps changes attributable and reduces the risk of improving a
leaderboard score through an untraceable mixture of prompt, data, and answer
edits.
