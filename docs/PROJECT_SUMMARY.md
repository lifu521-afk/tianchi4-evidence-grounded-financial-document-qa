# 项目总结 | Project Summary

## 1. 实验路线 | Experiment Roadmap

### A 榜 | A Track

项目从“题干 + 召回片段 + 单轮 Qwen 作答”开始，逐步加入：

The project started with “question + retrieved passages + one Qwen call” and evolved through:

1. PDF/HTML/TXT 解析和表格文本保留 / PDF, HTML, and TXT parsing with table preservation
2. 文档、页码、段落和 span 追踪 / document, page, paragraph, and span tracking
3. 题干级与选项级召回 / question-level and option-level retrieval
4. 跨文档覆盖和邻近片段扩展 / cross-document coverage and neighboring-context expansion
5. 金融领域提示词 / domain-specific financial prompts
6. 风险题复核和 evidence gate / risk-based review and evidence gating
7. checkpoint、缓存、停止和续跑 / checkpoints, caching, stop signals, and resume
8. 低 token 上下文和输出约束 / compact contexts and output limits

### B 榜 | B Track

B 榜在 A 榜基础上增加了多答案槽位、计算/抽取题处理、reasoning、原始 usage 汇总和合规校验。

The B track added multi-answer slots, calculation and extraction handling, reasoning summaries, raw usage aggregation, and compliance validation on top of the A-track pipeline.

## 2. 代表性结果 | Representative Results

本地实验记录的代表性结果：A 榜从约 58 分逐步提升到 89 分以上；B 榜低 token 版本约 86.7 分，证据复核和合规版本最高记录约 91.7 分。

Representative local records include an A-track progression from roughly 58 to above 89, a B-track low-token result around 86.7, and a best locally recorded evidence-reviewed compliant result around 91.7.

这些结果受题目版本、候选答案、模型版本、网络重试和线上评分公式影响，不代表固定保证，也不应作为公开仓库的自动化测试基准。

These results depend on question versions, candidate answers, model versions, retries, and online scoring formulas. They are not guarantees and are not used as automated-test baselines in this repository.

## 3. 哪些方法最有效 | What Helped Most

### 选项级召回 | Option-level retrieval

将 A/B/C/D 选项分别作为检索查询，能够发现题干整体召回中被忽略的限制条件、数字和否定词。它对多选题、真假判断和跨文档比较尤其重要。

Retrieving A/B/C/D options separately exposes constraints, numbers, and negations that question-level retrieval can miss. This is especially useful for multi-choice, true/false, and cross-document comparison questions.

### 证据门控 | Evidence gating

复核模型提出新答案后，系统要求新答案有直接证据支持，或原答案有明确矛盾证据，才允许改变基线。这样牺牲了一些“探索性变化”，但显著降低了复核把正确答案改坏的风险。

When a review call proposes a new answer, the system requires direct supporting evidence for the change or explicit contradiction of the baseline. This reduces destructive rewrites at the cost of accepting fewer speculative changes.

### 领域约束 | Domain constraints

金融问答的错误经常不是语言理解错误，而是年份、单位、指标口径、条款义务或责任范围混淆。领域提示词把这些检查显式化，比泛化地要求“认真回答”更有效。

Many financial QA errors come from confusing periods, units, metric definitions, obligations, or coverage scope rather than from general language understanding. Explicit domain checks work better than a generic “answer carefully” instruction.

### 数据和提交审计 | Data and submission auditing

先检查解析、文档覆盖、选项证据和 CSV，再判断模型好坏。B 榜中 reasoning、summary 和 token 字段本身也是评分输入，不能在最后一步才补格式。

Validate parsing, document coverage, option evidence, and CSV structure before judging model quality. On the B track, reasoning, summary, and token fields affect evaluation and must be designed from the beginning.

## 4. 失败经验 | Failure Modes

- 全量复核不一定提升分数：模型可能改坏已正确答案 / full review can reduce accuracy by changing correct baseline answers。
- 只压缩 token 可能丢失关键证据 / aggressive context compression can remove decisive evidence。
- 手工修改答案或 token 会破坏审计链 / manual answer or token edits break the audit trail。
- 只看最终分数无法定位问题 / leaderboard scores alone cannot identify the failure source。
- 不同榜单字段不能混用 / A-track and B-track schemas must not be mixed。

## 5. 对多模态自监督睡眠分期的启发 | Implications for Multimodal Self-Supervised Sleep Staging

### 多视图证据融合 | Multi-view evidence fusion

金融检索中的“题干、选项、文档、邻近片段”对应睡眠研究中的原始 EEG、时频图、统计特征和上下文窗口。不同视图不应简单拼接，而应先独立编码、分别评估，再根据一致性融合。

The QA pipeline's question, option, document, and neighboring-context views have an analogue in raw EEG, time-frequency maps, statistical features, and temporal context. These views should be encoded and assessed independently before consistency-aware fusion.

### 不确定性驱动训练 | Uncertainty-driven training

证据门控可以迁移为跨模态冲突检测：当 EEG 和时频图预测不一致，或模型置信度低时，将样本加入重点自监督训练和人工复核集合。

Evidence gating can become cross-modal conflict detection: when raw EEG and time-frequency predictions disagree or confidence is low, prioritize those samples for self-supervised training and expert review.

### 可追溯实验账本 | Traceable experiment ledger

记录数据版本、切窗参数、增强方式、模型版本、随机种子、训练 token/时间、验证集结果和 checkpoint。只有这样才能区分架构改进、数据泄漏、随机波动和评估误差。

Track dataset versions, windowing parameters, augmentations, model versions, random seeds, training cost, validation results, and checkpoints. This is necessary to distinguish architectural gains from leakage, randomness, or evaluation artifacts.

## 6. 当前局限 | Current Limitations

- 检索仍是词法检索，语义同义改写可能召回不足 / retrieval is lexical and may miss semantic paraphrases。
- 线上成绩依赖比赛题目和模型服务，无法在无标注本地数据上严格估计真实准确率 / online scores depend on the question set and model service。
- 公开版不包含数据集和历史答案，因此克隆后需要用户自行准备本地输入 / the public release excludes datasets and historical answers, so users must provide local inputs。
- usage 缺失时的估算逻辑只适合非提交调试，不能用于 B 榜合规结果 / usage fallback is for local debugging only and must not be used for compliant B-track submissions。
