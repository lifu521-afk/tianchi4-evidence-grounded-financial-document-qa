# 天池金融长文档问答 Agent | Tianchi Financial Long-Document QA Agent

一个面向金融文档知识检索与问答的证据增强型 Qwen Agent。项目最初服务于天池金融长文档问答 A/B 榜，但当前代码的技术能力可以抽象为：从企业金融文档中定位事实、核对条款、比较指标、提取答案，并生成可审计的依据和结构化结果。

An evidence-grounded Qwen agent for financial document retrieval and question
answering. It was developed for the Tianchi financial long-document QA
competition, but its reusable technical capability is broader: locating facts,
verifying clauses, comparing metrics, extracting answers, and producing
auditable structured outputs from enterprise financial documents.

> **定位边界 | Scope boundary**
>
> 本项目是文档知识检索与问答系统，不是交易系统、投资顾问、承保系统、自动法律意见系统或监管审批系统。它不能替代金融机构的专业人员、合规审查和最终决策。
>
> This is a document intelligence and QA system. It is not a trading system,
> investment adviser, underwriting system, automated legal-opinion system, or
> regulatory approval system. It does not replace professional review,
> compliance controls, or final business decisions.

## 1. 企业要解决的核心问题 | Enterprise Problem

金融企业通常拥有大量年报、合同、募集说明书、保险条款、监管文件和研究报告。业务人员需要回答“原文在哪里、具体数字是多少、条款如何约束、不同文档是否一致”，但人工逐页查找耗时，直接让大模型阅读整份文档又容易出现幻觉、漏证据、数字错位和成本失控。

Financial organizations maintain large collections of annual reports,
contracts, offering documents, insurance policy wordings, regulatory texts,
and research reports. Analysts need to answer “where is the source, what is
the exact number, what does the clause require, and do the documents agree?”
Manual page-by-page review is slow, while sending entire documents to an LLM
can cause hallucinations, missing evidence, numeric mismatches, and excessive
cost.

本项目针对的是一个受规则约束的企业文档问答问题：

The project addresses a governed document-QA problem with four constraints:

1. **答案必须基于指定原文**：不能用模型常识替代来源证据。
2. **答案必须精确匹配题型**：单选、多选、判断、计算和抽取题有不同输出约束。
3. **过程必须可审计**：B 榜还要求 reasoning 和真实 API usage。
4. **结果必须可提交、可恢复、可复盘**：CSV、summary、token、证据和 checkpoint 必须一致。

1. **Answers must be grounded in the supplied sources**, not model memory.
2. **Answers must respect the question type**, with different constraints for
   choices, true/false, calculations, and extraction.
3. **The process must be auditable**, including reasoning and raw API usage on
   the B track.
4. **The workflow must be recoverable and reviewable**, with consistent CSV,
   summary, token, evidence, and checkpoint artifacts.

## 2. 重点覆盖的金融业务场景 | Financial Business Areas

代码中的 `agent/domain.py` 明确配置了五类金融文本领域。以下是项目已经实际实现的业务核查重点，不是泛化的金融宣传。

The domain profiles in `agent/domain.py` explicitly cover five financial text
areas. The following are the implemented business checks, not a claim of
general financial coverage.

### 2.1 金融合同与债券条款 | Financial Contracts and Bond Terms

**面向的业务问题：**

**Business questions addressed:**

- 发行人、债券名称和发行主体是否对应；
- 发行规模、期限、票面利率和付息兑付安排是什么；
- 主体评级、债项评级、担保安排如何描述；
- 是否存在回售、赎回、转股或募集资金用途约束；
- 多份募集说明书或合同的条款是否一致。

- Whether issuer, bond, and issuing entity match.
- What the issue size, maturity, coupon, payment, and redemption terms are.
- How issuer ratings, bond ratings, and guarantees are described.
- Whether put, call, conversion, or use-of-proceeds constraints apply.
- Whether terms agree across offering documents or contracts.

**适合的企业任务 | Suitable enterprise tasks:** 条款定位、合同对比、发行材料核对、尽调资料检索。

Suitable for clause lookup, contract comparison, offering-document review,
and due-diligence research.

**不应直接用于 | Should not be used directly for:** 自动决定是否投资、自动签约或替代法务出具法律意见。

It must not directly decide whether to invest, execute a contract, or replace
legal counsel.

### 2.2 财务报告与经营指标 | Financial Reports and Operating Metrics

**面向的业务问题：**

**Business questions addressed:**

- 某一报告年度的营业收入、归母净利润和经营现金流；
- 财报单位是元、万元、亿元还是其他口径；
- 同比、占比、研发投入比例等指标如何计算；
- 分红、回购、研发投入和现金流数据在不同年度如何变化；
- 多家公司或多年度指标的横向、纵向比较。

- Revenue, attributable net profit, and operating cash flow for a reporting year.
- Whether figures are in yuan, ten-thousand yuan, hundred-million yuan, or
  another unit.
- How year-on-year, ratio, and R&D-intensity metrics should be calculated.
- How dividends, buybacks, R&D, and cash flow change across years.
- Cross-company and multi-year comparisons.

**适合的企业任务 | Suitable enterprise tasks:** 财报问答、指标抽取、经营数据核对、投研资料初筛。

Suitable for report QA, metric extraction, operating-data checks, and initial
research screening.

**关键风险 | Key risk:** 该系统能辅助定位和计算，但不能替代财务人员对会计准则、合并口径和审计结论的确认。

It can assist with locating and calculating figures, but does not replace
accounting, consolidation, or audit review.

### 2.3 保险产品条款 | Insurance Product Terms

**面向的业务问题：**

**Business questions addressed:**

- 保险责任和身故保险金的触发条件；
- 等待期、年龄或期间限制；
- 现金价值、已交保费、保单账户价值和领取条件；
- 退保、合同终止和免责条款；
- 多个保险产品条款的责任范围和限制比较。

- Conditions triggering coverage and death benefits.
- Waiting periods, age limits, and time restrictions.
- Cash value, paid premiums, policy-account value, and withdrawal conditions.
- Surrender, termination, and exclusions.
- Comparisons of coverage and restrictions across products.

**适合的企业任务 | Suitable enterprise tasks:** 条款检索、产品资料对比、客服知识库初步问答、核保前资料定位。

Suitable for policy lookup, product-document comparison, first-line knowledge
support, and pre-underwriting document location.

**关键风险 | Key risk:** 保险责任解释具有法律和业务后果，最终结果必须经过保险产品、精算、法务或合规人员确认。

Coverage interpretation has legal and business consequences and requires final
review by product, actuarial, legal, or compliance teams.

### 2.4 监管规则与公司治理 | Regulation and Corporate Governance

**面向的业务问题：**

**Business questions addressed:**

- “应当、不得、可以”等义务强度的差异；
- 条款编号、适用对象、例外条件和生效范围；
- 工作日、报告期限和保存期限；
- 普通决议、特别决议、表决比例；
- 对外担保、募集资金用途、独立董事和章程修改等条件。

- Differences between obligations, prohibitions, and permissions.
- Clause identifiers, applicable parties, exceptions, and effective scope.
- Business-day, reporting, and retention deadlines.
- Ordinary resolutions, special resolutions, and voting ratios.
- Conditions concerning guarantees, use of proceeds, independent directors, and
  charter amendments.

**适合的企业任务 | Suitable enterprise tasks:** 监管文件检索、制度条款定位、公司治理规则核对、合规人员初筛。

Suitable for regulatory lookup, policy-clause location, governance-rule
checks, and compliance-team first-pass review.

**关键风险 | Key risk:** 这是规则文本辅助检索，不是自动合规结论；法规更新、生效日期和适用主体必须由合规人员确认。

This is regulatory-text retrieval, not an automated compliance conclusion.
Updates, effective dates, and applicable entities require compliance review.

### 2.5 研究报告与投研资料 | Research Reports and Investment Research

**面向的业务问题：**

**Business questions addressed:**

- 研究结论、投资评级、目标价和风险提示；
- 行业规模、增速、市场份额和毛利率；
- 预测年份、指标口径和公司比较；
- 观点改写后是否仍然保持原报告方向和限定条件。

- Research conclusions, ratings, target prices, and risk warnings.
- Market size, growth, market share, and gross margin.
- Forecast years, metric definitions, and company comparisons.
- Whether a paraphrased claim preserves the source direction and limitations.

**适合的企业任务 | Suitable enterprise tasks:** 投研资料检索、研究观点核对、公司比较和风险提示抽取。

Suitable for research retrieval, thesis verification, company comparison, and
risk-warning extraction.

**关键风险 | Key risk:** 项目不生成投资建议，也不验证研究报告观点的真实性或未来收益。

The project does not produce investment advice or validate the truth or future
returns of a research opinion.

## 3. 已完成的方法 | Implemented Methods

### 3.1 文档解析与结构化 | Document Parsing and Structuring

已完成：

Implemented:

- PDF、HTML、TXT 文档发现和读取；
- PDF 页面文本抽取；
- PDF 表格行转换为可检索文本；
- 文档 ID、领域、源路径、页码和字符 span 保存；
- 问题、选项和文档引用标准化；
- 预处理报告和缺失文档检查。

- PDF, HTML, and TXT discovery and reading.
- Page-level PDF text extraction.
- Conversion of PDF table rows into searchable text.
- Preservation of document ID, domain, source path, page, and character spans.
- Normalization of questions, options, and document references.
- Preprocessing reports and missing-document checks.

### 3.2 词法检索与证据定位 | Lexical Retrieval and Evidence Location

项目使用不依赖 embedding 的词法检索：关键词、数字、条款编号、中文字符/二元片段和 BM25 风格评分。检索不是最终答案，而是为模型提供可追溯的证据候选。

The project uses embedding-free lexical retrieval over keywords, numbers,
clause identifiers, Chinese character/bigram fragments, and BM25-style scores.
Retrieval is not the final answer; it supplies traceable evidence candidates to
the model.

已完成四类召回：

Four retrieval layers are implemented:

1. 题干级召回 / question-level retrieval
2. 选项级召回 / option-level retrieval
3. 跨文档覆盖 / cross-document coverage
4. 命中片段邻近上下文扩展 / neighboring-context expansion

### 3.3 领域配置与提示词 | Domain Profiles and Prompts

`agent/domain.py` 为五类领域配置检索数量、上下文上限、关键词和核查清单；提示词要求区分直接支持、直接否定和证据不足，避免使用金融常识替代原文。

`agent/domain.py` configures retrieval limits, context budgets, query terms,
and checklists for the five domains. Prompts distinguish direct support, direct
contradiction, and insufficient evidence instead of replacing source text with
general financial knowledge.

### 3.4 结构化答案与风险复核 | Structured Answers and Risk Review

已完成：

Implemented:

- 结构化 JSON 结果解析；
- 单选、多选、判断、计算、抽取题答案规范化；
- 选项判断与最终答案一致性检查；
- 选项证据 ID 和跨文档覆盖检查；
- targeted、broad、precision 等风险范围；
- evidence gate 基线保护；
- 可配置的 thinking、上下文长度和输出长度。

- Structured JSON result parsing.
- Normalization for choice, true/false, calculation, and extraction answers.
- Consistency checks between option judgments and final answers.
- Option-evidence ID and cross-document coverage checks.
- targeted, broad, and precision risk scopes.
- Baseline protection through an evidence gate.
- Configurable thinking, context, and output limits.

### 3.5 Token、reasoning 与提交治理 | Token, Reasoning, and Submission Governance

B 榜链路已经实现：

The B-track path implements:

- `answer_1` 至 `answer_4` 答案槽位；
- `reasoning` 生成和基本一致性检查；
- 原始 API usage 读取；
- 多次调用按题累加；
- 每题和 summary 的 token 等式；
- qid、表头、行数和字段校验。

- `answer_1` through `answer_4` answer slots.
- Reasoning generation and basic consistency checks.
- Raw API usage capture.
- Per-question aggregation across multiple calls.
- Per-question and summary token equations.
- qid, header, row-count, and field validation.

### 3.6 运行恢复与工程化 | Recovery and Engineering Operations

已完成：

Implemented:

- VSCode 任务和统一 `train.py` 入口；
- 每题缓存和 checkpoint；
- Ctrl+C 和停止文件安全暂停；
- `resume` 续跑；
- checkpoint 重建；
- pytest 测试；
- GitHub Actions 在 Windows/Python 3.10、3.12 上编译和测试；
- `.gitignore`、空 key 配置、双语规则和数据布局说明。

- VS Code tasks and the unified `train.py` entry point.
- Per-question caches and checkpoints.
- Safe pause through Ctrl+C and stop files.
- Resume execution.
- Checkpoint rebuilding.
- pytest coverage.
- GitHub Actions compilation and tests on Windows/Python 3.10 and 3.12.
- `.gitignore`, credential-free config, bilingual rules, and data-layout docs.

## 4. 企业视角下的交付价值 | Enterprise Value

### 4.1 提升资料检索效率 | Faster Document Research

将“人工翻页查找”变成“问题—证据片段—答案—来源位置”的结构化流程，适合研究、尽调、合同审阅和合规初筛中的资料定位环节。

It turns manual page-by-page lookup into a structured
“question-evidence-answer-source” workflow for research, due diligence,
contract review, and compliance first-pass analysis.

### 4.2 降低无依据回答风险 | Lower Unsupported-Answer Risk

选项级召回、文档覆盖和 evidence gate 让答案修改必须有证据路径，便于复核人员判断“模型为什么这样答”。

Option-level retrieval, document coverage, and evidence gating require an
evidence path for answer changes, making model decisions easier to review.

### 4.3 支持成本与质量权衡 | Quality-Cost Trade-off

企业可以按业务风险选择：低风险问题使用紧凑上下文，高风险问题使用更多证据和复核；不必对所有问题使用同样的模型调用成本。

Organizations can route low-risk questions through compact contexts and
high-risk questions through richer evidence and review, instead of paying the
same model cost for every question.

### 4.4 形成可审计记录 | Auditable Records

证据、模型输出、reasoning、usage 和 checkpoint 形成运行记录，有利于内部复盘、问题定位和提交前检查。该记录仍需按照企业数据保留和访问控制制度管理。

Evidence, model outputs, reasoning, usage, and checkpoints form an operational
record for review, diagnosis, and pre-submission checks. They must still be
managed under the organization's retention and access-control policies.

## 5. 当前需要修正的方法 | Methods That Need Correction

以下问题是基于当前代码边界得出的工程结论，不是把未实现能力包装成成果。

The following items are engineering conclusions from the current codebase, not
claims of already implemented production capability.

| 优先级 / Priority | 当前问题 / Current issue | 影响 / Impact | 建议修正 / Recommended correction |
| --- | --- | --- | --- |
| P0 | 词法检索为主，缺少可评估的混合语义召回 | 同义改写、跨语言表达或长距离语义关系可能漏召回 | 在允许使用的企业环境中增加可插拔向量召回和 reranker；用离线 `recall@k`、文档覆盖率和选项证据覆盖率评估。比赛环境是否允许必须单独确认。 |
| P0 | 扫描 PDF/OCR 能力未形成完整链路 | 图片型表格或扫描合同可能没有有效文本 | 增加 OCR 适配层、页级质量分数、人工抽样和“文本缺失即阻断”的质量门。 |
| P0 | B 榜脚本包含题目/文档特定约束 | 适合比赛复现，但不适合直接作为通用企业服务，规则变更也可能失效 | 将特定约束移出代码，改成版本化外部配置；建立通用模板与数据集适配层。 |
| P0 | 缺少真实业务标注集和离线评估 | 只能依赖比赛分数，无法证明企业场景的准确率、召回率和拒答质量 | 建立经过授权的黄金集，评估 evidence recall、answer accuracy、unsupported-answer rate、拒答准确率和领域分项表现。 |
| P1 | `usage_or_estimate` 存在调试估算路径 | API 未返回 usage 时，估算值不能用于合规提交或严肃成本核算 | 生产/B 榜模式应 fail-closed；估算只允许显式 debug 模式，并在结果中标记 `estimated=true`。 |
| P1 | 复核策略主要是规则和模型提示，缺少校准 | 置信度未必等于真实正确率，高风险筛选可能漏题或过度复核 | 用带标签验证集做置信度校准、风险阈值选择和分领域阈值评估。 |
| P1 | 缺少企业级访问控制、脱敏和多租户隔离 | 合同、客户和监管资料可能包含敏感信息 | 增加身份认证、租户隔离、文档权限继承、PII 脱敏、加密存储、审计日志和数据删除策略。 |
| P1 | API 稳定性治理仍偏轻量 | 并发、限流、服务降级和成本异常缺少统一控制 | 增加 rate limit、指数退避上限、断路器、请求幂等键、预算告警和延迟/错误监控。 |
| P2 | 当前以本地 CLI 为主 | 尚未形成企业内部服务或工作流系统 | 在核心函数之上增加稳定 API、任务队列、结果数据库和人工审核界面；CLI 继续作为运维入口。 |
| P2 | 证据输出格式偏比赛审计 | 企业用户还需要引用片段、页面预览、版本和权限信息 | 建立统一 Evidence 对象：`document_id`、版本、页码、span、引用文本、权限和生成时间。 |
| P2 | 规则文档和模型白名单不会自动同步 | 官方规则变化可能导致旧校验器误判 | 将字段 schema、模型白名单和评分版本配置化，并为每个规则版本保留校验测试。 |

## 6. 比赛适配与企业产品的区别 | Competition Adapter vs Enterprise Product

当前代码可以作为企业文档智能原型和比赛适配器，但还不是开箱即用的生产系统。原因是比赛与企业的目标不同：

The current code can serve as a prototype for enterprise document
intelligence and as a competition adapter, but it is not an out-of-the-box
production system. Competition and enterprise requirements differ:

| 维度 / Dimension | 比赛适配 / Competition adapter | 企业生产 / Enterprise production |
| --- | --- | --- |
| 数据 | 本地授权数据、固定目录 | 文档平台、权限和版本治理 |
| 评估 | 线上分数和提交格式 | 黄金集、SLA、领域指标和人工抽检 |
| 模型 | 赛事允许的 Qwen 模型 | 经审批的模型路由和供应商治理 |
| 证据 | 满足题目和 reasoning 审计 | 可点击引用、版本、权限和保留策略 |
| 成本 | prompt/completion token | 预算、并发、延迟和部门成本归属 |
| 风险 | 防止提交失分 | 防止错误业务决策、数据泄露和合规违规 |

## 7. 不应声称的能力 | Claims the Project Should Not Make

为保持专业和真实，项目不应声称：

To remain accurate and professional, the project should not claim that it:

- 保证所有答案正确 / guarantees every answer;
- 自动完成投资、交易、承保或监管审批 / automatically makes investment,
  trading, underwriting, or approval decisions;
- 已完成企业级权限、脱敏、加密和多租户 / already provides enterprise
  authorization, redaction, encryption, or multi-tenancy;
- 已完成 OCR 对所有扫描件的可靠识别 / reliably handles every scanned document;
- 通过“训练模式”对 Qwen 参数进行了微调 / fine-tunes Qwen parameters through
  the `train.py` modes;
- 可以使用手工修改的 token 生成合规提交 / can create compliant submissions
  with manually edited token values。

## 8. 推荐企业化迭代路线 | Recommended Enterprise Roadmap

### 阶段一：评估和数据质量 | Phase 1: Evaluation and Data Quality

建立授权黄金集、文档版本、题型标签和人工证据标注；先测解析成功率、证据 `recall@k`、跨文档覆盖率和答案准确率。

Build an authorized golden set, document versions, question-type labels, and
human evidence annotations. Measure parsing success, evidence `recall@k`,
cross-document coverage, and answer accuracy first.

### 阶段二：检索和拒答 | Phase 2: Retrieval and Abstention

评估词法检索与向量/重排组合，增加证据不足时的拒答或人工转审，并校准风险阈值。

Evaluate lexical retrieval against a hybrid vector/reranking design, add
abstention or human escalation when evidence is insufficient, and calibrate
risk thresholds.

### 阶段三：安全和服务化 | Phase 3: Security and Serviceization

加入权限继承、脱敏、加密、审计日志、任务队列、限流、监控、预算告警和结果版本管理。

Add permission inheritance, redaction, encryption, audit logs, job queues,
rate limits, monitoring, budget alerts, and result versioning.

### 阶段四：人工闭环 | Phase 4: Human-in-the-Loop

让业务专家确认高风险条款、数字和拒答样本，把审核结果反馈为检索测试、提示词测试和领域规则，而不是直接手工覆盖答案。

Let business experts review high-risk clauses, numbers, and abstentions. Feed
their decisions back into retrieval tests, prompt tests, and domain rules
rather than manually overwriting answers.

## 9. 公开交付物 | Public Deliverables

当前仓库已经包含：

The repository includes:

- A/B 榜核心处理和运行代码 / A/B processing and runner code
- 五类金融领域配置 / five financial domain profiles
- 分层证据检索 / layered evidence retrieval
- Qwen OpenAI-compatible 客户端 / Qwen-compatible client
- reasoning、token 和 CSV 校验 / reasoning, token, and CSV validation
- checkpoint、停止和续跑 / checkpoints, stop, and resume
- VS Code 任务和 GitHub Actions / VS Code tasks and GitHub Actions
- 中英文规则、架构、数据布局、项目总结和本能力清单 / bilingual rules,
  architecture, data layout, project summary, and this capability inventory

数据集、标准答案、运行结果、证据、缓存和 API key 不在仓库中。目标 GitHub 仓库为：

Datasets, answer keys, run outputs, evidence, caches, and API keys are not in
the repository. The GitHub repository is:

[lifu521-afk/tianchi-financial-qa](https://github.com/lifu521-afk/tianchi-financial-qa)
