---
name: qwen-full-audit
description: Run, resume, review, or improve the Tianchi 100-question Qwen full ensemble audit. Use when preprocessing the financial long-document dataset, independently auditing every answer option against exact source locations, comparing candidates with the protected leaderboard baseline, building conservative answer.csv layers, reducing semantic false positives, or validating evidence and submission format before a manual test.
---

# Qwen Full Audit

Use this workflow for evidence-grounded review of all 100 questions and every
actual option. The runtime is `script/run_full_ensemble.py`, candidate
post-processing is `script/build_ensemble_candidates.py`, and executable prompts
are in `agent/ensemble_prompts.py`. Read
[references/prompts.md](references/prompts.md) before changing role behavior,
schemas, evidence rules, or answer gating.

## Safety

- Treat `最优/answer89.6178.csv` and root `answer.csv` as read-only baselines.
- Write experiments only under new `runs/full_ensemble_*` directories.
- Never promote or submit a candidate automatically.
- Keep API keys in `local_config.py` or environment variables. Never print or
  persist an unmasked key.
- Do not promise 100% correctness. Report unresolved semantic or evidence gaps.
- Preserve known constraints `fc_a_015=A` and `fc_a_020=ABD` unless direct,
  verified source evidence proves a change.

## Full Workflow

1. Hash and validate the protected baseline.
2. Rebuild `processed_data` if source spans are absent or extraction changed.
3. Require 100 questions, 68 referenced documents, no missing document IDs, and
   source character spans on every chunk.
4. Run four isolated roles per question:
   - Evidence locator: support, contradiction, and targeted follow-up retrieval.
   - Independent analyst: option decisions without seeing the baseline.
   - Adversarial reviewer: independent proof and disproof attempts.
   - Final judge: source-grounded decision with the baseline treated only as a
     conservative prior.
5. Derive answers in code from `true + entailed`; do not trust answer text alone.
6. Keep `answer.raw_ensemble.csv` separate from evidence-gated `answer.csv`.
7. Require two-role agreement, judge confidence, required-document coverage, a
   materialized location, and a verified verbatim quote before automatic change.
8. Run deterministic candidate post-processing and submission checks.
9. Manually inspect all changed options before testing a candidate.

## Semantic Guardrails

- Restore a subject uniquely determined by the question, title, or paragraph
  before judging an option. Omission alone is not an error.
- Accept synonymous summaries unless they change legal effect, economic meaning,
  calculation convention, or responsibility scope.
- Check document definitions before rejecting terms such as `欠款` versus
  `借款及借款利息`.
- For category-limited questions, first decide whether a statement is true, then
  whether it belongs to the category requested by the stem.
- Mark context or synonym disputes `unresolved` and keep the baseline unless the
  source proves a material difference.

## Insurance Calculation

For every multi-policy calculation, explicitly verify:

1. Insured-person scope.
2. Payment order.
3. Medical insurance and earlier commercial-insurance compensation.
4. Deduction of other commercial-insurance payments.
5. Expense-compensation and no-double-recovery rules.
6. Shared versus individual deductible and whether its threshold is reached.

Require step-by-step arithmetic in the analyst, reviewer, and judge output.

## Evidence Fallback

Use this source order for each of the 360 actual options:

1. Judge quote verified inside the cited chunk.
2. Analyst verified quote.
3. Adversarial reviewer verified quote.
4. Exact source-derived excerpt at a stable page/chunk/character span.
5. Explicitly labeled lexical fallback.

Levels 4-5 provide traceability but are weaker than a verified model quote.
`option_audit_400.csv` is only a legacy filename; the current dataset has 360
actual options.

## Commands

Reprocess and smoke-test one question:

```powershell
python script\run_full_ensemble.py --preprocess --limit 1
```

Run all 100 questions using `local_config.py`:

```powershell
python script\run_full_ensemble.py
```

Resume the same output directory:

```powershell
python script\run_full_ensemble.py --output-dir "D:\tianchi\runs\full_ensemble_TIMESTAMP" --resume
```

Build conservative candidates and full evidence reports without new model calls:

```powershell
python script\build_ensemble_candidates.py --audit-dir "D:\tianchi\runs\full_ensemble_TIMESTAMP"
```

Validate a candidate:

```powershell
python script\check_submission.py --file "D:\tianchi\runs\full_ensemble_TIMESTAMP\candidates\high_confidence_6\answer.csv"
```

## Required Outputs

- `manifest.json`: model, non-secret endpoint, hashes, prompt version, settings,
  and run status.
- `cache/questions/<qid>.json`: resumable stage cache.
- `answer.raw_ensemble.csv`: unrestricted judge-derived result.
- `answer.csv`: automatic evidence-gated result.
- `differences.csv`: baseline comparison.
- `option_audit.csv`: one row per actual option with role votes, source path,
  document, page, chunk, character span, excerpt, and verification state.
- `audit.json`: complete role outputs, evidence, disagreements, and usage.
- `summary.json`: coverage and change counts.
- `candidates/<layer>/answer.csv`: protected-baseline-derived candidate.
- `candidates/review_report.md`: all disputed changes and their evidence.
- `candidates/all_options_evidence.md`: all 100 questions and 360 option locations.

## Candidate Policy

- `high_confidence_6`: direct numerical, scope, threshold, exception, or metric
  contradictions. Test first.
- `balanced_8`: adds a verified insurance calculation and a category-limited
  regulatory decision.
- `full_gate_10`: preserves all automatic gate changes, including two semantic
  disputes. Keep for comparison; do not test first.
