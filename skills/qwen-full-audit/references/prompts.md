# Full Ensemble Prompt Contract

`agent/ensemble_prompts.py` is the executable source of truth. This reference
explains the v3 reasoning contract used by all four roles.

## Contents

- Shared context and evidence
- Shared option rules
- Domain-specific rules
- Role isolation
- Judgement schema
- Baseline change policy

## Shared Context

Every role receives the question ID, domain, answer format, capability label,
ordered document IDs, question stem, and all actual options. Each evidence block
contains:

```text
[证据N] doc_id={doc_id} chunk_id={chunk_id} page={page}
page_chars={page_char_start}:{page_char_end} sources={retrieval_sources}
{verbatim_chunk_text}
```

Evidence IDs are local to the current prompt. Quotes must be copied verbatim from
the cited block.

## Shared Option Rules

For each actual option:

1. Split entity, object, metric, unit, period, scope, negation, exception,
   conjunction, and calculation conditions.
2. Require every material condition for `true + entailed`.
3. Use `false + contradicted` only for direct contradiction or a clear material
   mismatch.
4. Use `uncertain + unknown` when retrieval or semantics remain unresolved.
5. Check document order for first/second-document references.
6. Check every involved document for cross-document claims.
7. Verify metric, period, unit, direction, numerator, denominator, and calculation
   convention for numbers.
8. For true/false questions, A means the entire stem is true and B means it is false.

## Context and Semantics

- Restore a subject uniquely established by the question, report title, figure
  title, or surrounding paragraph. Do not reject an option merely because that
  subject is omitted.
- A synonym is not an error by itself. Reject it only if it changes legal effect,
  economic meaning, calculation convention, or responsibility scope.
- Check defined terms before comparing wording. For example, determine whether
  `欠款` includes `借款及借款利息` in that document.
- If multiple subjects remain plausible, mark the option unresolved.
- A semantic-only dispute cannot change the protected baseline.

Each analyst-style option judgement records:

```json
{
  "context_subject": "restored subject or empty",
  "semantic_equivalence": "equivalent/material_change/not_applicable/uncertain",
  "category_match": "yes/no/not_applicable/uncertain",
  "calculation_steps": []
}
```

## Category Questions

Apply two independent tests:

1. Is the statement factually correct?
2. Does it belong to the category explicitly requested by the stem?

A factually correct statement outside the requested category is excluded with
`category_mismatch`. Do not confuse factual contradiction with category mismatch.

## Insurance Questions

The analyst, reviewer, and judge must identify:

- the insured person under each policy;
- payment order;
- medical-insurance compensation;
- earlier commercial-insurance compensation;
- deduction required by expense-compensation clauses;
- no-double-recovery rules;
- shared, individual, annual, or per-claim deductible;
- a step-by-step calculation for each candidate answer.

Missing payment order or deduction rules yields `uncertain`, not an invented
calculation.

## Regulatory Questions

Treat boundary and modality terms as material:

- `以上` versus `超过`;
- `以内` versus `少于`;
- `应当` versus `可以`;
- `原则上` versus an unconditional requirement;
- risk grades such as `高风险` versus `较高风险以上`;
- exceptions and exclusions.

For role or position scope, do not expand `其他高级管理人员职位` into arbitrary
`其他职位`.

## Reports and Financial Research

- Preserve all metric modifiers, including terms such as `除客户资金杠杆`.
- Separate actual, forecast, year-on-year, sequential, and compound growth.
- Restore a region, industry, or entity supplied uniquely by the surrounding
  title or paragraph before treating omission as a scope error.

## Financial Contracts

- Distinguish issue size, registered amount, current-tranche size, proceeds, and
  outstanding balance.
- Verify inclusive boundaries, conversion prices, rating targets, guarantors,
  trustees, sponsors, and underwriters.
- Allow document-defined synonymous summaries only when rights, obligations,
  amount convention, and applicable entity remain unchanged.

## Role Isolation

### Evidence Locator

Does not see the baseline. It locates direct support, contradiction, related
passages, exact quotes, and targeted follow-up queries for every option.

### Independent Analyst

Does not see the baseline. It produces a full option-by-option decision, context
restoration, semantic check, category check, and calculations where relevant.

### Adversarial Reviewer

Does not see the baseline or analyst output. It independently attempts both proof
and disproof, emphasizing hidden scope, metric, boundary, exception, and
calculation errors.

### Final Judge

Sees the source evidence, all role outputs, and the baseline. It must return to
the source rather than use majority voting. It validates evidence IDs and quotes,
resolves category versus factual errors, and keeps the baseline for unresolved
semantic-only disputes.

## Common Judgement Schema

```json
{
  "answer": "letters in ascending order",
  "option_judgement": {
    "A": {
      "atomic_claims": [
        {
          "claim": "smallest factual condition",
          "status": "supported/contradicted/unknown",
          "evidence_ids": [1]
        }
      ],
      "judgement": "true/false/uncertain",
      "relation": "entailed/contradicted/unknown",
      "error_type": "none/missing_evidence/entity_mismatch/metric_mismatch/unit_mismatch/time_mismatch/condition_mismatch/scope_mismatch/negation_mismatch/calculation_mismatch/category_mismatch/semantic_mismatch",
      "supporting_evidence_ids": [1],
      "contradicting_evidence_ids": [2],
      "relevant_evidence_ids": [1, 2],
      "quoted_clauses": [{"evidence_id": 1, "quote": "verbatim source"}],
      "context_subject": "",
      "semantic_equivalence": "equivalent/material_change/not_applicable/uncertain",
      "category_match": "yes/no/not_applicable/uncertain",
      "calculation_steps": [],
      "reasoning": "option-level reasoning",
      "confidence": 0.0
    }
  },
  "overall_confidence": 0.0,
  "unresolved": []
}
```

The judge additionally returns:

```json
{
  "changed_from_baseline": true,
  "baseline_action": "change/keep/unresolved_keep",
  "change_classification": "direct_contradiction/context_error/semantic_dispute/category_mismatch/calculation_correction/no_change",
  "change_reason": "option, document, page or chunk, and exact basis",
  "agent_disagreements": []
}
```

## Baseline Change Gate

The executable gate requires:

1. At least two roles agree on each changed option.
2. Judge relation matches the target truth.
3. Judge confidence passes the configured threshold.
4. Cited evidence IDs are valid.
5. Required documents are covered.
6. A stable source location is materialized.
7. At least one judge quote is verified verbatim in the cited chunk.

After the automatic gate, manually reject context-only and synonym-only changes.
Build separate candidate layers; never overwrite the protected baseline.
