# Agent Evaluation And Cost Governance

## Offline quality indicators

`agent.runtime.evaluate_records` computes metrics that do not require a live
API or private competition labels:

- **Accuracy**: exact answer match when an `expected_answer` is available.
- **Evidence coverage**: fraction of tasks with at least one source chunk.
- **Answer validity**: fraction of tasks with a normalized non-empty answer.
- **Review rate**: fraction flagged by reflection for human or policy review.
- **Token usage**: per-run total and average from raw model usage records.

These metrics complement, rather than replace, leaderboard evaluation. A high
score with weak evidence coverage should be investigated before deployment.

## Cost and audit rules

1. Record raw `prompt_tokens`, `completion_tokens`, and `total_tokens` for
   each contributing model call.
2. Aggregate calls per task through `CostLedger`; never invent estimates for a
   compliant submission.
3. Enforce a task budget before adding reflection or retry calls.
4. Preserve the trace with tool calls and failures alongside the output.
5. Treat a missing evidence link, invalid output, or token-budget overrun as a
   reviewable state, not an invisible fallback.

The lightweight runtime does not assign a currency cost because Qwen and relay
pricing varies by provider. A deployment can multiply `CostLedger.usage` by
its approved provider price card in its own environment.
