"""Run the lightweight agent workflow against preprocessed A-track data.

By default this is an API-free trace and retrieval smoke test. Pass
``--execute-llm`` only after configuring an allowed Qwen endpoint in
``local_config.py`` or the environment. The generated trace is an internal
audit artifact, not a competition submission file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import llm_config_from_env
from agent.io_utils import read_jsonl, write_json
from agent.qwen_client import OpenAICompatibleClient
from agent.retrieval import LexicalIndex
from agent.runtime import EvidenceGroundedOrchestrator, OrchestratorConfig, evaluate_records
from agent.runtime.rag import EvidenceRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evidence-grounded agent runtime over preprocessed questions.")
    parser.add_argument("--processed-dir", default="processed_data")
    parser.add_argument("--output", default="runs/agent_runtime/trace.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--evidence-mode", default="compact", choices=["full", "compact", "minimal", "micro", "nano"])
    parser.add_argument("--token-budget-per-task", type=int, default=0)
    parser.add_argument("--execute-llm", action="store_true", help="Call the configured Qwen-compatible endpoint.")
    args = parser.parse_args()

    processed = (ROOT / args.processed_dir).resolve()
    questions = read_jsonl(processed / "questions.jsonl")
    chunks = read_jsonl(processed / "chunks.jsonl")
    if not questions or not chunks:
        raise SystemExit(f"No preprocessed questions/chunks under {processed}. Run: python train.py --mode preprocess")
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    client = OpenAICompatibleClient(llm_config_from_env()) if args.execute_llm else None
    rag = EvidenceRAG(LexicalIndex(chunks))
    runtime = EvidenceGroundedOrchestrator(
        retrieve=lambda question, mode: rag.retrieve(question, mode),
        client=client,
        config=OrchestratorConfig(
            evidence_mode=args.evidence_mode,
            token_budget_per_task=args.token_budget_per_task or None,
            enable_llm=args.execute_llm,
        ),
    )
    records = [runtime.run(str(question.get("qid", index)), question).as_dict() for index, question in enumerate(questions[: args.limit], start=1)]
    report = evaluate_records(records).as_dict()
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"mode": "llm" if args.execute_llm else "offline_trace", "report": report, "records": records})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Agent trace: {output}")


if __name__ == "__main__":
    main()
