from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import default_paths, llm_config_from_env
from agent.qwen_client import OpenAICompatibleClient
from agent.solver import RunStopped, solve_questions, write_retrieval_preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Run answer generation and write answer.csv.")
    parser.add_argument("--root", default=str(ROOT), help="Project root directory.")
    parser.add_argument("--limit", type=int, default=None, help="Debug limit for question count.")
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=0,
        help="Maximum retrieved raw context chars before prompt compression. 0 means domain defaults.",
    )
    parser.add_argument(
        "--evidence-mode",
        choices=["nano", "micro", "minimal", "compact", "full"],
        default="compact",
        help="nano/micro/minimal/compact send shortened evidence excerpts to save tokens; full sends full chunks.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only write retrieval preview; do not call the model.")
    parser.add_argument("--review", action="store_true", help="Compatibility shortcut for --review-mode always.")
    parser.add_argument(
        "--review-mode",
        choices=["off", "auto", "broad", "always"],
        default="auto",
        help="off saves tokens; auto reviews high-risk outputs; broad also reviews all multi-choice outputs; always reviews every question.",
    )
    parser.add_argument(
        "--review-policy",
        choices=["replace", "evidence_gate"],
        default="evidence_gate",
        help="evidence_gate accepts review changes only with direct support/contradiction; replace trusts the review answer directly.",
    )
    parser.add_argument(
        "--initial-policy",
        choices=["replace", "evidence_gate", "preserve"],
        default="replace",
        help="evidence_gate compares the one-pass answer with --base-answer-csv; preserve keeps the base exactly.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint/cache files and skip completed qids.")
    parser.add_argument("--answer-csv", default="answer.csv", help="Final submission CSV output path.")
    parser.add_argument("--evidence-json", default="evidence.json", help="Final evidence audit JSON output path.")
    parser.add_argument("--checkpoint-csv", default="answer.checkpoint.csv", help="Incremental answer checkpoint CSV.")
    parser.add_argument("--checkpoint-evidence", default="evidence.checkpoint.jsonl", help="Incremental evidence checkpoint JSONL.")
    parser.add_argument("--cache-dir", default="run_cache/questions", help="Atomic per-question cache directory. Empty string disables it.")
    parser.add_argument("--stop-file", default="run_cache/STOP", help="If this file exists, stop after the current completed question. Empty string disables it.")
    parser.add_argument("--qid", action="append", default=[], help="Only rerun these qids. Can be repeated or comma/space separated.")
    parser.add_argument("--qid-file", default=None, help="Text file or CSV containing qids to rerun. CSV may have a qid column.")
    parser.add_argument("--base-answer-csv", default=None, help="Existing full answer CSV to merge unchanged non-target qids from.")
    parser.add_argument("--base-evidence-json", default=None, help="Existing full evidence JSON to merge unchanged non-target qids from.")
    parser.add_argument("--provider", default=None, help="Provider label, for example qwen or relay. Env: LLM_PROVIDER.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL, usually ending with /v1. Env: LLM_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="API key. Prefer env LLM_API_KEY/OPENAI_API_KEY/DASHSCOPE_API_KEY.")
    parser.add_argument("--model", default=None, help="Model name. Env: LLM_MODEL or QWEN_MODEL.")
    parser.add_argument("--disable-thinking", action="store_true", help="Disable Qwen thinking mode to reduce completion tokens.")
    parser.add_argument("--max-output-tokens", type=int, default=0, help="Limit completion tokens per request; 0 uses provider default.")
    parser.add_argument(
        "--allow-non-qwen",
        action="store_true",
        help="Allow non-Qwen model names. Competition submissions should not use this.",
    )
    args = parser.parse_args()

    paths = default_paths(args.root)
    if args.dry_run:
        report = write_retrieval_preview(paths.processed_dir, paths.root / "retrieval_preview.json", args.limit or 5)
        print(report)
        return

    answer_csv = resolve_output_path(paths.root, args.answer_csv)
    evidence_json = resolve_output_path(paths.root, args.evidence_json)
    checkpoint_csv = resolve_output_path(paths.root, args.checkpoint_csv)
    checkpoint_evidence = resolve_output_path(paths.root, args.checkpoint_evidence)
    cache_dir = resolve_output_path(paths.root, args.cache_dir) if args.cache_dir else None
    stop_file = resolve_output_path(paths.root, args.stop_file) if args.stop_file else None
    qid_filter = load_qid_filter(paths.root, args.qid, args.qid_file)
    base_answer_csv = resolve_output_path(paths.root, args.base_answer_csv) if args.base_answer_csv else None
    base_evidence_json = resolve_output_path(paths.root, args.base_evidence_json) if args.base_evidence_json else None
    if qid_filter and (not base_answer_csv or not base_evidence_json):
        print("Warning: --qid/--qid-file without --base-answer-csv and --base-evidence-json writes a partial output.")

    config = llm_config_from_env(
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )
    if args.disable_thinking or args.max_output_tokens:
        config = replace(
            config,
            enable_thinking=False if args.disable_thinking else config.enable_thinking,
            max_output_tokens=args.max_output_tokens or config.max_output_tokens,
        )
    validate_model_for_competition(config.model, args.allow_non_qwen)
    effective_review_mode = "always" if args.review else args.review_mode
    print(
        f"LLM provider={config.provider} model={config.model} endpoint={config.endpoint} "
        f"api_key={mask_key(config.api_key)} evidence_mode={args.evidence_mode} "
        f"review_mode={effective_review_mode} review_policy={args.review_policy} "
        f"initial_policy={args.initial_policy} thinking={config.enable_thinking} "
        f"max_output_tokens={config.max_output_tokens or 'provider-default'} resume={args.resume}"
    )
    print(f"Answer CSV: {answer_csv}")
    print(f"Evidence JSON: {evidence_json}")
    print(f"Checkpoint CSV: {checkpoint_csv}")
    print(f"Checkpoint evidence: {checkpoint_evidence}")
    print(f"Question cache: {cache_dir}")
    print(f"Stop file: {stop_file}")
    print(f"QID filter: {len(qid_filter) if qid_filter else 0} qids")
    print(f"Base answer CSV: {base_answer_csv}")
    print(f"Base evidence JSON: {base_evidence_json}")
    print("Stop safely with Ctrl+C, or run: python script\\request_stop.py")
    print("Resume later with: python script\\run_answer.py --resume")

    client = OpenAICompatibleClient(config)
    try:
        report = solve_questions(
            processed_dir=paths.processed_dir,
            answer_csv=answer_csv,
            evidence_json=evidence_json,
            client=client,
            limit=args.limit,
            max_context_chars=args.max_context_chars or None,
            review=args.review,
            review_mode=args.review_mode,
            review_policy=args.review_policy,
            initial_policy=args.initial_policy,
            evidence_mode=args.evidence_mode,
            checkpoint_csv=checkpoint_csv,
            checkpoint_evidence_jsonl=checkpoint_evidence,
            cache_dir=cache_dir,
            stop_file=stop_file,
            resume=args.resume,
            qid_filter=qid_filter,
            base_answer_csv=base_answer_csv,
            base_evidence_json=base_evidence_json,
        )
    except RunStopped as stopped:
        print(f"\nStop requested by {stopped.stop_file}.")
        print(f"Completed {stopped.completed}/{stopped.total}; checkpoint/cache are saved.")
        print("Clear stop file then resume:")
        print("  python script\\request_stop.py --clear")
        print("  python script\\run_answer.py --resume")
        raise SystemExit(0)
    except KeyboardInterrupt:
        print("\nInterrupted. Already completed rows are saved in checkpoint/cache files.")
        print("Resume command:")
        print("  python script\\run_answer.py --resume")
        raise SystemExit(130)
    print(report)


def resolve_output_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def split_qid_values(values: list[str]) -> set[str]:
    qids: set[str] = set()
    for value in values:
        for part in re.split(r"[\s,;]+", value.strip()):
            if part and part.lower() != "summary":
                qids.add(part)
    return qids


def load_qid_filter(root: Path, qid_values: list[str], qid_file: str | None) -> set[str] | None:
    qids = split_qid_values(qid_values)
    if qid_file:
        path = resolve_output_path(root, qid_file)
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and "qid" in reader.fieldnames:
                    for row in reader:
                        qid = (row.get("qid") or "").strip()
                        if qid and qid.lower() != "summary":
                            qids.add(qid)
                else:
                    f.seek(0)
                    for row in csv.reader(f):
                        if row and row[0].strip().lower() != "summary":
                            qids.add(row[0].strip())
        else:
            qids.update(split_qid_values([path.read_text(encoding="utf-8-sig")]))
    return qids or None


def validate_model_for_competition(model: str, allow_non_qwen: bool) -> None:
    if allow_non_qwen:
        return
    if "qwen" not in model.lower():
        raise SystemExit(
            f"Model '{model}' does not look like a Qwen model. The competition rules require Qwen-series models.\n"
            "Use a Qwen model such as qwen3.6-plus, qwen-plus, or the exact Qwen model exposed by your relay.\n"
            "For non-competition testing only, rerun with --allow-non-qwen."
        )


def mask_key(api_key: str) -> str:
    if not api_key:
        return "<missing>"
    if len(api_key) <= 8:
        return "***"
    return api_key[:4] + "..." + api_key[-4:]


if __name__ == "__main__":
    main()
