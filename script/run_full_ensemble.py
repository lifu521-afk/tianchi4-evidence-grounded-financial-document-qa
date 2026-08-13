from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import default_paths, llm_config_from_env, load_code_settings
from agent.ensemble_audit import (
    PROMPT_VERSION,
    EnsembleSettings,
    EnsembleStopped,
    atomic_write_json,
    run_ensemble_audit,
)
from agent.preprocess import build_processed_data
from agent.qwen_client import OpenAICompatibleClient


RUN_PREFIX = "full_ensemble_"
EXPECTED_QUESTIONS = 100
EXPECTED_DOCUMENTS = 68


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def resolve_path(value: str | Path, root: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def mask_key(value: str) -> str:
    if not value:
        return "(missing)"
    if len(value) <= 10:
        return value[:2] + "***"
    return value[:4] + "..." + value[-4:]


def bool_setting(settings: dict[str, Any], name: str, default: bool) -> bool:
    value = settings.get(name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def int_setting(settings: dict[str, Any], name: str, default: int) -> int:
    value = settings.get(name, default)
    return int(value) if value not in {None, ""} else default


def float_setting(settings: dict[str, Any], name: str, default: float) -> float:
    value = settings.get(name, default)
    return float(value) if value not in {None, ""} else default


def default_output_dir(settings: dict[str, Any], resume: bool) -> Path:
    configured = str(settings.get("ENSEMBLE_OUTPUT_DIR") or "").strip()
    if configured:
        return resolve_path(configured)
    runs_dir = ROOT / "runs"
    if resume:
        candidates = sorted(
            (path for path in runs_dir.glob(f"{RUN_PREFIX}*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0].resolve()
        raise SystemExit("--resume was requested, but no full_ensemble run directory exists.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (runs_dir / f"{RUN_PREFIX}{stamp}").resolve()


def ensure_isolated_output(output_dir: Path, baseline_csv: Path) -> None:
    forbidden = {
        ROOT.resolve(),
        baseline_csv.parent.resolve(),
        (ROOT / "最优").resolve(),
    }
    if output_dir in forbidden:
        raise SystemExit(
            f"Unsafe output directory: {output_dir}. "
            "Use a new directory under D:\\tianchi\\runs."
        )
    try:
        output_dir.relative_to((ROOT / "runs").resolve())
    except ValueError as exc:
        raise SystemExit("Full ensemble outputs must stay under D:\\tianchi\\runs.") from exc


def read_qid_filter(values: list[str], qid_file: str | None) -> set[str] | None:
    qids = {value.strip() for value in values if value.strip()}
    if qid_file:
        path = resolve_path(qid_file)
        text = path.read_text(encoding="utf-8-sig")
        try:
            rows = list(csv.DictReader(text.splitlines()))
        except csv.Error:
            rows = []
        if rows and "qid" in (rows[0].keys() if rows else []):
            qids.update(str(row.get("qid") or "").strip() for row in rows)
        else:
            qids.update(
                token.strip()
                for line in text.splitlines()
                for token in line.replace(",", " ").split()
            )
    qids.discard("")
    return qids or None


def validate_processed_data(processed_dir: Path) -> dict[str, Any]:
    required = [
        processed_dir / "questions.jsonl",
        processed_dir / "documents.jsonl",
        processed_dir / "chunks.jsonl",
        processed_dir / "preprocess_report.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Processed data is incomplete: {missing}")

    report = json.loads((processed_dir / "preprocess_report.json").read_text(encoding="utf-8"))
    questions = sum(
        1
        for line in (processed_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    documents = sum(
        1
        for line in (processed_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    chunks_with_spans = 0
    chunk_count = 0
    with (processed_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk_count += 1
            row = json.loads(line)
            if row.get("page_char_start") is not None and row.get("page_char_end") is not None:
                chunks_with_spans += 1

    missing_docs = report.get("missing_doc_ids") or []
    if questions != EXPECTED_QUESTIONS:
        raise RuntimeError(f"Expected {EXPECTED_QUESTIONS} questions, found {questions}.")
    if documents != EXPECTED_DOCUMENTS:
        raise RuntimeError(f"Expected {EXPECTED_DOCUMENTS} selected documents, found {documents}.")
    if missing_docs:
        raise RuntimeError(f"Missing referenced documents: {missing_docs}")
    if chunks_with_spans != chunk_count:
        raise RuntimeError(
            f"Only {chunks_with_spans}/{chunk_count} chunks contain source character spans. "
            "Run with --preprocess."
        )
    return {
        "questions": questions,
        "documents": documents,
        "chunks": chunk_count,
        "chunks_with_spans": chunks_with_spans,
        "dataset_sha256": combined_sha256(required[:3]),
        "preprocess_report": report,
    }


def build_manifest(
    *,
    baseline_csv: Path,
    baseline_sha256: str,
    output_dir: Path,
    processed: dict[str, Any],
    llm_config: Any,
    settings: EnsembleSettings,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": output_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "initialized",
        "prompt_version": PROMPT_VERSION,
        "baseline_path": str(baseline_csv),
        "baseline_sha256": baseline_sha256,
        "dataset_sha256": processed["dataset_sha256"],
        "dataset_counts": {
            "questions": processed["questions"],
            "documents": processed["documents"],
            "chunks": processed["chunks"],
        },
        "llm": {
            "provider": llm_config.provider,
            "model": llm_config.model,
            "base_url": llm_config.base_url,
            "temperature": llm_config.temperature,
            "enable_thinking": llm_config.enable_thinking,
            "max_output_tokens": llm_config.max_output_tokens,
        },
        "ensemble_settings": asdict(settings),
    }


def assert_resume_manifest(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    checks = (
        "prompt_version",
        "baseline_sha256",
        "dataset_sha256",
        "ensemble_settings",
        "llm",
    )
    mismatches = [key for key in checks if existing.get(key) != expected.get(key)]
    if mismatches:
        raise RuntimeError(
            "Resume manifest does not match the current run configuration: "
            + ", ".join(mismatches)
        )


def parse_args(settings: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a full evidence-grounded Qwen ensemble audit without overwriting answer.csv."
    )
    parser.add_argument("--baseline-csv", default=str(settings.get("ENSEMBLE_BASE_ANSWER_CSV") or "最优/answer89.6178.csv"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preprocess", action="store_true", default=bool_setting(settings, "ENSEMBLE_PREPROCESS", False))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--qid-file", default=None)
    parser.add_argument("--stop-file", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", action="store_true", dest="enable_thinking")
    thinking.add_argument("--disable-thinking", action="store_false", dest="enable_thinking")
    parser.set_defaults(enable_thinking=None)
    parser.add_argument("--initial-max-context-chars", type=int, default=int_setting(settings, "ENSEMBLE_INITIAL_MAX_CONTEXT_CHARS", 52000))
    parser.add_argument("--final-max-context-chars", type=int, default=int_setting(settings, "ENSEMBLE_FINAL_MAX_CONTEXT_CHARS", 90000))
    parser.add_argument("--initial-max-chunks", type=int, default=int_setting(settings, "ENSEMBLE_INITIAL_MAX_CHUNKS", 34))
    parser.add_argument("--final-max-chunks", type=int, default=int_setting(settings, "ENSEMBLE_FINAL_MAX_CHUNKS", 58))
    parser.add_argument("--top-global", type=int, default=int_setting(settings, "ENSEMBLE_TOP_GLOBAL", 16))
    parser.add_argument("--top-per-option", type=int, default=int_setting(settings, "ENSEMBLE_TOP_PER_OPTION", 9))
    parser.add_argument("--top-per-doc", type=int, default=int_setting(settings, "ENSEMBLE_TOP_PER_DOC", 6))
    parser.add_argument("--neighbor-radius", type=int, default=int_setting(settings, "ENSEMBLE_NEIGHBOR_RADIUS", 2))
    parser.add_argument("--locator-query-top-k", type=int, default=int_setting(settings, "ENSEMBLE_LOCATOR_QUERY_TOP_K", 5))
    parser.add_argument("--max-locator-queries", type=int, default=int_setting(settings, "ENSEMBLE_MAX_LOCATOR_QUERIES", 24))
    parser.add_argument("--confidence-threshold", type=float, default=float_setting(settings, "ENSEMBLE_CONFIDENCE_THRESHOLD", 0.82))
    parser.add_argument("--workers", type=int, default=int_setting(settings, "ENSEMBLE_QUESTION_WORKERS", 3))
    return parser.parse_args()


def main() -> None:
    code_settings = load_code_settings()
    args = parse_args(code_settings)
    paths = default_paths(ROOT)
    baseline_csv = resolve_path(args.baseline_csv)
    output_dir = resolve_path(args.output_dir) if args.output_dir else default_output_dir(code_settings, args.resume)
    ensure_isolated_output(output_dir, baseline_csv)
    if not baseline_csv.exists():
        raise SystemExit(f"Baseline answer CSV not found: {baseline_csv}")

    baseline_sha256 = sha256_file(baseline_csv)
    if args.preprocess:
        report = build_processed_data(
            questions_dir=paths.questions_dir,
            raw_dir=paths.raw_dir,
            processed_dir=paths.processed_dir,
            only_question_docs=True,
            progress=True,
        )
        print(f"Preprocess complete: {report}")
    processed = validate_processed_data(paths.processed_dir)

    llm_config = llm_config_from_env(
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )
    thinking_default = bool_setting(code_settings, "ENSEMBLE_ENABLE_THINKING", True)
    enable_thinking = thinking_default if args.enable_thinking is None else args.enable_thinking
    llm_config = replace(
        llm_config,
        enable_thinking=enable_thinking,
        max_output_tokens=0,
    )
    settings = EnsembleSettings(
        initial_max_chars=args.initial_max_context_chars,
        final_max_chars=args.final_max_context_chars,
        initial_max_chunks=args.initial_max_chunks,
        final_max_chunks=args.final_max_chunks,
        top_global=args.top_global,
        top_per_option=args.top_per_option,
        top_per_doc=args.top_per_doc,
        neighbor_radius=args.neighbor_radius,
        locator_query_top_k=args.locator_query_top_k,
        max_locator_queries=args.max_locator_queries,
        confidence_threshold=args.confidence_threshold,
        question_workers=max(1, args.workers),
    )
    expected_manifest = build_manifest(
        baseline_csv=baseline_csv,
        baseline_sha256=baseline_sha256,
        output_dir=output_dir,
        processed=processed,
        llm_config=llm_config,
        settings=settings,
    )

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise SystemExit(
                f"Run directory already exists: {output_dir}. Use --resume or choose a new directory."
            )
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert_resume_manifest(existing_manifest, expected_manifest)
        manifest = existing_manifest
    else:
        if args.resume:
            raise SystemExit(f"Cannot resume because manifest is missing: {manifest_path}")
        output_dir.mkdir(parents=True, exist_ok=False)
        inputs_dir = output_dir / "inputs"
        inputs_dir.mkdir(parents=True)
        shutil.copy2(baseline_csv, inputs_dir / "baseline_answer.csv")
        (inputs_dir / "baseline.sha256").write_text(baseline_sha256 + "\n", encoding="ascii")
        manifest = expected_manifest
        atomic_write_json(manifest_path, manifest)

    stop_file = resolve_path(args.stop_file) if args.stop_file else output_dir / "STOP"
    if args.resume and stop_file.exists():
        stop_file.unlink()
        print(f"Cleared stale stop file: {stop_file}")
    qid_filter = read_qid_filter(args.qid, args.qid_file)

    manifest["status"] = "running"
    manifest["started_or_resumed_at"] = datetime.now().astimezone().isoformat()
    manifest["stop_file"] = str(stop_file)
    atomic_write_json(manifest_path, manifest)

    print(
        f"LLM provider={llm_config.provider} model={llm_config.model} "
        f"endpoint={llm_config.endpoint} api_key={mask_key(llm_config.api_key)} "
        f"thinking={llm_config.enable_thinking}"
    )
    print(f"Baseline: {baseline_csv}")
    print(f"Baseline SHA256: {baseline_sha256}")
    print(
        "Processed data: "
        f"questions={processed['questions']} documents={processed['documents']} "
        f"chunks={processed['chunks']} spans={processed['chunks_with_spans']}"
    )
    print(f"Output directory: {output_dir}")
    print(f"Stop safely: New-Item -ItemType File -Force \"{stop_file}\"")
    print(f"Resume later: python script\\run_full_ensemble.py --output-dir \"{output_dir}\" --resume")

    client = OpenAICompatibleClient(llm_config)
    try:
        summary = run_ensemble_audit(
            processed_dir=paths.processed_dir,
            baseline_csv=baseline_csv,
            output_dir=output_dir,
            client=client,
            settings=settings,
            limit=args.limit,
            qid_filter=qid_filter,
            stop_file=stop_file,
        )
    except EnsembleStopped as exc:
        manifest["status"] = "stopped"
        manifest["stopped_at"] = datetime.now().astimezone().isoformat()
        manifest["stop_reason"] = str(exc)
        atomic_write_json(manifest_path, manifest)
        print(str(exc))
        print(f"Saved completed stages under: {output_dir}")
        return
    except Exception:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now().astimezone().isoformat()
        atomic_write_json(manifest_path, manifest)
        raise

    if sha256_file(baseline_csv) != baseline_sha256:
        manifest["status"] = "failed_baseline_changed"
        atomic_write_json(manifest_path, manifest)
        raise RuntimeError("The protected baseline changed during the run; candidate output is not trusted.")

    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    manifest["summary"] = summary
    atomic_write_json(manifest_path, manifest)
    print(f"Completed: {summary}")
    print(f"Candidate answer: {output_dir / 'answer.csv'}")
    print(f"Raw ensemble answer: {output_dir / 'answer.raw_ensemble.csv'}")
    print(f"Differences: {output_dir / 'differences.csv'}")
    print(f"Option evidence audit: {output_dir / 'option_audit.csv'}")


if __name__ == "__main__":
    main()
