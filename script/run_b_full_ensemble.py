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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.b_ensemble import (  # noqa: E402
    B_PROMPT_VERSION,
    BEnsembleSettings,
    BEnsembleStopped,
    run_b_ensemble,
)
from agent.b_preprocess import (  # noqa: E402
    DEFAULT_PROCESSED_DIR,
    DEFAULT_QUESTIONS_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_SUBMIT_CSV,
    build_b_processed_data,
)
from agent.b_submission import (  # noqa: E402
    load_submission_spec,
    validate_b_submission,
    write_b_submission,
)
from agent.config import llm_config_from_env, load_code_settings  # noqa: E402
from agent.ensemble_audit import atomic_write_json  # noqa: E402
from agent.qwen_client import OpenAICompatibleClient  # noqa: E402


RUN_PREFIX = "b_full_"
EXPECTED_QUESTIONS = 100
EXPECTED_DOCUMENTS = 573


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def mask_key(value: str) -> str:
    if not value:
        return "(missing)"
    return value[:4] + "..." + value[-4:] if len(value) > 10 else value[:2] + "***"


def ps_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def bool_setting(settings: dict[str, Any], name: str, default: bool) -> bool:
    value = settings.get(name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def int_setting(settings: dict[str, Any], name: str, default: int) -> int:
    value = settings.get(name, default)
    return int(value) if value not in {None, ""} else default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_sha256(processed_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("questions.jsonl", "documents.jsonl", "chunks.jsonl"):
        path = processed_dir / name
        digest.update(name.encode("ascii"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def default_output_dir(resume: bool) -> Path:
    runs_dir = ROOT / "runs"
    if resume:
        candidates = sorted(
            (path for path in runs_dir.glob(f"{RUN_PREFIX}*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0].resolve()
        raise SystemExit("--resume was requested, but no B full run directory exists.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (runs_dir / f"{RUN_PREFIX}{stamp}").resolve()


def ensure_isolated_output(output_dir: Path) -> None:
    runs_dir = (ROOT / "runs").resolve()
    try:
        output_dir.relative_to(runs_dir)
    except ValueError as exc:
        raise SystemExit("B outputs must stay under D:\\tianchi\\runs.") from exc
    protected = {
        ROOT.resolve(),
        (ROOT / "最优").resolve(),
        (ROOT / "upload_b").resolve(),
    }
    if output_dir in protected:
        raise SystemExit(f"Unsafe B output directory: {output_dir}")


def read_qid_filter(values: list[str], qid_file: str | None) -> set[str] | None:
    qids = {value.strip() for value in values if value.strip()}
    if qid_file:
        path = resolve_path(qid_file)
        text = path.read_text(encoding="utf-8-sig")
        rows = list(csv.DictReader(text.splitlines()))
        if rows and "qid" in rows[0]:
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
        raise RuntimeError(f"B processed data is incomplete: {missing}")

    questions = [
        json.loads(line)
        for line in required[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    documents = [
        json.loads(line)
        for line in required[1].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunk_count = 0
    chunks_with_spans = 0
    with required[2].open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_count += 1
            if (
                row.get("page_char_start") is not None
                and row.get("page_char_end") is not None
                and row.get("source_path")
            ):
                chunks_with_spans += 1

    doc_ids = [str(row.get("doc_id") or "") for row in documents]
    qids = [str(row.get("qid") or "") for row in questions]
    if len(questions) != EXPECTED_QUESTIONS or len(set(qids)) != len(qids):
        raise RuntimeError(
            f"Expected {EXPECTED_QUESTIONS} unique B questions, found {len(questions)}."
        )
    if len(documents) != EXPECTED_DOCUMENTS or len(set(doc_ids)) != len(doc_ids):
        raise RuntimeError(
            f"Expected {EXPECTED_DOCUMENTS} unique B documents, found {len(documents)}."
        )
    if not chunk_count or chunks_with_spans != chunk_count:
        raise RuntimeError(
            f"Only {chunks_with_spans}/{chunk_count} B chunks have source spans."
        )
    return {
        "questions": len(questions),
        "documents": len(documents),
        "chunks": chunk_count,
        "chunks_with_spans": chunks_with_spans,
        "dataset_sha256": dataset_sha256(processed_dir),
    }


def submission_from_answers(output_dir: Path) -> Path:
    answer_rows = json.loads((output_dir / "answers.json").read_text(encoding="utf-8"))
    spec = load_submission_spec(
        DEFAULT_SUBMIT_CSV,
        question_dir=DEFAULT_QUESTIONS_DIR,
    )
    answers: dict[str, list[str]] = {}
    for row in answer_rows:
        qid = row["qid"]
        values = [str(value).strip() for value in row["answers"]]
        for index, kind in enumerate(spec.questions[qid].kinds):
            if kind == "percent":
                values[index] = (
                    values[index]
                    .replace(",", "")
                    .replace("，", "")
                    .replace("％", "%")
                )
        answers[qid] = values
    usage = {row["qid"]: row["usage"] for row in answer_rows}
    answer_csv = write_b_submission(
        output_dir / "answer.csv",
        answers,
        template_path=DEFAULT_SUBMIT_CSV,
        question_dir=DEFAULT_QUESTIONS_DIR,
        usage_by_qid=usage,
    )
    report = validate_b_submission(
        answer_csv,
        template_path=DEFAULT_SUBMIT_CSV,
        question_dir=DEFAULT_QUESTIONS_DIR,
    )
    report.require_valid()
    atomic_write_json(
        output_dir / "submission_validation.json",
        {
            "path": str(answer_csv),
            "question_rows": report.question_rows,
            "expected_questions": report.expected_questions,
            "errors": report.errors,
            "warnings": report.warnings,
        },
    )
    return answer_csv


def parse_args(settings: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the independent full-token B leaderboard ensemble audit."
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=bool_setting(settings, "B_PREPROCESS", True),
    )
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--qid-file")
    parser.add_argument("--stop-file")
    parser.add_argument("--provider")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument("--prompt-version", default=B_PROMPT_VERSION)
    parser.add_argument("--initial-max-chars", type=int)
    parser.add_argument("--final-max-chars", type=int)
    parser.add_argument("--initial-max-chunks", type=int)
    parser.add_argument("--final-max-chunks", type=int)
    parser.add_argument("--top-global", type=int)
    parser.add_argument("--top-per-option", type=int)
    parser.add_argument("--top-per-candidate-doc", type=int)
    parser.add_argument("--max-candidate-docs", type=int)
    parser.add_argument("--neighbor-radius", type=int)
    parser.add_argument("--locator-query-top-k", type=int)
    parser.add_argument("--max-locator-queries", type=int)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", action="store_true", dest="enable_thinking")
    thinking.add_argument("--disable-thinking", action="store_false", dest="enable_thinking")
    parser.set_defaults(enable_thinking=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=int_setting(settings, "B_QUESTION_WORKERS", 3),
    )
    return parser.parse_args()


def main() -> None:
    code_settings = load_code_settings()
    args = parse_args(code_settings)
    processed_dir = resolve_path(args.processed_dir)
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else default_output_dir(args.resume)
    )
    ensure_isolated_output(output_dir)

    if args.preprocess or not (processed_dir / "preprocess_report.json").exists():
        report = build_b_processed_data(
            questions_dir=DEFAULT_QUESTIONS_DIR,
            raw_dir=DEFAULT_RAW_DIR,
            processed_dir=processed_dir,
            submit_csv=DEFAULT_SUBMIT_CSV,
            progress=True,
            use_cache=True,
        )
        print(f"B preprocess complete: {report}", flush=True)
    processed = validate_processed_data(processed_dir)

    llm_config = llm_config_from_env(
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )
    default_thinking = bool_setting(code_settings, "B_ENABLE_THINKING", True)
    enable_thinking = (
        default_thinking if args.enable_thinking is None else args.enable_thinking
    )
    llm_config = replace(
        llm_config,
        enable_thinking=enable_thinking,
        max_output_tokens=0,
    )
    settings = BEnsembleSettings(
        initial_max_chars=args.initial_max_chars
        or int_setting(code_settings, "B_INITIAL_MAX_CONTEXT_CHARS", 48000),
        final_max_chars=args.final_max_chars
        or int_setting(code_settings, "B_FINAL_MAX_CONTEXT_CHARS", 90000),
        initial_max_chunks=args.initial_max_chunks
        or int_setting(code_settings, "B_INITIAL_MAX_CHUNKS", 30),
        final_max_chunks=args.final_max_chunks
        or int_setting(code_settings, "B_FINAL_MAX_CHUNKS", 56),
        top_global=args.top_global
        or int_setting(code_settings, "B_TOP_GLOBAL", 20),
        top_per_option=args.top_per_option
        or int_setting(code_settings, "B_TOP_PER_OPTION", 7),
        top_per_candidate_doc=args.top_per_candidate_doc
        or int_setting(code_settings, "B_TOP_PER_CANDIDATE_DOC", 4),
        max_candidate_docs=args.max_candidate_docs
        or int_setting(code_settings, "B_MAX_CANDIDATE_DOCS", 12),
        neighbor_radius=(
            args.neighbor_radius
            if args.neighbor_radius is not None
            else int_setting(code_settings, "B_NEIGHBOR_RADIUS", 2)
        ),
        locator_query_top_k=args.locator_query_top_k
        or int_setting(code_settings, "B_LOCATOR_QUERY_TOP_K", 6),
        max_locator_queries=args.max_locator_queries
        or int_setting(code_settings, "B_MAX_LOCATOR_QUERIES", 28),
        question_workers=max(1, args.workers),
        prompt_version=args.prompt_version,
    )

    manifest_path = output_dir / "manifest.json"
    expected = {
        "schema_version": 1,
        "run_id": output_dir.name,
        "prompt_version": settings.prompt_version,
        "dataset_sha256": processed["dataset_sha256"],
        "dataset_counts": {
            key: processed[key] for key in ("questions", "documents", "chunks")
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
    if manifest_path.exists():
        if not args.resume:
            raise SystemExit(
                f"Run directory already exists: {output_dir}. Use --resume."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("prompt_version", "dataset_sha256", "llm", "ensemble_settings"):
            if manifest.get(key) != expected[key]:
                raise RuntimeError(f"Resume manifest mismatch: {key}")
    else:
        if args.resume:
            raise SystemExit(f"Cannot resume without manifest: {manifest_path}")
        output_dir.mkdir(parents=True, exist_ok=False)
        inputs_dir = output_dir / "inputs"
        inputs_dir.mkdir()
        shutil.copy2(DEFAULT_SUBMIT_CSV, inputs_dir / "submit_template.csv")
        manifest = {
            **expected,
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "initialized",
        }
        atomic_write_json(manifest_path, manifest)

    stop_file = resolve_path(args.stop_file) if args.stop_file else output_dir / "STOP"
    if args.resume and stop_file.exists():
        stop_file.unlink()
    qid_filter = read_qid_filter(args.qid, args.qid_file)

    manifest["status"] = "running"
    manifest["started_or_resumed_at"] = datetime.now().astimezone().isoformat()
    manifest["stop_file"] = str(stop_file)
    atomic_write_json(manifest_path, manifest)

    print(
        f"LLM provider={llm_config.provider} model={llm_config.model} "
        f"endpoint={llm_config.endpoint} api_key={mask_key(llm_config.api_key)} "
        f"thinking={llm_config.enable_thinking}",
        flush=True,
    )
    print(
        f"B processed: questions={processed['questions']} "
        f"documents={processed['documents']} chunks={processed['chunks']} "
        f"spans={processed['chunks_with_spans']}",
        flush=True,
    )
    print(f"Output directory: {output_dir}", flush=True)
    print(
        f'Stop safely: New-Item -ItemType File -Force "{stop_file}"',
        flush=True,
    )
    resume_parts = [
        f"& {ps_quote(sys.executable)}",
        ps_quote(Path(__file__).resolve()),
        "--output-dir",
        ps_quote(output_dir),
        "--resume",
        "--no-preprocess",
        "--processed-dir",
        ps_quote(processed_dir),
        "--prompt-version",
        ps_quote(settings.prompt_version),
        "--workers",
        str(settings.question_workers),
        "--initial-max-chars",
        str(settings.initial_max_chars),
        "--final-max-chars",
        str(settings.final_max_chars),
        "--initial-max-chunks",
        str(settings.initial_max_chunks),
        "--final-max-chunks",
        str(settings.final_max_chunks),
        "--top-global",
        str(settings.top_global),
        "--top-per-option",
        str(settings.top_per_option),
        "--top-per-candidate-doc",
        str(settings.top_per_candidate_doc),
        "--max-candidate-docs",
        str(settings.max_candidate_docs),
        "--neighbor-radius",
        str(settings.neighbor_radius),
        "--locator-query-top-k",
        str(settings.locator_query_top_k),
        "--max-locator-queries",
        str(settings.max_locator_queries),
        "--enable-thinking" if llm_config.enable_thinking else "--disable-thinking",
    ]
    if args.qid_file:
        resume_parts.extend(["--qid-file", ps_quote(resolve_path(args.qid_file))])
    for qid in args.qid:
        resume_parts.extend(["--qid", ps_quote(qid)])
    if args.stop_file:
        resume_parts.extend(["--stop-file", ps_quote(stop_file)])
    resume_command = " ".join(resume_parts)
    (output_dir / "resume_command.ps1").write_text(
        resume_command + "\n",
        encoding="utf-8",
    )
    print(f"Resume command saved: {output_dir / 'resume_command.ps1'}", flush=True)

    client = OpenAICompatibleClient(llm_config)
    try:
        summary = run_b_ensemble(
            processed_dir=processed_dir,
            output_dir=output_dir,
            client=client,
            settings=settings,
            limit=args.limit,
            qid_filter=qid_filter,
            stop_file=stop_file,
        )
    except BEnsembleStopped as exc:
        manifest["status"] = "stopped"
        manifest["stopped_at"] = datetime.now().astimezone().isoformat()
        manifest["stop_reason"] = str(exc)
        atomic_write_json(manifest_path, manifest)
        print(str(exc), flush=True)
        print(f"Completed stages are saved under: {output_dir}", flush=True)
        return
    except Exception:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now().astimezone().isoformat()
        atomic_write_json(manifest_path, manifest)
        raise

    manifest["summary"] = summary
    if summary["questions_completed"] == EXPECTED_QUESTIONS:
        answer_csv = submission_from_answers(output_dir)
        manifest["status"] = "completed"
        manifest["answer_csv"] = str(answer_csv)
        print(f"B full submission ready: {answer_csv}", flush=True)
    else:
        manifest["status"] = "completed_partial"
        print(
            f"Partial audit saved: {summary['questions_completed']}/"
            f"{EXPECTED_QUESTIONS}; no submission CSV was generated.",
            flush=True,
        )
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    atomic_write_json(manifest_path, manifest)
    print(f"Summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
