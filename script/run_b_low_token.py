from __future__ import annotations

import argparse
from dataclasses import replace
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
    B_LOW_PROMPT_VERSION,
    BEnsembleStopped,
    run_b_low_token,
)
from agent.b_preprocess import (  # noqa: E402
    DEFAULT_PROCESSED_DIR,
    DEFAULT_QUESTIONS_DIR,
    DEFAULT_SUBMIT_CSV,
)
from agent.b_submission import (  # noqa: E402
    assert_answer_fields_exact_match,
    load_submission_spec,
    validate_b_submission,
    write_b_submission,
)
from agent.config import llm_config_from_env, load_code_settings  # noqa: E402
from agent.ensemble_audit import atomic_write_json  # noqa: E402
from agent.qwen_client import OpenAICompatibleClient  # noqa: E402


RUN_PREFIX = "b_low_token_"
FULL_PREFIX = "b_full_"
EXPECTED_QUESTIONS = 100


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def mask_key(value: str) -> str:
    if not value:
        return "(missing)"
    return value[:4] + "..." + value[-4:] if len(value) > 10 else value[:2] + "***"


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


def latest_full_run() -> Path:
    candidates = sorted(
        (
            path
            for path in (ROOT / "runs").glob(f"{FULL_PREFIX}*")
            if path.is_dir()
            and (path / "audit.json").exists()
            and (path / "answer.csv").exists()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            "No completed B full run exists. Run script\\run_b_full_ensemble.py first."
        )
    return candidates[0].resolve()


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
        raise SystemExit("--resume was requested, but no B low-token run exists.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (runs_dir / f"{RUN_PREFIX}{stamp}").resolve()


def ensure_isolated_output(output_dir: Path) -> None:
    try:
        output_dir.relative_to((ROOT / "runs").resolve())
    except ValueError as exc:
        raise SystemExit("B outputs must stay under D:\\tianchi\\runs.") from exc


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
    return answer_csv


def parse_args(settings: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run compact B verification and preserve full-run answers exactly."
    )
    parser.add_argument("--full-run-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--stop-file")
    parser.add_argument("--provider")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", action="store_true", dest="enable_thinking")
    thinking.add_argument("--disable-thinking", action="store_false", dest="enable_thinking")
    parser.set_defaults(enable_thinking=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=int_setting(settings, "B_LOW_TOKEN_WORKERS", 3),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=int_setting(settings, "B_LOW_TOKEN_MAX_OUTPUT_TOKENS", 160),
    )
    return parser.parse_args()


def main() -> None:
    code_settings = load_code_settings()
    args = parse_args(code_settings)
    processed_dir = resolve_path(args.processed_dir)
    if not (processed_dir / "questions.jsonl").exists():
        raise SystemExit(
            f"B processed data is missing: {processed_dir}. Run the full script first."
        )

    full_run_dir = (
        resolve_path(args.full_run_dir) if args.full_run_dir else latest_full_run()
    )
    full_audit = full_run_dir / "audit.json"
    full_answer = full_run_dir / "answer.csv"
    if not full_audit.exists() or not full_answer.exists():
        raise SystemExit(f"Full B run is incomplete: {full_run_dir}")
    full_report = validate_b_submission(
        full_answer,
        template_path=DEFAULT_SUBMIT_CSV,
        question_dir=DEFAULT_QUESTIONS_DIR,
    )
    full_report.require_valid()

    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else default_output_dir(args.resume)
    )
    ensure_isolated_output(output_dir)

    llm_config = llm_config_from_env(
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )
    default_thinking = bool_setting(
        code_settings, "B_LOW_TOKEN_ENABLE_THINKING", False
    )
    enable_thinking = (
        default_thinking if args.enable_thinking is None else args.enable_thinking
    )
    llm_config = replace(
        llm_config,
        enable_thinking=enable_thinking,
        max_output_tokens=max(1, args.max_output_tokens),
    )

    manifest_path = output_dir / "manifest.json"
    expected = {
        "schema_version": 1,
        "run_id": output_dir.name,
        "prompt_version": B_LOW_PROMPT_VERSION,
        "full_run_dir": str(full_run_dir),
        "full_audit_sha256": sha256_file(full_audit),
        "full_answer_sha256": sha256_file(full_answer),
        "llm": {
            "provider": llm_config.provider,
            "model": llm_config.model,
            "base_url": llm_config.base_url,
            "temperature": llm_config.temperature,
            "enable_thinking": llm_config.enable_thinking,
            "max_output_tokens": llm_config.max_output_tokens,
        },
        "workers": max(1, args.workers),
    }
    if manifest_path.exists():
        if not args.resume:
            raise SystemExit(
                f"Run directory already exists: {output_dir}. Use --resume."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "prompt_version",
            "full_audit_sha256",
            "full_answer_sha256",
            "llm",
            "workers",
        ):
            if manifest.get(key) != expected[key]:
                raise RuntimeError(f"Resume manifest mismatch: {key}")
    else:
        if args.resume:
            raise SystemExit(f"Cannot resume without manifest: {manifest_path}")
        output_dir.mkdir(parents=True, exist_ok=False)
        inputs_dir = output_dir / "inputs"
        inputs_dir.mkdir()
        shutil.copy2(full_answer, inputs_dir / "teacher_answer.csv")
        manifest = {
            **expected,
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "initialized",
        }
        atomic_write_json(manifest_path, manifest)

    stop_file = resolve_path(args.stop_file) if args.stop_file else output_dir / "STOP"
    if args.resume and stop_file.exists():
        stop_file.unlink()
    qid_filter = {qid.strip() for qid in args.qid if qid.strip()} or None

    manifest["status"] = "running"
    manifest["started_or_resumed_at"] = datetime.now().astimezone().isoformat()
    manifest["stop_file"] = str(stop_file)
    atomic_write_json(manifest_path, manifest)

    print(
        f"LLM provider={llm_config.provider} model={llm_config.model} "
        f"endpoint={llm_config.endpoint} api_key={mask_key(llm_config.api_key)} "
        f"thinking={llm_config.enable_thinking} "
        f"max_output_tokens={llm_config.max_output_tokens}",
        flush=True,
    )
    print(f"Teacher full run: {full_run_dir}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(
        f'Stop safely: New-Item -ItemType File -Force "{stop_file}"',
        flush=True,
    )
    print(
        f'Update later: python script\\run_b_low_token.py '
        f'--full-run-dir "{full_run_dir}" --output-dir "{output_dir}" --resume',
        flush=True,
    )

    client = OpenAICompatibleClient(llm_config)
    try:
        summary = run_b_low_token(
            processed_dir=processed_dir,
            full_audit_path=full_audit,
            output_dir=output_dir,
            client=client,
            stop_file=stop_file,
            limit=args.limit,
            qid_filter=qid_filter,
            workers=max(1, args.workers),
        )
    except BEnsembleStopped as exc:
        manifest["status"] = "stopped"
        manifest["stopped_at"] = datetime.now().astimezone().isoformat()
        manifest["stop_reason"] = str(exc)
        atomic_write_json(manifest_path, manifest)
        print(str(exc), flush=True)
        return
    except Exception:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now().astimezone().isoformat()
        atomic_write_json(manifest_path, manifest)
        raise

    manifest["summary"] = summary
    if summary["questions_completed"] == EXPECTED_QUESTIONS:
        answer_csv = submission_from_answers(output_dir)
        assert_answer_fields_exact_match(
            full_answer,
            answer_csv,
            template_path=DEFAULT_SUBMIT_CSV,
            question_dir=DEFAULT_QUESTIONS_DIR,
        )
        manifest["status"] = "completed"
        manifest["answer_csv"] = str(answer_csv)
        manifest["answers_exact_match"] = True
        print(f"B low-token submission ready: {answer_csv}", flush=True)
        print("Full/low answer_1..answer_4 exact match: YES", flush=True)
    else:
        manifest["status"] = "completed_partial"
        print(
            f"Partial low-token run saved: {summary['questions_completed']}/"
            f"{EXPECTED_QUESTIONS}; no submission CSV was generated.",
            flush=True,
        )
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    atomic_write_json(manifest_path, manifest)
    print(f"Summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
