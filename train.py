from __future__ import annotations

import argparse
import csv
from datetime import datetime
import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent.config import llm_config_from_env


NO_API_MODES = {
    "preprocess",
    "dry-run",
    "agent",
    "evaluate",
    "trace",
    "harness",
    "check",
    "stop",
    "clear-stop",
}
MODE_ALIASES = {
    "train": "targeted",
    "answer": "full",
    "refine": "targeted",
    "target": "targeted",
    "wide": "broad",
    "final": "precision",
    "polish": "precision",
    "continue": "resume",
    "validate": "check",
    "cheap": "low-token",
    "low": "low-token",
    "micro": "micro",
    "ultra": "micro",
    "cheapest": "micro",
    "super-low": "super-low",
    "superlow": "super-low",
    "nano": "super-low",
    "runtime": "agent",
    "eval": "evaluate",
    "workflow": "harness",
}


def load_local_config() -> Any:
    try:
        return importlib.import_module("local_config")
    except ModuleNotFoundError:
        return None


LOCAL = load_local_config()


def setting(name: str, default: Any = None) -> Any:
    return getattr(LOCAL, name, default) if LOCAL is not None else default


def normalize_mode(mode: str | None) -> str:
    raw = (mode or setting("RUN_MODE", "targeted") or "targeted").strip().lower()
    return MODE_ALIASES.get(raw, raw)


def bool_setting(name: str, default: bool = False) -> bool:
    value = setting(name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def optional_int(value: Any) -> int | None:
    if value in {None, "", 0, "0"}:
        return None
    return int(value)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def mask_key(api_key: str) -> str:
    if not api_key:
        return "<missing>"
    if len(api_key) <= 8:
        return "***"
    return api_key[:4] + "..." + api_key[-4:]


def run_step(args: list[str], title: str) -> None:
    print(f"\n=== {title} ===")
    print(" ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def script(name: str) -> str:
    return str(ROOT / "script" / name)


def ensure_api_ready(allow_non_qwen: bool) -> None:
    config = llm_config_from_env()
    print(
        f"LLM from local_config/env: provider={config.provider} model={config.model} "
        f"endpoint={config.endpoint} api_key={mask_key(config.api_key)}"
    )
    if not config.api_key:
        raise SystemExit("请先打开 D:\\tianchi\\local_config.py，填写 API_KEY。")
    if not allow_non_qwen and "qwen" not in config.model.lower():
        raise SystemExit(
            f"当前 MODEL='{config.model}' 不像 Qwen 系列。比赛提交请在 local_config.py 中设置 Qwen 模型名，"
            "或者仅本地测试时加 --allow-non-qwen。"
        )


def output_dir_for(mode: str, args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = project_path(args.output_dir)
    else:
        root = project_path(str(setting("OUTPUT_DIR", "runs")))
        run_name = str(setting("RUN_NAME", "") or "").strip()
        if not run_name and bool_setting("TIMESTAMP_RUN_DIR", False):
            run_name = datetime.now().strftime(f"%Y%m%d_%H%M%S_{mode}")
        output_dir = root / (run_name or mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def answer_path_for(mode: str, args: argparse.Namespace) -> Path:
    return project_path(args.answer_csv) if args.answer_csv else output_dir_for(mode, args) / "answer.csv"


def evidence_path_for(mode: str, args: argparse.Namespace) -> Path:
    return project_path(args.evidence_json) if args.evidence_json else output_dir_for(mode, args) / "evidence.json"


def checkpoint_csv_for(answer_csv: Path) -> Path:
    return answer_csv.with_name("answer.checkpoint.csv")


def checkpoint_evidence_for(evidence_json: Path) -> Path:
    return evidence_json.with_name("evidence.checkpoint.jsonl")


def cache_dir_for(answer_csv: Path) -> Path:
    return answer_csv.parent / "cache" / "questions"


def common_answer_args(
    answer_csv: Path,
    evidence_json: Path,
    review_mode: str,
    args: argparse.Namespace,
    *,
    evidence_mode: str | None = None,
    initial_policy: str = "replace",
    default_max_context_chars: int | None = None,
    disable_thinking: bool = False,
    max_output_tokens: int | None = None,
) -> list[str]:
    answer_csv.parent.mkdir(parents=True, exist_ok=True)
    evidence_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        script("run_answer.py"),
        "--evidence-mode",
        evidence_mode or str(setting("EVIDENCE_MODE", "compact")),
        "--initial-policy",
        initial_policy,
        "--review-mode",
        review_mode,
        "--review-policy",
        str(setting("REVIEW_POLICY", "evidence_gate")),
        "--answer-csv",
        str(answer_csv),
        "--evidence-json",
        str(evidence_json),
        "--checkpoint-csv",
        str(checkpoint_csv_for(answer_csv)),
        "--checkpoint-evidence",
        str(checkpoint_evidence_for(evidence_json)),
        "--cache-dir",
        str(cache_dir_for(answer_csv)),
    ]
    configured_max_context = (
        args.max_context_chars
        if args.max_context_chars is not None
        else default_max_context_chars
        if default_max_context_chars is not None
        else setting("MAX_CONTEXT_CHARS", 0)
    )
    max_context_chars = optional_int(configured_max_context)
    if max_context_chars:
        cmd += ["--max-context-chars", str(max_context_chars)]
    limit = optional_int(args.limit if args.limit is not None else setting("LIMIT", None))
    if limit:
        cmd += ["--limit", str(limit)]
    if args.allow_non_qwen or bool_setting("ALLOW_NON_QWEN", False):
        cmd.append("--allow-non-qwen")
    if disable_thinking:
        cmd.append("--disable-thinking")
    if max_output_tokens:
        cmd += ["--max-output-tokens", str(max_output_tokens)]
    return cmd


def ensure_preprocessed(force: bool = False) -> None:
    chunks = ROOT / "processed_data" / "chunks.jsonl"
    questions = ROOT / "processed_data" / "questions.jsonl"
    if force or not chunks.exists() or not questions.exists():
        run_step([sys.executable, script("run_preprocess.py")], "preprocess dataset")


def require_file(path: str | Path, hint: str) -> None:
    full_path = project_path(path)
    if not full_path.exists():
        raise SystemExit(f"缺少 {full_path}。{hint}")


def qid_count_in_csv(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return sum(1 for row in csv.DictReader(f) if (row.get("qid") or "").strip())


def final_answer_path() -> Path:
    return project_path(str(setting("SUBMISSION_ANSWER_CSV", "answer.csv")))


def final_evidence_path() -> Path:
    return project_path(str(setting("SUBMISSION_EVIDENCE_JSON", "evidence.json")))


def same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def finalize_outputs(answer_csv: Path, evidence_json: Path, args: argparse.Namespace) -> None:
    run_step([sys.executable, script("check_submission.py"), "--file", str(answer_csv)], "check run folder answer.csv")
    if args.no_sync or not bool_setting("SYNC_TO_SUBMISSION", True):
        print(f"Run result kept in folder: {answer_csv.parent}")
        print(f"Submission sync skipped. Current final answer remains: {final_answer_path()}")
        return

    final_answer = final_answer_path()
    final_evidence = final_evidence_path()
    final_answer.parent.mkdir(parents=True, exist_ok=True)
    final_evidence.parent.mkdir(parents=True, exist_ok=True)

    if not same_file(answer_csv, final_answer):
        shutil.copy2(answer_csv, final_answer)
    if evidence_json.exists() and not same_file(evidence_json, final_evidence):
        shutil.copy2(evidence_json, final_evidence)

    run_step([sys.executable, script("check_submission.py"), "--file", str(final_answer)], "check final answer.csv")
    print("\nFinal files ready:")
    print(f"  archived answer:  {answer_csv}")
    print(f"  archived evidence: {evidence_json}")
    print(f"  submit answer.csv: {final_answer}")
    print(f"  latest evidence:   {final_evidence}")


def run_full(args: argparse.Namespace, resume: bool = False) -> tuple[Path, Path]:
    answer_csv = answer_path_for("full", args)
    evidence_json = evidence_path_for("full", args)
    review_mode = args.review_mode or setting("REVIEW_MODE", "auto")
    cmd = common_answer_args(answer_csv, evidence_json, review_mode, args)
    if resume:
        cmd.append("--resume")
    run_step(cmd, "generate full answer")
    return answer_csv, evidence_json


def run_targeted(args: argparse.Namespace, *, broad: bool = False, resume: bool = False) -> tuple[Path, Path]:
    scope = "broad" if broad else "targeted"
    output_dir = output_dir_for(scope, args)
    suspect_csv = output_dir / f"answer_suspects_{scope}.csv"
    base_answer = project_path(str(setting("BASE_ANSWER_CSV", setting("SUBMISSION_ANSWER_CSV", "answer.csv"))))
    base_evidence = project_path(str(setting("BASE_EVIDENCE_JSON", setting("SUBMISSION_EVIDENCE_JSON", "evidence.json"))))
    require_file(base_answer, "如果还没有基础答案，先运行：python train.py --mode full")
    require_file(base_evidence, "如果还没有基础证据，先运行：python train.py --mode full")

    if not resume:
        run_step(
            [
                sys.executable,
                script("analyze_answer_quality.py"),
                "--answer-csv",
                str(base_answer),
                "--evidence-json",
                str(base_evidence),
                "--analysis-stage",
                "initial",
                "--risk-scope",
                scope,
                "--suspect-output",
                str(suspect_csv),
            ],
            f"build {scope} risk list",
        )

    answer_csv = answer_path_for(scope, args)
    evidence_json = evidence_path_for(scope, args)
    review_mode = args.review_mode or ("broad" if broad else setting("REVIEW_MODE", "auto"))
    cmd = common_answer_args(answer_csv, evidence_json, review_mode, args)
    cmd += [
        "--qid-file",
        str(suspect_csv),
        "--base-answer-csv",
        str(base_answer),
        "--base-evidence-json",
        str(base_evidence),
    ]
    if resume:
        cmd.append("--resume")
    run_step(cmd, f"generate refined {scope} answer")
    return answer_csv, evidence_json


def run_precision(args: argparse.Namespace, resume: bool = False) -> tuple[Path, Path]:
    scope = "precision"
    output_dir = output_dir_for(scope, args)
    suspect_csv = output_dir / "remaining_final_risks.csv"
    base_answer = final_answer_path()
    base_evidence = final_evidence_path()
    require_file(base_answer, "请先完成 targeted 或 broad，再运行 precision。")
    require_file(base_evidence, "请先完成 targeted 或 broad，再运行 precision。")

    if not resume:
        run_step(
            [
                sys.executable,
                script("analyze_answer_quality.py"),
                "--answer-csv",
                str(base_answer),
                "--evidence-json",
                str(base_evidence),
                "--analysis-stage",
                "final",
                "--risk-scope",
                "targeted",
                "--suspect-output",
                str(suspect_csv),
            ],
            "build final precision risk list",
        )

    remaining = qid_count_in_csv(suspect_csv)
    print(f"Precision remaining qids: {remaining}")
    if remaining == 0:
        print("No final structural risks remain; precision API run skipped.")
        return base_answer, base_evidence

    answer_csv = answer_path_for(scope, args)
    evidence_json = evidence_path_for(scope, args)
    review_mode = args.review_mode or str(setting("PRECISION_REVIEW_MODE", "always"))
    cmd = common_answer_args(answer_csv, evidence_json, review_mode, args)
    cmd += [
        "--qid-file",
        str(suspect_csv),
        "--base-answer-csv",
        str(base_answer),
        "--base-evidence-json",
        str(base_evidence),
    ]
    if resume:
        cmd.append("--resume")
    run_step(cmd, "generate precision answer")
    return answer_csv, evidence_json


def run_low_token(args: argparse.Namespace, resume: bool = False) -> tuple[Path, Path]:
    scope = "low_token"
    output_dir = output_dir_for(scope, args)
    answer_csv = answer_path_for(scope, args)
    evidence_json = evidence_path_for(scope, args)
    base_answer = project_path(
        str(setting("LOW_TOKEN_BASE_ANSWER_CSV", "runs/preserved_score_65_6064/answer.csv"))
    )
    require_file(base_answer, "缺少已得分答案归档，请先确认 runs/preserved_score_65_6064/answer.csv 存在。")

    cmd = common_answer_args(
        answer_csv,
        evidence_json,
        "off",
        args,
        evidence_mode="minimal",
        initial_policy=str(setting("LOW_TOKEN_INITIAL_POLICY", "preserve")),
        default_max_context_chars=int(setting("LOW_TOKEN_MAX_CONTEXT_CHARS", 12000)),
        disable_thinking=True,
        max_output_tokens=int(setting("LOW_TOKEN_MAX_OUTPUT_TOKENS", 600)),
    )
    cmd += ["--base-answer-csv", str(base_answer)]
    if resume:
        cmd.append("--resume")
    run_step(cmd, "generate low-token evidence-gated answer")
    return answer_csv, evidence_json


def run_micro(args: argparse.Namespace, resume: bool = False) -> tuple[Path, Path]:
    scope = "micro"
    output_dir = output_dir_for(scope, args)
    suspect_csv = output_dir / "answer_suspects_micro.csv"
    base_answer = project_path(str(setting("MICRO_BASE_ANSWER_CSV", setting("SUBMISSION_ANSWER_CSV", "answer.csv"))))
    base_evidence = project_path(str(setting("MICRO_BASE_EVIDENCE_JSON", setting("SUBMISSION_EVIDENCE_JSON", "evidence.json"))))
    risk_scope = str(setting("MICRO_RISK_SCOPE", "targeted"))
    require_file(base_answer, "Run a baseline first, then use micro mode to audit only the risky questions.")
    require_file(base_evidence, "Run a baseline first, then use micro mode to reuse existing evidence.")

    if risk_scope not in {"targeted", "broad"}:
        raise SystemExit("MICRO_RISK_SCOPE must be targeted or broad")

    if not resume:
        run_step(
            [
                sys.executable,
                script("analyze_answer_quality.py"),
                "--answer-csv",
                str(base_answer),
                "--evidence-json",
                str(base_evidence),
                "--analysis-stage",
                "final",
                "--risk-scope",
                risk_scope,
                "--suspect-output",
                str(suspect_csv),
            ],
            f"build micro {risk_scope} risk list",
        )

    remaining = qid_count_in_csv(suspect_csv)
    print(f"Micro qids: {remaining}")
    if remaining == 0:
        print("No structural risks found; micro API run skipped.")
        return base_answer, base_evidence

    answer_csv = answer_path_for(scope, args)
    evidence_json = evidence_path_for(scope, args)
    cmd = common_answer_args(
        answer_csv,
        evidence_json,
        args.review_mode or str(setting("MICRO_REVIEW_MODE", "off")),
        args,
        evidence_mode="micro",
        initial_policy=str(setting("MICRO_INITIAL_POLICY", "preserve")),
        default_max_context_chars=int(setting("MICRO_MAX_CONTEXT_CHARS", 7000)),
        disable_thinking=True,
        max_output_tokens=int(setting("MICRO_MAX_OUTPUT_TOKENS", 220)),
    )
    cmd += [
        "--qid-file",
        str(suspect_csv),
        "--base-answer-csv",
        str(base_answer),
        "--base-evidence-json",
        str(base_evidence),
    ]
    if resume:
        cmd.append("--resume")
    run_step(cmd, "generate micro token audit")
    return answer_csv, evidence_json


def run_super_low(args: argparse.Namespace, resume: bool = False) -> tuple[Path, Path]:
    scope = "super_low_token"
    answer_csv = answer_path_for(scope, args)
    evidence_json = evidence_path_for(scope, args)
    base_answer = project_path(
        str(setting("SUPER_LOW_BASE_ANSWER_CSV", setting("SUBMISSION_ANSWER_CSV", "answer.csv")))
    )
    require_file(base_answer, "A complete baseline answer.csv is required for super-low mode.")

    cmd = common_answer_args(
        answer_csv,
        evidence_json,
        "off",
        args,
        evidence_mode="nano",
        initial_policy="preserve",
        default_max_context_chars=int(setting("SUPER_LOW_MAX_CONTEXT_CHARS", 4000)),
        disable_thinking=True,
        max_output_tokens=int(setting("SUPER_LOW_MAX_OUTPUT_TOKENS", 64)),
    )
    cmd += ["--base-answer-csv", str(base_answer)]
    if resume:
        cmd.append("--resume")
    run_step(cmd, "generate complete super-low-token answer")
    return answer_csv, evidence_json


def run_check(args: argparse.Namespace) -> None:
    check_file = project_path(args.check_file) if args.check_file else final_answer_path()
    run_step([sys.executable, script("check_submission.py"), "--file", str(check_file)], "check submission format")


def run_agent_runtime(args: argparse.Namespace, *, execute_llm: bool = False) -> None:
    cmd = [sys.executable, script("run_agent_runtime.py")]
    limit = optional_int(args.limit if args.limit is not None else setting("LIMIT", 5))
    if limit:
        cmd += ["--limit", str(limit)]
    if args.agent_output:
        cmd += ["--output", str(project_path(args.agent_output))]
    if args.agent_llm or execute_llm:
        cmd.append("--execute-llm")
    if args.harness_attempts != 1:
        cmd += ["--max-attempts", str(args.harness_attempts)]
    if args.fail_fast:
        cmd.append("--fail-fast")
    run_step(cmd, "run agent runtime")


def main() -> None:
    parser = argparse.ArgumentParser(description="VSCode-friendly one-command runner for the Tianchi QA project.")
    parser.add_argument("--mode", default=None, help="targeted/broad/precision/micro/super-low/full/low-token/resume/preprocess/dry-run/agent/evaluate/trace/harness/check/stop/clear-stop")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions for debugging.")
    parser.add_argument("--review-mode", choices=["off", "auto", "broad", "always"], default=None)
    parser.add_argument("--max-context-chars", type=int, default=None)
    parser.add_argument("--answer-csv", default=None, help="Override archived run answer path. Default: runs/<mode>/answer.csv")
    parser.add_argument("--evidence-json", default=None, help="Override archived run evidence path. Default: runs/<mode>/evidence.json")
    parser.add_argument("--output-dir", default=None, help="Override archived run folder. Default: OUTPUT_DIR/<mode>")
    parser.add_argument("--check-file", default=None)
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing before answer generation.")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip automatic preprocessing checks.")
    parser.add_argument("--no-sync", action="store_true", help="Do not copy archived run result to final answer.csv/evidence.json.")
    parser.add_argument("--allow-non-qwen", action="store_true", help="Only for non-competition local tests.")
    parser.add_argument("--agent-llm", action="store_true", help="For --mode agent: execute Qwen calls instead of the API-free trace.")
    parser.add_argument("--agent-output", default=None, help="For --mode agent: JSON trace output path.")
    parser.add_argument("--harness-attempts", type=int, default=1, help="Agent Harness attempts per task; raw usage is aggregated.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop Agent Harness batch execution after a failed task.")
    args = parser.parse_args()

    mode = normalize_mode(args.mode)
    if mode not in {"targeted", "broad", "precision", "micro", "super-low", "full", "low-token", "resume", "preprocess", "dry-run", "agent", "evaluate", "trace", "harness", "check", "stop", "clear-stop"}:
        raise SystemExit(f"Unknown mode: {mode}")

    if mode not in NO_API_MODES or args.agent_llm:
        ensure_api_ready(args.allow_non_qwen or bool_setting("ALLOW_NON_QWEN", False))

    if mode == "preprocess":
        ensure_preprocessed(force=True)
        return
    if mode == "dry-run":
        ensure_preprocessed(force=args.preprocess)
        cmd = [sys.executable, script("run_answer.py"), "--dry-run"]
        limit = optional_int(args.limit if args.limit is not None else setting("LIMIT", 5))
        if limit:
            cmd += ["--limit", str(limit)]
        run_step(cmd, "dry-run retrieval preview")
        return
    if mode in {"agent", "trace", "harness"}:
        ensure_preprocessed(force=args.preprocess)
        run_agent_runtime(args)
        return
    if mode == "evaluate":
        ensure_preprocessed(force=args.preprocess)
        run_agent_runtime(args, execute_llm=False)
        return
    if mode == "check":
        run_check(args)
        return
    if mode == "stop":
        run_step([sys.executable, script("request_stop.py")], "request safe stop")
        return
    if mode == "clear-stop":
        run_step([sys.executable, script("request_stop.py"), "--clear"], "clear stop signal")
        return
    if not args.no_preprocess and (args.preprocess or bool_setting("PREPROCESS_BEFORE_RUN", False)):
        ensure_preprocessed(force=args.preprocess)
    else:
        ensure_preprocessed(force=False)

    if mode == "full":
        answer_csv, evidence_json = run_full(args)
    elif mode == "targeted":
        answer_csv, evidence_json = run_targeted(args, broad=False)
    elif mode == "broad":
        answer_csv, evidence_json = run_targeted(args, broad=True)
    elif mode == "precision":
        answer_csv, evidence_json = run_precision(args)
    elif mode == "micro":
        answer_csv, evidence_json = run_micro(args)
    elif mode == "super-low":
        answer_csv, evidence_json = run_super_low(args)
    elif mode == "low-token":
        answer_csv, evidence_json = run_low_token(args)
    else:
        resume_mode = normalize_mode(setting("RESUME_MODE", setting("RUN_MODE", "targeted")))
        if resume_mode == "resume":
            resume_mode = "targeted"
        if resume_mode == "broad":
            answer_csv, evidence_json = run_targeted(args, broad=True, resume=True)
        elif resume_mode == "precision":
            answer_csv, evidence_json = run_precision(args, resume=True)
        elif resume_mode == "micro":
            answer_csv, evidence_json = run_micro(args, resume=True)
        elif resume_mode == "super-low":
            answer_csv, evidence_json = run_super_low(args, resume=True)
        elif resume_mode == "full":
            answer_csv, evidence_json = run_full(args, resume=True)
        elif resume_mode == "low-token":
            answer_csv, evidence_json = run_low_token(args, resume=True)
        else:
            answer_csv, evidence_json = run_targeted(args, broad=False, resume=True)

    if mode in {"precision", "micro", "super-low", "low-token"}:
        args.no_sync = True
        print(f"{mode} is experimental: result archived but not copied to the root submission file.")
    if args.limit:
        print(f"Debug limit={args.limit}: partial output kept at {answer_csv}; full submission validation skipped.")
        return
    finalize_outputs(answer_csv, evidence_json, args)


if __name__ == "__main__":
    main()
