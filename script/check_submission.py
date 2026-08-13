from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FIELDS = ["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"]


def load_questions(root: Path) -> dict[str, dict]:
    qdir = root / "public_dataset_a" / "public_dataset_upload" / "questions" / "group_a"
    questions: list[dict] = []
    for path in sorted(qdir.glob("*.json")):
        questions.extend(json.loads(path.read_text(encoding="utf-8")))
    return {q["qid"]: q for q in questions}


def is_uint(value: str | None) -> bool:
    return bool(re.fullmatch(r"\d+", str(value or "")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate answer.csv submission format.")
    parser.add_argument("--root", default=str(ROOT), help="Project root directory.")
    parser.add_argument("--file", default="answer.csv", help="Submission CSV path.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    csv_path = Path(args.file)
    if not csv_path.is_absolute():
        csv_path = root / csv_path

    if not csv_path.exists():
        raise SystemExit(f"NOT FOUND: {csv_path}")

    qmap = load_questions(root)
    errors: list[str] = []
    warnings: list[str] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)

    if fields != EXPECTED_FIELDS:
        errors.append(f"column mismatch: {fields} != {EXPECTED_FIELDS}")
    if not rows:
        errors.append("csv has no data rows")
    else:
        summary = rows[0]
        if summary.get("qid") != "summary":
            errors.append("first row qid must be summary")
        if summary.get("answer", ""):
            errors.append("summary answer must be empty")
        for col in EXPECTED_FIELDS[2:]:
            if not is_uint(summary.get(col)):
                errors.append(f"summary {col} must be a non-negative integer: {summary.get(col)!r}")

    seen: set[str] = set()
    sums = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for line_no, row in enumerate(rows[1:], start=3):
        qid = row.get("qid", "")
        if qid in seen:
            errors.append(f"line {line_no}: duplicated qid {qid}")
        seen.add(qid)

        question = qmap.get(qid)
        if question is None:
            errors.append(f"line {line_no}: unknown qid {qid}")
            answer_format = "multi"
        else:
            answer_format = question.get("answer_format", "multi")

        answer = row.get("answer", "") or ""
        if answer_format == "multi":
            if not re.fullmatch(r"[ABCD]+", answer):
                errors.append(f"line {line_no}: invalid multi answer for {qid}: {answer!r}")
            elif "".join(sorted(set(answer))) != answer:
                errors.append(f"line {line_no}: multi answer must be unique and sorted for {qid}: {answer!r}")
        elif answer_format == "mcq":
            if not re.fullmatch(r"[ABCD]", answer):
                errors.append(f"line {line_no}: invalid mcq answer for {qid}: {answer!r}")
        elif answer_format == "tf":
            if not re.fullmatch(r"[AB]", answer):
                errors.append(f"line {line_no}: invalid tf answer for {qid}: {answer!r}")
        else:
            warnings.append(f"line {line_no}: unknown answer_format {answer_format!r} for {qid}")

        token_values: dict[str, int] = {}
        for col in EXPECTED_FIELDS[2:]:
            value = row.get(col)
            if not is_uint(value):
                errors.append(f"line {line_no}: {qid}.{col} must be a non-negative integer: {value!r}")
            else:
                token_values[col] = int(value or 0)
                sums[col] += token_values[col]
        if len(token_values) == 3:
            if token_values["prompt_tokens"] + token_values["completion_tokens"] != token_values["total_tokens"]:
                errors.append(f"line {line_no}: token sum mismatch for {qid}")

    missing = sorted(set(qmap) - seen)
    extra = sorted(seen - set(qmap))
    if missing:
        errors.append(f"missing qids: {len(missing)}; first 10={missing[:10]}")
    if extra:
        errors.append(f"extra qids: {len(extra)}; first 10={extra[:10]}")
    if len(rows) > 1 and len(rows[1:]) != len(qmap):
        errors.append(f"question row count mismatch: {len(rows[1:])} != {len(qmap)}")

    if rows and all(is_uint(rows[0].get(col)) for col in EXPECTED_FIELDS[2:]):
        summary_sums = {col: int(rows[0][col]) for col in EXPECTED_FIELDS[2:]}
        if summary_sums != sums:
            errors.append(f"summary tokens mismatch: summary={summary_sums}, sum={sums}")
        if summary_sums["prompt_tokens"] + summary_sums["completion_tokens"] != summary_sums["total_tokens"]:
            errors.append("summary token sum mismatch")

    print(f"file: {csv_path}")
    print(f"columns: {fields}")
    print(f"rows: {len(rows)} including summary")
    print(f"question_rows: {max(len(rows) - 1, 0)} / expected {len(qmap)}")
    print(f"errors: {len(errors)}")
    for error in errors[:80]:
        print(f"ERROR: {error}")
    if len(errors) > 80:
        print(f"ERROR_MORE: {len(errors) - 80}")
    print(f"warnings: {len(warnings)}")
    for warning in warnings[:30]:
        print(f"WARN: {warning}")

    if errors:
        sys.exit(1)
    print("OK: submission format looks valid.")


if __name__ == "__main__":
    main()
