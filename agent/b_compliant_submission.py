from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


OFFICIAL_FIELDS = (
    "qid",
    "answer_1",
    "answer_2",
    "answer_3",
    "answer_4",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning",
)
ANSWER_FIELDS = ("answer_1", "answer_2", "answer_3", "answer_4")
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


@dataclass
class ComplianceReport:
    path: Path
    expected_questions: int
    question_rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    token_totals: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def require_valid(self) -> None:
        if self.errors:
            preview = "\n".join(f"- {item}" for item in self.errors[:20])
            raise ValueError(f"B compliant submission is invalid:\n{preview}")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.reader(handle))
    if not raw_rows:
        return [], [], ["csv is empty"]

    fields = raw_rows[0]
    rows: list[dict[str, str]] = []
    errors: list[str] = []
    for line_no, values in enumerate(raw_rows[1:], start=2):
        if len(values) != len(fields):
            errors.append(
                f"line {line_no}: expected {len(fields)} columns, got {len(values)}"
            )
        values = (values + [""] * len(fields))[: len(fields)]
        rows.append(dict(zip(fields, values)))
    return fields, rows, errors


def template_qids(template_path: str | Path) -> list[str]:
    path = Path(template_path).resolve()
    fields, rows, errors = _read_csv(path)
    if errors:
        raise ValueError("\n".join(errors))
    if not fields or fields[0] != "qid":
        raise ValueError(f"invalid B template header: {fields}")
    qids = [str(row.get("qid") or "").strip() for row in rows[1:]]
    if not qids or any(not qid for qid in qids):
        raise ValueError("B template has missing qids")
    if len(qids) != len(set(qids)):
        raise ValueError("B template has duplicate qids")
    return qids


def read_legacy_answers(
    answer_csv: str | Path,
    *,
    expected_qids: Iterable[str],
) -> dict[str, list[str]]:
    """Read answers from either the official underscored or legacy compact header."""
    path = Path(answer_csv).resolve()
    fields, rows, errors = _read_csv(path)
    if errors:
        raise ValueError("\n".join(errors))
    compact_fields = ("answer1", "answer2", "answer3", "answer4")
    source_fields = ANSWER_FIELDS if set(ANSWER_FIELDS).issubset(fields) else compact_fields
    if not set(source_fields).issubset(fields):
        raise ValueError(
            f"{path} has no recognized answer columns; found {fields}"
        )

    answers: dict[str, list[str]] = {}
    for row in rows:
        qid = str(row.get("qid") or "").strip()
        if not qid or qid == "summary":
            continue
        values = [str(row.get(field) or "").strip() for field in source_fields]
        while values and not values[-1]:
            values.pop()
        if not values or any(not value for value in values):
            raise ValueError(f"{path}: {qid} has non-contiguous answer slots")
        answers[qid] = values

    expected = list(expected_qids)
    missing = [qid for qid in expected if qid not in answers]
    extra = sorted(set(answers) - set(expected))
    if missing or extra:
        raise ValueError(
            f"{path}: answer qids mismatch; missing={missing[:10]} extra={extra[:10]}"
        )
    return {qid: answers[qid] for qid in expected}


def _parse_uint(value: Any, label: str) -> int:
    text = str(value if value is not None else "").strip()
    if not text.isdigit():
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
    return int(text)


def build_rows(
    *,
    expected_qids: Iterable[str],
    records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    totals = {field: 0 for field in TOKEN_FIELDS}
    question_rows: list[dict[str, str]] = []

    for qid in expected_qids:
        record = records.get(qid)
        if not record:
            raise ValueError(f"missing completed record for {qid}")
        answers = record.get("answers")
        if not isinstance(answers, list) or not 1 <= len(answers) <= 4:
            raise ValueError(f"{qid}: answers must contain one to four values")
        answers = [str(value).strip() for value in answers]
        if any(not value for value in answers):
            raise ValueError(f"{qid}: answers include an empty slot")
        usage = record.get("usage")
        if not isinstance(usage, Mapping):
            raise ValueError(f"{qid}: missing raw API usage")
        reasoning = str(record.get("reasoning") or "").strip()
        if len(reasoning) < 20:
            raise ValueError(f"{qid}: reasoning must contain at least 20 characters")
        prompt = _parse_uint(usage.get("prompt_tokens"), f"{qid}.prompt_tokens")
        completion = _parse_uint(
            usage.get("completion_tokens"), f"{qid}.completion_tokens"
        )
        total = _parse_uint(usage.get("total_tokens"), f"{qid}.total_tokens")
        if prompt <= 0 or completion <= 0:
            raise ValueError(f"{qid}: raw API usage must be positive")
        if prompt + completion != total:
            raise ValueError(f"{qid}: total_tokens must equal prompt + completion")

        row = {field: "" for field in OFFICIAL_FIELDS}
        row["qid"] = qid
        for index, answer in enumerate(answers):
            row[ANSWER_FIELDS[index]] = answer
        row["prompt_tokens"] = str(prompt)
        row["completion_tokens"] = str(completion)
        row["total_tokens"] = str(total)
        row["reasoning"] = reasoning
        question_rows.append(row)
        totals["prompt_tokens"] += prompt
        totals["completion_tokens"] += completion
        totals["total_tokens"] += total

    summary = {field: "" for field in OFFICIAL_FIELDS}
    summary["qid"] = "summary"
    for field in TOKEN_FIELDS:
        summary[field] = str(totals[field])
    return [summary, *question_rows]


def write_submission(output_path: str | Path, rows: list[dict[str, str]]) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OFFICIAL_FIELDS,
            lineterminator="\r\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)
    return path


def validate_submission(
    csv_path: str | Path,
    *,
    expected_qids: Iterable[str],
    min_total_tokens: int = 500_000,
    max_total_tokens: int = 5_000_000,
) -> ComplianceReport:
    path = Path(csv_path).resolve()
    expected = list(expected_qids)
    report = ComplianceReport(path=path, expected_questions=len(expected))
    if not path.exists():
        report.errors.append("submission file does not exist")
        return report

    fields, rows, read_errors = _read_csv(path)
    report.errors.extend(read_errors)
    if tuple(fields) != OFFICIAL_FIELDS:
        report.errors.append(
            f"header mismatch: {fields} != {list(OFFICIAL_FIELDS)}"
        )
    if not rows:
        report.errors.append("submission has no data rows")
        return report

    summary = rows[0]
    if summary.get("qid") != "summary":
        report.errors.append("first data row must be summary")
    for field in ANSWER_FIELDS:
        if str(summary.get(field) or ""):
            report.errors.append(f"summary.{field} must be empty")
    if str(summary.get("reasoning") or ""):
        report.errors.append("summary.reasoning must be empty")

    question_rows = rows[1:]
    report.question_rows = len(question_rows)
    actual_qids = [str(row.get("qid") or "") for row in question_rows]
    if actual_qids != expected:
        missing = [qid for qid in expected if qid not in actual_qids]
        extra = [qid for qid in actual_qids if qid not in set(expected)]
        if missing:
            report.errors.append(f"missing qids: {missing[:10]}")
        if extra:
            report.errors.append(f"unknown qids: {extra[:10]}")
        if not missing and not extra:
            report.errors.append("question rows are not in official template order")

    sums = {field: 0 for field in TOKEN_FIELDS}
    for line_no, row in enumerate(question_rows, start=3):
        qid = str(row.get("qid") or "")
        answers = [str(row.get(field) or "") for field in ANSWER_FIELDS]
        used = [index for index, answer in enumerate(answers) if answer]
        if not used:
            report.errors.append(f"line {line_no}: {qid} has no answer")
        elif used != list(range(max(used) + 1)):
            report.errors.append(f"line {line_no}: {qid} answer slots are not contiguous")
        reasoning = str(row.get("reasoning") or "").strip()
        if len(reasoning) < 20:
            report.errors.append(
                f"line {line_no}: {qid} reasoning must contain at least 20 characters"
            )
        parsed: dict[str, int] = {}
        for field in TOKEN_FIELDS:
            try:
                parsed[field] = _parse_uint(row.get(field), f"line {line_no}.{field}")
                sums[field] += parsed[field]
            except ValueError as exc:
                report.errors.append(str(exc))
        if len(parsed) == len(TOKEN_FIELDS):
            if parsed["prompt_tokens"] <= 0 or parsed["completion_tokens"] <= 0:
                report.errors.append(f"line {line_no}: {qid} usage must be positive")
            if parsed["prompt_tokens"] + parsed["completion_tokens"] != parsed["total_tokens"]:
                report.errors.append(f"line {line_no}: {qid} token sum mismatch")

    summary_values: dict[str, int] = {}
    for field in TOKEN_FIELDS:
        try:
            summary_values[field] = _parse_uint(summary.get(field), f"summary.{field}")
        except ValueError as exc:
            report.errors.append(str(exc))
    if len(summary_values) == len(TOKEN_FIELDS):
        if summary_values["prompt_tokens"] + summary_values["completion_tokens"] != summary_values["total_tokens"]:
            report.errors.append("summary token sum mismatch")
        if summary_values != sums:
            report.errors.append(
                f"summary token totals mismatch: summary={summary_values} rows={sums}"
            )
        total = summary_values["total_tokens"]
        if min_total_tokens and total < min_total_tokens:
            report.warnings.append(
                f"total_tokens={total} is below the optional local threshold "
                f"{min_total_tokens}; the current official formula still accepts it"
            )
        if total > max_total_tokens:
            report.warnings.append(
                f"total_tokens={total} exceeds {max_total_tokens}; "
                "the current official formula gives TokenScore=0 rather than invalidating the CSV"
            )
        report.token_totals = summary_values
    return report
