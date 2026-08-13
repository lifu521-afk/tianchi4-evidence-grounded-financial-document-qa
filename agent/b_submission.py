from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_PATH = PROJECT_ROOT / "upload_b" / "submit.csv"
PROTECTED_A_ANSWER_PATH = PROJECT_ROOT / "answer.csv"

ANSWER_FIELDS = ("answer_1", "answer_2", "answer_3", "answer_4")
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
EXPECTED_FIELDS = ("qid", *ANSWER_FIELDS, *TOKEN_FIELDS)
CHOICE_TYPES = {"单选题", "多选题", "判断题"}
OPEN_TYPES = {"计算题", "抽取题"}

_UINT_RE = re.compile(r"\d+")
_CHOICE_RE = re.compile(r"[A-D]+")
_PERCENT_RE = re.compile(r"-?(?:0|[1-9]\d*)\.\d{2}%")
_NUMBER_TEMPLATE_RE = re.compile(r"-?(?:0|[1-9]\d*)\.(\d+)")
_PARSEABLE_NUMBER_RE = re.compile(
    r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:笔|天|分|人|户|家|个|次|元|万元|亿元|万人)?"
)
_DATE_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
_DATE_QUESTION_RE = re.compile(r"(?:何时|哪一天|哪日|日期|从何时)")
_ONE_DECIMAL_RE = re.compile(r"保留一位小数")
_PERCENT_QUESTION_RE = re.compile(r"(?:同比)?增速(?:最接近)?多少")


@dataclass(frozen=True)
class QuestionSpec:
    qid: str
    slot_count: int
    examples: tuple[str, ...]
    kinds: tuple[str, ...]
    question_type: str = ""
    question: str = ""
    option_letters: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubmissionSpec:
    template_path: Path
    qids: tuple[str, ...]
    questions: Mapping[str, QuestionSpec]


@dataclass
class ValidationReport:
    path: Path | None = None
    expected_questions: int = 0
    question_rows: int = 0
    fields: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def require_valid(self) -> None:
        if self.errors:
            preview = "\n".join(f"- {error}" for error in self.errors[:20])
            if len(self.errors) > 20:
                preview += f"\n- ... and {len(self.errors) - 20} more"
            raise ValueError(f"Invalid B submission:\n{preview}")


@dataclass(frozen=True)
class AnswerMismatch:
    qid: str
    full_answers: tuple[str, str, str, str]
    low_answers: tuple[str, str, str, str]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], list[str]]:
    errors: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.reader(handle))
    if not raw_rows:
        return [], [], ["csv is empty"]

    fields = raw_rows[0]
    rows: list[dict[str, str]] = []
    for line_no, values in enumerate(raw_rows[1:], start=2):
        if len(values) != len(fields):
            errors.append(
                f"line {line_no}: expected {len(fields)} columns, got {len(values)}"
            )
        normalized = (values + [""] * len(fields))[: len(fields)]
        rows.append(dict(zip(fields, normalized)))
    return fields, rows, errors


def _load_question_metadata(question_dir: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    if not question_dir.exists():
        return metadata

    for path in sorted(question_dir.iterdir()):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        if path.suffix == ".jsonl":
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
        else:
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
            records = loaded if isinstance(loaded, list) else [loaded]
        for record in records:
            qid = str(record.get("qid", "")).strip()
            if qid:
                metadata[qid] = record
    return metadata


def _slot_kind(
    example: str,
    slot_index: int,
    question_type: str,
    question: str,
) -> str:
    if slot_index == 0 and question_type in CHOICE_TYPES:
        return "choice"
    if slot_index == 0 and question_type in OPEN_TYPES and _DATE_QUESTION_RE.search(question):
        return "date"
    if example.endswith("%") or (
        slot_index == 0
        and question_type in OPEN_TYPES
        and _PERCENT_QUESTION_RE.search(question)
    ):
        return "percent"
    if ">" in example:
        return "sort"
    number_match = _NUMBER_TEMPLATE_RE.fullmatch(example)
    if number_match:
        decimals = 1 if _ONE_DECIMAL_RE.search(question) else len(number_match.group(1))
        return f"number:{decimals}"
    return "open"


def load_submission_spec(
    template_path: str | Path = DEFAULT_TEMPLATE_PATH,
    *,
    question_dir: str | Path | None = None,
) -> SubmissionSpec:
    path = Path(template_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"B submission template not found: {path}")

    fields, rows, read_errors = _read_csv(path)
    errors = list(read_errors)
    if fields != list(EXPECTED_FIELDS):
        errors.append(f"template columns mismatch: {fields} != {list(EXPECTED_FIELDS)}")
    if not rows or rows[0].get("qid") != "summary":
        errors.append("template first data row must be summary")

    if question_dir is None:
        question_path = path.parent / "question_b"
    else:
        question_path = Path(question_dir).resolve()
    metadata = _load_question_metadata(question_path)

    qids: list[str] = []
    questions: dict[str, QuestionSpec] = {}
    seen: set[str] = set()
    for line_no, row in enumerate(rows[1:], start=3):
        qid = row.get("qid", "").strip()
        if not qid:
            errors.append(f"template line {line_no}: qid is empty")
            continue
        if qid in seen:
            errors.append(f"template line {line_no}: duplicated qid {qid}")
            continue
        seen.add(qid)

        examples = tuple(row.get(field, "") for field in ANSWER_FIELDS)
        used = [index for index, value in enumerate(examples) if value != ""]
        if not used:
            errors.append(f"template line {line_no}: {qid} has no answer slots")
            continue
        slot_count = max(used) + 1
        if used != list(range(slot_count)):
            errors.append(f"template line {line_no}: {qid} answer slots are not contiguous")
            continue

        item = metadata.get(qid, {})
        question_type = str(item.get("type", "")).strip()
        question = str(item.get("question", "")).strip()
        options = item.get("options") if isinstance(item.get("options"), dict) else {}
        option_letters = tuple(
            key for key in ("A", "B", "C", "D") if key in options
        )
        kinds = tuple(
            _slot_kind(examples[index], index, question_type, question)
            for index in range(slot_count)
        )
        qids.append(qid)
        questions[qid] = QuestionSpec(
            qid=qid,
            slot_count=slot_count,
            examples=examples[:slot_count],
            kinds=kinds,
            question_type=question_type,
            question=question,
            option_letters=option_letters,
        )

    if metadata:
        missing_metadata = [qid for qid in qids if qid not in metadata]
        extra_metadata = sorted(set(metadata) - set(qids))
        if missing_metadata:
            errors.append(
                f"template qids missing question metadata: {missing_metadata[:10]}"
            )
        if extra_metadata:
            errors.append(
                f"question metadata qids missing from template: {extra_metadata[:10]}"
            )
    if errors:
        raise ValueError("Invalid B submission template:\n- " + "\n- ".join(errors))
    return SubmissionSpec(path, tuple(qids), questions)


def _is_uint(value: Any) -> bool:
    return bool(_UINT_RE.fullmatch(str(value if value is not None else "")))


def _token_value(value: Any, label: str) -> int:
    if isinstance(value, bool) or not _is_uint(value):
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
    return int(value)


def _answer_values(value: Any, slot_count: int, qid: str) -> list[str]:
    if isinstance(value, Mapping):
        if any(field in value for field in ANSWER_FIELDS):
            raw_values = [value.get(field, "") for field in ANSWER_FIELDS[:slot_count]]
        elif "answers" in value:
            raw_values = value["answers"]
        elif "answer" in value:
            raw_values = [value["answer"]]
        else:
            raise ValueError(f"{qid}: answer mapping has no answer fields")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_values = list(value)
    else:
        raw_values = [value]

    if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
        raw_values = [raw_values]
    values = ["" if item is None else str(item) for item in raw_values]
    if len(values) != slot_count:
        raise ValueError(
            f"{qid}: expected {slot_count} answer value(s), got {len(values)}"
        )
    return values


def _usage_values(
    qid: str,
    answer_value: Any,
    usage_by_qid: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, int]:
    source: Mapping[str, Any] = {}
    if usage_by_qid is not None and qid in usage_by_qid:
        source = usage_by_qid[qid]
    elif isinstance(answer_value, Mapping):
        source = answer_value
    prompt = _token_value(source.get("prompt_tokens", 0), f"{qid}.prompt_tokens")
    completion = _token_value(
        source.get("completion_tokens", 0), f"{qid}.completion_tokens"
    )
    supplied_total = source.get("total_tokens")
    total = prompt + completion
    if supplied_total not in (None, ""):
        parsed_total = _token_value(supplied_total, f"{qid}.total_tokens")
        if parsed_total != total:
            raise ValueError(
                f"{qid}: total_tokens {parsed_total} != {prompt} + {completion}"
            )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def build_submission_rows(
    answers: Mapping[str, Any],
    *,
    template_path: str | Path = DEFAULT_TEMPLATE_PATH,
    question_dir: str | Path | None = None,
    usage_by_qid: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, str]]:
    spec = load_submission_spec(template_path, question_dir=question_dir)
    missing = [qid for qid in spec.qids if qid not in answers]
    extra = sorted(set(answers) - set(spec.qids))
    if missing:
        raise ValueError(f"missing B answers: {missing[:20]}")
    if extra:
        raise ValueError(f"unknown B answer qids: {extra[:20]}")
    if usage_by_qid is not None:
        extra_usage = sorted(set(usage_by_qid) - set(spec.qids))
        if extra_usage:
            raise ValueError(f"unknown B usage qids: {extra_usage[:20]}")

    totals = {field: 0 for field in TOKEN_FIELDS}
    question_rows: list[dict[str, str]] = []
    for qid in spec.qids:
        question_spec = spec.questions[qid]
        answer_value = answers[qid]
        values = _answer_values(answer_value, question_spec.slot_count, qid)
        usage = _usage_values(qid, answer_value, usage_by_qid)
        row = {field: "" for field in EXPECTED_FIELDS}
        row["qid"] = qid
        for index, value in enumerate(values):
            row[ANSWER_FIELDS[index]] = value
        for field in TOKEN_FIELDS:
            row[field] = str(usage[field])
            totals[field] += usage[field]
        question_rows.append(row)

    summary = {field: "" for field in EXPECTED_FIELDS}
    summary["qid"] = "summary"
    for field in TOKEN_FIELDS:
        summary[field] = str(totals[field])
    rows = [summary, *question_rows]
    report = validate_submission_rows(rows, spec=spec)
    report.require_valid()
    return rows


def _validate_answer(
    value: str,
    kind: str,
    question_spec: QuestionSpec,
    field: str,
    line_no: int,
    errors: list[str],
    warnings: list[str],
) -> None:
    label = f"line {line_no}: {question_spec.qid}.{field}"
    if value == "":
        errors.append(f"{label} is required")
        return
    if value != value.strip():
        errors.append(f"{label} must not have leading or trailing whitespace")

    if kind == "choice":
        if not _CHOICE_RE.fullmatch(value):
            errors.append(f"{label} must contain only uppercase A-D letters: {value!r}")
            return
        if "".join(sorted(set(value))) != value:
            errors.append(f"{label} choice letters must be unique and sorted: {value!r}")
        allowed = set(question_spec.option_letters or ("A", "B", "C", "D"))
        if not set(value).issubset(allowed):
            errors.append(f"{label} contains a letter outside the question options")
        if question_spec.question_type in {"单选题", "判断题"} and len(value) != 1:
            errors.append(f"{label} requires exactly one choice letter")
        if question_spec.question_type == "判断题" and value not in {"A", "B"}:
            errors.append(f"{label} judgement answer must be A or B")
        return

    if kind == "percent":
        if not _PERCENT_RE.fullmatch(value):
            errors.append(
                f"{label} must be a percentage with % and exactly two decimals: {value!r}"
            )
        return

    if kind == "date":
        match = _DATE_RE.fullmatch(value)
        if not match:
            errors.append(f"{label} must use YYYY年M月D日 format: {value!r}")
            return
        try:
            date(*(int(part) for part in match.groups()))
        except ValueError:
            errors.append(f"{label} is not a valid calendar date: {value!r}")
        return

    if kind.startswith("number:"):
        decimals = int(kind.partition(":")[2])
        number_re = re.compile(rf"-?(?:0|[1-9]\d*)\.\d{{{decimals}}}")
        if not number_re.fullmatch(value):
            if _PARSEABLE_NUMBER_RE.fullmatch(value):
                warnings.append(
                    f"{label} is numerically parseable but official formatting asks "
                    f"for exactly {decimals} decimals and no unit: {value!r}"
                )
            else:
                errors.append(
                    f"{label} must contain a parseable number; official formatting "
                    f"asks for exactly {decimals} decimals and no unit: {value!r}"
                )
        return

    if kind == "sort":
        if "＞" in value or re.search(r"\s>\s|>\s|\s>", value):
            errors.append(f"{label} must use half-width > with no surrounding spaces")
            return
        parts = value.split(">")
        if len(parts) < 2 or any(not part for part in parts):
            errors.append(f"{label} must be a non-empty ranking joined by >")
        return

    if "％" in value:
        errors.append(f"{label} must use the half-width % character")
    if "%" in value and not _PERCENT_RE.fullmatch(value):
        errors.append(
            f"{label} percentage must end with % and have exactly two decimals"
        )
    if any(marker in value for marker in ("年", "月", "日")):
        match = _DATE_RE.fullmatch(value)
        if not match:
            errors.append(f"{label} date must use YYYY年M月D日 format")
        else:
            try:
                date(*(int(part) for part in match.groups()))
            except ValueError:
                errors.append(f"{label} is not a valid calendar date: {value!r}")
    if "＞" in value:
        errors.append(f"{label} must use the half-width > character")
    if ">" in value and re.search(r"\s>\s|>\s|\s>", value):
        errors.append(f"{label} must not contain spaces around >")


def validate_submission_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: SubmissionSpec,
    path: Path | None = None,
    initial_errors: Sequence[str] = (),
    fields: Sequence[str] = EXPECTED_FIELDS,
) -> ValidationReport:
    report = ValidationReport(
        path=path,
        expected_questions=len(spec.qids),
        question_rows=max(len(rows) - 1, 0),
        fields=list(fields),
        errors=list(initial_errors),
    )
    if list(fields) != list(EXPECTED_FIELDS):
        report.errors.append(
            f"column mismatch: {list(fields)} != {list(EXPECTED_FIELDS)}"
        )
    if not rows:
        report.errors.append("csv has no data rows")
        return report

    summary = rows[0]
    if str(summary.get("qid", "")) != "summary":
        report.errors.append("first data row must be summary")
    for field in ANSWER_FIELDS:
        if str(summary.get(field, "") or ""):
            report.errors.append(f"summary.{field} must be empty")

    expected_order = list(spec.qids)
    actual_order = [str(row.get("qid", "")) for row in rows[1:]]
    if actual_order != expected_order:
        missing = [qid for qid in expected_order if qid not in actual_order]
        extra = [qid for qid in actual_order if qid not in set(expected_order)]
        if missing:
            report.errors.append(f"missing qids: {missing[:20]}")
        if extra:
            report.errors.append(f"unknown qids: {extra[:20]}")
        if not missing and not extra:
            report.errors.append("question rows are not in upload_b/submit.csv order")
    if len(actual_order) != len(expected_order):
        report.errors.append(
            f"question row count mismatch: {len(actual_order)} != {len(expected_order)}"
        )

    sums = {field: 0 for field in TOKEN_FIELDS}
    seen: set[str] = set()
    for line_no, row in enumerate(rows[1:], start=3):
        qid = str(row.get("qid", ""))
        if qid in seen:
            report.errors.append(f"line {line_no}: duplicated qid {qid}")
        seen.add(qid)
        question_spec = spec.questions.get(qid)
        if question_spec is not None:
            for index, field in enumerate(ANSWER_FIELDS):
                value = str(row.get(field, "") or "")
                if index < question_spec.slot_count:
                    _validate_answer(
                        value,
                        question_spec.kinds[index],
                        question_spec,
                        field,
                        line_no,
                        report.errors,
                        report.warnings,
                    )
                elif value:
                    report.errors.append(
                        f"line {line_no}: {qid}.{field} must be empty; "
                        f"template allows {question_spec.slot_count} answer slot(s)"
                    )

        parsed_tokens: dict[str, int] = {}
        for field in TOKEN_FIELDS:
            value = row.get(field)
            if not _is_uint(value):
                report.errors.append(
                    f"line {line_no}: {qid}.{field} must be a non-negative integer: "
                    f"{value!r}"
                )
            else:
                parsed_tokens[field] = int(str(value))
                sums[field] += parsed_tokens[field]
        if len(parsed_tokens) == len(TOKEN_FIELDS):
            if (
                parsed_tokens["prompt_tokens"]
                + parsed_tokens["completion_tokens"]
                != parsed_tokens["total_tokens"]
            ):
                report.errors.append(f"line {line_no}: token sum mismatch for {qid}")

    summary_tokens: dict[str, int] = {}
    for field in TOKEN_FIELDS:
        value = summary.get(field)
        if not _is_uint(value):
            report.errors.append(
                f"summary.{field} must be a non-negative integer: {value!r}"
            )
        else:
            summary_tokens[field] = int(str(value))
    if len(summary_tokens) == len(TOKEN_FIELDS):
        if (
            summary_tokens["prompt_tokens"]
            + summary_tokens["completion_tokens"]
            != summary_tokens["total_tokens"]
        ):
            report.errors.append("summary token sum mismatch")
        if summary_tokens != sums:
            report.errors.append(
                f"summary tokens mismatch: summary={summary_tokens}, rows={sums}"
            )
    return report


def validate_b_submission(
    csv_path: str | Path,
    *,
    template_path: str | Path = DEFAULT_TEMPLATE_PATH,
    question_dir: str | Path | None = None,
) -> ValidationReport:
    path = Path(csv_path).resolve()
    if not path.exists():
        report = ValidationReport(path=path)
        report.errors.append(f"submission file not found: {path}")
        return report
    spec = load_submission_spec(template_path, question_dir=question_dir)
    fields, rows, read_errors = _read_csv(path)
    return validate_submission_rows(
        rows,
        spec=spec,
        path=path,
        initial_errors=read_errors,
        fields=fields,
    )


def write_b_submission(
    output_path: str | Path,
    answers: Mapping[str, Any],
    *,
    template_path: str | Path = DEFAULT_TEMPLATE_PATH,
    question_dir: str | Path | None = None,
    usage_by_qid: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    path = Path(output_path).resolve()
    template = Path(template_path).resolve()
    if path == PROTECTED_A_ANSWER_PATH.resolve():
        raise ValueError(
            f"refusing to overwrite protected A leaderboard answer: {path}"
        )
    if path == template:
        raise ValueError(f"refusing to overwrite official B submission template: {path}")

    rows = build_submission_rows(
        answers,
        template_path=template,
        question_dir=question_dir,
        usage_by_qid=usage_by_qid,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(EXPECTED_FIELDS),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)
    return path


def _validated_rows(
    csv_path: str | Path,
    template_path: str | Path,
    question_dir: str | Path | None,
) -> tuple[SubmissionSpec, dict[str, dict[str, str]]]:
    report = validate_b_submission(
        csv_path,
        template_path=template_path,
        question_dir=question_dir,
    )
    report.require_valid()
    spec = load_submission_spec(template_path, question_dir=question_dir)
    _, rows, _ = _read_csv(Path(csv_path).resolve())
    return spec, {row["qid"]: row for row in rows[1:]}


def compare_answer_fields(
    full_csv: str | Path,
    low_csv: str | Path,
    *,
    template_path: str | Path = DEFAULT_TEMPLATE_PATH,
    question_dir: str | Path | None = None,
) -> list[AnswerMismatch]:
    full_spec, full_rows = _validated_rows(full_csv, template_path, question_dir)
    low_spec, low_rows = _validated_rows(low_csv, template_path, question_dir)
    if full_spec.qids != low_spec.qids:
        raise ValueError("full and low submissions use different B templates")

    mismatches: list[AnswerMismatch] = []
    for qid in full_spec.qids:
        full_answers = tuple(full_rows[qid][field] for field in ANSWER_FIELDS)
        low_answers = tuple(low_rows[qid][field] for field in ANSWER_FIELDS)
        if full_answers != low_answers:
            mismatches.append(
                AnswerMismatch(
                    qid=qid,
                    full_answers=full_answers,
                    low_answers=low_answers,
                )
            )
    return mismatches


def assert_answer_fields_exact_match(
    full_csv: str | Path,
    low_csv: str | Path,
    *,
    template_path: str | Path = DEFAULT_TEMPLATE_PATH,
    question_dir: str | Path | None = None,
) -> None:
    mismatches = compare_answer_fields(
        full_csv,
        low_csv,
        template_path=template_path,
        question_dir=question_dir,
    )
    if mismatches:
        qids = [item.qid for item in mismatches]
        raise ValueError(
            f"full/low answer fields differ for {len(qids)} qid(s): {qids[:20]}"
        )
