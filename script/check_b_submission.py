from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.b_submission import (  # noqa: E402
    DEFAULT_TEMPLATE_PATH,
    compare_answer_fields,
    validate_b_submission,
)


def _print_report(label: str, report) -> None:
    print(f"{label}: {report.path}")
    print(f"columns: {report.fields}")
    print(
        f"question_rows: {report.question_rows} / expected "
        f"{report.expected_questions}"
    )
    print(f"errors: {len(report.errors)}")
    for error in report.errors[:80]:
        print(f"ERROR: {error}")
    if len(report.errors) > 80:
        print(f"ERROR_MORE: {len(report.errors) - 80}")
    print(f"warnings: {len(report.warnings)}")
    for warning in report.warnings[:30]:
        print(f"WARN: {warning}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a B leaderboard answer.csv and optionally compare answers."
    )
    parser.add_argument("--file", required=True, help="B submission CSV to validate.")
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE_PATH),
        help="Official upload_b/submit.csv template.",
    )
    parser.add_argument(
        "--compare-with",
        "--match",
        dest="compare_with",
        help="Second B CSV whose answer_1..answer_4 must exactly match --file.",
    )
    args = parser.parse_args(argv)

    report = validate_b_submission(args.file, template_path=args.template)
    _print_report("file", report)
    if not report.ok:
        return 1

    if args.compare_with:
        other = validate_b_submission(args.compare_with, template_path=args.template)
        _print_report("compare_with", other)
        if not other.ok:
            return 1
        mismatches = compare_answer_fields(
            args.file,
            args.compare_with,
            template_path=args.template,
        )
        print(f"answer_field_mismatches: {len(mismatches)}")
        for mismatch in mismatches[:80]:
            print(
                f"MISMATCH: {mismatch.qid}: "
                f"full={mismatch.full_answers} low={mismatch.low_answers}"
            )
        if mismatches:
            return 1
        print("EXACT MATCH: answer_1..answer_4 are identical for every qid.")

    print("OK: B submission format is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

