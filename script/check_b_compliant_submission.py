from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.b_compliant_submission import template_qids, validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the B leaderboard submission format introduced on 2026-07-23."
    )
    parser.add_argument("csv_path")
    parser.add_argument(
        "--template",
        default=str(ROOT / "upload_b" / "submit.csv"),
    )
    parser.add_argument("--min-total-tokens", type=int, default=0)
    parser.add_argument("--max-total-tokens", type=int, default=5_000_000)
    args = parser.parse_args()

    qids = template_qids(args.template)
    report = validate_submission(
        args.csv_path,
        expected_qids=qids,
        min_total_tokens=args.min_total_tokens,
        max_total_tokens=args.max_total_tokens,
    )
    print(f"path: {report.path}")
    print(f"question_rows: {report.question_rows} / expected {report.expected_questions}")
    print(f"token_totals: {report.token_totals}")
    print(f"errors: {len(report.errors)}")
    print(f"warnings: {len(report.warnings)}")
    for item in report.errors:
        print(f"ERROR: {item}")
    for item in report.warnings:
        print(f"WARNING: {item}")
    if report.ok:
        print("OK: B compliant submission format and token range are valid.")
        total_tokens = report.token_totals["total_tokens"]
        budget = 5_000_000
        token_score = max(0.0, min(1.0, (budget - total_tokens) / budget))
        accuracy_multiplier = 0.7 + 0.3 * token_score
        print(f"TokenScore: {token_score:.6f}")
        print(f"accuracy_multiplier: {accuracy_multiplier:.6f}")
        print(
            "final_score_per_correct_answer: "
            f"{100 * accuracy_multiplier / report.expected_questions:.6f}"
        )
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
