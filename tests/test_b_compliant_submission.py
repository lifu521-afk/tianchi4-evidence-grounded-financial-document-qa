from __future__ import annotations

import csv
from pathlib import Path

from agent.b_compliant_submission import (
    OFFICIAL_FIELDS,
    build_rows,
    validate_submission,
    write_submission,
)


def _record(answer: str) -> dict:
    return {
        "answers": [answer],
        "reasoning": "Internal audit reasoning remains in the cache record.",
        "usage": {
            "prompt_tokens": 300000,
            "completion_tokens": 1000,
            "total_tokens": 301000,
        },
    }


def test_official_header_uses_exact_eight_columns(tmp_path: Path) -> None:
    rows = build_rows(
        expected_qids=["fc_b_001", "fc_b_002"],
        records={
            "fc_b_001": _record("A"),
            "fc_b_002": _record("B"),
        },
    )
    path = write_submission(tmp_path / "answer.csv", rows)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        assert tuple(next(reader)) == OFFICIAL_FIELDS
        assert len(OFFICIAL_FIELDS) == 8

    report = validate_submission(
        path,
        expected_qids=["fc_b_001", "fc_b_002"],
    )
    assert report.ok
    assert report.token_totals["total_tokens"] == 602000


def test_compact_answer_headers_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "old.csv"
    path.write_text(
        "qid,answer1,answer2,answer3,answer4,prompt_tokens,"
        "completion_tokens,total_tokens\n"
        "summary,,,,,600000,2000,602000\n"
        "fc_b_001,A,,,,300000,1000,301000\n"
        "fc_b_002,B,,,,300000,1000,301000\n",
        encoding="utf-8-sig",
    )

    report = validate_submission(
        path,
        expected_qids=["fc_b_001", "fc_b_002"],
    )
    assert not report.ok
    assert any("header mismatch" in error for error in report.errors)


def test_extra_reasoning_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nine_columns.csv"
    path.write_text(
        "qid,answer_1,answer_2,answer_3,answer_4,prompt_tokens,"
        "completion_tokens,total_tokens,reasoning\n"
        "summary,,,,,600000,2000,602000,\n"
        "fc_b_001,A,,,,300000,1000,301000,reason one\n"
        "fc_b_002,B,,,,300000,1000,301000,reason two\n",
        encoding="utf-8-sig",
    )

    report = validate_submission(
        path,
        expected_qids=["fc_b_001", "fc_b_002"],
    )
    assert not report.ok
    assert any("header mismatch" in error for error in report.errors)
