from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from agent.b_submission import (
    ANSWER_FIELDS,
    EXPECTED_FIELDS,
    PROTECTED_A_ANSWER_PATH,
    compare_answer_fields,
    load_submission_spec,
    validate_b_submission,
    write_b_submission,
)
from script.check_b_submission import main as checker_main


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "upload_b" / "submit.csv"


class BSubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_submission_spec(TEMPLATE)

    def setUp(self) -> None:
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_obj.name)

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def _valid_answers(self) -> dict[str, str | list[str]]:
        answers: dict[str, str | list[str]] = {}
        for qid in self.spec.qids:
            question = self.spec.questions[qid]
            values: list[str] = []
            for kind in question.kinds:
                if kind == "choice":
                    values.append("A")
                elif kind == "percent":
                    values.append("12.34%")
                elif kind == "date":
                    values.append("2026年3月30日")
                elif kind == "sort":
                    values.append("甲公司>乙公司")
                elif kind.startswith("number:"):
                    decimals = int(kind.partition(":")[2])
                    values.append(f"{12.34:.{decimals}f}")
                else:
                    values.append("12.34")
            answers[qid] = values[0] if len(values) == 1 else values
        return answers

    def _usage(self, prompt: int = 2, completion: int = 1):
        return {
            qid: {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }
            for qid in self.spec.qids
        }

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(EXPECTED_FIELDS), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_official_template_defines_order_and_slot_counts(self) -> None:
        self.assertEqual(len(self.spec.qids), 100)
        self.assertEqual(self.spec.qids[0], "fc_b_001")
        self.assertEqual(self.spec.questions["fc_b_001"].slot_count, 4)
        self.assertEqual(self.spec.questions["fc_b_005"].slot_count, 2)
        self.assertEqual(self.spec.questions["fin_b_014"].slot_count, 3)
        self.assertEqual(self.spec.questions["fin_b_020"].slot_count, 3)

    def test_writer_places_summary_second_and_supports_multiple_fields(self) -> None:
        output = self.temp_dir / "b_full" / "answer.csv"
        write_b_submission(
            output,
            self._valid_answers(),
            template_path=TEMPLATE,
            usage_by_qid=self._usage(),
        )
        rows = self._read_rows(output)
        self.assertEqual(rows[0]["qid"], "summary")
        self.assertEqual(rows[0]["prompt_tokens"], "200")
        self.assertEqual(rows[0]["completion_tokens"], "100")
        self.assertEqual(rows[0]["total_tokens"], "300")
        fc001 = next(row for row in rows if row["qid"] == "fc_b_001")
        self.assertEqual(
            [fc001[field] for field in ANSWER_FIELDS],
            ["12.34%", "12.34%", "12.34%", "12.34%"],
        )
        self.assertTrue(validate_b_submission(output, template_path=TEMPLATE).ok)

    def test_choice_percentage_date_and_sort_formats_are_strict(self) -> None:
        answers = self._valid_answers()
        invalid_values = {
            "fc_b_002": "CA",
            "fc_b_001": ["12.3%", "12.34%", "12.34%", "12.34%"],
            "reg_b_003": "2026-03-30",
            "fin_b_015": ["甲公司 ＞ 乙公司", "12.34"],
        }
        for qid, value in invalid_values.items():
            with self.subTest(qid=qid):
                changed = dict(answers)
                changed[qid] = value
                with self.assertRaises(ValueError):
                    write_b_submission(
                        self.temp_dir / f"{qid}.csv",
                        changed,
                        template_path=TEMPLATE,
                    )

    def test_numeric_template_deviations_warn_but_remain_parseable(self) -> None:
        answers = self._valid_answers()
        warning_values = {
            "ins_b_001": "333.2",
            "reg_b_016": "2笔",
            "res_b_007": "61.984",
        }
        for qid, value in warning_values.items():
            with self.subTest(qid=qid):
                changed = dict(answers)
                changed[qid] = value
                output = self.temp_dir / f"{qid}.csv"
                write_b_submission(output, changed, template_path=TEMPLATE)
                report = validate_b_submission(output, template_path=TEMPLATE)
                self.assertTrue(report.ok)
                self.assertTrue(
                    any(qid in warning for warning in report.warnings),
                    report.warnings,
                )

        changed = dict(answers)
        changed["ins_b_001"] = "not-a-number"
        with self.assertRaises(ValueError):
            write_b_submission(
                self.temp_dir / "invalid_numeric.csv",
                changed,
                template_path=TEMPLATE,
            )

        answers["res_b_012"] = "67.1"
        write_b_submission(
            self.temp_dir / "explicit_one_decimal.csv",
            answers,
            template_path=TEMPLATE,
        )

    def test_validator_rejects_token_and_summary_mismatches(self) -> None:
        output = self.temp_dir / "bad_tokens.csv"
        write_b_submission(
            output,
            self._valid_answers(),
            template_path=TEMPLATE,
            usage_by_qid=self._usage(),
        )
        rows = self._read_rows(output)
        rows[1]["total_tokens"] = "99"
        rows[0]["prompt_tokens"] = "-1"
        self._write_rows(output, rows)
        report = validate_b_submission(output, template_path=TEMPLATE)
        self.assertFalse(report.ok)
        self.assertTrue(any("non-negative integer" in item for item in report.errors))
        self.assertTrue(any("token sum mismatch" in item for item in report.errors))

    def test_full_low_exact_match_ignores_token_differences(self) -> None:
        full = self.temp_dir / "full.csv"
        low = self.temp_dir / "low.csv"
        answers = self._valid_answers()
        write_b_submission(
            full,
            answers,
            template_path=TEMPLATE,
            usage_by_qid=self._usage(20, 5),
        )
        write_b_submission(
            low,
            answers,
            template_path=TEMPLATE,
            usage_by_qid=self._usage(1, 0),
        )
        self.assertEqual(
            compare_answer_fields(full, low, template_path=TEMPLATE),
            [],
        )
        self.assertEqual(
            checker_main(
                [
                    "--file",
                    str(full),
                    "--template",
                    str(TEMPLATE),
                    "--compare-with",
                    str(low),
                ]
            ),
            0,
        )

        changed = dict(answers)
        changed["fc_b_002"] = "B"
        write_b_submission(low, changed, template_path=TEMPLATE)
        mismatches = compare_answer_fields(full, low, template_path=TEMPLATE)
        self.assertEqual([item.qid for item in mismatches], ["fc_b_002"])

    def test_writer_refuses_to_overwrite_root_a_answer(self) -> None:
        with self.assertRaisesRegex(ValueError, "protected A leaderboard answer"):
            write_b_submission(
                PROTECTED_A_ANSWER_PATH,
                self._valid_answers(),
                template_path=TEMPLATE,
            )


if __name__ == "__main__":
    unittest.main()
