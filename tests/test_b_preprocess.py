from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.b_preprocess import (
    build_b_processed_data,
    discover_b_raw_documents,
    load_b_questions,
)
from agent.io_utils import load_json, read_jsonl


class BPreprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.questions_dir = self.root / "upload_b" / "question_b"
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed_data_b"
        self.submit_csv = self.root / "upload_b" / "submit.csv"
        self.questions_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_fixture_questions(self) -> None:
        (self.questions_dir / "part_a.json").write_text(
            "\ufeff"
            + json.dumps(
                [
                    {
                        "qid": "calc_b_001",
                        "domain": "finance",
                        "split": "B",
                        "question": "Calculate two values.",
                        "type": "calculation",
                        "options": {},
                    }
                ]
            ),
            encoding="utf-8",
        )
        (self.questions_dir / "part_b.jsonl").write_text(
            json.dumps(
                {
                    "qid": "choice_b_001",
                    "domain": "rules",
                    "split": "B",
                    "question": "Choose the correct option.",
                    "type": "multiple_choice",
                    "options": {"A": "First", "B": "Second"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.submit_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.submit_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "qid",
                    "answer_1",
                    "answer_2",
                    "answer_3",
                    "answer_4",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                ]
            )
            writer.writerow(["summary", "", "", "", "", 0, 0, 0])
            writer.writerow(["choice_b_001", "A", "", "", "", 0, 0, 0])
            writer.writerow(["calc_b_001", "999999.99%", "999999.99%", "", "", 0, 0, 0])

    def _write_fixture_documents(self) -> None:
        fixtures = {
            self.raw_dir / "finance" / "shared.txt": "Revenue was 100.\n\nProfit was 20.",
            self.raw_dir / "rules" / "shared.txt": "Rule one applies.\n\nRule two does not.",
            self.raw_dir / "finance" / "shared.html": (
                "<html><body><h1>Shared title</h1><p>HTML source text.</p></body></html>"
            ),
        }
        for path, text in fixtures.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    def test_loads_json_and_jsonl_with_submit_slot_metadata(self) -> None:
        self._write_fixture_questions()

        questions = load_b_questions(self.questions_dir, self.submit_csv)

        self.assertEqual(["choice_b_001", "calc_b_001"], [question.qid for question in questions])
        self.assertEqual("multiple_choice", questions[0].type)
        self.assertEqual({"A": "First", "B": "Second"}, questions[0].options)
        self.assertEqual(1, questions[0].answer_slots)
        self.assertEqual(("A",), questions[0].answer_template)
        self.assertEqual(2, questions[1].answer_slots)
        self.assertEqual(("999999.99%", "999999.99%"), questions[1].template_answers)

    def test_discovers_all_same_stem_documents_with_unique_ids(self) -> None:
        self._write_fixture_documents()

        documents = discover_b_raw_documents(self.raw_dir)

        self.assertEqual(3, len(documents))
        self.assertEqual(3, len({document.doc_id for document in documents}))
        self.assertEqual(
            {
                "finance/shared.html",
                "finance/shared.txt",
                "rules/shared.txt",
            },
            {document.relative_path for document in documents},
        )

    def test_builds_all_artifacts_with_source_spans_and_reuses_cache(self) -> None:
        self._write_fixture_questions()
        self._write_fixture_documents()

        first_report = build_b_processed_data(
            questions_dir=self.questions_dir,
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
            submit_csv=self.submit_csv,
            progress=False,
            chunk_chars=30,
            overlap=5,
        )

        expected_outputs = {
            "questions.jsonl",
            "documents.jsonl",
            "chunks.jsonl",
            "preprocess_report.json",
        }
        self.assertTrue(
            expected_outputs.issubset(
                {path.name for path in self.processed_dir.iterdir() if path.is_file()}
            )
        )
        self.assertEqual(2, first_report["questions"])
        self.assertEqual(3, first_report["raw_documents"])
        self.assertEqual(3, first_report["documents"])
        self.assertEqual(0, first_report["cache_hits"])
        self.assertEqual(3, first_report["cache_misses"])

        questions = read_jsonl(self.processed_dir / "questions.jsonl")
        documents = read_jsonl(self.processed_dir / "documents.jsonl")
        chunks = read_jsonl(self.processed_dir / "chunks.jsonl")
        report = load_json(self.processed_dir / "preprocess_report.json")

        self.assertEqual(["choice_b_001", "calc_b_001"], [row["qid"] for row in questions])
        self.assertEqual(2, questions[1]["answer_slots"])
        self.assertEqual(3, len({row["doc_id"] for row in documents}))
        self.assertEqual(3, report["selected_documents"])
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn("title", chunk)
            self.assertIn("domain", chunk)
            self.assertTrue(Path(chunk["source_path"]).is_absolute())
            self.assertIn("page", chunk)
            self.assertGreaterEqual(chunk["page_char_start"], 0)
            self.assertGreaterEqual(chunk["page_char_end"], chunk["page_char_start"])
            self.assertLessEqual(chunk["page_char_end"], len(chunk["text"]) + chunk["page_char_start"])

        with patch(
            "agent.b_preprocess.extract_document",
            side_effect=AssertionError("cached documents should not be extracted again"),
        ):
            second_report = build_b_processed_data(
                questions_dir=self.questions_dir,
                raw_dir=self.raw_dir,
                processed_dir=self.processed_dir,
                submit_csv=self.submit_csv,
                progress=False,
                chunk_chars=18,
                overlap=3,
            )

        self.assertEqual(3, second_report["cache_hits"])
        self.assertEqual(0, second_report["cache_misses"])
        self.assertGreaterEqual(second_report["chunks"], first_report["chunks"])

    def test_rejects_question_template_qid_mismatch(self) -> None:
        self._write_fixture_questions()
        lines = self.submit_csv.read_text(encoding="utf-8").splitlines()
        self.submit_csv.write_text(
            "\n".join(line for line in lines if not line.startswith("calc_b_001,")) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "qid mismatch"):
            load_b_questions(self.questions_dir, self.submit_csv)


if __name__ == "__main__":
    unittest.main()
