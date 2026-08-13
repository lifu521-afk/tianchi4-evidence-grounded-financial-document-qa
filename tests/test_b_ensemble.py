from __future__ import annotations

import unittest

from agent.b_ensemble import (
    answer_slot_count,
    collect_locator_queries,
    materialize_citations,
    normalize_choice_answer,
    normalize_open_answer,
    parsed_answer_values,
    question_kind,
    valid_final_payload,
)


class BEnsembleTests(unittest.TestCase):
    def test_choice_answer_is_sorted_and_filtered(self) -> None:
        question = {
            "type": "多选题",
            "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "answer_slots": 1,
        }
        self.assertEqual(normalize_choice_answer("D,A,C,A", question), "ACD")
        self.assertEqual(parsed_answer_values({"answers": ["DCA"]}, question), ["ACD"])
        self.assertTrue(valid_final_payload({"answers": ["DCA"]}, question))

    def test_single_choice_keeps_one_letter(self) -> None:
        question = {
            "type": "单选题",
            "options": {"A": "a", "B": "b"},
            "answer_slots": 1,
        }
        self.assertEqual(parsed_answer_values({"answer": "BA"}, question), ["A"])

    def test_open_answer_respects_multiple_slots(self) -> None:
        question = {
            "type": "计算题",
            "options": {},
            "answer_slots": 3,
        }
        parsed = {"answers": ["1.23", "4.56", "3.33"]}
        self.assertEqual(question_kind(question), "open")
        self.assertEqual(answer_slot_count(question), 3)
        self.assertEqual(parsed_answer_values(parsed, question), parsed["answers"])
        self.assertTrue(valid_final_payload(parsed, question))
        self.assertFalse(valid_final_payload({"answers": ["1.23"]}, question))

    def test_open_numeric_answer_follows_template_precision_and_removes_unit(self) -> None:
        two_decimal_question = {
            "type": "计算题",
            "question": "需核实几笔？",
            "options": {},
            "answer_slots": 1,
            "answer_template": ["999999.99"],
        }
        one_decimal_question = {
            "type": "计算题",
            "question": "普通用户人数约为多少万人？保留一位小数。",
            "options": {},
            "answer_slots": 1,
            "answer_template": ["999999.99"],
        }
        self.assertEqual(
            normalize_open_answer("2笔", two_decimal_question, 0),
            "2.00",
        )
        self.assertEqual(
            parsed_answer_values({"answers": ["61.984"]}, two_decimal_question),
            ["61.98"],
        )
        self.assertEqual(
            parsed_answer_values({"answers": ["67.1"]}, one_decimal_question),
            ["67.1"],
        )

    def test_locator_query_collection_handles_nested_fields(self) -> None:
        locator = {
            "global_search_queries": ["公司债券"],
            "option_search": {
                "A": {"search_queries": ["交叉保护", "宽限期"]},
            },
        }
        self.assertEqual(
            collect_locator_queries(locator),
            ["公司债券", "交叉保护", "宽限期"],
        )

    def test_materialized_citation_verifies_quote_and_location(self) -> None:
        evidence = [
            {
                "doc_id": "d1",
                "chunk_id": "d1#c0001",
                "page": 8,
                "page_char_start": 100,
                "page_char_end": 300,
                "source_path": "d1.pdf",
                "text": "发行人应当在十个交易日内采取措施恢复承诺。",
            }
        ]
        parsed = {
            "citations": [
                {
                    "evidence_id": 1,
                    "quote": "十个交易日内采取措施恢复承诺",
                }
            ]
        }
        citations = materialize_citations(parsed, evidence)
        self.assertEqual(citations[0]["page"], 8)
        self.assertTrue(citations[0]["quote_verified"])

    def test_materializes_option_level_citations(self) -> None:
        evidence = [
            {
                "doc_id": "d1",
                "chunk_id": "d1#c0001",
                "page": 3,
                "page_char_start": 20,
                "page_char_end": 120,
                "source_path": "d1.pdf",
                "text": "保险合同成立后，投保人按照约定交付保险费。",
            }
        ]
        parsed = {
            "option_judgement": {
                "A": {
                    "citations": [
                        {
                            "evidence_id": "E1",
                            "quote": "投保人按照约定交付保险费",
                        }
                    ]
                }
            }
        }
        citations = materialize_citations(parsed, evidence)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["option"], "A")
        self.assertEqual(citations[0]["chunk_id"], "d1#c0001")
        self.assertTrue(citations[0]["quote_verified"])


if __name__ == "__main__":
    unittest.main()
