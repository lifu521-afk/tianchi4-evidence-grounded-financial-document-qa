from __future__ import annotations

import unittest

from agent.solver import (
    extract_claim_numbers,
    merge_answer_with_evidence_gate,
    option_numbers_are_supported,
    option_semantics_are_supported,
)


class EvidenceGateNumberTests(unittest.TestCase):
    def test_document_identifier_is_not_a_claim_number(self) -> None:
        self.assertEqual(
            extract_claim_numbers("fc_text_003 中违约利息计算公式包含 150% 的系数"),
            ["150%"],
        )

    def test_direct_percentage_evidence_passes_numeric_gate(self) -> None:
        question = {
            "answer_format": "multi",
            "options": {
                "A": "兑付日约定正确",
                "B": "利润总额逐年下降",
                "C": "资产负债率总体下降",
                "D": "fc_text_003 中违约利息计算公式包含 150% 的系数",
            },
        }
        parsed = {
            "option_judgement": {
                "D": {
                    "judgement": "true",
                    "supporting_evidence_ids": [1],
                }
            }
        }
        evidence = [
            {
                "text": "违约金具体计算方式为延迟支付的本金和利息×票面利率×150%×违约天数/365。"
            }
        ]

        self.assertTrue(option_numbers_are_supported(question, "D", parsed, evidence))

    def test_metric_name_mismatch_blocks_an_addition(self) -> None:
        question = {
            "answer_format": "multi",
            "options": {
                "A": "兑付日约定正确",
                "B": "利润总额逐年下降",
                "C": "资产负债率总体下降",
                "D": "fc_text_003 中违约利息计算公式包含 150% 的系数",
            },
        }
        parsed = {
            "option_judgement": {
                "D": {
                    "judgement": "true",
                    "relation": "entailed",
                    "error_type": "none",
                    "supporting_evidence_ids": [1],
                    "reasoning": "违约金公式包含150%。",
                }
            }
        }
        evidence = [
            {
                "doc_id": "text03",
                "text": "违约金具体计算方式为本金和利息×票面利率×150%。",
            }
        ]

        self.assertFalse(option_semantics_are_supported(question, "D", parsed, evidence))
        self.assertEqual(merge_answer_with_evidence_gate("AC", parsed, question, evidence), "AC")

    def test_missing_evidence_cannot_delete_a_base_option(self) -> None:
        question = {"answer_format": "multi", "options": {"A": "2025年营业收入实现正增长"}}
        parsed = {
            "option_judgement": {
                "A": {
                    "judgement": "false",
                    "relation": "unknown",
                    "error_type": "missing_evidence",
                    "supporting_evidence_ids": [1],
                    "reasoning": "现有证据未提供2025年营业收入。",
                }
            }
        }
        evidence = [{"doc_id": "annual_report", "text": "2025年度报告摘要"}]

        self.assertEqual(merge_answer_with_evidence_gate("A", parsed, question, evidence), "A")


if __name__ == "__main__":
    unittest.main()
