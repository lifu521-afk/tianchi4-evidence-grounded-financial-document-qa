from __future__ import annotations

import unittest

from agent.ensemble_audit import (
    EnsembleSettings,
    audit_option_citations,
    conservative_decision,
    derive_answer,
    materialize_option_citations,
    resolve_option_doc_scope,
)
from agent.preprocess import split_text_with_spans


def question(answer_format: str = "multi") -> dict:
    return {
        "qid": "q1",
        "domain": "regulatory",
        "answer_format": answer_format,
        "question": "根据文档判断。",
        "options": {
            "A": "第一份文档规定应当备案。",
            "B": "第二份文档规定可以免于备案。",
        },
        "doc_ids": ["reg_text_001", "reg_text_002"],
    }


def option_payload(
    *,
    a_truth: str = "true",
    a_relation: str = "entailed",
    confidence: float = 0.95,
    supporting: list[int] | None = None,
    contradicting: list[int] | None = None,
    quote: str = "公司应当备案",
) -> dict:
    return {
        "option_judgement": {
            "A": {
                "judgement": a_truth,
                "relation": a_relation,
                "confidence": confidence,
                "supporting_evidence_ids": supporting or [],
                "contradicting_evidence_ids": contradicting or [],
                "relevant_evidence_ids": sorted(
                    set((supporting or []) + (contradicting or []))
                ),
                "quoted_clauses": [{"evidence_id": 1, "quote": quote}],
            },
            "B": {
                "judgement": "false",
                "relation": "contradicted",
                "confidence": 0.95,
                "supporting_evidence_ids": [],
                "contradicting_evidence_ids": [2],
                "relevant_evidence_ids": [2],
                "quoted_clauses": [
                    {"evidence_id": 2, "quote": "不得免于备案"}
                ],
            },
        },
        "overall_confidence": confidence,
    }


class EnsembleAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = [
            {
                "doc_id": "reg_text_001",
                "chunk_id": "reg_text_001#c0001",
                "page": 3,
                "page_char_start": 100,
                "page_char_end": 140,
                "source_path": "d1.pdf",
                "text": "依据第十条，公司应当备案，并保存相关材料。",
            },
            {
                "doc_id": "reg_text_002",
                "chunk_id": "reg_text_002#c0002",
                "page": 8,
                "page_char_start": 200,
                "page_char_end": 230,
                "source_path": "d2.pdf",
                "text": "本办法明确规定不得免于备案。",
            },
        ]

    def test_resolve_option_doc_scope_uses_document_order(self) -> None:
        q = question()
        self.assertEqual(
            resolve_option_doc_scope(q, "第一份文档规定应当备案"),
            ["reg_text_001"],
        )
        self.assertEqual(
            resolve_option_doc_scope(q, "第二份文档规定可以免于备案"),
            ["reg_text_002"],
        )

    def test_derive_answer_requires_true_and_entailed(self) -> None:
        parsed = {
            "option_judgement": {
                "A": {"judgement": "true", "relation": "unknown"},
                "B": {"judgement": "true", "relation": "entailed"},
            }
        }
        self.assertEqual(derive_answer(parsed, question()), "B")

    def test_materialize_citation_verifies_exact_quote_and_location(self) -> None:
        parsed = option_payload(supporting=[1])
        citations = materialize_option_citations(
            question(),
            "A",
            parsed,
            self.evidence,
        )
        self.assertEqual(len(citations), 1)
        self.assertTrue(citations[0]["model_quote_verified"])
        self.assertEqual(citations[0]["page"], 3)
        self.assertEqual(citations[0]["page_char_start"], 100)
        self.assertEqual(citations[0]["quote"], "公司应当备案")

    def test_audit_citation_falls_back_to_analyst_location(self) -> None:
        citations = audit_option_citations(
            question(),
            "A",
            option_payload(supporting=[1]),
            {},
            {},
            self.evidence,
        )
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["citation_origin"], "analyst")
        self.assertTrue(citations[0]["source_excerpt_verified"])

    def test_audit_citation_uses_labeled_lexical_fallback(self) -> None:
        citations = audit_option_citations(
            question(),
            "A",
            {},
            {},
            {},
            self.evidence,
        )
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["citation_origin"], "lexical_fallback")
        self.assertEqual(citations[0]["doc_id"], "reg_text_001")
        self.assertTrue(citations[0]["source_excerpt_verified"])

    def test_conservative_gate_rejects_low_confidence_change(self) -> None:
        analyst = option_payload(confidence=0.95, supporting=[1])
        skeptic = option_payload(confidence=0.95, supporting=[1])
        judge = option_payload(confidence=0.50, supporting=[1])
        answer, passed, reasons, _ = conservative_decision(
            question(),
            "",
            "A",
            analyst,
            skeptic,
            judge,
            self.evidence,
            EnsembleSettings(confidence_threshold=0.82),
        )
        self.assertEqual(answer, "")
        self.assertFalse(passed)
        self.assertIn("A:confidence_below_threshold", reasons)

    def test_conservative_gate_allows_supported_two_agent_change(self) -> None:
        analyst = option_payload(confidence=0.95, supporting=[1])
        skeptic = option_payload(confidence=0.90, supporting=[1])
        judge = option_payload(confidence=0.94, supporting=[1])
        answer, passed, reasons, details = conservative_decision(
            question(),
            "",
            "A",
            analyst,
            skeptic,
            judge,
            self.evidence,
            EnsembleSettings(confidence_threshold=0.82),
        )
        self.assertEqual(answer, "A")
        self.assertTrue(passed)
        self.assertEqual(reasons, ["passed"])
        self.assertEqual(details["A"]["cited_doc_ids"], ["reg_text_001"])

    def test_conservative_gate_rejects_unverified_model_quote(self) -> None:
        analyst = option_payload(confidence=0.95, supporting=[1])
        skeptic = option_payload(confidence=0.90, supporting=[1])
        judge = option_payload(
            confidence=0.94,
            supporting=[1],
            quote="这句话不在原文里",
        )
        answer, passed, reasons, _ = conservative_decision(
            question(),
            "",
            "A",
            analyst,
            skeptic,
            judge,
            self.evidence,
            EnsembleSettings(confidence_threshold=0.82),
        )
        self.assertEqual(answer, "")
        self.assertFalse(passed)
        self.assertIn("A:no_verified_verbatim_quote", reasons)

    def test_conservative_gate_respects_unresolved_keep(self) -> None:
        analyst = option_payload(confidence=0.95, supporting=[1])
        skeptic = option_payload(confidence=0.90, supporting=[1])
        judge = option_payload(confidence=0.94, supporting=[1])
        judge["baseline_action"] = "unresolved_keep"
        judge["change_classification"] = "semantic_dispute"
        answer, passed, reasons, _ = conservative_decision(
            question(),
            "",
            "A",
            analyst,
            skeptic,
            judge,
            self.evidence,
            EnsembleSettings(confidence_threshold=0.82),
        )
        self.assertEqual(answer, "")
        self.assertFalse(passed)
        self.assertEqual(reasons, ["judge_baseline_action_unresolved_keep"])

    def test_conservative_gate_rejects_unresolved_semantic_mismatch(self) -> None:
        analyst = option_payload(confidence=0.95, supporting=[1])
        skeptic = option_payload(confidence=0.90, supporting=[1])
        judge = option_payload(confidence=0.94, supporting=[1])
        judge["option_judgement"]["A"]["error_type"] = "semantic_mismatch"
        judge["option_judgement"]["A"]["semantic_equivalence"] = "uncertain"
        answer, passed, reasons, _ = conservative_decision(
            question(),
            "",
            "A",
            analyst,
            skeptic,
            judge,
            self.evidence,
            EnsembleSettings(confidence_threshold=0.82),
        )
        self.assertEqual(answer, "")
        self.assertFalse(passed)
        self.assertIn("A:semantic_equivalence_not_materially_resolved", reasons)

    def test_split_text_with_spans_points_back_to_source(self) -> None:
        text = "第一段包含备案要求。\n\n第二段包含保存要求。" * 12
        rows = split_text_with_spans(text, chunk_chars=70, overlap=12)
        self.assertGreater(len(rows), 1)
        for row in rows:
            start = row["page_char_start"]
            end = row["page_char_end"]
            self.assertGreaterEqual(start, 0)
            self.assertGreater(end, start)
            self.assertLessEqual(end, len(text))
            self.assertEqual(text[start:end], row["text"])


if __name__ == "__main__":
    unittest.main()
