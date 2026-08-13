from __future__ import annotations

import unittest

from agent.retrieval import build_option_anchor_query, option_document_ids


class RetrievalAnchorTests(unittest.TestCase):
    def test_option_document_reference_selects_matching_doc(self) -> None:
        question = {"doc_ids": ["text03", "text13"]}

        self.assertEqual(
            option_document_ids(question, "fc_text_003 中违约利息公式包含150%"),
            ["text03"],
        )

    def test_anchor_query_drops_document_alias_but_keeps_claim(self) -> None:
        query = build_option_anchor_query("fc_text_003 中违约利息公式包含 150%")

        self.assertNotIn("fc_text_003", query)
        self.assertIn("违约利息", query)
        self.assertIn("150%", query)


if __name__ == "__main__":
    unittest.main()
