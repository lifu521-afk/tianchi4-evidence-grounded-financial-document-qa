from __future__ import annotations

import unittest

from agent.solver import normalize_answer_for_question


class AnswerConstraintTests(unittest.TestCase):
    def test_tf_answer_only_allows_a_or_b(self) -> None:
        question = {"answer_format": "tf", "options": {"A": "正确", "B": "错误"}}

        self.assertEqual(normalize_answer_for_question("CD", question), "")
        self.assertEqual(normalize_answer_for_question("CA", question), "A")

    def test_multi_answer_is_sorted_deduped_and_filtered_to_existing_options(self) -> None:
        question = {"answer_format": "multi", "options": {"A": "alpha", "C": "gamma"}}

        self.assertEqual(normalize_answer_for_question("DCBAAC", question), "AC")


if __name__ == "__main__":
    unittest.main()