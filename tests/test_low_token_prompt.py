from __future__ import annotations

import unittest

from agent.solver import build_messages


class LowTokenPromptTests(unittest.TestCase):
    def test_minimal_prompt_limits_and_compacts_evidence(self) -> None:
        question = {
            "qid": "demo",
            "answer_format": "multi",
            "question": "以下哪些选项正确？",
            "options": {"A": "条款甲", "B": "条款乙", "C": "条款丙", "D": "条款丁"},
        }
        evidence = [
            {
                "doc_id": f"doc{i}",
                "chunk_id": f"doc{i}#c1",
                "page": i,
                "score": 10 - i,
                "sources": ["global"],
                "text": (f"第{i}条 条款原文。" * 100),
            }
            for i in range(1, 13)
        ]

        messages = build_messages(question, evidence, "minimal", "AC")
        prompt = messages[1]["content"]

        self.assertIn("[10]", prompt)
        self.assertNotIn("[11]", prompt)
        self.assertIn("已有答案:AC", prompt)
        self.assertLess(len(prompt), 7000)


    def test_micro_prompt_is_shorter_than_minimal_prompt(self) -> None:
        question = {
            "qid": "demo",
            "answer_format": "multi",
            "question": "Which options are supported?",
            "options": {"A": "alpha 10", "B": "beta 20", "C": "gamma 30", "D": "delta 40"},
        }
        evidence = [
            {
                "doc_id": f"doc{i}",
                "chunk_id": f"doc{i}#c1",
                "page": i,
                "score": 10 - i,
                "sources": ["global"],
                "text": (f"clause {i} alpha beta gamma delta 10 20 30 40. " * 160),
            }
            for i in range(1, 13)
        ]

        minimal = build_messages(question, evidence, "minimal", "AC")[1]["content"]
        micro = build_messages(question, evidence, "micro", "AC")[1]["content"]

        self.assertIn("[6]", micro)
        self.assertNotIn("[7]", micro)
        self.assertLess(len(micro), len(minimal))
        self.assertLess(len(micro), 3600)


    def test_nano_prompt_is_smaller_than_micro_prompt(self) -> None:
        question = {
            "qid": "demo",
            "answer_format": "multi",
            "question": "Which options are supported?",
            "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        }
        evidence = [
            {
                "doc_id": f"doc{i}",
                "chunk_id": f"doc{i}#c1",
                "page": i,
                "score": 10 - i,
                "sources": ["global"],
                "text": (f"clause {i} alpha beta gamma delta. " * 160),
            }
            for i in range(1, 13)
        ]

        micro = build_messages(question, evidence, "micro", "AC")[1]["content"]
        nano = build_messages(question, evidence, "nano", "AC")[1]["content"]

        self.assertIn("[4]", nano)
        self.assertNotIn("[5]", nano)
        self.assertLess(len(nano), len(micro))
        self.assertIn('{"answer":"LETTERS"}', nano)


if __name__ == "__main__":
    unittest.main()
