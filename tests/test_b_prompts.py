from __future__ import annotations

import unittest

from agent.b_prompts import (
    build_compact_teacher_review_messages,
    build_evidence_analyst_messages,
    build_final_judge_messages,
    build_independent_solver_messages,
    build_locator_messages,
    build_skeptic_messages,
    format_evidence,
    question_kind,
)


def choice_question() -> dict:
    return {
        "qid": "reg_b_001",
        "domain": "regulatory",
        "split": "B",
        "question": "根据规定，下列哪些说法正确？",
        "type": "多选题",
        "options": {"A": "应当备案。", "B": "可以免于备案。"},
    }


def open_question() -> dict:
    return {
        "qid": "fin_b_014",
        "domain": "financial_reports",
        "split": "B",
        "question": "计算三个年度的增长率并按模板顺序填写。",
        "type": "计算题",
        "options": {},
    }


def evidence() -> list[dict]:
    return [
        {
            "doc_id": "doc_1",
            "source_path": "reports/demo.pdf",
            "page": 8,
            "chunk_id": "doc_1#c0003",
            "page_char_start": 120,
            "page_char_end": 180,
            "table_id": "table_2",
            "row_label": "营业收入",
            "column_label": "2025年",
            "year": "2025",
            "unit": "万元",
            "text": "2025年营业收入为120万元，2024年为100万元。",
        }
    ]


class BPromptTests(unittest.TestCase):
    def test_open_audit_requires_full_year_component_check(self) -> None:
        question = {
            "qid": "fin_b_test",
            "domain": "financial_reports",
            "question": "计算全年现金分红。",
            "type": "计算题",
            "options": {},
            "answer_slots": 1,
        }

        messages = build_locator_messages(question, [])
        body = messages[-1]["content"]

        self.assertIn("中期", body)
        self.assertIn("特别分红", body)
        self.assertIn("排除重复计算", body)

    def assert_json_only_instruction(self, messages: list[dict[str, str]]) -> None:
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        self.assertIn("合法 JSON 对象", messages[0]["content"])
        self.assertIn("禁止 Markdown", messages[0]["content"])

    def test_question_kind_supports_choice_and_open(self) -> None:
        self.assertEqual(question_kind(choice_question()), "choice")
        self.assertEqual(question_kind(open_question()), "open")
        extraction = {**open_question(), "type": "抽取题"}
        self.assertEqual(question_kind(extraction), "open")

    def test_evidence_format_preserves_exact_location_and_table_metadata(self) -> None:
        rendered = format_evidence(evidence())
        for expected in (
            "[E1]",
            '"source_path": "reports/demo.pdf"',
            '"page": 8',
            '"chunk_id": "doc_1#c0003"',
            '"row_label": "营业收入"',
            '"column_label": "2025年"',
            '"unit": "万元"',
        ):
            self.assertIn(expected, rendered)

    def test_all_five_roles_require_json_and_verbatim_locations(self) -> None:
        q = choice_question()
        ev = evidence()
        stages = [
            build_locator_messages(q, ev),
            build_evidence_analyst_messages(q, ev, {"option_targets": {}}),
            build_independent_solver_messages(q, ev, {}, {}),
            build_skeptic_messages(q, ev, {}, {}, {}),
            build_final_judge_messages(q, ev, {}, {}, {}, {}),
        ]
        role_names = ("locator", "evidence analyst", "independent solver", "skeptic", "final judge")
        for messages, role_name in zip(stages, role_names):
            self.assert_json_only_instruction(messages)
            prompt = messages[0]["content"] + messages[1]["content"]
            self.assertIn(role_name, prompt)
            self.assertIn("逐字", prompt)
            self.assertIn("page", prompt)
            self.assertIn("chunk_id", prompt)
            self.assertIn("年份", prompt)
            self.assertIn("单位", prompt)

    def test_choice_prompts_require_per_option_relation_judgement_and_citations(self) -> None:
        q = choice_question()
        ev = evidence()
        analyst = build_evidence_analyst_messages(q, ev)[1]["content"]
        judge = build_final_judge_messages(q, ev)[1]["content"]
        for prompt in (analyst, judge):
            self.assertIn('"A"', prompt)
            self.assertIn('"B"', prompt)
            self.assertIn('"relation"', prompt)
            self.assertIn('"judgement"', prompt)
            self.assertIn('"citations"', prompt)
        self.assertIn('"answers"', judge)
        self.assertIn("先完成每个选项的 relation/judgement/citations", judge)

    def test_open_final_schema_contains_all_required_fields(self) -> None:
        prompt = build_final_judge_messages(
            open_question(),
            evidence(),
            answer_slots=3,
        )[1]["content"]
        required = (
            '"answers"',
            '"raw_values"',
            '"formula"',
            '"calculation_steps"',
            '"rounding_stage"',
            '"format_spec"',
            '"citations"',
            '"confidence"',
            '"unresolved"',
        )
        for field in required:
            self.assertIn(field, prompt)
        self.assertIn("answer_3", prompt)
        self.assertIn("表格行列", prompt)
        self.assertIn("中间步骤不得提前四舍五入", prompt)
        self.assertIn("最后一步", prompt)
        self.assertIn("YYYY年M月D日", prompt)
        self.assertIn("半角 >", prompt)

    def test_open_roles_keep_calculation_order_and_final_format_visible(self) -> None:
        q = open_question()
        ev = evidence()
        stages = [
            build_locator_messages(q, ev, answer_slots=3),
            build_evidence_analyst_messages(q, ev, answer_slots=3),
            build_independent_solver_messages(q, ev, answer_slots=3),
            build_skeptic_messages(q, ev, answer_slots=3),
            build_final_judge_messages(q, ev, answer_slots=3),
        ]
        for messages in stages:
            prompt = messages[1]["content"]
            self.assertIn("answer_slots: 3", prompt)
            self.assertIn("表格", prompt)
            self.assertIn("年份", prompt)
            self.assertIn("单位", prompt)
            self.assertIn("舍入", prompt)
            self.assertIn("格式", prompt)

    def test_compact_teacher_prompt_outputs_answers_only(self) -> None:
        messages = build_compact_teacher_review_messages(
            open_question(),
            {
                "answers": ["20.00%", "2025年7月1日", "A>B>C"],
                "key_evidence": [{"quote": "营业收入为120万元"}],
                "formula": "(120-100)/100",
                "format_spec": {"answer_slots": 3},
            },
            answer_slots=3,
        )
        system = messages[0]["content"]
        user = messages[1]["content"]
        self.assertIn("只能包含 answers 这一个键", system)
        self.assertIn("不重新解题", system)
        self.assertIn('"answers"', user)
        self.assertIn("20.00%", user)
        self.assertIn("2025年7月1日", user)
        self.assertIn("A>B>C", user)
        self.assertIn("顺序、字符、百分号", user)
        self.assertNotIn('"confidence"', user)
        self.assertNotIn('"reasoning"', user)

    def test_answer_slots_must_be_between_one_and_four(self) -> None:
        with self.assertRaises(ValueError):
            build_final_judge_messages(open_question(), evidence(), answer_slots=0)
        with self.assertRaises(ValueError):
            build_final_judge_messages(open_question(), evidence(), answer_slots=5)


if __name__ == "__main__":
    unittest.main()
