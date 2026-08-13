from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.qwen_client import ChatResult
from script.run_b_compliant_reasoning import (
    _CLEAN_QID_CHUNK_ANCHORS,
    _answers_equivalent,
    _clean_report_plan,
    _has_fin_b_016_formulas,
    _missing_declared_evidence_ids,
    _reasoning_has_correction_trace,
    _reasoning_supports_answers,
    _selected_option_letters,
    add_usage,
    append_attempt,
    select_evidence,
    select_clean_evidence,
)


class BCompliantReasoningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_obj.name)

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def test_answer_and_named_document_beat_old_retrieval_score(self) -> None:
        cache_path = self.temp_dir / "fc_b_001.json"
        cache_path.write_text(
            json.dumps(
                {
                    "retrieved_evidence": [
                        {
                            "doc_id": "wrong.pdf",
                            "chunk_id": "wrong#1",
                            "score": 9999,
                            "text": "另一发行人主营业务毛利率情况表，金融服务为81.26%。",
                        },
                        {
                            "doc_id": "right.pdf",
                            "chunk_id": "right#69",
                            "score": 1,
                            "text": (
                                "广东省广晟控股集团有限公司2026年面向专业投资者公开发行"
                                "科技创新公司债券（第一期）募集说明书。"
                                "矿业：2025年1-6月7.35%，2024年度7.68%，"
                                "2023年度5.36%，2022年度5.55%。"
                            ),
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        question = {
            "question": (
                "根据《广东省广晟控股集团有限公司2026年面向专业投资者公开发行"
                "科技创新公司债券（第一期）募集说明书》，2022-2024年及"
                "2025年1-6月，矿业板块主营业务毛利率分别是多少？"
            ),
            "type": "计算题",
            "options": {},
        }

        selected = select_evidence(
            cache_path,
            5000,
            question=question,
            answers=["5.55%", "5.36%", "7.68%", "7.35%"],
        )

        self.assertEqual(selected[0]["chunk_id"], "right#69")
        self.assertNotIn("wrong#1", {item["chunk_id"] for item in selected})

    def test_api_attempt_ledger_accumulates_original_usage(self) -> None:
        ledger_path = self.temp_dir / "api_attempts" / "q1.json"
        first = ChatResult(
            content='{"answers":["B"]}',
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
            raw={},
        )
        second = ChatResult(
            content='{"answers":["A"]}',
            usage={
                "prompt_tokens": 110,
                "completion_tokens": 30,
                "total_tokens": 140,
            },
            raw={},
        )

        append_attempt(ledger_path, qid="q1", result=first)
        _, totals = append_attempt(ledger_path, qid="q1", result=second)

        self.assertEqual(
            totals,
            {
                "prompt_tokens": 210,
                "completion_tokens": 50,
                "total_tokens": 260,
            },
        )
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        self.assertEqual(len(ledger["attempts"]), 2)
        self.assertEqual(ledger["attempts"][0]["raw_model_output"], first.content)

    def test_audit_citations_are_selected_before_lexical_matches(self) -> None:
        cache_path = self.temp_dir / "calculation.json"
        cache_path.write_text(
            json.dumps(
                {
                    "retrieved_evidence": [
                        {
                            "doc_id": "report-a.pdf",
                            "chunk_id": "lexical",
                            "score": 9999,
                            "text": "营业收入和现金流量的概括说明，没有本题所需原始值。",
                        },
                        {
                            "doc_id": "report-b.pdf",
                            "chunk_id": "cited-value",
                            "score": 1,
                            "text": "经营活动产生的现金流量净额为 53,345,930 千元。",
                        },
                    ],
                    "result": {
                        "citations": {
                            "doc_id": "report-b.pdf",
                            "chunk_id": "cited-value",
                        },
                        "raw_values": [],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        selected = select_evidence(
            cache_path,
            5000,
            question={
                "question": "比较两家公司的经营现金流率。",
                "type": "计算题",
                "options": {},
            },
            answers=["19.75"],
        )

        self.assertEqual(selected[0]["chunk_id"], "cited-value")

    def test_answer_comparison_ignores_only_whitespace(self) -> None:
        self.assertTrue(
            _answers_equivalent(
                ["2026 年 3 月 30 日"],
                ["2026年3月30日"],
            )
        )
        self.assertFalse(_answers_equivalent(["76.920"], ["76.92"]))

    def test_reasoning_must_include_non_choice_answers(self) -> None:
        good = (
            "依据E1和E2，宁德时代中期10.07元与期末69.57元合计79.64元；"
            "排序为宁德时代79.64元>美的集团43元>招商银行20.16元>"
            "中国建筑2.718元，"
            "最高与最低差额79.64-2.718=76.922，保留两位为76.92。"
        )
        bad = "依据E1，最高69.57元，最低2.718元，差额为66.85。"
        answers = ["宁德时代>美的集团>招商银行>中国建筑", "76.92"]
        self.assertTrue(_reasoning_supports_answers(good, answers))
        self.assertFalse(_reasoning_supports_answers(bad, answers))

    def test_correction_traces_are_rejected(self) -> None:
        self.assertTrue(
            _reasoning_has_correction_trace(
                "先算得322万元，重新检查后修正为366万元。"
            )
        )
        self.assertTrue(
            _reasoning_has_correction_trace(
                "假设使用季度加总，题目要求填写19.75。"
            )
        )
        self.assertFalse(
            _reasoning_has_correction_trace(
                "依据E1，四项分别为144、75、72、75，合计366万元。"
            )
        )

    def test_fin_b_016_accepts_explicit_or_semantic_full_year_formula(self) -> None:
        explicit = (
            "依据E1和E2，宁德时代每10股现金分红为"
            "10.07+69.57=79.64元；差额为79.64-2.718=76.922，"
            "保留两位为76.92。"
        )
        semantic = (
            "依据E2，宁德时代中期每10股分红10.07元；依据E1，"
            "年度及特别分红为每10股69.57元，故全年合计79.64元。"
            "差额计算为79.64-2.718=76.922，保留两位为76.92。"
        )
        incomplete = (
            "宁德时代全年分红为79.64元，差额为"
            "79.64-2.718=76.922，保留两位为76.92。"
        )

        self.assertTrue(_has_fin_b_016_formulas(explicit))
        self.assertTrue(_has_fin_b_016_formulas(semantic))
        self.assertFalse(_has_fin_b_016_formulas(incomplete))

    def test_missing_declared_evidence_ids_are_reported_exactly(self) -> None:
        reasoning = "依据 E1、E2 和 E7 核对选项，最终支持 AB。"

        self.assertEqual(
            _missing_declared_evidence_ids(reasoning, ["E1", "E2"]),
            ["E7"],
        )
        self.assertEqual(
            _missing_declared_evidence_ids(reasoning, ["e7", "E2", "E1"]),
            [],
        )

    def test_usage_addition_preserves_raw_api_totals(self) -> None:
        combined = add_usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
            {
                "prompt_tokens": 200,
                "completion_tokens": 30,
                "total_tokens": 230,
            },
            qid="q1",
        )

        self.assertEqual(
            combined,
            {
                "prompt_tokens": 300,
                "completion_tokens": 50,
                "total_tokens": 350,
            },
        )

    def test_cross_report_plan_uses_b_2025_reports_in_answer_order(self) -> None:
        plan = _clean_report_plan(
            {"qid": "fin_b_016", "question": "跨公司年度现金分红比较"},
            ["宁德时代>美的集团>招商银行>中国建筑", "76.92"],
        )

        self.assertEqual(
            [(company, doc_id) for company, doc_id, _ in plan],
            [
                (
                    "宁德时代",
                    "b::financial_reports/annual_catl_2025_report.PDF",
                ),
                (
                    "美的集团",
                    "b::financial_reports/annual_midea_2025_report.PDF",
                ),
                (
                    "招商银行",
                    "b::financial_reports/annual_cmb_2025_report.PDF",
                ),
                (
                    "中国建筑",
                    "b::financial_reports/annual_cscec_2025_report.pdf",
                ),
            ],
        )

    def test_choice_retrieval_identifies_only_locked_option_letters(self) -> None:
        question = {
            "options": {
                "A": "selected A",
                "B": "selected B",
                "C": "rejected C",
                "D": "rejected D",
            }
        }

        self.assertEqual(
            _selected_option_letters(question, ["AB"]),
            {"A", "B"},
        )
        self.assertEqual(
            _selected_option_letters(question, ["76.92"]),
            set(),
        )

    def test_ins_b_010_clean_retrieval_excludes_unselected_product_docs(self) -> None:
        question = {
            "qid": "ins_b_010",
            "domain": "insurance",
            "question": "关于未成年人身故保险金限制，下列产品明确提及该限制的是？",
            "type": "多选题",
            "options": {
                "A": "国寿增益宝",
                "B": "平安安佑福重疾险",
                "C": "众安个人急性白血病复发医疗保险",
                "D": "平安富鸿金生养老年金保险",
            },
        }

        selected = select_clean_evidence(
            9000,
            question=question,
            answers=["AB"],
        )

        selected_doc_ids = {str(item["doc_id"]) for item in selected}
        self.assertEqual(
            selected_doc_ids,
            {
                "b::insurance/2.pdf",
                "b::insurance/4.pdf",
            },
        )

    def test_failed_clean_questions_include_all_decisive_anchors(self) -> None:
        questions = {
            json.loads(line)["qid"]: json.loads(line)
            for line in (
                Path("processed_data_b/questions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            if line.strip()
        }
        baseline_answers: dict[str, list[str]] = {}
        import csv

        baseline_path = Path(
            "runs/b_morning_submit_20260723_v2/"
            "03_fc_b_019_plus_res_b_006_AB/answer.csv"
        )
        with baseline_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                qid = str(row.get("qid") or "")
                if qid in _CLEAN_QID_CHUNK_ANCHORS:
                    baseline_answers[qid] = [
                        str(row.get(f"answer_{number}") or "")
                        for number in range(1, 5)
                        if str(row.get(f"answer_{number}") or "")
                    ]

        for qid, expected_anchor_ids in _CLEAN_QID_CHUNK_ANCHORS.items():
            with self.subTest(qid=qid):
                selected = select_clean_evidence(
                    9000,
                    question=questions[qid],
                    answers=baseline_answers[qid],
                )
                selected_ids = {
                    str(item.get("chunk_id") or "") for item in selected
                }
                self.assertTrue(set(expected_anchor_ids) <= selected_ids)


if __name__ == "__main__":
    unittest.main()
