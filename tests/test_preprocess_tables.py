from __future__ import annotations

import unittest

from agent.preprocess import table_rows_to_text


class PreprocessTableTests(unittest.TestCase):
    def test_two_column_key_value_table_is_not_header_expanded(self) -> None:
        text = table_rows_to_text(
            1,
            [
                ["注册金额", "不超过180亿元"],
                ["本期发行金额", "不超过20亿元"],
                ["发行人", "西部证券股份有限公司"],
            ],
        )

        self.assertIn("注册金额 | 不超过180亿元", text)
        self.assertIn("本期发行金额 | 不超过20亿元", text)
        self.assertNotIn("注册金额:本期发行金额", text)

    def test_matrix_table_keeps_header_value_pairs(self) -> None:
        text = table_rows_to_text(
            2,
            [
                ["年份", "营业收入", "净利润"],
                ["2024", "100", "8"],
                ["2025", "120", "10"],
            ],
        )

        self.assertIn("年份:2024；营业收入:100；净利润:8", text)
        self.assertIn("年份:2025；营业收入:120；净利润:10", text)


if __name__ == "__main__":
    unittest.main()