"""Focused smoke tests for the trained Chinese text path."""

from __future__ import annotations

import unittest
from pathlib import Path

from inference import DetectorService


ROOT = Path(__file__).resolve().parents[2]


class ChineseTextInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = DetectorService(ROOT)
        cls.service.algorithm_policy = {}  # Preserve the historical baseline smoke checks.

    def test_long_chinese_text_has_model_score(self) -> None:
        text = "今年中秋，我又走过村口那条小路。树叶落在石阶上，母亲在门前等我回来。" * 55
        self.assertGreater(len(text), 1000)
        result = self.service.predict_text(text)
        self.assertFalse(result["out_of_scope"])
        self.assertGreaterEqual(result["risk_score"], 0)
        self.assertLessEqual(result["risk_score"], 1)
        self.assertEqual(result["model_version"], "zh-char-tfidf-lr-v1")
        self.assertEqual(result["applicability"], "unverified_platform")

    def test_short_chinese_text_abstains(self) -> None:
        result = self.service.predict_text("这段中文内容太短，无法使用已评估的中文模型可靠筛查。" * 2)
        self.assertTrue(result["out_of_scope"])
        self.assertEqual(result["decision"], "uncertain")

    def test_english_still_uses_original_path(self) -> None:
        result = self.service.predict_text(
            "The experiment needs more evidence before we can draw a reliable conclusion."
        )
        self.assertNotIn("model_version", result)
        self.assertFalse(result["out_of_scope"])

if __name__ == "__main__":
    unittest.main()
