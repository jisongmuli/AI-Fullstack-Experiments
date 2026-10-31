"""Equivalent course text must use the same features as the training corpus."""
import sys
import unittest
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/models"))
from course_models import CourseDetectors


class CourseTextPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = CourseDetectors(ROOT)

    def test_nfkc_and_whitespace_variants_have_identical_scores(self):
        text = ("研究团队核对样本来源，保留训练、验证和测试的划分记录。模型结果经过逐项检查，报告保留原文与模型版本。"
                " 实验记录 2026 ABC 123 展示对比结果。") * 3
        variant = " \n\t" + "\u3000\n".join(text.split(" ")) + "\t "
        variant = "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in variant)
        canonical = " ".join(unicodedata.normalize("NFKC", variant).split())
        self.assertEqual(canonical, " ".join(unicodedata.normalize("NFKC", text).split()))
        for stage in ("trained", "optimized"):
            with self.subTest(stage=stage):
                expected = self.models.predict_text(canonical, stage)
                actual = self.models.predict_text(variant, stage)
                self.assertEqual(actual["risk_score"], expected["risk_score"])

    def test_whitespace_does_not_pad_short_content_into_scoring_scope(self):
        value = ("研究团队检查样本来源" * 5) + " " * 200
        value = value[:20] + "\n" * 200 + value[20:]
        self.assertLess(len(" ".join(value.split())), 80)
        self.assertTrue(self.models.predict_text(value)["out_of_scope"])


if __name__ == "__main__":
    unittest.main()
