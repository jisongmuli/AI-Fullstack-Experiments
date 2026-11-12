"""Regression cases for numeric meaning changes that were previously accepted."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from editing import edit_text, meaning_check  # noqa: E402
from llm_preview import screen_candidate  # noqa: E402


class NumericMeaningTests(unittest.TestCase):
    def assert_blocked(self, original, candidate):
        checked = meaning_check(original, candidate)
        self.assertFalse(checked["checks"]["numbers_preserved"])
        self.assertEqual(checked["status"], "blocked")
        preview = screen_candidate(original, candidate)
        self.assertEqual(preview["status"], "rejected")
        self.assertIsNone(preview["candidate"])

    def test_dropped_percent_and_changed_percent_unit_are_blocked(self):
        for before, after in [("20%", "20"), ("20%", "20‰"), ("20％", "20")]:
            with self.subTest(before=before, after=after):
                self.assert_blocked(f"The measured rate was {before} in both groups.",
                                    f"The measured rate was {after} in both groups.")

    def test_lost_or_changed_numeric_sign_is_blocked(self):
        for before, after in [("-5", "5"), ("-5", "+5"), ("−5", "5")]:
            with self.subTest(before=before, after=after):
                self.assert_blocked(f"The recorded temperature was {before} degrees during the experiment.",
                                    f"The recorded temperature was {after} degrees during the experiment.")

    def test_leading_decimal_point_is_preserved(self):
        self.assert_blocked("The observed value was .5 in both groups.",
                            "The observed value was 5 in both groups.")

    def test_scientific_notation_exponent_sign_is_preserved(self):
        self.assert_blocked("The measured value was 1e-3 in both groups.",
                            "The measured value was 1e+3 in both groups.")

    def test_numbers_next_to_chinese_text_are_checked(self):
        self.assert_blocked("小组记录本次结果为20%，并保留了原始实验记录。",
                            "小组记录本次结果为20，并保留了原始实验记录。")

    def test_equivalent_unit_spacing_and_unicode_signs_are_accepted(self):
        original = "The measured rate was 20 % at −5 degrees during the experiment."
        candidate = "The measured rate was 20% at -5 degrees during the experiment."
        self.assertTrue(meaning_check(original, candidate)["checks"]["numbers_preserved"])
        self.assertEqual(screen_candidate(original, candidate)["status"], "review_candidate")

    def test_rule_editing_preserves_numbers_and_can_still_apply_safe_edits(self):
        original = "In order to review 1,000 samples, the the team recorded a 20% rate at -5 degrees."
        result = edit_text(original)
        self.assertTrue(result["applied_edits"])
        self.assertTrue(result["meaning_check"]["checks"]["numbers_preserved"])
        for value in ("1,000", "20%", "-5"):
            self.assertIn(value, result["edited_text"])


if __name__ == "__main__":
    unittest.main()
