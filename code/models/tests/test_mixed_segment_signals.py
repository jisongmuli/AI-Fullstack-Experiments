"""Mixed segment signals should not become a confident document score."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import DetectorService  # noqa: E402


class TextSegmentTests(unittest.TestCase):
    def test_conflicting_paragraphs_abstain_on_document(self):
        service = DetectorService.__new__(DetectorService)

        def fake_prediction(text: str) -> dict:
            score = 0.01 if "乙" not in text else 0.99
            return {"risk_score": score, "decision": "human" if score < 0.5 else "ai",
                    "out_of_scope": False, "signals": ["test pattern"]}

        service.predict_text_batch = lambda texts: [fake_prediction(text) for text in texts]
        result = service.predict_text_segments("甲" * 100 + "\n\n" + "乙" * 100)
        self.assertIsNone(result["risk_score"])
        self.assertEqual(result["decision"], "uncertain")
        self.assertEqual(result["applicability"], "mixed_segment_signals")
        self.assertTrue(result["segment_summary"]["mixed_signals"])
        self.assertEqual([item["highlighted"] for item in result["segments"]], [False, True])


if __name__ == "__main__":
    unittest.main()
