"""Focused size-policy checks without loading the downloaded model weights."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import DetectorService  # noqa: E402


class FakeCrossCheck:
    def __init__(self):
        self.local_evidence_requested = None

    def predict(self, image: Image.Image, *, include_local_evidence: bool = False) -> dict:
        self.local_evidence_requested = include_local_evidence
        return {"modality": "image", "risk_score": .75, "decision": "uncertain",
                "signals": [], "out_of_scope": False}


class LowResolutionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.service = DetectorService.__new__(DetectorService)
        self.service.highres_image_model = FakeCrossCheck()

    def test_289_by_202_returns_provisional_score_and_local_evidence_request(self):
        result = self.service.predict_image(Image.new("RGB", (289, 202)),
                                            include_local_evidence=True)
        self.assertEqual(result["risk_score"], .75)
        self.assertEqual(result["applicability"], "low_resolution_experimental")
        self.assertFalse(result["out_of_scope"])
        self.assertEqual(result["decision"], "uncertain")
        self.assertIsNone(result["threshold"])
        self.assertTrue(self.service.highres_image_model.local_evidence_requested)
        self.assertIn("不是 AI 创作比例", result["limitations"])

    def test_below_200_short_side_receives_a_score(self):
        result = self.service.predict_image(Image.new("RGB", (400, 199)))
        self.assertFalse(result["out_of_scope"])
        self.assertEqual(result["risk_score"], .75)
        self.assertEqual(result["applicability"], "low_resolution_experimental")

    def test_normal_resolution_keeps_experimental_scope(self):
        result = self.service.predict_image(Image.new("RGB", (512, 512)))
        self.assertEqual(result["applicability"], "experimental_cross_source")
        self.assertFalse(self.service.highres_image_model.local_evidence_requested)


if __name__ == "__main__":
    unittest.main()
