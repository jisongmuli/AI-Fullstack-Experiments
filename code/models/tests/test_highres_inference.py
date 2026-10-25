"""Focused tests for cross-check output and metadata privacy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from highres_inference import HighResolutionCrossCheck, blur_grid, metadata_hints, screening_consensus  # noqa: E402


class FakeDetector:
    def __init__(self, score: float):
        self.score = score

    def predict(self, image: Image.Image) -> dict:
        return {"risk_score": self.score}


class FakeBatchDetector:
    def __init__(self, scores: list[float]):
        self.scores = scores
        self.batch_size = None

    def predict_batch(self, images: list[Image.Image], *, batch_size: int) -> list[float]:
        self.batch_size = batch_size
        assert len(images) == len(self.scores)
        return self.scores

    def predict(self, image: Image.Image) -> dict:
        return {"risk_score": self.scores[0]}


class CrossCheckTests(unittest.TestCase):
    def test_joint_high_requires_both_models(self):
        self.assertEqual(screening_consensus((0.95, 0.85)), "joint_high")
        self.assertEqual(screening_consensus((0.99, 0.65)), "mixed")
        self.assertEqual(screening_consensus((0.1, 0.19)), "joint_low")

    def test_reports_independent_scores_and_disagreement(self):
        detector = HighResolutionCrossCheck.__new__(HighResolutionCrossCheck)
        detector.detectors = (("SigLIP", FakeDetector(0.9)), ("ViT", FakeDetector(0.1)))
        result = detector.predict(Image.new("RGB", (800, 600)))
        self.assertEqual(result["risk_score"], 0.5)
        self.assertEqual(result["model_disagreement"], 0.8)
        self.assertEqual(result["decision"], "uncertain")
        self.assertTrue(any("未共同达到" in signal for signal in result["signals"]))

    def test_metadata_reports_field_name_without_private_value(self):
        image = Image.new("RGB", (800, 600))
        image.info["parameters"] = "private prompt with a real person's name"
        image.info["Software"] = "ComfyUI; user@example.com"
        hints = metadata_hints(image)
        self.assertIn("parameters", " ".join(hints))
        self.assertIn("comfyui", " ".join(hints))
        self.assertNotIn("private prompt", " ".join(hints))
        self.assertNotIn("user@example.com", " ".join(hints))

    def test_local_evidence_is_bounded_sensitivity_not_region_probability(self):
        detector = HighResolutionCrossCheck.__new__(HighResolutionCrossCheck)
        first = FakeBatchDetector([.9, .6, .8, 1.0, .9, .9, .9, .9, .9, .9])
        second = FakeBatchDetector([.7, .6, .6, .8, .7, .7, .7, .7, .7, .7])
        detector.detectors = (("SigLIP", first), ("ViT", second))
        result = detector.predict(Image.new("RGB", (289, 202)), include_local_evidence=True)
        evidence = result["local_evidence"]
        self.assertEqual((evidence["grid_rows"], evidence["grid_cols"]), (3, 3))
        self.assertEqual(len(evidence["values"]), 9)
        self.assertEqual(len(evidence["raw_deltas"]), 9)
        self.assertEqual(evidence["analysis_size"], [289, 202])
        self.assertEqual(first.batch_size, 3)
        self.assertEqual(second.batch_size, 3)
        self.assertEqual(evidence["raw_deltas"][:3], [.2, .1, -.1])
        self.assertEqual(evidence["values"][:3], [1.0, .5, 0.0])
        self.assertIn("不是区域由 AI 生成的概率", evidence["meaning"])

        without_overlay = detector.predict(Image.new("RGB", (3840, 2160)))
        with_overlay = detector.predict(Image.new("RGB", (3840, 2160)),
                                        include_local_evidence=True)
        self.assertEqual(with_overlay["risk_score"], without_overlay["risk_score"])
        self.assertEqual(with_overlay["model_scores"], without_overlay["model_scores"])
        self.assertLessEqual(max(with_overlay["local_evidence"]["analysis_size"]), 512)

    def test_blur_grid_changes_only_selected_cell(self):
        image = Image.new("RGB", (90, 90))
        for x in range(90):
            for y in range(90):
                image.putpixel((x, y), (255 if (x + y) % 2 else 0, 0, 0))
        probes = blur_grid(image)
        self.assertEqual(len(probes), 9)
        self.assertNotEqual(probes[0].getpixel((10, 10)), image.getpixel((10, 10)))
        self.assertEqual(probes[0].getpixel((80, 80)), image.getpixel((80, 80)))

    def test_small_score_jitter_does_not_become_colored_region(self):
        detector = HighResolutionCrossCheck.__new__(HighResolutionCrossCheck)
        detector.detectors = (("SigLIP", FakeBatchDetector([.5] + [.499] * 9)),
                              ("ViT", FakeBatchDetector([.5] + [.499] * 9)))
        evidence = detector.predict(Image.new("RGB", (289, 202)),
                                    include_local_evidence=True)["local_evidence"]
        self.assertEqual(evidence["raw_deltas"], [.001] * 9)
        self.assertEqual(evidence["values"], [0.0] * 9)


if __name__ == "__main__":
    unittest.main()
