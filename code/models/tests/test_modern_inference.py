"""Bounded native crops, full token-window coverage and modern adapter contracts."""
from __future__ import annotations

import sys
import threading
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_detectors import DearDetector, TasteDetector, MpuDetector
from modern_inference import ModernTextAdapter, ModernImageAdapter


EN = "The research team compared the source records before the final evaluation."
ZH = "研究团队核对了原始样本的来源和标签，记录清洗流程与验证结果，然后检查每一个数据字段。" * 3


class CropTests(unittest.TestCase):
    def test_all_positive_shapes_have_bounded_native_crops(self):
        detector = DearDetector.__new__(DearDetector)
        detector.tile_size = 256
        for width, height in [(1, 1), (1, 4096), (4096, 1), (17, 29), (289, 202),
                              (3840, 2160), (2160, 3840), (4096, 2160), (4096, 4096)]:
            with self.subTest(size=(width, height)):
                boxes = detector.tile_boxes(width, height)
                self.assertTrue(1 <= len(boxes) <= 9)
                self.assertEqual(min(b[0] for b in boxes), 0)
                self.assertEqual(min(b[1] for b in boxes), 0)
                self.assertEqual(max(b[2] for b in boxes), width)
                self.assertEqual(max(b[3] for b in boxes), height)
                self.assertEqual(len(set(boxes)), len(boxes))
                for left, top, right, bottom in boxes:
                    self.assertTrue(0 <= left < right <= width and 0 <= top < bottom <= height)
                    self.assertLessEqual((right-left)*(bottom-top), 256*256)

    def test_zero_and_negative_dimensions_are_rejected(self):
        detector = DearDetector.__new__(DearDetector)
        detector.tile_size = 256
        for size in [(0, 100), (100, 0), (-1, 20)]:
            with self.assertRaises(ValueError):
                detector.tile_boxes(*size)


class FakeVocabulary:
    def transform(self, texts):
        return types.SimpleNamespace(getnnz=lambda axis: np.array([0 if "qzx" in t or set(t) == {"龘"} else 1 for t in texts]))


class FakeTaste:
    def __init__(self):
        self.calls = []

    def representations(self, texts):
        self.calls.append(list(texts))
        values = np.zeros((len(texts), 768), dtype=np.float32)
        values[:, 0] = 3.
        return [4.] * len(texts), values


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ModernTextAdapter.__new__(ModernTextAdapter)
        self.adapter.engine = FakeTaste()
        self.adapter.vocabularies = dict.fromkeys(["english", "chinese"], FakeVocabulary())
        self.adapter.policy = {"chinese_head": "adapted", "chinese_calibration": {"coefficient": 2., "intercept": 0.},
                               "english_calibration": {"coefficient": 1., "intercept": 0.}}
        self.adapter.coefficient = np.zeros(768)
        self.adapter.coefficient[0] = 1.
        self.adapter.intercept = 0.

    def test_language_specific_heads_and_input_order(self):
        results = self.adapter.predict_many([EN, ZH, EN])
        self.assertAlmostEqual(results[0]["risk_score"], .982014, places=6)
        self.assertAlmostEqual(results[1]["risk_score"], .997527, places=6)
        self.assertEqual(results[0]["risk_score"], results[2]["risk_score"])
        self.assertEqual(self.adapter.engine.calls, [[EN], [ZH]])
        self.assertIn("en-official", results[0]["model_version"])
        self.assertIn("zh-adapted", results[1]["model_version"])

    def test_duplicate_outputs_have_independent_evidence_lists(self):
        results = self.adapter.predict_many([EN, EN])
        results[0]["signals"].clear()
        self.assertTrue(results[1]["signals"])

    def test_context_and_zero_feature_inputs_never_enter_the_encoder(self):
        texts = ["Please write an essay about science.", "> " + EN, "```python\nprint(1)\n```", "qzxvjk " * 9, "龘" * 100]
        results = self.adapter.predict_many(texts)
        self.assertTrue(all(r["risk_score"] is None and r["out_of_scope"] for r in results))
        self.assertEqual(self.adapter.engine.calls, [])
        self.assertEqual(results[-1]["applicability"], "insufficient_model_features")

    def test_short_fragments_stay_unscored(self):
        results = self.adapter.predict_many(["A brief note.", "中文短句。"])
        self.assertTrue(all(r["risk_score"] is None for r in results))
        self.assertEqual(self.adapter.engine.calls, [])

    def test_mpu_uses_the_selected_language_calibration(self):
        self.adapter.pipeline = {"english_backend": "mpu", "chinese_backend": "mpu",
            "english_calibration": {"coefficient": 1., "intercept": 0.},
            "chinese_calibration": {"coefficient": 2., "intercept": -3.}}
        calls = []
        def engine(language, backend):
            calls.append((language, backend))
            return types.SimpleNamespace(logits=lambda texts: [2.] * len(texts))
        self.adapter._engine = engine
        results = self.adapter.predict_many([EN, ZH])
        self.assertAlmostEqual(results[0]["risk_score"], .880797, places=6)
        self.assertAlmostEqual(results[1]["risk_score"], .731059, places=6)
        self.assertEqual(calls, [("english", "mpu"), ("chinese", "mpu")])
        self.assertEqual(self.adapter.engine.calls, [])

    def test_low_resolution_selects_its_validated_detector_and_calibration(self):
        adapter = ModernImageAdapter.__new__(ModernImageAdapter)
        adapter.policy = {"variant": "dear_r", "calibration": {"coefficient": 1., "intercept": 0.},
            "low_resolution": {"variant": "dear_c", "short_side_below": 512, "tile_size": 256,
                               "calibration": {"coefficient": 2., "intercept": 1.}}}
        adapter.engine = types.SimpleNamespace(logit=lambda image: 1.)
        adapter._secondary = types.SimpleNamespace(logit=lambda image: 2.)
        adapter._load_lock = threading.Lock()
        self.assertAlmostEqual(adapter.score(Image.new("RGB", (320, 240))), .993307149)
        self.assertAlmostEqual(adapter.score(Image.new("RGB", (512, 512))), .731058579)

    def test_preview_heatmap_keeps_the_whole_image_detector_selection(self):
        adapter = ModernImageAdapter.__new__(ModernImageAdapter)
        adapter.policy = {"variant": "dear_r", "calibration": {"coefficient": 1., "intercept": 0.}}
        adapter.engine = types.SimpleNamespace(tile_size=256, tile_boxes=lambda w, h: [(0, 0, 256, 256)])
        seen = []
        adapter.score = lambda image, selection=None: seen.append((image.size, selection[2])) or .6
        result = adapter.predict(Image.new("RGB", (512, 512)), include_local_evidence=True)
        self.assertEqual(len(seen), 11)
        self.assertTrue(all(variant == "dear_r" for _, variant in seen))
        self.assertEqual(result["local_evidence"]["analysis_size"], [256, 256])

    def test_missing_classifier_weights_are_a_load_error(self):
        with patch("research_detectors.AutoModelForSequenceClassification.from_pretrained",
                   return_value=(object(), {"missing_keys": ["classifier.weight"]})):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                MpuDetector(Path("."), "chinese")

    def test_4k_input_uses_modern_image_adapter_without_resize(self):
        adapter = ModernImageAdapter.__new__(ModernImageAdapter)
        adapter.engine = types.SimpleNamespace(tile_size=256, tile_boxes=lambda w, h: [(0, 0, min(w, 256), min(h, 256))])
        adapter.policy = {"variant": "dear_r", "calibration": {"coefficient": 1., "intercept": 0.}}
        seen = []
        adapter.score = lambda image, selection=None: seen.append(image.size) or .9
        result = adapter.predict(Image.new("RGB", (3840, 2160)))
        self.assertEqual(seen, [(3840, 2160)])
        self.assertEqual(result["input_size"], [3840, 2160])
        self.assertEqual(result["decision"], "ai")
        self.assertEqual(len(result["model_scores"]), 1)


class TokenWindowTests(unittest.TestCase):
    def test_tail_windows_are_included_without_double_counting_overlap(self):
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                self.arguments = kwargs
                return {"input_ids": [[2] * 512, [6] * 180, [4] * 5], "overflow_to_sample_mapping": [0, 0, 1]}

            def pad(self, windows, **kwargs):
                return {"input_ids": torch.tensor([[r["input_ids"][0]] for r in windows])}

        class Model:
            def bert(self, input_ids):
                value = torch.zeros((len(input_ids), 768))
                value[:, 0] = input_ids[:, 0]
                return types.SimpleNamespace(pooler_output=value)

            dropout = staticmethod(lambda value: value)
            classifier = staticmethod(lambda value: torch.stack([torch.zeros(len(value)), value[:, 0]], dim=1))

        detector = TasteDetector.__new__(TasteDetector)
        detector.tokenizer, detector.model = Tokenizer(), Model()
        detector.batch_size, detector.device, detector._lock = 2, torch.device("cpu"), threading.Lock()
        logits, features = detector.representations(["long", "short"])
        self.assertAlmostEqual(logits[0], (2 * 510 + 6 * 114) / 624)
        self.assertEqual(logits[1], 4.)
        self.assertAlmostEqual(features[0, 0], logits[0], places=6)
        self.assertEqual(detector.tokenizer.arguments["stride"], 64)
        self.assertTrue(detector.tokenizer.arguments["return_overflowing_tokens"])


if __name__ == "__main__":
    unittest.main()
