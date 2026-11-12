"""Lazy startup, complete active assets, and independent modality loading."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import DetectorService
from modern_inference import ModernTextAdapter
from runtime_assets import asset_problems, required_assets


def fixture(root):
    directory = root / "models"
    directory.mkdir()
    values = {
        "algorithm_policy.json": {"text_backend": "transformer_2026", "image_backend": "dear", "cpu_threads": 4},
        "text_policy_2026.json": {"english_backend": "mpu", "chinese_backend": "mpu"},
        "text_calibration_phase1.json": {"chinese_backend": "mpu", "chinese_calibration": {"coefficient": 1., "intercept": 0.}},
        "model_config.json": {"abstain_probability": .8, "text_calibration": {"coefficient": 1., "intercept": 0.}},
        "text_zh_config.json": {"model_version": "fixture", "calibration": {"coefficient": 1., "intercept": 0.}},
    }
    for name, value in values.items():
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
    for path in required_assets(root):
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in {".bin", ".pth"}:
            with path.open("wb") as stream:
                stream.truncate(1024 * 1024)
        else:
            path.write_text("fixture asset data with enough bytes", encoding="utf-8")


class LazyAssetsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fixture(self.root)

    def test_active_mpu_does_not_require_taste_or_legacy_image_weights(self):
        names = [path.relative_to(self.root).as_posix() for path in required_assets(self.root)]
        self.assertFalse(any("taste/" in name or name.endswith("image_detector.ts") for name in names))
        self.assertEqual(asset_problems(self.root), [])
        with patch("inference.joblib.load", side_effect=AssertionError("Eager vocabulary load")), \
             patch("inference.torch.jit.load", side_effect=AssertionError("Eager image load")):
            service = DetectorService(self.root)
        self.assertIsNone(service.text_model)
        self.assertIsNone(service.image_model)

    def test_missing_tokenizer_and_lfs_pointer_are_reported_before_inference(self):
        (self.root / "models/mpu_en_v3/merges.txt").unlink()
        (self.root / "models/mpu_zh_v3/pytorch_model.bin").write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:" + "a" * 64 + "\nsize 409147757\n", encoding="utf-8")
        problems = asset_problems(self.root)
        self.assertTrue(any("merges.txt" in problem for problem in problems))
        self.assertTrue(any("Restore the full LFS" in problem for problem in problems))
        with self.assertRaises(FileNotFoundError):
            DetectorService(self.root)

    def test_image_initialization_never_loads_text_vocabularies(self):
        service = DetectorService(self.root)
        with patch("modern_inference.ModernImageAdapter", return_value=object()) as image, \
             patch("inference.joblib.load", side_effect=AssertionError("Unrelated vocabulary")), \
             patch("inference.torch.jit.load", side_effect=AssertionError("Historical image model")):
            first = service._modern("image")
            self.assertIs(service._modern("image"), first)
        image.assert_called_once()
        self.assertIsNone(service.text_model)

    def test_text_initialization_never_loads_historical_image_model(self):
        service = DetectorService(self.root)
        vocabulary = Mock()
        vocabulary.get_feature_names_out.return_value = ["word"]
        model = {"vectorizer": vocabulary}
        with patch("inference.joblib.load", return_value=model) as load, \
             patch("modern_inference.ModernTextAdapter", return_value=object()) as text, \
             patch("inference.torch.jit.load", side_effect=AssertionError("Unrelated image model")):
            first = service._modern("text")
            self.assertIs(service._modern("text"), first)
        self.assertEqual(load.call_count, 2)
        text.assert_called_once()
        self.assertIsNone(service.image_model)

    def test_concurrent_text_loading_publishes_one_complete_pair(self):
        service = DetectorService(self.root)
        started, release = threading.Event(), threading.Event()
        vocabulary = Mock()
        vocabulary.get_feature_names_out.return_value = ["word"]
        def load(path):
            if path.name == "text_detector.joblib":
                started.set()
                self.assertTrue(release.wait(2))
            return {"vectorizer": vocabulary}
        with patch("inference.joblib.load", side_effect=load) as mocked, ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(service._load_text_models)
            self.assertTrue(started.wait(2))
            second = workers.submit(service._load_text_models)
            self.assertIsNone(service.text_model)
            release.set()
            first.result(timeout=2); second.result(timeout=2)
        self.assertEqual(mocked.call_count, 2)
        self.assertIsNotNone(service.zh_text_model)

    def test_mpu_adapter_does_not_open_inactive_taste_head(self):
        with patch("modern_inference.np.load", side_effect=AssertionError("Inactive classifier head")):
            adapter = ModernTextAdapter(self.root, object(), object())
        self.assertEqual(adapter.policy, {})
        self.assertIsNone(adapter.engine)


if __name__ == "__main__":
    unittest.main()
