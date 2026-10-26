"""Model information stays cheap and named results avoid unused file scans."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import ROOT, ModelService, create_app


class ModelInformationTests(unittest.TestCase):
    def test_information_checks_assets_without_loading_models_or_hashing_weights(self):
        service = ModelService(ROOT)
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.suffix in {".pt", ".pth", ".bin", ".ts", ".joblib", ".safetensors"}:
                raise AssertionError("Model information opened a checkpoint")
            return original(path, *args, **kwargs)
        with patch.object(service, "get_bundle", side_effect=AssertionError("Eager bundle load")), \
             patch.object(service, "artifact_fingerprints", side_effect=AssertionError("Unused hashes")), \
             patch.object(Path, "open", guarded), \
             TestClient(create_app(service), base_url="http://localhost") as client:
            result = client.get("/api/models")
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.json()["available"])
        self.assertFalse(result.json()["loaded"])
        self.assertFalse(result.json()["fingerprints_verified"])
        self.assertEqual(result.json()["artifact_fingerprints"], {})
        self.assertIsNone(service._bundle)

    def test_missing_assets_are_reported_without_attempting_a_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            service = ModelService(Path(temporary))
            with patch.object(service, "get_bundle", side_effect=AssertionError("Unexpected load")), \
                 TestClient(create_app(service), base_url="http://localhost") as client:
                result = client.get("/api/models").json()
            self.assertFalse(result["available"])
            self.assertIn("Missing detector asset", result["reason"])

    def test_named_and_unscored_results_avoid_fingerprint_reads(self):
        class Bundle:
            def get_bundle(self):
                return self
            def predict_text(self, text):
                if "unscored" in text:
                    return {"out_of_scope": True, "scope_reason": "fixture"}
                return {"risk_score": .3, "decision": "human", "signals": [], "model_version": "fixture-named"}
            def artifact_fingerprints(self):
                raise AssertionError("Named result scanned unrelated model files")
        with TestClient(create_app(Bundle()), base_url="http://localhost") as client:
            named = client.post("/api/text/detect", json={"text": "The team compared every source record before publication."})
            unscored = client.post("/api/text/detect", json={"text": "The unscored input contains enough ordinary English words."})
        self.assertEqual(named.status_code, 200)
        self.assertEqual(named.json()["model_version"], "fixture-named")
        self.assertEqual(unscored.status_code, 200)
        self.assertIsNone(unscored.json()["risk_score"])

    def test_legacy_unnamed_results_retain_fingerprint_version(self):
        class Bundle:
            def get_bundle(self):
                return self
            def predict_text(self, text):
                return {"risk_score": .3, "decision": "human", "signals": []}
            def artifact_fingerprints(self):
                return {"text": "1234abcdef"}
        with TestClient(create_app(Bundle()), base_url="http://localhost") as client:
            result = client.post("/api/text/detect", json={"text": "The team compared every source record before publication."})
        self.assertEqual(result.json()["model_version"], "sha256:1234abcdef")


if __name__ == "__main__":
    unittest.main()
