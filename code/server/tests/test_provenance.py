"""Verify the provenance layer stays separate from model scores and private claims."""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenance import _safe_claims, inspect_provenance  # noqa: E402


class ProvenanceTests(unittest.TestCase):
    def test_unsigned_png_is_absent_not_ai(self):
        image = Image.new("RGB", (32, 32), (100, 100, 100))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        result = inspect_provenance(buffer.getvalue(), "PNG")
        self.assertEqual(result["status"], "credential_absent")
        self.assertNotIn("risk_score", result)

    def test_official_positive_and_negative_fixtures(self):
        root = Path(__file__).resolve().parents[3]
        directory = root / "data/raw/c2pa_official"
        if not directory.is_dir():
            self.skipTest("Official C2PA fixtures were not downloaded")
        expected = {
            "adobe-20220124-A.jpg": "credential_absent",
            "adobe-20220124-C.jpg": "valid_but_untrusted",
            "adobe-20220124-E-dat-CA.jpg": "invalid",
            "adobe-20220124-E-sig-CA.jpg": "invalid",
        }
        for name, status in expected.items():
            with self.subTest(name=name):
                result = inspect_provenance((directory / name).read_bytes(), "JPEG")
                self.assertEqual(result["status"], status)
                self.assertNotEqual(result["status"], "trusted")

    def test_only_known_claim_vocabulary_is_returned(self):
        manifest = {"assertions": [{"label": "c2pa.actions", "data": {"actions": [
            {"action": "c2pa.created", "digitalSourceType":
             "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
             "softwareAgent": {"name": "private uploader"}, "private_prompt": "secret"},
            {"action": "custom.secret", "digitalSourceType": "https://example.com/private/person"},
        ]}}]}
        actions, sources = _safe_claims(manifest)
        self.assertEqual(actions, ["创建"])
        self.assertEqual(sources, ["训练模型生成"])
        self.assertNotIn("secret", str((actions, sources)))


if __name__ == "__main__":
    unittest.main()
