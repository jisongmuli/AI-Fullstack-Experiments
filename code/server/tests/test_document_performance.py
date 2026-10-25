"""Regression checks for long-document cost and passive report expiry."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import ModelService  # noqa: E402
from documents import analyze_document, extract_pasted_text  # noqa: E402
from report_store import ReportStore  # noqa: E402


class DocumentPerformanceTests(unittest.TestCase):
    def test_artifact_hash_streaming_freezes_across_paragraph_callbacks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "models" / "text_detector.joblib"
            model_path.parent.mkdir()
            payload = (bytes(range(256)) * 8192) + b"final partial streaming block"
            model_path.write_bytes(payload)
            service = ModelService(root)
            fingerprints = []

            def detector(text):
                version = service.artifact_fingerprints()["text"]
                fingerprints.append(version)
                return {"risk_score": .5, "decision": "uncertain", "model_version": version,
                        "segments": [{"start": 0, "end": len(text), "text": text,
                                      "risk_score": .5, "decision": "uncertain", "evidence": []}]}

            source = extract_pasted_text("\n\n".join(["Measured document content. " * 12] * 40))
            with patch.object(service, "_compute_fingerprints", wraps=service._compute_fingerprints) as compute:
                # The method must stream bytes; whole-file reads are forbidden here.
                with patch.object(Path, "read_bytes", side_effect=AssertionError("Whole-file hash read")):
                    result = analyze_document(source, detector)
                self.assertEqual(compute.call_count, 1)
                self.assertGreater(len(fingerprints), 3)
                expected = hashlib.sha256(payload).hexdigest()[:16]
                self.assertEqual(set(fingerprints), {expected})
                self.assertEqual(result["document_summary"]["scored_character_percent"], 100)
                # Updating files cannot alter the identity of the loaded instance.
                updated = payload + b"replacement artifact"
                model_path.write_bytes(updated)
                self.assertEqual(service.artifact_fingerprints()["text"], expected)
                caller_copy = service.artifact_fingerprints()
                caller_copy["text"] = "caller mutation"
                self.assertEqual(service.artifact_fingerprints()["text"], expected)
                self.assertEqual(compute.call_count, 1)
                self.assertEqual(ModelService(root).artifact_fingerprints()["text"],
                                 hashlib.sha256(updated).hexdigest()[:16])

    def test_reports_expire_without_a_new_store_request(self):
        store = ReportStore(ttl_seconds=.05)
        pause = threading.Event()
        try:
            report = store.put({"extracted_text": "private document text"})
            report_id = report["report_id"]
            deadline = time.monotonic() + .75
            removed = False
            while time.monotonic() < deadline:
                # Inspect without calling get/put/delete: expiry must be passive.
                with store._lock:
                    removed = report_id not in store._items
                if removed:
                    break
                pause.wait(.01)
            self.assertTrue(removed, "Expired source text stayed in memory until a new request")
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
