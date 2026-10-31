"""Boundaries for expiring in-memory document reports."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_store import ReportStore  # noqa: E402


class ReportStoreTests(unittest.TestCase):
    def test_expiry_at_boundary_and_delete(self):
        now = [1_000.0]
        store = ReportStore(ttl_seconds=10, clock=lambda: now[0])
        result = store.put({"extracted_text": "secret"})
        report_id = result["report_id"]
        now[0] = 1009.99
        self.assertEqual(store.get(report_id)["extracted_text"], "secret")
        now[0] = 1010
        self.assertIsNone(store.get(report_id))
        self.assertFalse(store.delete(report_id))
        other = store.put({"extracted_text": "other"})
        self.assertTrue(store.delete(other["report_id"]))
        self.assertFalse(store.delete(other["report_id"]))

    def test_all_returned_and_input_objects_are_independent(self):
        source = {"detection": {"segments": [{"text": "original"}]}}
        store = ReportStore()
        returned = store.put(source)
        report_id = returned["report_id"]
        source["detection"]["segments"][0]["text"] = "source mutated"
        returned["detection"]["segments"][0]["text"] = "return mutated"
        fetched = store.get(report_id)
        self.assertEqual(fetched["detection"]["segments"][0]["text"], "original")
        fetched["detection"]["segments"].clear()
        self.assertEqual(store.get(report_id)["detection"]["segments"][0]["text"], "original")

    def test_count_evicts_oldest_without_extending_ttl_on_read(self):
        now = [0.0]
        store = ReportStore(ttl_seconds=10, max_items=2, clock=lambda: now[0])
        first = store.put({"value": "one"})
        now[0] = 1
        second = store.put({"value": "two"})
        self.assertIsNotNone(store.get(first["report_id"]))
        now[0] = 2
        third = store.put({"value": "three"})
        self.assertIsNone(store.get(first["report_id"]))
        self.assertIsNotNone(store.get(second["report_id"]))
        self.assertIsNotNone(store.get(third["report_id"]))
        now[0] = 11
        self.assertIsNone(store.get(second["report_id"]))

    def test_byte_budget_evicts_and_oversize_is_rejected_without_loss(self):
        store = ReportStore(max_items=20, max_bytes=1_600)
        first = store.put({"text": "x" * 700})
        second = store.put({"text": "y" * 700})
        self.assertIsNone(store.get(first["report_id"]))
        self.assertIsNotNone(store.get(second["report_id"]))
        with self.assertRaises(ValueError):
            store.put({"text": "large" * 1_000})
        self.assertEqual(store.get(second["report_id"])["text"], "y" * 700)
        sizes = sum(len(json.dumps(item[2], ensure_ascii=False, allow_nan=False).encode("utf-8"))
                    for item in store._items.values())
        self.assertLessEqual(sizes, store.max_bytes)

    def test_nonfinite_or_nonserializable_reports_do_not_enter_cache(self):
        store = ReportStore()
        saved = store.put({"value": "good"})
        for value in (float("nan"), float("inf"), float("-inf"), object()):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises((ValueError, TypeError)):
                    store.put({"value": value})
        self.assertEqual(len(store._items), 1)
        self.assertEqual(store.get(saved["report_id"])["value"], "good")

    def test_list_metadata_is_detached_and_excludes_source_and_detector_details(self):
        store = ReportStore()
        self.addCleanup(store.close)
        saved = store.put({"filename": "example.txt", "created_at": "2026-10-25T12:00:00Z",
                           "format": "txt", "extracted_text": "private original",
                           "source_sha256": "private fingerprint", "sections": [{"text": "private section"}],
                           "detection": {"segments": [{"text": "private segment"}]},
                           "document_summary": {"highlighted_total_percent": 12.5, "private_extra": "secret"}})
        listed = store.list_reports()
        self.assertEqual(set(listed[0]), {"report_id", "filename", "format", "created_at",
                                         "expires_at", "highlighted_total_percent"})
        self.assertEqual(listed[0]["highlighted_total_percent"], 12.5)
        self.assertNotIn("private", json.dumps(listed))
        listed[0]["filename"] = "changed.txt"
        self.assertEqual(store.get(saved["report_id"])["filename"], "example.txt")

    def test_list_is_newest_first_and_respects_expiry_eviction_and_deletion(self):
        now = [1000.0]
        store = ReportStore(ttl_seconds=10, max_items=2, clock=lambda: now[0])
        self.addCleanup(store.close)
        store.put({"filename": "evicted.txt"})
        now[0] += 1
        second = store.put({"filename": "second.txt"})
        now[0] += 1
        third = store.put({"filename": "third.txt"})
        self.assertEqual([item["filename"] for item in store.list_reports()], ["third.txt", "second.txt"])
        self.assertEqual(store.list_reports()[1]["expires_at"], second["expires_at"])
        now[0] = 1011
        self.assertEqual([item["report_id"] for item in store.list_reports()], [third["report_id"]])
        self.assertTrue(store.delete(third["report_id"]))
        self.assertEqual(store.list_reports(), [])


if __name__ == "__main__":
    unittest.main()
