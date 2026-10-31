"""Course split integrity, genuine physical compression and saved fine-tuning."""
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/llm"))
from agent_model import eligible, PLANS


class CourseLineageTests(unittest.TestCase):
    def test_text_pairs_stay_together_and_normalized_content_is_unique(self):
        manifest = json.loads((ROOT / "data/course_20261027/text.json").read_text(encoding="utf-8"))
        groups, hashes = {}, set()
        for row in manifest["records"]:
            self.assertEqual(groups.setdefault(row["group"], row["split"]), row["split"])
            self.assertNotIn(row["sha256"], hashes)
            hashes.add(row["sha256"])
        self.assertEqual(len(groups), 2000)
        self.assertEqual(len(hashes), 4000)

    def test_new_images_are_disjoint_from_prior_audit_ids_and_from_each_other(self):
        manifest = json.loads((ROOT / "data/course_20261027/images.json").read_text(encoding="utf-8"))
        previous = json.loads((ROOT / "data/benchmarks_2026/ntire/manifest.json").read_text(encoding="utf-8"))
        self.assertFalse({r["id"] for r in manifest["records"]} & {r["id"] for r in previous["records"]})
        self.assertEqual(len({r["pixel_sha256"] for r in manifest["records"]}), 400)
        self.assertEqual(manifest["split_counts"], {"train": 240, "validation": 80, "test": 80})

    def test_compression_changes_parameter_and_vocabulary_counts(self):
        directory = ROOT / "results/course_20261027"
        image = json.loads((directory / "experiment2-image.json").read_text(encoding="utf-8"))
        text = json.loads((directory / "experiment2-text.json").read_text(encoding="utf-8"))
        self.assertLess(image["optimized"]["parameters"], image["trained"]["parameters"])
        self.assertLess(text["optimized"]["features"], text["trained"]["features"])
        self.assertEqual(image["export_max_logit_difference"], 0.)

    def test_saved_finetuning_has_complete_steps_and_disjoint_cases(self):
        directory = ROOT / "results/course_20261027/agent"
        rows = json.loads((directory / "routing-data.json").read_text(encoding="utf-8"))["records"]
        self.assertEqual(len({row["sha256"] for row in rows}), len(rows))
        history = [json.loads(row) for row in (directory / "training-history.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["step"] for row in history], list(range(1, 65)))
        metrics = json.loads((directory / "finetune-metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["trainable_parameters"], 159744)
        self.assertEqual(metrics["after"]["samples"], 30)
        self.assertGreater(metrics["after"]["accuracy"], metrics["before"]["accuracy"])

    def test_only_plans_supported_by_real_inputs_are_eligible(self):
        none = eligible({})
        self.assertEqual(set(none), {"A", "B", "J"})
        self.assertNotIn("G", eligible({"image": True}))
        self.assertIn("G", eligible({"image": True, "text": True}))
        self.assertEqual(PLANS["I"], ["text_detection", "clarity_edit"])


if __name__ == "__main__": unittest.main()
