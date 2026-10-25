import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_acceptance import audit_file, gate_audit, wilson_interval


class AcceptanceAuditTests(unittest.TestCase):
    def test_abstention_does_not_hide_class_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("sample_id", "actual", "risk_score"))
                writer.writerows((
                    ("h1", 0, 0.95), ("h2", 0, 0.5), ("h3", 0, 0.1),
                    ("a1", 1, 0.05), ("a2", 1, 0.5), ("a3", 1, 0.9),
                    ("a4", 1, ""),
                ))
            audit = audit_file(path, 0.2, 0.8)
        self.assertEqual(audit["coverage"]["count"], 4)
        self.assertEqual(audit["coverage"]["denominator"], 7)
        self.assertEqual(audit["confusion_with_abstention"]["ai"]["uncertain"], 2)
        self.assertEqual(audit["false_positive_rate_all_human"]["count"], 1)
        self.assertEqual(audit["false_negative_rate_all_ai"]["count"], 1)
        self.assertEqual(audit["false_positive_rate_covered_human"]["denominator"], 2)
        self.assertEqual(gate_audit(audit, 0.01, 0.01, 0.8, 3)["status"], "fail")

    def test_wilson_zero_error_still_has_uncertainty(self):
        lower, upper = wilson_interval(0, 100)
        self.assertEqual(lower, 0)
        self.assertGreater(upper, 0.03)


if __name__ == "__main__":
    unittest.main()
