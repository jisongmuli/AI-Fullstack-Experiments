"""Collect completed checks without rerunning model inference."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    units = []
    for name, expected, skipped in [
        ("server-tests", 101, 1), ("model-tests", 35, 0),
        ("text-segments", 12, 0), ("chinese-text", 3, 0),
    ]:
        text = (ROOT / f"logs/stability-modern-{name}.log").read_text(encoding="utf-8", errors="replace")
        assert f"Ran {expected} tests" in text and re.search(r"^OK(?: \(skipped=1\))?$", text, re.M), name
        units.append({"suite": name, "run": expected, "passed": expected - skipped, "skipped": skipped})
    workflows = []
    for name in ["output/playwright/no-agent-check.json", "output/playwright/modern-workflow-check.json",
                 "results/upgrade_20261026/portable-smoke.json", "results/upgrade_20261026/jobs-smoke.json",
                 "results/upgrade_20261026/http-text-smoke.json", "results/upgrade_20261026/resolutions.json"]:
        data = load(name)
        assert data["status"] == "passed", name
        assert not data.get("errors"), name
        workflows.append({"record": name, "checks": len(data["checks"]), "status": "passed"})
    assets = load("results/upgrade_20261026/asset-verification.json")
    concurrency = load("results/upgrade_20261026/concurrency.json")
    assert assets["status"] == concurrency["status"] == "passed"
    result = {"status": "passed", "recorded_at": datetime.now(timezone.utc).isoformat(),
              "python": "3.12.2", "device": "CPU", "url": "http://127.0.0.1:8005/",
              "unit_suites": units, "tests_run": sum(row["run"] for row in units),
              "tests_passed": sum(row["passed"] for row in units), "tests_skipped": 1,
              "skip_reason": "Optional signed C2PA fixture is absent; unsigned image inspection was exercised.",
              "workflow_checks": workflows, "assets": assets, "concurrency": concurrency,
              "accuracy_record": "results/upgrade_20261026/final-evaluation.json",
              "accuracy_threshold": .5,
              "runtime_review_band": [.2, .8],
              "sample_roles": "NLPCC Phase 2 binary subset; September 2026 generated English with traceable human controls; NTIRE public training-shard samples with separate validation/test roles. Native and resized image cohorts share source images."}
    output = ROOT / "results/upgrade_20261026/verification.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "tests": result["tests_run"], "workflows": sum(row["checks"] for row in workflows)}))


if __name__ == "__main__":
    main()
