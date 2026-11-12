"""Index completed performance and workflow regression evidence."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/performance_20261026"


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    suites = []
    for name, count, skipped in [("server-tests", 105, 1), ("model-tests", 41, 0), ("segments", 12, 0), ("chinese", 3, 0)]:
        log = (ROOT / f"logs/stability-performance-{name}.log").read_text(encoding="utf-8", errors="replace")
        assert f"Ran {count} tests" in log and re.search(r"^OK(?: \(skipped=1\))?$", log, re.M), name
        suites.append({"suite": name, "run": count, "passed": count - skipped, "skipped": skipped})
    workflows = []
    for path in ["results/performance_20261026/resolutions.json", "results/performance_20261026/text-smoke.json",
                 "output/playwright/performance-workflow.json", "output/playwright/performance-regression.json"]:
        result = load(path)
        assert result["status"] == "passed" and not result.get("errors"), path
        workflows.append({"record": path, "checks": len(result["checks"]), "status": "passed"})
    parity = load("results/performance_20261026/score-parity.json")
    concurrency = load("results/performance_20261026/concurrency.json")
    assert parity["status"] == concurrency["status"] == "passed"
    result = {"status": "passed", "recorded_at": datetime.now(timezone.utc).isoformat(),
              "baseline_commit": "6ccf4a6", "python": "3.12.2", "device": "CPU",
              "url": "http://127.0.0.1:8005/", "unit_suites": suites,
              "tests_run": sum(row["run"] for row in suites), "tests_passed": sum(row["passed"] for row in suites),
              "tests_skipped": 1, "skip_reason": "Optional official signed C2PA fixture is absent",
              "workflow_checks": workflows, "workflow_checks_passed": sum(row["checks"] for row in workflows),
              "profile_before": load("results/performance_20261026/before.json"),
              "profile_after": load("results/performance_20261026/after.json"),
              "profile_scope": "Single first model-information request in separate fresh Python processes; not inference latency or an averaged benchmark",
              "score_regression": parity, "concurrency": concurrency,
              "accuracy_record": "results/upgrade_20261026/final-evaluation.json"}
    (RESULTS / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "tests_passed": result["tests_passed"], "workflow_checks": result["workflow_checks_passed"]}))


if __name__ == "__main__":
    main()
