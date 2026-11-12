"""Index evidence for all five course requirements after real tests finish."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/course_20261027"


def load(path): return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    suites = []
    for name, count, skipped in [("server-tests", 112, 1), ("model-tests", 46, 0), ("segments", 12, 0), ("chinese", 3, 0)]:
        value = (ROOT / f"logs/stability-course-{name}.log").read_text(encoding="utf-8", errors="replace")
        assert f"Ran {count} tests" in value and re.search(r"^OK(?: \(skipped=1\))?$", value, re.M), name
        suites.append({"name": name, "run": count, "passed": count-skipped, "skipped": skipped})
    browser = []
    for path in ("output/playwright/course-ui.json", "output/playwright/course-document-ui.json"):
        record = load(path)
        assert record["status"] == "passed" and not record["errors"]
        browser.append({"path": path, "passed": len(record["checks"])})
    http = load("results/course_20261027/http-verification.json")
    extra = load("results/course_20261027/agent/extra-http.json")
    assets = load("results/course_20261027/asset-verification.json")
    assert http["status"] == extra["status"] == assets["status"] == "passed"
    result = {"status": "passed", "recorded_at": datetime.now(timezone.utc).isoformat(), "url": "http://127.0.0.1:8006/#course",
              "unit_suites": suites, "tests_run": sum(row["run"] for row in suites),
              "tests_passed": sum(row["passed"] for row in suites), "tests_skipped": 1,
              "skip_reason": "Optional official C2PA signed test asset is absent",
              "browser": browser, "browser_checks_passed": sum(row["passed"] for row in browser),
              "course_api_prediction_replay": http["cohort_score_replay"],
              "actual_agent_tool_types": sorted({row["tool"] for response in [*http["agent_responses"], extra["history_response"], extra["editing_response"]] for row in response["tool_calls"]}),
              "actual_agent_requests": len(http["agent_responses"])+2,
              "asset_files_verified": assets["course_files_verified"], "official_agent_assets_verified": assets["official_agent_assets_verified"],
              "pipeline": {"experiment1": "scripts/prepare_course_2026.py", "experiment2_3": "scripts/train_course_2026.py",
                           "experiment4": "code/models/course_models.py", "experiment5": "scripts/train_agent_router.py"},
              "assessment": "The five functional stages and reproducible evidence are implemented; course and advanced model cohorts are reported separately."}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "tests_passed": result["tests_passed"], "browser_checks": result["browser_checks_passed"]}))


if __name__ == "__main__": main()
