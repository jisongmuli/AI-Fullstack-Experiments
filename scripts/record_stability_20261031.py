"""Index completed tests and HTTP/browser evidence for this stability release."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stability_20261031"


def main():
    suites = []
    for name in ("server-tests", "model-tests", "segments", "chinese"):
        text = (ROOT / f"logs/stability-20261031-{name}.log").read_text(encoding="utf-8", errors="replace")
        count = re.search(r"^Ran (\d+) tests?", text, re.M)
        result = re.search(r"^OK(?: \(skipped=(\d+)\))?$", text, re.M)
        assert count and result and not re.search(r"^FAILED", text, re.M), name
        total, skipped = int(count[1]), int(result[1] or 0)
        suites.append({"name": name, "run": total, "passed": total-skipped, "skipped": skipped})
    browser_path = "output/playwright/stability-20261031-ui.json"
    browser = json.loads((ROOT / browser_path).read_text(encoding="utf-8"))
    http = json.loads((OUT / "http-verification.json").read_text(encoding="utf-8"))
    assert browser["status"] == http["status"] == "passed" and not browser["errors"]
    names = ["code/models/course_models.py", "code/models/course_preprocessing.py", "scripts/prepare_course_2026.py",
             "code/server/app.py", "code/server/course_agent.py", "web/course.js", "web/index.html"]
    fingerprints = [{"path": name, "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()} for name in names]
    result = {"status": "passed", "recorded_at": datetime.now(timezone.utc).isoformat(), "url": http["url"],
              "unit_suites": suites, "tests_run": sum(row["run"] for row in suites),
              "tests_passed": sum(row["passed"] for row in suites), "tests_skipped": sum(row["skipped"] for row in suites),
              "browser": {"path": browser_path, "checks_passed": len(browser["checks"]), "actual_local_llm_retry": True},
              "http": {"path": "results/stability_20261031/http-verification.json", "checks_passed": len(http["checks"]), "replayed": http["replayed"]},
              "source_fingerprints": fingerprints}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "tests_run", "tests_passed", "tests_skipped", "browser")}))


if __name__ == "__main__":
    main()
