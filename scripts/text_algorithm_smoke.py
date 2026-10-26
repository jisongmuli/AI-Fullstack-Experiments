"""Exercise the installed text algorithm through HTTP using generated inputs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8003")
    parser.add_argument("--output", type=Path, default=Path("results/algorithm_20261026/http-smoke.json"))
    args = parser.parse_args()
    english = "The research team compared the source records and checked every label before the final evaluation."
    chinese = "研究团队先核对原始样本的来源与时间，再检查缺失字段、重复记录和异常数值。完成清洗后，团队按来源划分训练集与验证集，分别记录评估结果，整理误报样本并进行人工复核。"
    oov = "qzxvjk " * 9
    checks, reports, jobs = [], [], []
    with httpx.Client(base_url=args.url, timeout=180, trust_env=False) as client:
        try:
            for name, text, scoreable in [("English", english, True), ("Chinese", chinese, True),
                                          ("Unseen Latin features", oov, False), ("Unseen Han features", "龘" * 100, False)]:
                response = client.post("/api/text/detect", json={"text": text})
                response.raise_for_status()
                value = response.json()
                assert bool(value["segments"][0]["risk_score"] is not None) == scoreable
                if not scoreable:
                    assert value["risk_score"] is None and value["decision"] == "uncertain"
                    assert value["applicability"] == "insufficient_model_features"
                    assert value["segment_summary"]["scored_character_percent"] == 0
                checks.append({"name": name, "passed": True, "risk_score": value["risk_score"], "applicability": value["applicability"]})
            source = "\n\n".join((chinese + "\n\n" + english) for _ in range(100))
            accepted = client.post("/api/document/jobs", data={"text": source})
            assert accepted.status_code == 202, accepted.text
            job_id = accepted.json()["job_id"]
            jobs.append(job_id)
            deadline = time.monotonic() + 90
            while True:
                response = client.get(f"/api/document/jobs/{job_id}")
                response.raise_for_status()
                job = response.json()
                if job["status"] in {"completed", "failed", "cancelled"}:
                    break
                assert time.monotonic() < deadline, "Long-document task timed out"
                time.sleep(.03)
            assert job["status"] == "completed", job
            report_id = job["report_id"]
            reports.append(report_id)
            exported = client.get(f"/api/document/reports/{report_id}/json")
            exported.raise_for_status()
            report = exported.json()
            assert report["extracted_text"] == source
            assert report["detection"]["segments"]
            for segment in report["detection"]["segments"]:
                assert source[segment["start"]:segment["end"]] == segment["text"]
            checks.append({"name": "Long-document task and JSON export", "passed": True,
                           "characters": len(source), "segments": len(report["detection"]["segments"])})
            agent = client.post("/api/agent/run", json={"text": oov, "edit": False})
            agent.raise_for_status()
            findings = agent.json()["findings"]
            assert findings["text"]["risk_score"] is None
            checks.append({"name": "Agent preserves zero-feature abstention", "passed": True})
        finally:
            for job_id in jobs:
                value = client.get(f"/api/document/jobs/{job_id}")
                if value.status_code == 200 and value.json()["status"] not in {"completed", "failed", "cancelled"}:
                    client.post(f"/api/document/jobs/{job_id}/cancel")
                else:
                    client.delete(f"/api/document/jobs/{job_id}")
            for report_id in reports:
                client.delete(f"/api/document/reports/{report_id}")
        assert client.get("/api/runtime").json()["processing"]["active"] == 0
    result = {"status": "passed", "url": args.url, "actual_models": True,
              "checks": checks, "accuracy_evaluated": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
