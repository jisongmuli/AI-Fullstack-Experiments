"""Actual HTTP replay of course test cohorts and local-LLM Agent tools."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/course_20261027"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8006")
    args = parser.parse_args()
    checks, agents, reports = [], [], []
    with httpx.Client(base_url=args.url, timeout=180, trust_env=False) as client:
        overview = client.get("/api/course").json()
        assert all(row["ready"] for row in overview["stages"])
        checks.append({"name": "All five course stages have evidence", "passed": True})
        images = json.loads((ROOT / "data/course_20261027/images.json").read_text(encoding="utf-8"))["records"]
        expected = {row["id"]: row["score"] for row in json.loads((OUT / "image-predictions.json").read_text(encoding="utf-8")) if row["stage"] == "optimized"}
        count = 0
        for row in images:
            if row["split"] != "test": continue
            path = ROOT / row["path"]
            response = client.post("/api/course/image/detect", files={"file": (path.name, path.read_bytes(), "image/png")})
            response.raise_for_status()
            assert abs(response.json()["risk_score"] - expected[row["id"]]) <= .0001, row["id"]
            assert (response.json()["risk_score"] >= .5) == (expected[row["id"]] >= .5)
            count += 1
        checks.append({"name": "Optimized image API matches all heldout predictions", "samples": count, "passed": True})
        print("Image API replay", count, flush=True)
        expected = {row["id"]: row["score"] for row in json.loads((OUT / "text-predictions.json").read_text(encoding="utf-8")) if row["stage"] == "optimized"}
        rows = [json.loads(line) for line in (ROOT / "data/course_20261027/processed/text/test.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in rows:
            response = client.post("/api/course/text/detect", json={"text": row["text"]})
            response.raise_for_status()
            assert abs(response.json()["risk_score"] - expected[row["id"]]) <= .0001, row["id"]
            assert (response.json()["risk_score"] >= .5) == (expected[row["id"]] >= .5)
        checks.append({"name": "Optimized text API matches all heldout predictions", "samples": len(rows), "passed": True})
        print("Text API replay", len(rows), flush=True)
        # A real planning/generation request is kept in flight while health and
        # Agent admission are probed. No mocked LLM in this HTTP verification.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(client.post, "/api/agent/run", data={"instruction": "展示课程各阶段的可复查材料"})
            deadline = time.monotonic()+30
            while not future.done() and time.monotonic() < deadline:
                if client.get("/api/runtime").json()["agent"]["active"]: break
                time.sleep(.05)
            started = time.perf_counter()
            assert client.get("/health").json()["status"] == "ok"
            health_ms = (time.perf_counter()-started)*1000
            busy = client.post("/api/agent/run", data={"instruction": "查看模型信息"})
            assert busy.status_code == 503
            response = future.result(timeout=180)
        response.raise_for_status(); value = response.json()
        assert value["planner"]["tools"] == ["course_overview"], value
        assert value["answer_source"] == "local_llm", value
        agents.append(value)
        checks.append({"name": "Real LLM keeps health responsive and rejects a second Agent request", "health_ms": health_ms, "passed": True})
        image_row = next(row for row in images if row["split"] == "test")
        image_path = ROOT / image_row["path"]
        passage = rows[0]["text"][:350]
        cases = [
          ("看看这篇正文的生成信号", {"text": passage, "profile": "course"}, None, ["text_detection"]),
          ("文字和照片都需要检测结果", {"text": passage, "profile": "course"}, {"image": (image_path.name, image_path.read_bytes(), "image/png")}, ["image_detection", "text_detection"]),
          ("检查这份材料并保留原文标注", {}, {"document": ("course-text.txt", passage.encode("utf-8"), "text/plain")}, ["document_detection"]),
        ]
        try:
            for instruction, data, files, tools in cases:
                started = time.perf_counter()
                response = client.post("/api/agent/run", data={"instruction": instruction, **data}, files=files)
                response.raise_for_status(); value = response.json()
                assert value["planner"]["tools"] == tools, value
                assert all(row["http_status"] == 200 for row in value["tool_calls"]), value
                assert value["answer_source"] == "local_llm", value
                reports.extend(row["result"]["report_id"] for row in value["tool_calls"] if row["result"].get("report_id"))
                agents.append(value)
                checks.append({"name": "Real Agent: " + instruction, "tools": tools, "elapsed_seconds": round(time.perf_counter()-started, 2), "passed": True})
                print("Actual Agent tools", tools, "passed", flush=True)
        finally:
            for report in reports: client.delete("/api/document/reports/"+report)
        runtime = client.get("/api/runtime").json()
        assert runtime["processing"]["active"] == 0 and not runtime["agent"]["active"]
    result = {"status": "passed", "actual_models": True, "checks": checks, "agent_responses": agents,
              "cohort_score_replay": {"images": 80, "texts": 600, "absolute_score_tolerance": .0001,
                                      "binary_predictions_match": True}, "runtime_after": runtime}
    (OUT / "http-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Course and real LLM Agent verification passed", flush=True)


if __name__ == "__main__": main()
