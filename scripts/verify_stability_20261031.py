"""Recheck all heldout course predictions and actual HTTP error recovery."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stability_20261031"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8006")
    args = parser.parse_args()
    checks, counts = [], {}
    def check(value, name):
        if not value:
            raise AssertionError(name)
        checks.append({"name": name, "passed": True})
    prior = ROOT / "results/course_20261027"
    with httpx.Client(base_url=args.url, timeout=180, trust_env=False) as client:
        for filename, body, expected in [("old.doc", b"old", 415), ("broken.docx", b"broken", 422), ("empty.txt", b"", 400), ("blank.txt", b"   ", 422)]:
            response = client.post("/api/agent/run", data={"instruction": "检测文档"}, files={"document": (filename, body, "application/octet-stream")})
            check(response.status_code == expected and isinstance(response.json().get("detail"), str), "Agent ingestion: " + filename)
        for role in ([], {}):
            response = client.post("/api/agent/run", data={"instruction": "检测正文", "history": json.dumps([{"role": role, "content": "test"}])})
            check(response.status_code == 400, "Malformed conversation role: " + type(role).__name__)
        images = json.loads((ROOT / "data/course_20261027/images.json").read_text(encoding="utf-8"))["records"]
        expected_images = {row["id"]: row["score"] for row in json.loads((prior / "image-predictions.json").read_text(encoding="utf-8")) if row["stage"] == "optimized"}
        counts["images"] = 0
        for row in images:
            if row["split"] != "test":
                continue
            path = ROOT / row["path"]
            response = client.post("/api/course/image/detect", files={"file": (path.name, path.read_bytes(), "image/png")})
            response.raise_for_status()
            score = response.json()["risk_score"]
            expected = expected_images[row["id"]]
            assert abs(score - expected) <= .0001 and (score >= .5) == (expected >= .5), row["id"]
            counts["images"] += 1
        check(counts["images"] == 80, "All 80 optimized image predictions match saved measurements")
        print("Image API replay", counts["images"], flush=True)
        texts = [json.loads(line) for line in (ROOT / "data/course_20261027/processed/text/test.jsonl").read_text(encoding="utf-8").splitlines()]
        expected_texts = {row["id"]: row["score"] for row in json.loads((prior / "text-predictions.json").read_text(encoding="utf-8")) if row["stage"] == "optimized"}
        counts["texts"] = 0
        for row in texts:
            response = client.post("/api/course/text/detect", json={"text": row["text"]})
            response.raise_for_status()
            score, expected = response.json()["risk_score"], expected_texts[row["id"]]
            assert abs(score - expected) <= .0001 and (score >= .5) == (expected >= .5), row["id"]
            counts["texts"] += 1
        check(counts["texts"] == 600, "All 600 optimized text predictions match saved measurements")
        print("Text API replay", counts["texts"], flush=True)
        passage = "研究团队核对样本来源，保留训练、验证和测试的划分记录。 实验记录 2026 ABC 123 展示对比结果。" * 3
        variant = "\n\t" + "\u3000\n".join(passage.split(" "))
        variant = "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in variant)
        for stage in ("trained", "optimized"):
            a = client.post("/api/course/text/detect", json={"text": passage, "stage": stage})
            b = client.post("/api/course/text/detect", json={"text": variant, "stage": stage})
            a.raise_for_status(); b.raise_for_status()
            check(a.json()["risk_score"] == b.json()["risk_score"], "Equivalent Unicode and whitespace: " + stage)
        short = ("研究团队检查样本来源" * 5)
        response = client.post("/api/course/text/detect", json={"text": short[:20] + "\n" * 200 + short[20:]})
        response.raise_for_status()
        check(response.json()["risk_score"] is None, "Whitespace does not turn short text into a scorable sample")
        check(client.get("/health").json()["status"] == "ok", "Health remains available after invalid inputs and full replay")
        runtime = client.get("/api/runtime").json()
        check(runtime["processing"]["active"] == 0 and not runtime["agent"]["active"], "All processing and Agent slots are released")
    OUT.mkdir(parents=True, exist_ok=True)
    value = {"status": "passed", "recorded_at": datetime.now(timezone.utc).isoformat(), "url": args.url,
             "checks": checks, "actual_models": True, "replayed": counts, "score_tolerance": .0001, "runtime_after": runtime}
    (OUT / "http-verification.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "checks": len(checks), "replayed": counts}))


if __name__ == "__main__":
    main()
