"""Exercise document jobs against actual local models using generated content."""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import httpx
import pymupdf
from docx import Document


TEXT = "The team checked the source files before preparing the data. Each record received a stable identifier, and missing values were counted separately."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8002")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    word = Document(); word.add_paragraph(TEXT)
    stream = io.BytesIO(); word.save(stream)
    pdf = pymupdf.open(); page = pdf.new_page(); page.insert_textbox((40, 40, 550, 700), TEXT, fontsize=12)
    pdf_data = pdf.tobytes(); pdf.close()
    fixtures = [("fixture.txt", TEXT.encode(), "text/plain"),
                ("fixture.docx", stream.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                ("fixture.pdf", pdf_data, "application/pdf"),
                ("long-pasted-text", (TEXT + "\n\n") * 90, None)]
    record = {"status": "passed", "actual_models": True, "accuracy_evaluated": False, "checks": []}
    with httpx.Client(base_url=args.url, timeout=180, trust_env=False) as client:
        for name, data, mime in fixtures:
            started = time.monotonic()
            response = client.post("/api/document/jobs", files={"file": (name, data, mime)}) if mime else client.post("/api/document/jobs", data={"text": data})
            assert response.status_code == 202, response.text
            job = response.json(); states = [job["stage"]]; progress = [job["progress"]]
            deadline = time.monotonic() + 180
            while job["status"] not in {"completed", "failed", "cancelled"}:
                assert time.monotonic() < deadline, "Document job timed out"
                time.sleep(.05)
                status = client.get("/api/document/jobs/" + job["job_id"]); status.raise_for_status()
                job = status.json(); states.append(job["stage"]); progress.append(job["progress"])
            assert job["status"] == "completed", job
            report = client.get("/api/document/reports/" + job["report_id"]); report.raise_for_status()
            result = report.json()
            assert result["detection"]["segments"] and result["extracted_text"]
            if name == "long-pasted-text":
                assert result["extracted_text"] == data
            for format_ in ("docx", "html", "json"):
                exported = client.get(f"/api/document/reports/{job['report_id']}/{format_}")
                exported.raise_for_status()
            record["checks"].append({"input": name, "accepted_status": response.status_code,
                "final_status": job["status"], "observed_stages": list(dict.fromkeys(states)),
                "final_progress": progress[-1], "elapsed_seconds": round(time.monotonic() - started, 3),
                "original_characters": len(result["extracted_text"]), "exports": ["docx", "html", "json"]})
            client.delete("/api/document/reports/" + job["report_id"]).raise_for_status()
            client.delete("/api/document/jobs/" + job["job_id"]).raise_for_status()
        runtime = client.get("/api/runtime").json()
        assert runtime["processing"]["active"] == 0
        record["processing_after"] = runtime["processing"]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
