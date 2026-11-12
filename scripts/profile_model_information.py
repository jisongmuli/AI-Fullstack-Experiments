"""Measure the first model-information request in a fresh Python process."""
import argparse
import json
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/server"))
from app import ModelService, create_app
from fastapi.testclient import TestClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    service = ModelService(ROOT)
    process = psutil.Process()
    before = process.memory_info().rss
    started = time.perf_counter()
    application = create_app(service)
    client = TestClient(application, base_url="http://localhost")
    try:
        response = client.get("/api/models")
        response.raise_for_status()
        value = response.json()
        result = {"operation": "first_model_information", "seconds": round(time.perf_counter() - started, 4),
                  "rss_increase_mb": round((process.memory_info().rss - before) / 1048576, 2),
                  "bundle_loaded": service._bundle is not None, "http_status": response.status_code,
                  "fingerprint_count": len(value.get("artifact_fingerprints", {}))}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result))
    finally:
        client.close()
        application.state.document_jobs.close()
        application.state.report_store.close()


if __name__ == "__main__":
    main()
