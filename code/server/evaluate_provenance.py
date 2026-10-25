"""Evaluate the C2PA status layer on official positive and negative JPEGs."""

from __future__ import annotations

import hashlib
import io
import json
import statistics
import time
from pathlib import Path

from PIL import Image

from provenance import inspect_provenance


ROOT = Path(__file__).resolve().parents[2]
SAMPLES = {
    "adobe-20220124-A.jpg": "credential_absent",
    "adobe-20220124-C.jpg": "valid_but_untrusted",
    "adobe-20220124-E-dat-CA.jpg": "invalid",
    "adobe-20220124-E-sig-CA.jpg": "invalid",
}


def main() -> None:
    directory = ROOT / "data/raw/c2pa_official"
    rows = []
    for name, expected in SAMPLES.items():
        data = (directory / name).read_bytes()
        times = []
        for _ in range(3):
            start = time.perf_counter()
            result = inspect_provenance(data, "JPEG")
            times.append(round((time.perf_counter() - start) * 1000, 2))
        rows.append({
            "sample": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "expected_status": expected, "observed_status": result["status"],
            "correct": expected == result["status"],
            "validation_state": result.get("validation_state"),
            "failure_codes": result.get("failure_codes", []),
            "median_read_ms": round(statistics.median(times), 2),
        })

    # Re-encoding normally strips the embedded manifest. Its absence is not a forgery verdict.
    with Image.open(directory / "adobe-20220124-C.jpg") as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
    transformed = inspect_provenance(buffer.getvalue(), "JPEG")
    rows.append({"sample": "C_reencoded_with_Pillow", "bytes": len(buffer.getvalue()),
                 "expected_status": "credential_absent", "observed_status": transformed["status"],
                 "correct": transformed["status"] == "credential_absent"})
    report = {
        "source": "https://github.com/c2pa-org/public-testfiles/blob/main/legacy/1.4/image/README.md",
        "license": "CC BY-SA 4.0; attribution to C2PA public-testfiles / Adobe test files",
        "sdk": "c2pa-python==0.38.0", "scope": "C2PA credential status, not AI image detection accuracy",
        "correct": sum(row["correct"] for row in rows), "total": len(rows),
        "rows": rows,
    }
    output = ROOT / "results/provenance_evaluation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"correct": report["correct"], "total": report["total"],
                      "statuses": [(row["sample"], row["observed_status"]) for row in rows]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
