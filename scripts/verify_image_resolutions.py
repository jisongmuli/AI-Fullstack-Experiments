"""Real HTTP geometry/mode checks through 4K; synthetic fixtures carry no accuracy labels."""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import httpx
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8005")
    parser.add_argument("--output", type=Path, default=Path("results/upgrade_20261026/resolutions.json"))
    args = parser.parse_args()
    fixtures = Path("output/playwright/resolution-fixtures")
    fixtures.mkdir(parents=True, exist_ok=True)
    checks = []
    cases = [(1, 1, "RGB"), (1, 4096, "L"), (4096, 1, "RGBA"), (17, 29, "RGB"),
             (32, 32, "L"), (199, 199, "RGB"), (289, 202, "RGB"), (512, 512, "RGB"),
             (1920, 1080, "RGB"), (3840, 2160, "RGB"), (2160, 3840, "RGB"),
             (4096, 2160, "RGB"), (4096, 4096, "RGB")]
    with httpx.Client(base_url=args.url, timeout=180, trust_env=False) as client:
        for width, height, mode in cases:
            data = io.BytesIO()
            color = 128 if mode == "L" else (90, 120, 160, 200) if mode == "RGBA" else (90, 120, 160)
            with Image.new(mode, (width, height), color) as image:
                image.save(data, "PNG")
            name = f"{width}x{height}-{mode}.png"
            (fixtures / name).write_bytes(data.getvalue())
            started = time.perf_counter()
            response = client.post("/api/image/detect", files={"file": (name, data.getvalue(), "image/png")})
            response.raise_for_status()
            value = response.json()
            assert value["input_size"] == [width, height], value
            assert isinstance(value["risk_score"], (int, float)) and 0 <= value["risk_score"] <= 1
            assert value["model_version"].startswith("dear_"), value
            assert len(value["local_evidence"]["values"]) == 9
            checks.append({"size": [width, height], "mode": mode, "passed": True,
                           "bytes": len(data.getvalue()), "seconds": round(time.perf_counter()-started, 3),
                           "model_version": value["model_version"], "risk_score": value["risk_score"]})
            print(name, checks[-1]["seconds"], flush=True)
        assert client.get("/api/runtime").json()["processing"]["active"] == 0
    result = {"status": "passed", "url": args.url, "checks": checks, "actual_models": True,
              "fixture_scope": "Synthetic geometry/mode inputs; model accuracy is evaluated separately on labeled 2026 datasets",
              "accuracy_evaluated": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
