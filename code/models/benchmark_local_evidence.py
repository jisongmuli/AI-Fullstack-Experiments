"""Bounded-run latency and CUDA-memory check for optional 3x3 image overlay.

All sizes derive from a single retained sample. Timing is device-specific; no
accuracy conclusion is possible from this script.
"""
from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from highres_inference import HighResolutionCrossCheck


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/highres_holdout/communityforensics_firefly_512/4000_cf30112d338d1d10.jpg"
OUTPUT = ROOT / "results/highres/local_evidence_resource_benchmark.json"


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    detector = HighResolutionCrossCheck(ROOT)
    with Image.open(SOURCE) as loaded:
        source = loaded.convert("RGB")
    cases = {"289x202": source.resize((289, 202)),
             "512x512": source,
             "3840x2160": source.resize((3840, 2160))}
    results = {}
    for name, image in cases.items():
        for overlay in (False, True):
            # Warm one pass separately so timings exclude kernel and cache setup.
            detector.predict(image, include_local_evidence=overlay)
            sync()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            durations = []
            for _ in range(3):
                sync()
                start = time.perf_counter()
                result = detector.predict(image, include_local_evidence=overlay)
                sync()
                durations.append(round((time.perf_counter() - start) * 1000, 2))
            results[f"{name}_overlay_{overlay}"] = {
                "times_ms": durations,
                "median_ms": round(statistics.median(durations), 2),
                "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                if torch.cuda.is_available() else None,
                "risk_score": result["risk_score"],
                "local_evidence_present": "local_evidence" in result,
            }
        print(f"Benchmarked {name}", flush=True)
    output = {
        "purpose": "latency and memory only, not classifier accuracy or region localization validity",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "gpu_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1)
        if torch.cuda.is_available() else None,
        "source": str(SOURCE.relative_to(ROOT)),
        "method": "3x3 Gaussian blur masking, max three model inputs per forward, downscale analysis canvas to max 512 pixels",
        "results": results,
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
