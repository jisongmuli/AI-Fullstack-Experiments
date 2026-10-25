"""Measure input-size handling and GPU memory; this does not measure accuracy.

Run with a CUDA-enabled PyTorch environment from the project root:
    python code/models/benchmark_image_sizes.py --model ateeqq
"""
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import torch
from PIL import Image

from highres_inference import HighResolutionDetector


SIZES = [
    (100, 100), (200, 200), (400, 400), (800, 600), (1200, 800),
    (1080, 1080), (1080, 1350), (1280, 1280), (1200, 675),
    (1242, 1660), (1080, 1440), (1920, 1080), (2560, 1440),
    (3024, 1964), (3840, 2160),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model", choices=list(HighResolutionDetector.MODELS), default="ateeqq")
    args = parser.parse_args()
    root = args.root.resolve()
    archive_path = root / "data/raw/rhythmghai_real_ai_512.zip"
    with zipfile.ZipFile(archive_path) as archive:
        member = next(name for name in archive.namelist() if name.lower().startswith("real_dataset/")
                      and name.lower().endswith((".jpg", ".jpeg", ".png")))
        with Image.open(io.BytesIO(archive.read(member))) as original:
            source = original.convert("RGB")

    detector = HighResolutionDetector(root, args.model)
    detector.predict(source)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    rows = []
    for width, height in SIZES:
        scaled = source.resize((width, height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        scaled.save(buffer, format="JPEG", quality=88)
        payload = buffer.getvalue()
        start = time.perf_counter()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            result = detector.predict(image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        rows.append({"width": width, "height": height, "jpeg_bytes": len(payload),
                     "decode_and_inference_ms": round(elapsed_ms, 2),
                     "model_probability_unvalidated": result["risk_score"]})
        print(f"{width}x{height}: {elapsed_ms:.1f} ms", flush=True)

    report = {
        "purpose": "input compatibility and memory only; enlarged 512px source is not a 4K accuracy test",
        "source_archive": str(archive_path), "source_member": member,
        "model": detector.model_id, "model_internal_size": [224, 224],
        "device": str(detector.device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1)
        if torch.cuda.is_available() else None,
        "torch_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if torch.cuda.is_available() else None,
        "torch_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1)
        if torch.cuda.is_available() else None,
        "results": rows,
    }
    destination = root / "results/highres/image_size_benchmark.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
