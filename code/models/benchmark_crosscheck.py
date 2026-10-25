"""Benchmark the two-model image path on one resized source (not accuracy)."""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import torch
from PIL import Image

from highres_inference import HighResolutionCrossCheck


ROOT = Path(__file__).resolve().parents[2]
SIZES = ((512, 512), (1080, 1440), (3840, 2160))


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    archive_path = ROOT / "data/raw/rhythmghai_real_ai_512.zip"
    with zipfile.ZipFile(archive_path) as archive:
        member = next(name for name in archive.namelist()
                      if name.startswith("Ai_generated_dataset/") and name.lower().endswith(".png"))
        with Image.open(io.BytesIO(archive.read(member))) as image:
            source = image.convert("RGB")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    free_before = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None
    load_start = time.perf_counter()
    detector = HighResolutionCrossCheck(ROOT)
    sync()
    load_ms = round((time.perf_counter() - load_start) * 1000, 2)
    free_after_load = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None
    rows = []
    for width, height in SIZES:
        resized = source.resize((width, height), Image.Resampling.LANCZOS)
        payload = io.BytesIO()
        resized.save(payload, format="JPEG", quality=88)
        for run in range(3):
            start = time.perf_counter()
            with Image.open(io.BytesIO(payload.getvalue())) as image:
                image.load()
                detector.predict(image)
            sync()
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            if run:
                rows.append({"width": width, "height": height, "run": run,
                             "decode_and_inference_ms": elapsed_ms})

    report = {
        "purpose": "dual-model resource and speed only; resizing one source does not test accuracy",
        "source_archive": str(archive_path), "source_member": member,
        "device": device, "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1)
        if torch.cuda.is_available() else None,
        "model_load_ms": load_ms,
        "gpu_free_before_mib": round(free_before / 2**20, 1) if free_before is not None else None,
        "gpu_free_after_load_mib": round(free_after_load / 2**20, 1) if free_after_load is not None else None,
        "torch_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if torch.cuda.is_available() else None,
        "torch_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1)
        if torch.cuda.is_available() else None,
        "results": rows,
    }
    destination = ROOT / "results/highres/crosscheck_size_benchmark.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)
    print(json.dumps({"device": device, "model_load_ms": load_ms,
                      "torch_peak_reserved_mib": report["torch_peak_reserved_mib"],
                      "results": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
