"""Frozen image-policy audit on an independent Community Forensics sample.

Source: https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval
License: CC BY-NC-SA 4.0, research/education only. The source's Firefly_Image2
rows are selected by fixed viewer row ranges before any model prediction:
real 4000:4100 and 4500:4600, fake 8500:8550. The first 50 real images with
short side >=512 are selected. Both labels are decoded, center-cropped to a
512-pixel square, and JPEG encoded at quality 90 to reduce size/format leakage.

This is a small external-source research audit, not a platform acceptance test.
Do not tune thresholds or the selection rule after reading its labels/scores.
"""
from __future__ import annotations

import base64
import argparse
import csv
import hashlib
import io
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score

from highres_inference import HighResolutionCrossCheck, screening_consensus


ROOT = Path(__file__).resolve().parents[2]
DATASET = "OwensLab/CommunityForensics-Eval"
API = "https://datasets-server.huggingface.co/rows"
RAW_RANGES = ((4000, 100, 0), (4500, 100, 0), (8500, 50, 1))
PER_CLASS = 50
JPEG_QUALITY = 90


def fetch_rows(offset: int, length: int) -> list[dict]:
    params = {"dataset": DATASET, "config": "default", "split": "CompEval",
              "offset": offset, "length": length}
    for attempt in range(5):
        try:
            response = requests.get(API, params=params, timeout=(15, 120))
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                payload = response.json()
                if payload.get("partial") or len(payload["rows"]) != length:
                    raise ValueError(f"Incomplete source rows at offset {offset}")
                return payload["rows"]
        except requests.RequestException:
            if attempt == 4:
                raise
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Dataset viewer unavailable at offset {offset}")


def source_rows() -> list[dict]:
    real: list[dict] = []
    fake: list[dict] = []
    for offset, count, label in RAW_RANGES:
        # The viewer serves large image cells unreliably in batches above 10.
        batch_size = 100 if label == 0 else 10
        for start in range(offset, offset + count, batch_size):
            for item in fetch_rows(start, min(batch_size, offset + count - start)):
                row = item["row"]
                if item["truncated_cells"] or row["label"] != label or row["model_name"] != "Firefly_Image2":
                    raise ValueError(f"Unexpected or truncated Community Forensics row {item['row_idx']}")
                if label == 0:
                    if min(row["resolution"]) >= 512:
                        real.append(item)
                else:
                    if min(row["resolution"]) < 512:
                        raise ValueError(f"Generated row under 512 pixels: {item['row_idx']}")
                    fake.append(item)
    if len(real) < PER_CLASS or len(fake) != PER_CLASS:
        raise ValueError(f"Insufficient fixed-range samples: {len(real)} real, {len(fake)} fake")
    return real[:PER_CLASS] + fake


def normalize_and_save(rows: list[dict], directory: Path) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    seen_hashes: set[str] = set()
    for item in rows:
        row = item["row"]
        original = base64.b64decode(row["image_data"], validate=True)
        raw_sha = hashlib.sha256(original).hexdigest()
        if raw_sha in seen_hashes:
            raise ValueError(f"Duplicate source bytes at row {item['row_idx']}")
        seen_hashes.add(raw_sha)
        with Image.open(io.BytesIO(original)) as image:
            image.load()
            image = ImageOps.exif_transpose(image).convert("RGB")
            if min(image.size) < 512 or list(image.size) != row["resolution"]:
                raise ValueError(f"Source resolution mismatch at row {item['row_idx']}")
            prepared = ImageOps.fit(image, (512, 512), method=Image.Resampling.LANCZOS)
            filename = f"{item['row_idx']}_{raw_sha[:16]}.jpg"
            path = directory / filename
            prepared.save(path, "JPEG", quality=JPEG_QUALITY, subsampling=0, exif=b"")
        records.append({"row_idx": item["row_idx"], "actual": row["label"],
                        "model_name": row["model_name"], "real_source": row["real_source"],
                        "original_size": row["resolution"], "source_sha256": raw_sha,
                        "normalized_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "local_file": str(path.relative_to(ROOT))})
    return records


def load_local_records() -> list[dict]:
    """Verify retained JPEG bytes and metadata without accessing the network."""
    source = ROOT / "results/highres/communityforensics_firefly_holdout_predictions.csv"
    original_summary = json.loads((ROOT / "results/highres/communityforensics_firefly_holdout.json").read_text(encoding="utf-8"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != original_summary["prediction_csv_sha256"]:
        raise ValueError("The original prediction CSV no longer matches its recorded SHA-256")
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2 * PER_CLASS or sorted(int(r["actual"]) for r in rows) != [0] * PER_CLASS + [1] * PER_CLASS:
        raise ValueError("The retained audit must contain exactly 50 samples per class")
    records = []
    for row in rows:
        path = ROOT / row["local_file"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["normalized_sha256"]:
            raise ValueError(f"Missing or modified normalized image: {path}")
        if len(row["source_sha256"]) != 64:
            raise ValueError("Missing original source SHA-256")
        with Image.open(path) as image:
            if image.size != (512, 512) or image.format != "JPEG":
                raise ValueError(f"Unexpected normalized image format: {path}")
        records.append({key: row[key] for key in (
            "row_idx", "actual", "model_name", "real_source", "source_sha256",
            "normalized_sha256", "local_file")})
        records[-1]["row_idx"] = int(records[-1]["row_idx"])
        records[-1]["actual"] = int(records[-1]["actual"])
        records[-1]["original_size"] = json.loads(row["original_size"])
    if len({row["row_idx"] for row in records}) != len(records):
        raise ValueError("Duplicate viewer row index")
    return records


def wilson(successes: int, total: int) -> dict:
    p = successes / total
    z = 1.959963984540054
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"count": successes, "denominator": total, "value": p,
            "wilson_95": [center - radius, center + radius]}


def audit(records: list[dict], detector: HighResolutionCrossCheck, offline: bool = False) -> dict:
    results: list[dict] = []
    for index, record in enumerate(records, 1):
        with Image.open(ROOT / record["local_file"]) as image:
            prediction = detector.predict(image)
        scores = [item["score"] for item in prediction["model_scores"]]
        results.append({**record, "risk_score": prediction["risk_score"],
                        "siglip_score": scores[0], "vit_score": scores[1],
                        "consensus": screening_consensus(scores)})
        if index % 20 == 0:
            print(f"Evaluated {index}/{len(records)}", flush=True)
    stem = "communityforensics_firefly_holdout_recomputed" if offline else "communityforensics_firefly_holdout"
    output = ROOT / f"results/highres/{stem}_predictions.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        for result in results:
            writer.writerow({**result, "original_size": json.dumps(result["original_size"])})

    real = [r for r in results if r["actual"] == 0]
    fake = [r for r in results if r["actual"] == 1]
    mean_fps = sum(r["risk_score"] >= 0.8 for r in real)
    mean_tps = sum(r["risk_score"] >= 0.8 for r in fake)
    joint_fps = sum(r["consensus"] == "joint_high" for r in real)
    joint_tps = sum(r["consensus"] == "joint_high" for r in fake)
    summary = {
        "status": "offline_recomputation_of_saved_holdout" if offline else "independent_source_research_audit_not_platform_acceptance",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": f"https://huggingface.co/datasets/{DATASET}",
        "source_license": "CC BY-NC-SA 4.0; research/education only",
        "source_access": "Hugging Face dataset viewer rows API on main; upstream snapshot was not pinned",
        "selection_relation": "No local training, threshold setting, or model choice used these 100 rows before this audit; pretrained-model overlap cannot be ruled out",
        "selection": "fixed row ranges 4000:4100, 4500:4600 real and 8500:8550 fake; first 50 qualifying >=512 real",
        "preprocessing": "EXIF transpose, center square crop, 512x512, JPEG quality 90 with no metadata for both labels",
        "n": len(results), "per_class": PER_CLASS,
        "mean_score_auc": roc_auc_score([r["actual"] for r in results], [r["risk_score"] for r in results]),
        "mean_at_0_8": {"false_positive_rate": wilson(mean_fps, PER_CLASS),
                        "true_positive_rate": wilson(mean_tps, PER_CLASS)},
        "joint_at_0_8": {"false_positive_rate": wilson(joint_fps, PER_CLASS),
                         "true_positive_rate": wilson(joint_tps, PER_CLASS)},
        "prediction_csv": str(output.relative_to(ROOT)),
        "prediction_csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "limitations": "One Firefly generator paired with LAION real images; small n, source/content and cropping confounds remain. Metrics apply to the normalized JPEG images, not original uploads or platform traffic.",
    }
    if offline:
        original = ROOT / "results/highres/communityforensics_firefly_holdout_predictions.csv"
        with original.open(newline="", encoding="utf-8") as stream:
            expected = {int(row["row_idx"]): row for row in csv.DictReader(stream)}
        deltas = [abs(result["risk_score"] - float(expected[result["row_idx"]]["risk_score"])) for result in results]
        summary["max_abs_mean_score_delta_from_original"] = max(deltas)
    destination = ROOT / f"results/highres/{stem}.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-recompute", action="store_true",
                        help="Recompute from retained JPEGs after SHA-256 verification; never access or overwrite the original audit")
    args = parser.parse_args()
    if args.offline_recompute:
        records = load_local_records()
    else:
        rows = source_rows()
        records = normalize_and_save(rows, ROOT / "data/highres_holdout/communityforensics_firefly_512")
    detector = HighResolutionCrossCheck(ROOT)
    print(json.dumps(audit(records, detector, offline=args.offline_recompute), ensure_ascii=False, indent=2), flush=True)
