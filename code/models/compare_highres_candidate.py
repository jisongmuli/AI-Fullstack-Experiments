"""Compare an independently sourced detector on the same public test images.

Run after evaluate_highres.py. The source CSVs freeze the sample identities,
so model comparisons cannot quietly change the test set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from transformers import AutoImageProcessor, AutoModelForImageClassification

from evaluate_acceptance import audit_file
from evaluate_highres import ZIP_FILES


MODEL_ID = "haywoodsloan/ai-image-detector-deploy"
MODEL_REVISION = "4ae1822603cb1ec980c217cfde1ef1b8e0477a16"


def score_model(model, processor, image: Image.Image, artificial_index: int) -> float:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    inputs = processor(images=rgb, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits[0].float()
        return float(torch.softmax(logits, dim=-1)[artificial_index])


def evaluate(root: Path, name: str, model, processor, artificial_index: int, per_class: int) -> dict:
    baseline_file = root / f"results/highres/{name}_predictions.csv"
    source_zip = root / ZIP_FILES[name]
    output = root / f"results/highres/candidate_swinv2_{name}_predictions.csv"
    with baseline_file.open(encoding="utf-8", newline="") as stream:
        baseline = list(csv.DictReader(stream))
    selected = []
    for label in ("0", "1"):
        class_rows = [row for row in baseline if row["actual"] == label]
        selected.extend(class_rows if per_class == 0 else class_rows[:per_class])
    if len({row["sample_id"] for row in selected}) != len(selected):
        raise ValueError(f"Repeated sample identity in {baseline_file}")
    actual = []
    scores = []
    latencies = []
    with zipfile.ZipFile(source_zip) as archive, output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "sample_id", "actual", "risk_score", "decision_at_0.8", "member", "width", "height", "latency_ms"
        ))
        writer.writeheader()
        for index, row in enumerate(selected, 1):
            raw = archive.read(row["member"])
            if not row["sample_id"].endswith(hashlib.sha256(raw).hexdigest()[:16]):
                raise ValueError(f"Sample identity changed: {row['member']}")
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                started = time.perf_counter()
                score = score_model(model, processor, image, artificial_index)
                latency = (time.perf_counter() - started) * 1000
            actual.append(int(row["actual"]))
            scores.append(score)
            latencies.append(latency)
            writer.writerow({
                "sample_id": row["sample_id"], "actual": row["actual"],
                "risk_score": round(score, 6),
                "decision_at_0.8": "ai" if score >= 0.8 else "human" if score <= 0.2 else "uncertain",
                "member": row["member"], "width": row["width"], "height": row["height"],
                "latency_ms": round(latency, 2),
            })
            if index % 100 == 0 or index == len(selected):
                print(f"{name}: {index}/{len(selected)}", flush=True)
    latencies.sort()
    return {
        "source_zip": str(source_zip), "baseline_file": str(baseline_file),
        "predictions_csv": str(output), "n": len(selected), "per_class": per_class,
        "accuracy_at_0_5": accuracy_score(actual, [int(score >= 0.5) for score in scores]),
        "macro_f1_at_0_5": f1_score(actual, [int(score >= 0.5) for score in scores], average="macro"),
        "confusion_at_0_5": confusion_matrix(actual, [int(score >= 0.5) for score in scores], labels=[0, 1]).tolist(),
        "roc_auc": roc_auc_score(actual, scores),
        "latency_ms_p50": latencies[len(latencies) // 2],
        "latency_ms_p95": latencies[int(0.95 * (len(latencies) - 1))],
        "abstaining_audit": audit_file(output, 0.2, 0.8),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataset", action="append", choices=list(ZIP_FILES))
    parser.add_argument("--per-class", type=int, default=100, help="0 means all baseline rows")
    args = parser.parse_args()
    root = args.root.resolve()
    directory = root / "models/highres_candidate_swinv2"
    torch.set_num_threads(min(8, torch.get_num_threads()))
    processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(
        directory, local_files_only=True, use_safetensors=True
    ).eval()
    labels = {int(index): label.lower() for index, label in model.config.id2label.items()}
    artificial_index = next(index for index, label in labels.items() if label == "artificial")
    datasets = args.dataset or list(ZIP_FILES)
    results = {name: evaluate(root, name, model, processor, artificial_index, args.per_class) for name in datasets}
    destination = root / "results/highres/candidate_swinv2_evaluation.json"
    prior = json.loads(destination.read_text(encoding="utf-8")) if destination.is_file() else {}
    prior_results = prior.get("datasets", {})
    prior_results.update(results)
    destination.write_text(json.dumps({
        "model_id": MODEL_ID, "revision": MODEL_REVISION, "license_on_host": "Apache-2.0",
        "training_data_provenance": "not documented in model card", "datasets": prior_results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
