"""Post-hoc paired resize audit of the retained 100-image CommunityForensics sample.

This script measures score stability after resizing and JPEG re-encoding. The
same sample has already been inspected, so these metrics cannot validate a
new operating threshold or an independent real-platform accuracy claim.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from evaluate_communityforensics import ROOT, load_local_records
from highres_inference import HighResolutionCrossCheck, screening_consensus


BASELINE = ROOT / "results/highres/communityforensics_firefly_holdout_recomputed_predictions.csv"
OUTPUT = ROOT / "results/highres/lowres_paired_stability.json"
PREDICTIONS = ROOT / "results/highres/lowres_paired_stability_predictions.csv"
VARIANTS = {"square_202": (202, 202), "landscape_289x202": (289, 202),
            "square_160": (160, 160), "square_128": (128, 128)}


def prepare(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Use a direct resize to isolate size/codec effects, recording distortion."""
    encoded = io.BytesIO()
    image.convert("RGB").resize(size, Image.Resampling.LANCZOS).save(
        encoded, format="JPEG", quality=85, subsampling=0)
    encoded.seek(0)
    with Image.open(encoded) as result:
        return result.convert("RGB")


def summary(rows: list[dict], baseline: dict[int, dict]) -> dict:
    original = np.array([float(baseline[r["row_idx"]]["risk_score"]) for r in rows])
    current = np.array([r["risk_score"] for r in rows])
    labels = np.array([r["actual"] for r in rows])
    deltas = np.abs(current - original)
    original_consensus = [baseline[r["row_idx"]]["consensus"] for r in rows]
    current_consensus = [r["consensus"] for r in rows]
    real = labels == 0
    fake = labels == 1
    return {
        "n": len(rows),
        "mean_abs_score_delta": float(np.mean(deltas)),
        "median_abs_score_delta": float(np.median(deltas)),
        "p90_abs_score_delta": float(np.quantile(deltas, .9)),
        "max_abs_score_delta": float(np.max(deltas)),
        "fraction_abs_delta_over_0_2": float(np.mean(deltas > .2)),
        "consensus_changed_count": sum(a != b for a, b in zip(original_consensus, current_consensus)),
        "joint_high_to_other_count": sum(a == "joint_high" and b != "joint_high" for a, b in zip(original_consensus, current_consensus)),
        "other_to_joint_high_count": sum(a != "joint_high" and b == "joint_high" for a, b in zip(original_consensus, current_consensus)),
        "exploratory_auc": float(roc_auc_score(labels, current)),
        "exploratory_joint_high_fpr": float(np.mean([r["consensus"] == "joint_high" for r, is_real in zip(rows, real) if is_real])),
        "exploratory_joint_high_tpr": float(np.mean([r["consensus"] == "joint_high" for r, is_fake in zip(rows, fake) if is_fake])),
    }


def main() -> None:
    records = load_local_records()  # Verifies retained bytes against frozen hashes.
    with BASELINE.open(newline="", encoding="utf-8") as stream:
        baseline = {int(r["row_idx"]): r for r in csv.DictReader(stream)}
    if set(baseline) != {r["row_idx"] for r in records}:
        raise ValueError("Baseline rows differ from retained source rows")
    detector = HighResolutionCrossCheck(ROOT)
    output_rows = []
    for index, record in enumerate(records, 1):
        with Image.open(ROOT / record["local_file"]) as image:
            for name, size in VARIANTS.items():
                processed = prepare(image, size)
                prediction = detector.predict(processed)
                model_scores = [s["score"] for s in prediction["model_scores"]]
                output_rows.append({"row_idx": record["row_idx"], "actual": record["actual"],
                                    "variant": name, "risk_score": prediction["risk_score"],
                                    "siglip_score": model_scores[0], "vit_score": model_scores[1],
                                    "consensus": screening_consensus(model_scores)})
        if index % 20 == 0:
            print(f"Processed {index}/{len(records)} source images", flush=True)
    with PREDICTIONS.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    report = {
        "status": "post_hoc_same_sample_exploration_not_independent_accuracy",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval",
        "source_sample": "100 retained center-cropped 512x512 JPEG q90 images; 50 LAION real / 50 Firefly Image2",
        "transformation": "Direct Lanczos resize from 512x512, JPEG quality 85, 4:4:4. Landscape 289x202 distorts square geometry; no crop. This is a stress test, not a native low-resolution benchmark.",
        "baseline_file": str(BASELINE.relative_to(ROOT)),
        "baseline_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        "variants": {name: summary([r for r in output_rows if r["variant"] == name], baseline)
                     for name in VARIANTS},
        "prediction_csv": str(PREDICTIONS.relative_to(ROOT)),
        "prediction_csv_sha256": hashlib.sha256(PREDICTIONS.read_bytes()).hexdigest(),
        "limitations": "Sample and model behavior were already observed before this analysis. JPEG recoding, square-source aspect distortion, generator/source confounding and platform shift remain. These scores are not AI-authorship probabilities or region truth.",
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
