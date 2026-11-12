"""Compare released DEAR with the previous dual model on a pinned NTIRE 2026 sample."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/models"))
sys.path.insert(0, str(ROOT / "scripts"))
from research_detectors import DearDetector
from highres_inference import HighResolutionCrossCheck
from evaluate_2026_text import metrics

DATA = ROOT / "data/benchmarks_2026/ntire"
OUT = ROOT / "results/upgrade_20261026/image"


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    records = sorted(manifest["records"], key=lambda row: (row["split"] != "validation", row["id"]))
    hashes = [row["sha256"] for row in records]
    assert len(set(hashes)) == len(hashes), "Duplicate image bytes"
    plan = {"source_revision": manifest["revision"], "source_shard": 5, "records": records,
            "native_method": "DEAR-r; stride-1 ResNet50 and released fixed channel gate; up to 9 evenly spaced native 256px crops; area-weighted logit",
            "threshold_selection": "Sigmoid calibration on validation only; freeze before test metrics",
            "test_used_for_model_selection_or_calibration": False,
            "robustness_subset": "first 20 sorted test IDs per class; JPEG Q65 and 320x240 resize",
            "official_challenge_score": False}
    (OUT / "evaluation-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    cache_path = OUT / "predictions.jsonl"
    cached = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cached[row["id"]] = row
    dear = DearDetector(ROOT, "dear_r")
    legacy = HighResolutionCrossCheck(ROOT)
    for index, row in enumerate(records):
        if row["id"] in cached:
            continue
        data = (DATA / row["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        with Image.open(io.BytesIO(data)) as image:
            tick = time.perf_counter()
            new_logit = dear.logit(image)
            old = legacy.predict(image)["risk_score"]
            value = {**row, "dear_logit": new_logit, "legacy_score": old,
                     "seconds": round(time.perf_counter() - tick, 3)}
        with cache_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        cached[row["id"]] = value
        if index % 8 == 0:
            print(f"Images: {index + 1}/{len(records)}; last {value['seconds']}s", flush=True)
    rows = [cached[row["id"]] for row in records]
    labels = np.array([row["label"] for row in rows])
    logits = np.array([row["dear_logit"] for row in rows])
    old = np.array([row["legacy_score"] for row in rows])
    valid = np.array([row["split"] == "validation" for row in rows])
    calibrator = LogisticRegression(C=1., max_iter=1000).fit(logits[valid].reshape(-1, 1), labels[valid])
    coefficient, intercept = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])
    if coefficient <= 0:
        raise RuntimeError("Validation does not support the declared AI-score orientation")
    probability = lambda values: 1. / (1. + np.exp(-np.clip(values * coefficient + intercept, -60, 60)))
    scores = probability(logits)
    policy = {"method": "DEAR-r-ICML2026", "variant": "dear_r", "tile_size": 256,
              "maximum_native_crops": 9, "calibration": {"coefficient": coefficient, "intercept": intercept},
              "validation_samples": int(valid.sum()), "source_revision": manifest["revision"],
              "calibration_scope": "External labeled public NTIRE 2026 training-shard audit slice"}
    policy_path = ROOT / "models/dear_r/policy.json"
    if policy_path.exists():
        low_resolution = json.loads(policy_path.read_text(encoding="utf-8")).get("low_resolution")
        if low_resolution:
            policy["low_resolution"] = low_resolution
    policy_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    test = ~valid
    results = {"native": {"legacy": metrics(labels[test], old[test]), "dear2026": metrics(labels[test], scores[test])}}
    robustness_path = OUT / "robustness-predictions.jsonl"
    robustness = []
    if robustness_path.exists():
        robustness = [json.loads(line) for line in robustness_path.read_text(encoding="utf-8").splitlines()]
    done = {(row["id"], row["transform"]) for row in robustness}
    selected = [row for label in (0, 1) for row in sorted([r for r in rows if r["split"] == "test" and r["label"] == label], key=lambda r: r["id"])[:20]]
    for index, row in enumerate(selected):
        with Image.open(DATA / row["path"]) as original:
            image = original.convert("RGB")
            encoded = io.BytesIO(); image.save(encoded, "JPEG", quality=65)
            for name, transformed in [("jpeg_q65", Image.open(io.BytesIO(encoded.getvalue()))),
                                      ("resize_320x240", image.resize((320, 240), Image.Resampling.LANCZOS))]:
                if (row["id"], name) in done:
                    continue
                value = {"id": row["id"], "label": row["label"], "transform": name,
                         "dear_score": float(probability(dear.logit(transformed)))}
                robustness.append(value)
                with robustness_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(value) + "\n")
        if index % 8 == 0:
            print(f"Robustness: {index+1}/{len(selected)} sources", flush=True)
    for transform in ("jpeg_q65", "resize_320x240"):
        group = [row for row in robustness if row["transform"] == transform]
        results[transform] = metrics([row["label"] for row in group], [row["dear_score"] for row in group])
    result = {"status": "passed", "policy": policy, "heldout_results": results,
              "official_challenge_score": False,
              "comparison_scope": "200 reserved external images; robustness subsets derived from 40 of those sources"}
    (OUT / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
