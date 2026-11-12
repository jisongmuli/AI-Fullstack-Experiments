"""Fit resolution-specific calibration on validation images; audit fresh resized holdouts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/models"))
sys.path.insert(0, str(ROOT / "scripts"))
from research_detectors import DearDetector
from evaluate_2026_text import metrics


def main():
    torch.set_num_threads(4)
    data = ROOT / "data/benchmarks_2026/ntire"
    output = ROOT / "results/upgrade_20261026/image"
    records = json.loads((data / "manifest.json").read_text(encoding="utf-8"))["records"]
    valid = sorted([r for r in records if r["split"] == "validation"], key=lambda r: r["id"])
    fresh = [row for label in (0, 1) for row in sorted(
        [r for r in records if r["split"] == "test" and r["label"] == label], key=lambda r: r["id"])[20:80]]
    plan = {"calibration": [r["id"] for r in valid], "fresh_test": [r["id"] for r in fresh],
            "size": [320, 240], "selection": "validation-only choice between DEAR-c and DEAR-r; unchanged released weights",
            "development_note": "Earlier first-20-per-class resized probes revealed a calibration shift. Those diagnostic sources are excluded from this fresh resized holdout.",
            "test_used_for_calibration": False}
    (output / "lowres-plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    cache_path = output / "lowres-logits.jsonl"
    cache = {}
    if cache_path.exists():
        cache = {(r["id"], r["variant"]): r for r in map(json.loads, cache_path.read_text(encoding="utf-8").splitlines())}

    def infer(detector, rows, variant):
        values = []
        for index, row in enumerate(rows):
            key = row["id"], variant
            if key not in cache:
                with Image.open(data / row["path"]) as original:
                    image = original.convert("RGB").resize((320, 240), Image.Resampling.LANCZOS)
                    value = detector.logit(image)
                result = {"id": row["id"], "label": row["label"], "variant": variant, "logit": value}
                cache[key] = result
                with cache_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result) + "\n")
            values.append(cache[key]["logit"])
            if index % 24 == 0:
                print(f"{variant}: {index+1}/{len(rows)}", flush=True)
        return np.array(values)

    labels = np.array([r["label"] for r in valid])
    candidates, detectors = {}, {}
    for variant in ("dear_r", "dear_c"):
        detector = DearDetector(ROOT, variant)
        values = infer(detector, valid, variant)
        candidates[variant] = {"auc": float(roc_auc_score(labels, values)), "values": values}
        detectors[variant] = detector
    variant = max(candidates, key=lambda name: candidates[name]["auc"])
    fit = LogisticRegression(C=1., max_iter=1000).fit(candidates[variant]["values"].reshape(-1, 1), labels)
    coefficient, intercept = float(fit.coef_[0, 0]), float(fit.intercept_[0])
    if coefficient <= 0:
        raise RuntimeError("Validation does not support the AI-score orientation")
    raw = infer(detectors[variant], fresh, variant)
    score = 1. / (1. + np.exp(-np.clip(raw * coefficient + intercept, -60, 60)))
    result = {"status": "passed", "validation_auroc": {k: v["auc"] for k, v in candidates.items()},
              "selected_variant": variant, "calibration": {"coefficient": coefficient, "intercept": intercept},
              "fresh_holdout": metrics([r["label"] for r in fresh], score),
              "native_size_threshold": 512, "fresh_test_sources": len(fresh), "resize": [320, 240]}
    native_policy = json.loads((ROOT / "models/dear_r/policy.json").read_text(encoding="utf-8"))["calibration"]
    baseline = infer(detectors["dear_r"], fresh, "dear_r")
    baseline = 1. / (1. + np.exp(-np.clip(baseline * native_policy["coefficient"] + native_policy["intercept"], -60, 60)))
    result["same_sources_global_calibration_baseline"] = metrics([r["label"] for r in fresh], baseline)
    (output / "lowres-evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    policy_path = ROOT / "models/dear_r/policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["low_resolution"] = {"short_side_below": 512, "variant": variant, "tile_size": 256,
                                "calibration": result["calibration"], "validation_samples": 64,
                                "validation_auroc": result["validation_auroc"]}
    policy_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
