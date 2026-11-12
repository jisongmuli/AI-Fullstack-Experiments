"""Select/calibrate Chinese models on separate 2026 Phase 1 records, never Phase 2 labels."""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/models"))
sys.path.insert(0, str(ROOT / "scripts"))
from research_detectors import MpuDetector, TasteDetector
from evaluate_2026_text import metrics


def canonical(text):
    return hashlib.sha256(re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().encode("utf-8")).hexdigest()


def main():
    torch.set_num_threads(2)
    data, output = ROOT / "data/benchmarks_2026/nlpcc/data", ROOT / "results/upgrade_20261026/text"
    labels = {r["id"]: int(r["label"]) for r in json.loads((data / "testp1_testing_label.json").read_text(encoding="utf-8-sig"))}
    source = json.loads((data / "testp1.json").read_text(encoding="utf-8-sig"))
    forbidden = {canonical(r["text"]) for r in json.loads((data / "testp2_testing_label.json").read_text(encoding="utf-8-sig"))}
    groups = {label: [] for label in (0, 1)}
    seen = set()
    for row in source:
        label, sha = labels[row["id"]], canonical(row["text"])
        if label in groups and sha not in forbidden and sha not in seen:
            groups[label].append({"id": row["id"], "text": row["text"], "label": label, "canonical_sha256": sha})
            seen.add(sha)
    rng = random.Random(20261026)
    rows = [r for label in (0, 1) for r in rng.sample(sorted(groups[label], key=lambda r: r["id"]), 64)]
    plan = {"source": "Official NLPCC 2026 Phase 1", "seed": 20261026,
            "role": "development validation, separate from Phase 2 reporting",
            "selection": "64 random human + 64 random generated; canonical NFKC/whitespace duplicates against Phase 2 excluded",
            "criterion": "validation AUROC; positive sigmoid fit on the same validation only",
            "records": [{k: v for k, v in r.items() if k != "text"} for r in rows]}
    (output / "phase1-plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    keys = np.array([r["id"] for r in rows])
    path = output / "phase1-taste-cache.npz"
    if path.exists():
        saved = np.load(path, allow_pickle=False)
        assert np.array_equal(saved["ids"], keys)
        original, features = saved["logits"], saved["features"]
    else:
        detector, original, features = TasteDetector(ROOT), [], []
        for start in range(0, len(rows), 4):
            batch = rows[start:start+4]
            values, hidden = detector.representations([r["text"] for r in batch])
            original.extend(values); features.extend(hidden)
            if start % 32 == 0:
                print(f"Phase1 TASTE: {start+len(batch)}/{len(rows)}", flush=True)
        original, features = np.array(original), np.array(features)
        np.savez_compressed(path, ids=keys, logits=original, features=features)
        del detector
    path = output / "phase1-mpu-cache.npz"
    if path.exists():
        saved = np.load(path, allow_pickle=False)
        assert np.array_equal(saved["ids"], keys)
        mpu = saved["logits"]
    else:
        detector, mpu = MpuDetector(ROOT, "chinese"), []
        for start in range(0, len(rows), 4):
            batch = rows[start:start+4]
            mpu.extend(detector.logits([r["text"] for r in batch]))
            if start % 32 == 0:
                print(f"Phase1 MPU: {start+len(batch)}/{len(rows)}", flush=True)
        mpu = np.array(mpu)
        np.savez_compressed(path, ids=keys, logits=mpu)
        del detector
    head = np.load(ROOT / "models/taste/chinese_head.npz", allow_pickle=False)
    adapted = features @ head["coefficient"] + head["intercept"]
    labels = np.array([r["label"] for r in rows])
    candidates = {"taste_original": original, "taste_adapted": adapted, "mpu": mpu}
    summaries, calibration = {}, {}
    for name, raw in candidates.items():
        fit = LogisticRegression(C=1., max_iter=1000).fit(raw.reshape(-1, 1), labels)
        coefficient, intercept = float(fit.coef_[0, 0]), float(fit.intercept_[0])
        if coefficient <= 0:
            continue
        score = 1. / (1. + np.exp(-np.clip(raw * coefficient + intercept, -60, 60)))
        summaries[name] = metrics(labels, score)
        calibration[name] = {"coefficient": coefficient, "intercept": intercept}
    selected = max(summaries, key=lambda name: summaries[name]["auroc"])
    result = {"status": "passed", "selected_candidate": selected,
              "chinese_backend": "mpu" if selected == "mpu" else "taste",
              "chinese_taste_head": "adapted" if selected == "taste_adapted" else "original",
              "chinese_calibration": calibration[selected], "candidate_calibrations": calibration,
              "validation_results": summaries, "validation_source": "NLPCC2026 Phase1; 128 records",
              "validation_ids": keys.tolist(), "selection_used_phase2_labels": False}
    (ROOT / "models/text_calibration_phase1.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "phase1-calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
