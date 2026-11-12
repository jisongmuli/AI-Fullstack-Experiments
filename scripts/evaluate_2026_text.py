"""Train a small local TASTE head on training records; audit untouched 2026 test data."""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/models"))
from research_detectors import TasteDetector
from inference import DetectorService
from text_segments import segment_ranges
from evaluate_acceptance import wilson_interval

OUT = ROOT / "results/upgrade_20261026/text"
DATA = ROOT / "data/benchmarks_2026"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    total = len(labels)
    scored = np.isfinite(scores)
    labels, scores = labels[scored], scores[scored]
    pred = scores >= .5
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {"total_samples": total, "samples": len(labels), "scored_coverage": len(labels) / total,
            "human": int((labels == 0).sum()), "ai": int((labels == 1).sum()),
            "precision": float(precision_score(labels, pred, zero_division=0)),
            "recall": float(recall_score(labels, pred, zero_division=0)),
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "accuracy_wilson_95": wilson_interval(int(tn + tp), len(labels)),
            "fpr_wilson_95": wilson_interval(int(fp), int(tn + fp)),
            "fnr_wilson_95": wilson_interval(int(fn), int(tp + fn)),
            "accuracy": float(accuracy_score(labels, pred)), "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            "auroc": float(roc_auc_score(labels, scores)), "false_positive_rate": float(fp / (tn + fp)),
            "false_negative_rate": float(fn / (tp + fn)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    training = json.loads((DATA / "nlpcc/data/train_data.json").read_text(encoding="utf-8-sig"))
    rng = random.Random(20261026)
    rng.shuffle(training)
    records, seen = [], set()
    for group in training:
        if len(records) >= (128 + 64) * 2:
            break
        human, generated = group.get("HWT"), group.get("LGT", group.get("MGT"))
        if not isinstance(human, str) or not isinstance(generated, str):
            continue
        hashes = {digest(human), digest(generated)}
        if len(hashes) < 2 or seen.intersection(hashes):
            continue
        split = "train" if len(records) < 256 else "validation"
        seen.update(hashes)
        for label, text in [(0, human), (1, generated)]:
            records.append({"id": f"train-{group['ID']}-{label}", "text": text, "label": label,
                            "split": split, "source": "NLPCC2026-training", "sha256": digest(text)})
    assert len(records) == 384
    test = json.loads((DATA / "nlpcc/data/testp2_testing_label.json").read_text(encoding="utf-8-sig"))
    for row in test:
        records.append({"id": row["id"], "text": row["text"], "label": int(row["label"]),
                        "split": "test", "source": "NLPCC2026-phase2", "sha256": digest(row["text"])})
    for filename in ["ai_generated_2026-09-29.jsonl", "human_pmc_ccby_nonanglophone.jsonl", "human_wikipedia_pre2022.jsonl"]:
        for line in (DATA / "textsight/data" / filename).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            records.append({"id": row["id"], "text": row["text"], "label": 1 if row["label"] == "ai" else 0,
                            "split": "test", "source": "TextSight-September2026", "generator": row.get("source"),
                            "sha256": digest(row["text"])})
    train_hashes = {r["sha256"] for r in records if r["split"] == "train"}
    val_hashes = {r["sha256"] for r in records if r["split"] == "validation"}
    test_hashes = {r["sha256"] for r in records if r["split"] == "test"}
    assert not (train_hashes & val_hashes or train_hashes & test_hashes or val_hashes & test_hashes), "Exact text leakage"
    plan = {"seed": 20261026, "head_training_groups": 128, "validation_groups": 64,
            "method": "TASTE released multilingual encoder; compare original head and training-only logistic head on validation, then freeze",
            "head_logistic_C": .1, "test_used_for_training_or_calibration": False,
            "exact_duplicate_check": "SHA256 of original text; passed across splits",
            "records": [{k: v for k, v in r.items() if k != "text"} for r in records]}
    (OUT / "evaluation-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    cache = OUT / "taste-representations.npz"
    hashes = np.array([r["sha256"] for r in records])
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        assert np.array_equal(saved["hashes"], hashes)
        logits, features = saved["logits"], saved["features"]
    else:
        detector = TasteDetector(ROOT)
        logits, features = [], []
        for start in range(0, len(records), 8):
            group = records[start:start + 8]
            tick = time.perf_counter()
            values, hidden = detector.representations([r["text"] for r in group])
            logits.extend(values); features.extend(hidden)
            if start % 80 == 0:
                print(f"TASTE: {start + len(group)}/{len(records)} records; last batch {time.perf_counter()-tick:.2f}s", flush=True)
        logits, features = np.asarray(logits), np.asarray(features)
        np.savez_compressed(cache, hashes=hashes, logits=logits, features=features)
        del detector
    labels = np.array([r["label"] for r in records])
    train = np.array([r["split"] == "train" for r in records])
    valid = np.array([r["split"] == "validation" for r in records])
    scaler = StandardScaler().fit(features[train])
    head = LogisticRegression(C=.1, class_weight="balanced", max_iter=1000, random_state=20261026)
    head.fit(scaler.transform(features[train]), labels[train])
    coefficient = head.coef_[0] / scaler.scale_
    intercept = float(head.intercept_[0] - np.dot(coefficient, scaler.mean_))
    adapted = features @ coefficient + intercept
    sigmoid = lambda value: 1. / (1. + np.exp(-np.clip(value, -60, 60)))
    original_val = metrics(labels[valid], sigmoid(logits[valid]))
    adapted_val = metrics(labels[valid], sigmoid(adapted[valid]))
    use_adapted = adapted_val["balanced_accuracy"] > original_val["balanced_accuracy"]
    selected = adapted if use_adapted else logits
    calibrator = LogisticRegression(C=1., max_iter=1000).fit(selected[valid].reshape(-1, 1), labels[valid])
    if calibrator.coef_[0, 0] <= 0:
        raise RuntimeError("Validation does not support the declared AI score orientation")
    calibrated = sigmoid(selected * calibrator.coef_[0, 0] + calibrator.intercept_[0])
    # Only the Chinese branch uses the locally adapted head/calibration.
    english = np.array([r["source"] == "TextSight-September2026" for r in records])
    calibrated[english] = sigmoid(logits[english])
    np.savez_compressed(ROOT / "models/taste/chinese_head.npz", coefficient=coefficient, intercept=intercept)
    policy = {"method": "TASTE-ICLR2026", "chinese_head": "adapted" if use_adapted else "original",
              "chinese_calibration": {"coefficient": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0])},
              "english_head": "original", "english_calibration": {"coefficient": 1., "intercept": 0.},
              "validation_original": original_val, "validation_adapted": adapted_val,
              "training_groups": 128, "validation_groups": 64, "max_tokens": 512,
              "overflow_stride": 64, "long_text_aggregation": "unique-token-weighted mean logit"}
    (ROOT / "models/taste/policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    legacy = DetectorService(ROOT)
    legacy.algorithm_policy = {}
    baseline = []
    for index, row in enumerate(records):
        weighted, length = 0., 0
        for start, end in segment_ranges(row["text"]):
            result = legacy.predict_text(row["text"][start:end])
            if result["risk_score"] is not None and not result["out_of_scope"]:
                count = sum(not c.isspace() for c in row["text"][start:end])
                weighted += count * result["risk_score"]; length += count
        baseline.append(weighted / length if length else float("nan"))
    baseline = np.array(baseline)
    results = {}
    for source in ["NLPCC2026-phase2", "TextSight-September2026"]:
        mask = np.array([r["source"] == source and r["label"] in (0, 1) for r in records])
        results[source] = {"legacy": metrics(labels[mask], baseline[mask]), "taste2026": metrics(labels[mask], calibrated[mask])}
    output_rows = [{k: v for k, v in r.items() if k != "text"} | {"legacy_score": float(old) if np.isfinite(old) else None, "taste_score": float(new)}
                   for r, old, new in zip(records, baseline, calibrated)]
    result = {"status": "passed", "policy": policy, "heldout_results": results,
              "refined_text": "NLPCC label 2 retained separately; excluded from pure human-vs-generated accuracy",
              "predictions": output_rows}
    (OUT / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
