"""Compare official MPU v3 with TASTE on the same 2026 records, preserving all split IDs."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/models"))
sys.path.insert(0, str(ROOT / "scripts"))
from research_detectors import MpuDetector
from evaluate_2026_text import metrics


def main():
    torch.set_num_threads(4)
    data, out = ROOT / "data/benchmarks_2026", ROOT / "results/upgrade_20261026/text"
    evaluation = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    rows = evaluation["predictions"]
    required = {r["sha256"] for r in rows if r["split"] == "validation" or (r["split"] == "test" and r["label"] in (0, 1))}
    texts = {}
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    training = json.loads((data / "nlpcc/data/train_data.json").read_text(encoding="utf-8-sig"))
    for group in training:
        for key in ("HWT", "LGT"):
            value = group.get(key)
            if isinstance(value, str) and digest(value) in required:
                texts[digest(value)] = value
    del training
    for row in json.loads((data / "nlpcc/data/testp2_testing_label.json").read_text(encoding="utf-8-sig")):
        if digest(row["text"]) in required:
            texts[digest(row["text"])] = row["text"]
    for filename in ["ai_generated_2026-09-29.jsonl", "human_pmc_ccby_nonanglophone.jsonl", "human_wikipedia_pre2022.jsonl"]:
        for line in (data / "textsight/data" / filename).read_text(encoding="utf-8").splitlines():
            value = json.loads(line)["text"]
            if digest(value) in required:
                texts[digest(value)] = value
    cache_path = out / "mpu-logits.jsonl"
    cache = {}
    if cache_path.exists():
        cache = {r["id"]: r for r in map(json.loads, cache_path.read_text(encoding="utf-8").splitlines())}
    for language, source in [("chinese", "NLPCC2026-phase2"), ("english", "TextSight-September2026")]:
        group = [r for r in rows if r["sha256"] in required and (r["source"] == source or
                 (language == "chinese" and r["split"] == "validation"))]
        detector = MpuDetector(ROOT, language)
        missing = [r for r in group if r["id"] not in cache]
        for start in range(0, len(missing), 8):
            batch = missing[start:start + 8]
            logits = detector.logits([texts[r["sha256"]] for r in batch])
            with cache_path.open("a", encoding="utf-8") as stream:
                for row, value in zip(batch, logits):
                    record = {"id": row["id"], "sha256": row["sha256"], "language": language, "logit": value}
                    cache[row["id"]] = record
                    stream.write(json.dumps(record) + "\n")
            if start % 80 == 0:
                print(f"MPU {language}: {start + len(batch)}/{len(missing)}", flush=True)
        del detector
    valid = [r for r in rows if r["split"] == "validation"]
    x = np.array([cache[r["id"]]["logit"] for r in valid])
    y = np.array([r["label"] for r in valid])
    calibration = LogisticRegression(C=1., max_iter=1000).fit(x.reshape(-1, 1), y)
    coefficient, intercept = float(calibration.coef_[0, 0]), float(calibration.intercept_[0])
    if coefficient <= 0:
        raise RuntimeError("MPU validation contradicts the published AI-label orientation")
    sigmoid = lambda value: float(1. / (1. + np.exp(-np.clip(value, -60, 60))))
    valid_scores = [sigmoid(cache[r["id"]]["logit"] * coefficient + intercept) for r in valid]
    valid_metrics = metrics(y, valid_scores)
    taste_valid = metrics(y, [r["taste_score"] for r in valid])
    chinese_backend = "mpu" if valid_metrics["balanced_accuracy"] >= taste_valid["balanced_accuracy"] else "taste"
    # English uses the language-specific author checkpoint, chosen before its test metrics are computed.
    policy = {"chinese_backend": chinese_backend, "english_backend": "mpu",
              "chinese_calibration": {"coefficient": coefficient, "intercept": intercept},
              "english_calibration": {"coefficient": 1., "intercept": 0.},
              "selection": "Chinese validation balanced accuracy; English dedicated official MPU v3 short checkpoint",
              "validation_mpu": valid_metrics, "validation_taste": taste_valid,
              "mpu_release": "2025-06 author v3 short; latest release linked by the official project at retrieval"}
    results, predictions = {}, []
    for source in ("NLPCC2026-phase2", "TextSight-September2026"):
        group = [r for r in rows if r["source"] == source and r["label"] in (0, 1)]
        scores = [sigmoid(cache[r["id"]]["logit"] * coefficient + intercept) if source.startswith("NLPCC") else
                  sigmoid(cache[r["id"]]["logit"]) for r in group]
        results[source] = {"mpu": metrics([r["label"] for r in group], scores),
                           "taste": metrics([r["label"] for r in group], [r["taste_score"] for r in group])}
        predictions.extend({"id": r["id"], "label": r["label"], "source": source, "sha256": r["sha256"], "mpu_score": score}
                           for r, score in zip(group, scores))
    (ROOT / "models/text_policy_2026.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"status": "passed", "policy": policy, "heldout_results": results, "predictions": predictions}
    (out / "mpu-evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"policy": policy, "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
