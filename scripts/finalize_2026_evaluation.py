"""Freeze the validation-selected pipeline and summarize cached held-out measurements."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_2026_text import metrics


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    phase1 = read("models/text_calibration_phase1.json")
    image_policy = read("models/dear_r/policy.json")
    raw = read("results/upgrade_20261026/text/evaluation.json")["predictions"]
    mpu = {r["id"]: r for r in map(json.loads, (ROOT / "results/upgrade_20261026/text/mpu-logits.jsonl").read_text(encoding="utf-8").splitlines())}
    neural = np.load(ROOT / "results/upgrade_20261026/text/taste-representations.npz", allow_pickle=False)
    assert np.array_equal(neural["hashes"], np.array([r["sha256"] for r in raw]))
    head = np.load(ROOT / "models/taste/chinese_head.npz", allow_pickle=False)
    selected, calibration = phase1["selected_candidate"], phase1["chinese_calibration"]
    sigmoid = lambda value: float(1. / (1. + np.exp(-np.clip(value, -60, 60))))
    groups = {"chinese": [], "english": []}
    prediction_rows = []
    for index, row in enumerate(raw):
        if row["split"] != "test" or row["label"] not in (0, 1):
            continue
        language = "english" if row["source"] == "TextSight-September2026" else "chinese"
        if language == "english" or selected == "mpu":
            margin = mpu[row["id"]]["logit"]
        elif selected == "taste_original":
            margin = neural["logits"][index]
        else:
            margin = neural["features"][index] @ head["coefficient"] + head["intercept"]
        score = sigmoid(margin) if language == "english" else sigmoid(margin * calibration["coefficient"] + calibration["intercept"])
        item = {"id": row["id"], "sha256": row["sha256"], "label": row["label"], "source": row["source"],
                "score": score, "legacy_score": row["legacy_score"]}
        groups[language].append(item); prediction_rows.append(item)
    results = {}
    for language, rows in groups.items():
        legacy_scores = [r["legacy_score"] if r["legacy_score"] is not None else np.nan for r in rows]
        results[language] = {
            "current": metrics([r["label"] for r in rows], [r["score"] for r in rows]),
            "legacy": metrics([r["label"] for r in rows], legacy_scores)}
        scores, labels = np.array([r["score"] for r in rows]), np.array([r["label"] for r in rows])
        confident = (scores <= .2) | (scores >= .8)
        results[language]["runtime_confidence_band"] = {"low": .2, "high": .8, "coverage": float(confident.mean()),
            "conclusions": int(confident.sum()), "correct_conclusion_fraction": float(((scores[confident] >= .8) == labels[confident]).mean())}
    records = [json.loads(line) for line in (ROOT / "results/upgrade_20261026/image/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    test = [r for r in records if r["split"] == "test"]
    calibration = image_policy["calibration"]
    scores = [sigmoid(r["dear_logit"] * calibration["coefficient"] + calibration["intercept"]) for r in test]
    results["image_native"] = {"current": metrics([r["label"] for r in test], scores),
                               "legacy": metrics([r["label"] for r in test], [r["legacy_score"] for r in test])}
    lowres = read("results/upgrade_20261026/image/lowres-evaluation.json")
    results["image_low_resolution"] = {"current": lowres["fresh_holdout"],
                                       "global_calibration_baseline": lowres["same_sources_global_calibration_baseline"]}
    result = {"status": "passed", "test_threshold": .5, "runtime_low_threshold": .2, "runtime_high_threshold": .8,
              "chinese_selection": selected, "selection_source": phase1["validation_source"],
              "results": results, "text_predictions": prediction_rows,
              "evaluation_scope": "Public 2026 datasets; separate validation/test IDs. Model selection is documented; platform traffic is a separate target domain."}
    (ROOT / "results/upgrade_20261026/final-evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    names = {"chinese": "中文 · NLPCC 2026 Phase 2", "english": "英文 · 2026年9月生成样本与人工对照",
             "image_native": "图片 · NTIRE 2026 保留集", "image_low_resolution": "小图 · 320×240 保留集"}
    summary = [{"name": names[k], **v["current"]} for k, v in results.items()]
    policy = {"version": "2026-10-26", "cpu_threads": 4, "text_backend": "transformer_2026", "image_backend": "dear",
              "text_chinese": selected, "text_english": "MPU-v3-short", "image_native": "DEAR-r-ICML2026",
              "image_low_resolution": "DEAR-c-ICML2026", "evaluation_summary": summary,
              "evaluation_threshold": .5, "runtime_confidence_thresholds": [.2, .8],
              "text_description": "采用 MPU 项目作者当前公开的中英文专项 v3 Short 权重，中文通过 NLPCC 2026 Phase 1 独立校准。按 512 token 滑窗处理长文本，并提供逐段评分。",
              "model_release_note": "MPU paper ICLR 2024, latest author v3 weights 2025-06; DEAR ICML 2026. TASTE 2026 was evaluated as a candidate."}
    if selected != "mpu":
        policy["text_description"] = "中文采用验证集选定的 TASTE 2026 模型，英文采用 MPU 作者 v3 Short 权重；长文本按 512 token 滑窗覆盖。"
    (ROOT / "models/algorithm_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: {"accuracy": value["current"]["accuracy"], "fpr": value["current"]["false_positive_rate"],
                                "fnr": value["current"]["false_negative_rate"], "samples": value["current"]["samples"]}
                      for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
