"""Audit full versus compact saved text models with independent validation calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from train_and_optimize import calibrated, fit_calibration, read_jsonl, score_metrics, text_raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    data = root / "data/text"
    splits = {name: read_jsonl(data / filename) for name, filename in
              (("validation", "validation.jsonl"), ("test", "test.jsonl"),
               ("external_test", "external_test.jsonl"))}
    comparison = {"purpose": "post-training audit only; external results were not used to fit or select a model",
                  "models": {}}
    for name, filename in (("full", "text_detector_full.joblib"), ("compact", "text_detector.joblib")):
        model = joblib.load(root / "models" / filename)
        raw_val, actual_val = text_raw(model, splits["validation"])
        calibration = fit_calibration(raw_val, actual_val)
        scores = {}
        for split, records in splits.items():
            raw, actual = text_raw(model, records)
            scores[split] = score_metrics(actual, calibrated(raw, calibration))
        comparison["models"][name] = {"features": len(model["vectorizer"].vocabulary_),
                                       "bytes": (root / "models" / filename).stat().st_size,
                                       "calibration": calibration, "scores": scores}
    output = root / "results/exp2_exp3/text_full_vs_compact.json"
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: {split: value["accuracy"] for split, value in item["scores"].items()}
                      for name, item in comparison["models"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
