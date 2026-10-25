"""Train experiment-two detectors and measure experiment-three optimizations."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import random
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from architecture import ImageCNN

SEED = 42
ABSTAIN_PROBABILITY = 0.8


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score_metrics(actual: np.ndarray, probability: np.ndarray) -> dict:
    predicted = (probability >= 0.5).astype(int)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    covered = (probability <= 1 - ABSTAIN_PROBABILITY) | (probability >= ABSTAIN_PROBABILITY)
    selective_predicted = (probability[covered] >= ABSTAIN_PROBABILITY).astype(int)
    return {
        "n": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "ai_precision": float(precision_score(actual, predicted, zero_division=0)),
        "ai_recall": float(recall_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "brier": float(brier_score_loss(actual, probability)),
        "human_false_positive_rate": float(matrix[0, 1] / max(1, matrix[0].sum())),
        "confusion_matrix_rows_human_ai": matrix.tolist(),
        "abstention": {
            "probability_threshold": ABSTAIN_PROBABILITY,
            "coverage": float(covered.mean()),
            "covered_accuracy": float(accuracy_score(actual[covered], selective_predicted)) if covered.any() else None,
            "covered_count": int(covered.sum()),
        },
    }


def fit_calibration(raw: np.ndarray, actual: np.ndarray) -> dict:
    calibrator = LogisticRegression(random_state=SEED, C=1e4, max_iter=1000)
    calibrator.fit(raw.reshape(-1, 1), actual)
    return {"coefficient": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0]),
            "method": "sigmoid logistic calibration on validation; validation calibration scores are optimistic"}


def calibrated(raw: np.ndarray, calibration: dict) -> np.ndarray:
    z = np.clip(calibration["coefficient"] * raw + calibration["intercept"], -60, 60)
    return 1 / (1 + np.exp(-z))


def write_predictions(path: Path, records: list[dict], probability: np.ndarray) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "actual", "risk_score", "predicted_at_0.5", "decision_at_0.8"])
        for record, risk in zip(records, probability):
            decision = "ai" if risk >= ABSTAIN_PROBABILITY else "human" if risk <= 1 - ABSTAIN_PROBABILITY else "uncertain"
            writer.writerow([record["sample_id"], record["label"], f"{risk:.8f}", int(risk >= .5), decision])


class Images(Dataset):
    def __init__(self, root: Path, records: list[dict], training: bool):
        self.root = root
        self.records = records
        self.transform = transforms.Compose([
            *([transforms.RandomHorizontalFlip(), transforms.RandomRotation(8)] if training else []),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record["relative_path"]) as source:
            image = self.transform(source.convert("RGB"))
        return image, int(record["label"])


def image_raw_predictions(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    raw, actual = [], []
    with torch.inference_mode():
        for image, target in loader:
            logits = model(image)
            raw.extend((logits[:, 1] - logits[:, 0]).tolist())
            actual.extend(target.tolist())
    return np.asarray(raw), np.asarray(actual)


def benchmark_image(model: nn.Module, tensor: torch.Tensor) -> dict:
    model.eval()
    with torch.inference_mode():
        for _ in range(25):
            model(tensor)
        durations = []
        for _ in range(200):
            started = time.perf_counter_ns()
            model(tensor)
            durations.append((time.perf_counter_ns() - started) / 1e6)
    return {"n": len(durations), "p50_ms": float(np.median(durations)), "p95_ms": float(np.percentile(durations, 95))}


def image_experiment(root: Path, models: Path, results: Path) -> tuple[dict, dict]:
    with (root / "data/images/split_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        all_records = list(csv.DictReader(stream))
    partitions = {split: [row for row in all_records if row["split"] == split]
                  for split in ("train", "validation", "test")}
    loaders = {split: DataLoader(Images(root, rows, split == "train"), batch_size=128,
                                 shuffle=split == "train", num_workers=0,
                                 generator=torch.Generator().manual_seed(SEED))
               for split, rows in partitions.items()}
    model = ImageCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    best_state, best_f1, best_epoch = None, -1.0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, 9):
        model.train()
        loss_sum = 0.0
        for images, targets in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(images)
        scheduler.step()
        raw, actual = image_raw_predictions(model, loaders["validation"])
        probability = 1 / (1 + np.exp(-np.clip(raw, -60, 60)))
        macro_f1 = f1_score(actual, probability >= .5, average="macro")
        history.append({"epoch": epoch, "train_loss": loss_sum / len(partitions["train"]),
                        "validation_accuracy": float(accuracy_score(actual, probability >= .5)),
                        "validation_macro_f1": float(macro_f1), "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"image epoch={epoch} validation_macro_f1={macro_f1:.4f}", flush=True)
        if macro_f1 > best_f1:
            best_state, best_f1, best_epoch = copy.deepcopy(model.state_dict()), macro_f1, epoch
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "architecture": "ImageCNN", "seed": SEED,
                "best_epoch": best_epoch}, models / "image_detector.pt")
    script = torch.jit.trace(model.eval(), torch.rand(1, 3, 32, 32))
    buffer = io.BytesIO()
    torch.jit.save(script, buffer)
    (models / "image_detector.ts").write_bytes(buffer.getvalue())
    raw_val, actual_val = image_raw_predictions(model, loaders["validation"])
    calibration = fit_calibration(raw_val, actual_val)
    evaluations = {}
    for split in ("validation", "test"):
        raw, actual = image_raw_predictions(model, loaders[split])
        probability = calibrated(raw, calibration)
        evaluations[split] = score_metrics(actual, probability)
        write_predictions(results / f"image_{split}_predictions.csv", partitions[split], probability)
    first_image = Images(root, partitions["test"], False)[0][0].unsqueeze(0)
    eager_benchmark = benchmark_image(model, first_image)
    scripted_benchmark = benchmark_image(script, first_image)
    with torch.inference_mode():
        parity_max_abs_error = float((model(first_image) - script(first_image)).abs().max())
    metrics = {
        "dataset": "CIFAKE 32x32 only", "seed": SEED, "architecture": "five convolution layers with batch normalization and global average pooling",
        "optimizer": "AdamW lr=0.001, weight_decay=0.0001; StepLR every four epochs",
        "selection": "best validation macro-F1 over eight epochs; test untouched until final evaluation",
        "counts": {name: len(rows) for name, rows in partitions.items()},
        "history": history, "best_epoch": best_epoch, "scores": evaluations,
        "training_seconds": round(time.perf_counter() - started, 2),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "torch_version": torch.__version__,
    }
    optimization = {
        "method": "TorchScript trace on fixed 1x3x32x32 input", "eager": eager_benchmark,
        "torchscript": scripted_benchmark,
        "eager_bytes": (models / "image_detector.pt").stat().st_size,
        "torchscript_bytes": (models / "image_detector.ts").stat().st_size,
        "max_abs_logit_difference_on_benchmark_image": parity_max_abs_error,
        "note": "Latency is CPU-only, batch one, model computation excluding image decoding; process load changes timing.",
    }
    return metrics, {"calibration": calibration, "optimization": optimization}


def train_text_candidate(train: list[dict], kind: str, max_features: int) -> dict:
    if kind == "word":
        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=max_features,
                                     min_df=2, sublinear_tf=True, stop_words="english")
    else:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=max_features,
                                     min_df=2, sublinear_tf=True)
    features = vectorizer.fit_transform([row["text"] for row in train])
    classifier = LogisticRegression(C=1, max_iter=1000, class_weight="balanced", random_state=SEED)
    classifier.fit(features, [row["label"] for row in train])
    return {"vectorizer": vectorizer, "classifier": classifier, "kind": kind,
            "max_features": max_features}


def text_raw(model: dict, records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = model["vectorizer"].transform([row["text"] for row in records])
    return model["classifier"].decision_function(features), np.asarray([row["label"] for row in records])


def benchmark_text(model: dict, text: str) -> dict:
    for _ in range(20):
        text_raw(model, [{"text": text, "label": 0}])
    durations = []
    for _ in range(200):
        started = time.perf_counter_ns()
        text_raw(model, [{"text": text, "label": 0}])
        durations.append((time.perf_counter_ns() - started) / 1e6)
    return {"n": len(durations), "p50_ms": float(np.median(durations)), "p95_ms": float(np.percentile(durations, 95))}


def text_experiment(root: Path, models: Path, results: Path) -> tuple[dict, dict]:
    data = root / "data/text"
    train = read_jsonl(data / "train.jsonl") + read_jsonl(data / "train_augmented.jsonl")
    validation = read_jsonl(data / "validation.jsonl")
    test = read_jsonl(data / "test.jsonl")
    external = read_jsonl(data / "external_test.jsonl")
    trials, candidate_models = [], {}
    for kind, count in (("word", 20000), ("char", 30000)):
        started = time.perf_counter()
        candidate = train_text_candidate(train, kind, count)
        raw, actual = text_raw(candidate, validation)
        raw_probability = 1 / (1 + np.exp(-np.clip(raw, -60, 60)))
        score = float(f1_score(actual, raw_probability >= .5, average="macro"))
        trials.append({"kind": kind, "max_features": count, "vocabulary_size": len(candidate["vectorizer"].vocabulary_),
                       "validation_macro_f1_uncalibrated": score, "fit_seconds": round(time.perf_counter() - started, 2)})
        candidate_models[kind] = candidate
        print(f"text {kind} validation_macro_f1={score:.4f}", flush=True)
    selected = max(trials, key=lambda item: item["validation_macro_f1_uncalibrated"])
    full_model = candidate_models[selected["kind"]]
    joblib.dump(full_model, models / "text_detector_full.joblib", compress=3)
    full_val_f1 = selected["validation_macro_f1_uncalibrated"]
    compact_trials = []
    for count in ((5000, 10000) if selected["kind"] == "word" else (10000, 18000)):
        candidate = train_text_candidate(train, selected["kind"], count)
        raw, actual = text_raw(candidate, validation)
        val_f1 = float(f1_score(actual, raw >= 0, average="macro"))
        compact_trials.append({"max_features": count, "validation_macro_f1_uncalibrated": val_f1,
                               "within_one_percentage_point": val_f1 >= full_val_f1 - 0.01, "model": candidate})
        print(f"text compact features={count} validation_macro_f1={val_f1:.4f}", flush=True)
    viable = [item for item in compact_trials if item["within_one_percentage_point"]]
    optimized = viable[0]["model"] if viable else full_model
    optimized_count = viable[0]["max_features"] if viable else selected["max_features"]
    joblib.dump(optimized, models / "text_detector.joblib", compress=3)
    raw_val, actual_val = text_raw(optimized, validation)
    calibration = fit_calibration(raw_val, actual_val)
    evaluations = {}
    for split, records in (("validation", validation), ("test", test), ("external_test", external)):
        raw, actual = text_raw(optimized, records)
        probability = calibrated(raw, calibration)
        evaluations[split] = score_metrics(actual, probability)
        write_predictions(results / f"text_{split}_predictions.csv", records, probability)
    for split, records in (("test", test), ("external_test", external)):
        raw, actual = text_raw(full_model, records)
        evaluations[f"full_{split}"] = score_metrics(actual, calibrated(raw, calibration)) if optimized is full_model else {
            "uncalibrated_macro_f1": float(f1_score(actual, raw >= 0, average="macro")),
            "uncalibrated_accuracy": float(accuracy_score(actual, raw >= 0)),
        }
    benchmark_example = test[0]["text"]
    optimization = {
        "method": "reduce TF-IDF vocabulary under <=1 percentage point validation macro-F1 loss",
        "selected_full_features": selected["max_features"], "chosen_features": optimized_count,
        "compact_trials": [{key: value for key, value in item.items() if key != "model"} for item in compact_trials],
        "full_bytes": (models / "text_detector_full.joblib").stat().st_size,
        "optimized_bytes": (models / "text_detector.joblib").stat().st_size,
        "full_inference": benchmark_text(full_model, benchmark_example),
        "optimized_inference": benchmark_text(optimized, benchmark_example),
        "note": "CPU batch-one vectorization plus classification on one fixed held-out text; no external labels used for model selection.",
    }
    metrics = {
        "dataset": "length-matched English text; primary and external sources differ",
        "seed": SEED,
        "training_count": len(train), "validation_count": len(validation), "test_count": len(test),
        "external_test_count": len(external),
        "training_augmentation_count": len(read_jsonl(data / "train_augmented.jsonl")),
        "candidate_trials": trials,
        "selection": "maximum uncalibrated validation macro-F1; external data never used for selection",
        "selected_kind": selected["kind"], "deployed_features": optimized_count,
        "scores": evaluations,
        "sklearn_version": sklearn.__version__,
        "limitations": ["Primary CSV has no per-record source field.",
                        "Primary same-source test can overstate transfer; see external_test.",
                        "Scores cannot establish author identity or editing history."],
    }
    return metrics, {"calibration": calibration, "optimization": optimization}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    models, results = root / "models", root / "results/exp2_exp3"
    models.mkdir(exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    set_seed()
    image_metrics, image_extra = image_experiment(root, models, results)
    (results / "experiment2_image_metrics.json").write_text(json.dumps(image_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    text_metrics, text_extra = text_experiment(root, models, results)
    (results / "experiment2_text_metrics.json").write_text(json.dumps(text_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {"labels": {"0": "human_or_real", "1": "ai_generated"}, "abstain_probability": ABSTAIN_PROBABILITY,
              "image_calibration": image_extra["calibration"], "text_calibration": text_extra["calibration"],
              "image_input": "32x32 RGB", "text_input": "English excerpt; 25-1000 characters and >=5 words"}
    (models / "model_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (results / "experiment3_optimization.json").write_text(json.dumps({
        "image": image_extra["optimization"], "text": text_extra["optimization"],
        "abstention_probability_threshold": ABSTAIN_PROBABILITY,
        "environment": {"torch": torch.__version__, "sklearn": sklearn.__version__, "cpu_threads": torch.get_num_threads()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved models and experiment 2/3 results", flush=True)


if __name__ == "__main__":
    main()
