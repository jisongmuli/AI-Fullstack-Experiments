"""Fixed image/text probes for experiment-one augmentation comparisons."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from PIL import Image
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SEED = 42
SYNONYMS = {
    "important": "significant", "various": "diverse", "however": "nevertheless",
    "shows": "demonstrates", "use": "employ", "many": "numerous",
    "help": "assist", "different": "distinct", "because": "since",
}


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def scores(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    human_count = int((y_true == 0).sum())
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "human_false_positive_rate": float(matrix[0, 1] / human_count) if human_count else None,
        "confusion_matrix_rows_human_ai": matrix.tolist(),
    }


class ImageDataset(Dataset):
    def __init__(self, root: Path, frame: pd.DataFrame, augmentation: bool):
        self.root = root
        self.frame = frame.reset_index(drop=True)
        steps = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(8)] if augmentation else []
        self.transform = transforms.Compose([*steps, transforms.ToTensor()])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.root / row.relative_path) as source:
            tensor = self.transform(source.convert("RGB"))
        return tensor, int(row.label)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Linear(64 * 8 * 8, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.classifier(self.features(x))


def predict_cnn(model: nn.Module, loader: DataLoader):
    model.eval()
    actual, prediction = [], []
    with torch.inference_mode():
        for images, labels in loader:
            result = model(images)
            actual.extend(labels.tolist())
            prediction.extend(result.argmax(1).tolist())
    return actual, prediction


def image_experiment(root: Path) -> dict:
    seed_all()
    manifest = pd.read_csv(root / "data/images/split_manifest.csv")
    val = manifest[manifest.split == "validation"]
    test = manifest[manifest.split == "test"]
    val_loader = DataLoader(ImageDataset(root, val, False), batch_size=128, shuffle=False)
    test_loader = DataLoader(ImageDataset(root, test, False), batch_size=128, shuffle=False)
    results = {}
    for label, augment in (("original", False), ("augmented", True)):
        seed_all()
        train = manifest[manifest.split == "train"]
        train_loader = DataLoader(ImageDataset(root, train, augment), batch_size=128, shuffle=True,
                                  generator=torch.Generator().manual_seed(SEED), num_workers=0)
        model = SmallCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        best = None
        history = []
        started = time.perf_counter()
        for epoch in range(1, 4):
            model.train()
            total_loss = 0.0
            for images, targets in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(images), targets)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(images)
            y, pred = predict_cnn(model, val_loader)
            val_scores = scores(y, pred)
            history.append({"epoch": epoch, "train_loss": total_loss / len(train),
                            "validation_accuracy": val_scores["accuracy"],
                            "validation_macro_f1": val_scores["macro_f1"]})
            if best is None or val_scores["macro_f1"] > best[0]:
                best = (val_scores["macro_f1"], epoch, copy.deepcopy(model.state_dict()))
            print("image", label, "epoch", epoch, "val", round(val_scores["accuracy"], 4), flush=True)
        model.load_state_dict(best[2])
        torch.save({"state_dict": model.state_dict(), "seed": SEED, "best_epoch": best[1],
                    "classes": {0: "human", 1: "ai"}}, root / "results" / f"image_probe_{label}.pt")
        vy, vp = predict_cnn(model, val_loader)
        ty, tp = predict_cnn(model, test_loader)
        results[label] = {
            "augmentation": "RandomHorizontalFlip(0.5) + RandomRotation(8 degrees) on training only" if augment else "none",
            "epochs": 3, "best_epoch_by_validation_macro_f1": best[1],
            "history": history, "validation": scores(vy, vp), "test": scores(ty, tp),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        pd.DataFrame({"sample_id": test.sample_id, "actual": ty, "prediction": tp}).to_csv(
            root / "results" / f"image_probe_{label}_test_predictions.csv", index=False, encoding="utf-8-sig")
    payload = {
        "purpose": "small fixed experiment-one probe, not deployment model",
        "seed": SEED, "train_count": int((manifest.split == "train").sum()),
        "validation_count": len(val), "test_count": len(test),
        "architecture": "3-convolution small CNN with 64-unit classifier",
        "training": "Adam lr=0.001, batch=128, 3 epochs; best epoch selected only on validation macro F1",
        "results": results, "torch_version": torch.__version__,
    }
    (root / "results/image_augmentation_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def replace_once(text: str) -> str | None:
    for before, after in SYNONYMS.items():
        match = re.search(rf"\b{before}\b", text, flags=re.IGNORECASE)
        if match:
            # Alter one conservative lexical item only; meaning still needs manual review.
            replacement = after.capitalize() if match.group(0)[0].isupper() else after
            return text[:match.start()] + replacement + text[match.end():]
    return None


def text_experiment(root: Path) -> dict:
    seed_all()
    data = root / "data/text"
    train = read_jsonl(data / "train.jsonl")
    validation = read_jsonl(data / "validation.jsonl")
    test = read_jsonl(data / "test.jsonl")
    external = read_jsonl(data / "external_test.jsonl")
    rng = np.random.default_rng(SEED)
    eligible = [i for i, record in enumerate(train) if replace_once(record["text"]) is not None]
    chosen = rng.choice(eligible, size=min(round(len(train) * .25), len(eligible)), replace=False)
    augmented = []
    bleu = []
    smoother = SmoothingFunction().method1
    for index in chosen:
        original = train[int(index)]
        replaced = replace_once(original["text"])
        assert replaced and replaced != original["text"]
        item = dict(original)
        item["sample_id"] = "aug_" + original["sample_id"]
        item["text"] = replaced
        item["augmented_from"] = original["sample_id"]
        augmented.append(item)
        bleu.append(float(sentence_bleu([original["text"].split()], replaced.split(),
                                        smoothing_function=smoother)))
    with (data / "train_augmented.jsonl").open("w", encoding="utf-8") as stream:
        for item in augmented:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    results = {}
    for name, training in (("original", train), ("augmented", train + augmented)):
        started = time.perf_counter()
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000,
                                     min_df=2, stop_words="english", sublinear_tf=True)
        features = vectorizer.fit_transform([x["text"] for x in training])
        model = LogisticRegression(max_iter=1000, random_state=SEED)
        model.fit(features, [x["label"] for x in training])
        evaluations = {}
        for split_name, records in (("validation", validation), ("test", test), ("external_test", external)):
            prediction = model.predict(vectorizer.transform([x["text"] for x in records]))
            evaluations[split_name] = scores([x["label"] for x in records], prediction)
            if split_name in ("test", "external_test"):
                pd.DataFrame({"sample_id": [x["sample_id"] for x in records],
                              "actual": [x["label"] for x in records],
                              "prediction": prediction}).to_csv(
                    root / "results" / f"text_probe_{name}_{split_name}_predictions.csv", index=False, encoding="utf-8-sig")
        results[name] = {"train_count": len(training), "vocabulary_size": len(vectorizer.vocabulary_),
                         "scores": evaluations, "elapsed_seconds": round(time.perf_counter() - started, 2)}
        print("text", name, "val", round(evaluations["validation"]["accuracy"], 4),
              "external", round(evaluations["external_test"]["accuracy"], 4), flush=True)
    payload = {
        "purpose": "fixed experiment-one augmentation probe, not authorship proof",
        "seed": SEED,
        "augmentation": "one curated English synonym replacement in 25% of original training texts; validation/test untouched",
        "augmented_samples": len(augmented),
        "mean_sentence_bleu_vs_original": float(np.mean(bleu)) if bleu else None,
        "results": results,
        "sklearn_version": sklearn.__version__,
        "limitations": ["Primary CSV has no per-record source column despite its web description.",
                        "External corpus provenance is incomplete and may differ strongly by topic.",
                        "Simple synonym swaps can alter style and meaning; this is a controlled teaching comparison."],
    }
    (root / "results/text_augmentation_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    image_experiment(root)
    text_experiment(root)


if __name__ == "__main__":
    main()
