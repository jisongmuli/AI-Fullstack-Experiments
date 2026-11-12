"""Experiments 2/3: train, physically prune, compress, calibrate and benchmark."""
from __future__ import annotations
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/models"))
from course_models import CourseCNN, image_tensor

DATA = ROOT / "data/course_20261027"
OUT = ROOT / "results/course_20261027"
MODELS = ROOT / "models/course_20261027"


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metrics(labels, scores):
    predictions = np.asarray(scores) >= .5
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {"samples": len(labels), "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")), "auroc": float(roc_auc_score(labels, scores)),
            "false_positive_rate": float(fp/max(tn+fp, 1)), "false_negative_rate": float(fn/max(fn+tp, 1)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def calibration(raw, labels):
    fitted = LogisticRegression(C=1., random_state=20261027).fit(np.array(raw).reshape(-1, 1), labels)
    return {"coefficient": float(fitted.coef_[0, 0]), "intercept": float(fitted.intercept_[0])}


def probabilities(raw, calibration):
    return 1/(1+np.exp(-np.clip(np.asarray(raw)*calibration["coefficient"]+calibration["intercept"], -60, 60)))


class Images(Dataset):
    def __init__(self, rows, config, training=False):
        self.rows, self.config, self.training = rows, config, training
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row["path"]) as image:
            value = image_tensor(image, self.config)
        if self.training and torch.rand(()) < .5:
            value = value.flip(-1)
        return value, row["label"]


def image_raw(model, loader):
    model.eval(); outputs, labels = [], []
    with torch.inference_mode():
        for values, target in loader:
            logits = model(values)
            outputs.extend((logits[:, 1]-logits[:, 0]).tolist()); labels.extend(target.tolist())
    return np.array(outputs), labels


def fit_image(model, train, validation, epochs, name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    best, state, history = -1., None, []
    for epoch in range(epochs):
        model.train(); loss_sum = 0.
        for values, labels in train:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(values), labels)
            loss.backward(); optimizer.step(); loss_sum += float(loss.detach())*len(labels)
        raw, labels = image_raw(model, validation)
        macro = f1_score(labels, raw >= 0, average="macro")
        record = {"epoch": epoch+1, "train_loss": loss_sum/len(train.dataset), "validation_macro_f1": float(macro)}
        history.append(record)
        if macro > best:
            best, state = macro, copy.deepcopy(model.state_dict())
        print(name, record, flush=True)
    model.load_state_dict(state)
    save(name + "-history.json", history)
    return model.eval()


def prune_model(model):
    compact = CourseCNN((12, 24, 48))
    previous = torch.arange(3)
    with torch.no_grad():
        for old, new, width in zip(model.blocks, compact.blocks, (12, 24, 48)):
            selected = old[0].weight.abs().sum((1, 2, 3)).topk(width).indices.sort().values
            new[0].weight.copy_(old[0].weight[selected][:, previous]); new[0].bias.copy_(old[0].bias[selected])
            for key in ("weight", "bias", "running_mean", "running_var"):
                getattr(new[1], key).copy_(getattr(old[1], key)[selected])
            new[1].num_batches_tracked.copy_(old[1].num_batches_tracked)
            previous = selected
        compact.head.weight.copy_(model.head.weight[:, previous]); compact.head.bias.copy_(model.head.bias)
    return compact


def benchmark(action):
    for _ in range(8): action()
    measurements = []
    for _ in range(40):
        start = time.perf_counter(); action(); measurements.append((time.perf_counter()-start)*1000)
    return {"median_ms": float(np.median(measurements)), "p95_ms": float(np.percentile(measurements, 95)), "runs": 40}


def main():
    random.seed(20261027); np.random.seed(20261027); torch.manual_seed(20261027); torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    images = json.loads((DATA / "images.json").read_text(encoding="utf-8"))
    loaders = {split: DataLoader(Images([row for row in images["records"] if row["split"] == split], images, split == "train"),
               batch_size=24, shuffle=split == "train", num_workers=0) for split in ("train", "validation", "test")}
    full = fit_image(CourseCNN(), loaders["train"], loaders["validation"], 15, "image-trained")
    torch.save(full.state_dict(), MODELS / "image_trained.pt")
    compact = fit_image(prune_model(full), loaders["train"], loaders["validation"], 6, "image-pruned")
    raw_full, y_val = image_raw(full, loaders["validation"]); raw_small, _ = image_raw(compact, loaders["validation"])
    accepted = f1_score(y_val, raw_small >= 0, average="macro") >= f1_score(y_val, raw_full >= 0, average="macro")-.05
    selected = compact if accepted else full
    torch.save(selected.state_dict(), MODELS / "image_optimized.pt")
    scripted = torch.jit.trace(selected, torch.zeros(1, 3, 64, 64)); torch.jit.save(scripted, str(MODELS / "image_optimized.ts"))
    image_config = {"mean": images["mean"], "std": images["std"], "trained_channels": [16, 32, 64],
                    "optimized_channels": [12, 24, 48] if accepted else [16, 32, 64]}
    image_result = {"selection_rule": "pruned validation macro-F1 loss <=5 percentage points; no test selection", "pruning_selected": bool(accepted)}
    image_predictions = []
    for stage, model, raw in (("trained", full, raw_full), ("optimized", selected, raw_small if accepted else raw_full)):
        fit = calibration(raw, y_val); image_config[stage + "_calibration"] = fit
        test_raw, labels = image_raw(model, loaders["test"])
        scores = probabilities(test_raw, fit)
        image_result[stage] = metrics(labels, scores) | {"parameters": sum(p.numel() for p in model.parameters())}
        for row, score in zip(loaders["test"].dataset.rows, scores):
            image_predictions.append({"id": row["id"], "label": row["label"], "stage": stage, "score": float(score)})
    with torch.inference_mode():
        sample = next(iter(loaders["test"]))[0][:1]
        image_result["trained"]["latency"] = benchmark(lambda: full(sample))
        image_result["optimized"]["latency"] = benchmark(lambda: scripted(sample))
        image_result["export_max_logit_difference"] = float((selected(sample)-scripted(sample)).abs().max())
    save("image-predictions.json", image_predictions)
    save("experiment2-image.json", image_result)

    text = {split: [json.loads(line) for line in (DATA / "processed/text" / (split+".jsonl")).read_text(encoding="utf-8").splitlines()]
            for split in ("train", "validation", "test")}
    def fit_text(features):
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=features, min_df=2, sublinear_tf=True)
        matrix = vectorizer.fit_transform([row["text"] for row in text["train"]])
        classifier = LogisticRegression(C=4., max_iter=500, random_state=20261027).fit(matrix, [row["label"] for row in text["train"]])
        return {"vectorizer": vectorizer, "classifier": classifier}
    def raw_text(model, split): return model["classifier"].decision_function(model["vectorizer"].transform([row["text"] for row in text[split]]))
    full_text = fit_text(12000)
    labels_val = [row["label"] for row in text["validation"]]
    full_f1 = f1_score(labels_val, raw_text(full_text, "validation") >= 0, average="macro")
    candidates = []
    for width in (2000, 4000, 8000):
        model = fit_text(width)
        score = f1_score(labels_val, raw_text(model, "validation") >= 0, average="macro")
        candidates.append((width, float(score), model))
    viable = [row for row in candidates if row[1] >= full_f1-.02]
    width, _, compact_text = viable[0] if viable else (12000, full_f1, full_text)
    joblib.dump(full_text, MODELS / "text_full.joblib", compress=3); joblib.dump(compact_text, MODELS / "text_compact.joblib", compress=3)
    text_result = {"selection_rule": "smallest vocabulary with validation macro-F1 loss <=2 percentage points",
                   "candidates": [{"features": n, "validation_macro_f1": f} for n, f, _ in candidates]}
    text_config, text_predictions = {}, []
    for stage, model in (("trained", full_text), ("optimized", compact_text)):
        fit = calibration(raw_text(model, "validation"), labels_val); text_config[stage+"_calibration"] = fit
        scores = probabilities(raw_text(model, "test"), fit)
        text_result[stage] = metrics([row["label"] for row in text["test"]], scores) | {"features": len(model["vectorizer"].vocabulary_)}
        example = [text["test"][0]["text"]]
        text_result[stage]["latency"] = benchmark(lambda: model["classifier"].decision_function(model["vectorizer"].transform(example)))
        text_predictions.extend({"id": row["id"], "label": row["label"], "score": float(score), "stage": stage} for row, score in zip(text["test"], scores))
    save("text-predictions.json", text_predictions); save("experiment2-text.json", text_result)
    config = {"version": "2026-10-27", "image": image_config, "text": text_config}
    (MODELS / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    assets = []
    for path in sorted(MODELS.iterdir()):
        with path.open("rb") as stream: digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assets.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest})
    save("experiment3.json", {"image": image_result, "text": text_result, "assets": assets,
         "pipeline": ["scripts/prepare_course_2026.py", "scripts/train_course_2026.py", "code/models/course_models.py"],
         "environment": {"torch": torch.__version__, "threads": torch.get_num_threads(), "device": "cpu"}})
    print("Completed both trained and physically compressed course models", flush=True)


if __name__ == "__main__": main()
