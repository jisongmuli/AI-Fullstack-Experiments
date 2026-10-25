"""Train and audit a Chinese AIGC screening baseline from two public corpora.

Usage from the project root: py code/models/train_chinese_text.py
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "text_chinese" / "raw"
OUT = ROOT / "results" / "chinese_text"
MODELS = ROOT / "models"
SEED = 42
LENGTH_EDGES = (80, 160, 320, 640, 1280, 2560, 4001)
ABSTAIN = 0.8
HC3_REVISION = "09a687b8dc164b89e7df95abf15df3b216bc31c2"
ANX_REVISION = "558d97ab9453a1390841105b3742dea327ff349c"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[*#`_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_question(question: str) -> str:
    bucket = int(digest(normalize(question))[:8], 16) % 100
    return "train" if bucket < 76 else "validation" if bucket < 88 else "test"


def valid_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and 80 <= len(value) <= 4000
        and "\ufffd" not in value
        and "[SEP]" not in value
        and sum("\u4e00" <= c <= "\u9fff" for c in value) >= 40
    )


def read_hc3() -> list[dict]:
    records = []
    with (RAW / "hc3_chinese_all.jsonl").open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            item = json.loads(line)
            split = split_question(item["question"])
            for label, key in ((0, "human_answers"), (1, "chatgpt_answers")):
                for answer in item.get(key) or []:
                    if not answer:
                        continue
                    text = normalize(answer)
                    if valid_text(text):
                        records.append({
                            "id": f"hc3_{index}_{key}_{len(records)}", "text": text,
                            "hash": digest(text), "label": label, "split": split,
                            "source": "hc3_" + item["source"],
                            "generator": "ChatGPT-2023" if label else "human",
                            "corpus": "HC3-Chinese",
                        })
    return records


def read_anx() -> list[dict]:
    records = []
    for split, file_name in (("train", "anx_train.csv"),
                             ("validation", "anx_val.csv"), ("test", "anx_test.csv")):
        frame = pd.read_csv(RAW / file_name, usecols=["text", "source", "label", "category"])
        selected = frame[
            frame["category"].isna()
            & ((frame["source"].eq("thucnews") & frame["label"].eq(0))
               | (frame["source"].fillna("").str.startswith("parallel_") & frame["label"].eq(1)))
        ]
        for index, item in selected.iterrows():
            if not isinstance(item.text, str):
                continue
            text = normalize(item.text)
            if valid_text(text):
                records.append({
                    "id": f"anx_{split}_{index}", "text": text,
                    "hash": digest(text), "label": int(item.label), "split": split,
                    "source": str(item.source),
                    "generator": str(item.source) if item.label else "human",
                    "corpus": "AnxForever-chinese-ai-detection-dataset",
                })
    return records


def remove_leakage(records: list[dict]) -> tuple[dict[str, list[dict]], dict]:
    labels: dict[str, set[int]] = defaultdict(set)
    for row in records:
        labels[row["hash"]].add(row["label"])
    conflicting = {key for key, values in labels.items() if len(values) > 1}
    clean: dict[str, list[dict]] = {s: [] for s in ("train", "validation", "test")}
    seen: set[str] = set()
    for split in clean:
        for row in records:
            if row["split"] == split and row["hash"] not in seen and row["hash"] not in conflicting:
                clean[split].append(row)
                seen.add(row["hash"])
    return clean, {
        "raw_candidate_rows": len(records), "contradictory_hashes_quarantined": len(conflicting),
        "duplicate_or_conflicting_rows_removed": len(records) - sum(map(len, clean.values())),
        "unique_hashes_by_split": {split: len(rows) for split, rows in clean.items()},
        "split_hash_overlap": 0,
    }


def length_bin(length: int) -> int:
    return int(np.searchsorted(LENGTH_EDGES, length, side="right") - 1)


def balance_train(rows: list[dict]) -> tuple[list[dict], dict]:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(length_bin(len(row["text"])), row["label"])].append(row)
    rng = np.random.default_rng(SEED)
    selected = []
    audit = {}
    for bin_index in range(len(LENGTH_EDGES) - 1):
        human, ai = groups[(bin_index, 0)], groups[(bin_index, 1)]
        take = min(len(human), len(ai), 6000)
        audit[f"{LENGTH_EDGES[bin_index]}-{LENGTH_EDGES[bin_index+1]-1}"] = {
            "human_before": len(human), "ai_before": len(ai), "each_selected": take,
        }
        if take:
            selected.extend(human[i] for i in rng.choice(len(human), take, replace=False))
            selected.extend(ai[i] for i in rng.choice(len(ai), take, replace=False))
    rng.shuffle(selected)
    return selected, audit


def probabilities(raw: np.ndarray, calibration: dict) -> np.ndarray:
    logit = np.clip(calibration["coefficient"] * raw + calibration["intercept"], -60, 60)
    return 1 / (1 + np.exp(-logit))


def metrics(rows: list[dict], probability: np.ndarray) -> dict:
    actual = np.asarray([r["label"] for r in rows])
    predicted = (probability >= .5).astype(int)
    covered = (probability <= 1 - ABSTAIN) | (probability >= ABSTAIN)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    return {
        "n": len(rows), "counts": dict(Counter(map(str, actual))),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "ai_precision": float(precision_score(actual, predicted, zero_division=0)),
        "ai_recall": float(recall_score(actual, predicted, zero_division=0)),
        "human_false_positive_rate": float(matrix[0, 1] / max(1, matrix[0].sum())),
        "roc_auc": float(roc_auc_score(actual, probability)) if len(set(actual)) == 2 else None,
        "brier": float(brier_score_loss(actual, probability)),
        "confusion_matrix_human_ai": matrix.tolist(),
        "abstention_at_0.8": {
            "coverage": float(covered.mean()),
            "covered_accuracy": float(accuracy_score(actual[covered], predicted[covered])) if covered.any() else None,
            "covered_count": int(covered.sum()),
        },
    }


def sha256_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    rows, audit = remove_leakage(read_hc3() + read_anx())
    train, balancing = balance_train(rows["train"])
    print("clean split sizes", {k: len(v) for k, v in rows.items()}, flush=True)
    print("balanced training rows", len(train), flush=True)
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 4), max_features=180_000,
        min_df=3, sublinear_tf=True, dtype=np.float32,
    )
    train_x = vectorizer.fit_transform([r["text"] for r in train])
    classifier = LogisticRegression(C=2.0, max_iter=1000, solver="liblinear", random_state=SEED)
    classifier.fit(train_x, [r["label"] for r in train])
    del train_x
    print("vocabulary", len(vectorizer.vocabulary_), flush=True)

    def raw_scores(items: list[dict]) -> np.ndarray:
        return classifier.decision_function(vectorizer.transform([r["text"] for r in items]))

    validation_raw = raw_scores(rows["validation"])
    calibrator = LogisticRegression(C=100.0, max_iter=1000, random_state=SEED)
    calibrator.fit(validation_raw.reshape(-1, 1), [r["label"] for r in rows["validation"]])
    calibration = {
        "coefficient": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
        "method": "Platt sigmoid fit on validation only",
    }
    evaluation = {}
    for split in ("validation", "test"):
        items = rows[split]
        risk = probabilities(validation_raw if split == "validation" else raw_scores(items), calibration)
        evaluation[split] = metrics(items, risk)
        frame = pd.DataFrame({
            "id": [r["id"] for r in items], "corpus": [r["corpus"] for r in items],
            "source": [r["source"] for r in items], "label": [r["label"] for r in items],
            "length": [len(r["text"]) for r in items], "risk_score": risk,
        })
        frame.to_csv(OUT / f"{split}_predictions.csv", index=False, encoding="utf-8-sig")
        if split == "test":
            evaluation["test_hc3"] = metrics(
                [r for r in items if r["corpus"] == "HC3-Chinese"],
                risk[frame.corpus.eq("HC3-Chinese").to_numpy()],
            )
            evaluation["test_anx_pure"] = metrics(
                [r for r in items if r["corpus"] != "HC3-Chinese"],
                risk[frame.corpus.ne("HC3-Chinese").to_numpy()],
            )
            long_mask = (frame.length >= 1000).to_numpy()
            if long_mask.any():
                evaluation["test_long_ge_1000_chars"] = metrics(
                    [r for r, keep in zip(items, long_mask) if keep], risk[long_mask],
                )
            for generator in ("parallel_Kimi-K2", "parallel_cursor2-gpt-5"):
                mask = (frame.source.eq(generator) | frame.source.eq("thucnews")).to_numpy()
                evaluation[f"test_{generator}_vs_news"] = metrics(
                    [r for r, keep in zip(items, mask) if keep], risk[mask],
                )
    model_path = MODELS / "text_detector_zh.joblib"
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, model_path, compress=3)
    config = {
        "model_version": "zh-char-tfidf-lr-v1", "language": "Simplified Chinese",
        "scikit_learn_version": sklearn.__version__,
        "training_corpora": ["HC3-Chinese", "AnxForever non-HC3 pure text"],
        "character_range_evaluated": [80, 4000],
        "abstain_probability": ABSTAIN, "calibration": calibration,
        "limitations": "QA and news/generated longform only. Human fiction, newer generators, paraphrases and platform traffic lack independent blind evaluation. Score is not an authorship determination.",
    }
    (MODELS / "text_zh_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "datasets": [
            {"name": "Hello-SimpleAI/HC3-Chinese", "revision": HC3_REVISION,
             "url": "https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese",
             "license_declared": "CC BY-SA 4.0, underlying sources may have stricter terms",
             "file": "hc3_chinese_all.jsonl", "sha256": sha256_file(RAW / "hc3_chinese_all.jsonl")},
            {"name": "AnxForever/chinese-ai-detection-dataset", "revision": ANX_REVISION,
             "url": "https://huggingface.co/datasets/AnxForever/chinese-ai-detection-dataset",
             "license_declared": "MIT; derived-source licensing needs independent review",
             "files": {name: sha256_file(RAW / name) for name in ("anx_train.csv", "anx_val.csv", "anx_test.csv")}},
        ],
        "filter": "Pure human/AI only; Anx HC3 copies and mixed labels excluded; 80-4000 Chinese characters; U+FFFD excluded",
        "splitting": "HC3 by question hash; Anx original splits with global exact normalized-text deduplication",
        "risk": "Anx human longform is mainly THUCNews, while AI longform uses parallel models; genre/source confounding remains. Question-level and exact-match leakage removed, but prompt-near-duplicates are not proven absent.",
        "audit": audit, "length_balance": balancing,
        "training_counts": dict(Counter((r["corpus"], r["label"]) for r in train)),
        "artifact_sha256": sha256_file(model_path),
    }
    # Convert tuple-key counters to stable JSON keys.
    manifest["training_counts"] = {f"{corpus}|{label}": count for (corpus, label), count in manifest["training_counts"].items()}
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "seed": SEED, "algorithm": "character 2-4gram TF-IDF + L2 logistic regression",
        "scikit_learn_version": sklearn.__version__,
        "train_n": len(train), "vocabulary": len(vectorizer.vocabulary_),
        "calibration": calibration, "scores": evaluation,
        "training_seconds": round(time.perf_counter() - started, 2),
        "model_bytes": model_path.stat().st_size,
        "warning": manifest["risk"],
    }
    (OUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("test", evaluation["test"], flush=True)
    print("long", evaluation.get("test_long_ge_1000_chars"), flush=True)
    print("seconds", result["training_seconds"], flush=True)


if __name__ == "__main__":
    main()
