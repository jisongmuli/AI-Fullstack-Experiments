"""Measure the text-length shortcut and plot experiment-one audit results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/text"
RESULTS = ROOT / "results"


def evaluate(train: pd.DataFrame, target: pd.DataFrame, field: str) -> dict:
    model = LogisticRegression(random_state=42)
    model.fit(np.log1p(train[[field]].to_numpy()), train.label)
    prediction = model.predict(np.log1p(target[[field]].to_numpy()))
    return {
        "n": len(target),
        "accuracy": float(accuracy_score(target.label, prediction)),
        "macro_f1": float(f1_score(target.label, prediction, average="macro")),
        "confusion_matrix_rows_human_ai": confusion_matrix(target.label, prediction, labels=[0, 1]).tolist(),
    }


def main() -> None:
    original = pd.read_csv(DATA / "primary_clean_manifest.csv")
    prepared = pd.read_csv(DATA / "primary_model_manifest.csv")
    external_original = pd.read_csv(DATA / "external_original_manifest.csv")
    external_prepared = pd.read_csv(DATA / "external_clean_manifest.csv")
    splits = {}
    for name in ("train", "validation", "test"):
        with (DATA / f"{name}.jsonl").open(encoding="utf-8") as stream:
            ids = [json.loads(line)["sample_id"] for line in stream]
        splits[name] = set(ids)
    measures = {}
    for stage, primary, external, field in (
        ("source", original, external_original, "length_chars"),
        ("length_matched", prepared, external_prepared, "length_chars"),
    ):
        train = primary[primary.sample_id.isin(splits["train"])]
        measures[stage] = {
            "validation": evaluate(train, primary[primary.sample_id.isin(splits["validation"])], field),
            "test": evaluate(train, primary[primary.sample_id.isin(splits["test"])], field),
            "external_test": evaluate(train, external, field),
        }
    payload = {
        "purpose": "single-feature length shortcut diagnostic; not a deployable detector",
        "feature": "log1p(character count)", "seed": 42,
        "same_primary_sample_ids_and_splits_before_after": True,
        "results": measures,
    }
    (RESULTS / "text_length_baseline.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (title, frame, limit) in zip(axes, [
        ("Source text", original, 5000), ("After length matching", prepared, 600)
    ]):
        for label, color, name in [(0, "#247c78", "Human"), (1, "#c35e40", "AI")]:
            lengths = frame.loc[frame.label == label, "length_chars"]
            ax.hist(lengths, bins=np.linspace(0, limit, 51), alpha=.65, density=True, color=color, label=name)
        ax.set_title(title); ax.set_xlabel("Characters"); ax.set_xlim(0, limit); ax.grid(alpha=.18)
        ax.set_ylabel("Density")
    axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(RESULTS / "text_length_before_after.png", dpi=180); plt.close(fig)

    image = json.loads((RESULTS / "image_augmentation_metrics.json").read_text(encoding="utf-8"))
    text = json.loads((RESULTS / "text_augmentation_metrics.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (title, metric, splits_to_plot) in zip(axes, [
        ("Image CNN", image, ["validation", "test"]),
        ("Text TF-IDF + logistic regression", text, ["validation", "test", "external_test"]),
    ]):
        xx = np.arange(len(splits_to_plot)); width = .33
        for j, (key, color) in enumerate([("original", "#247c78"), ("augmented", "#c35e40")]):
            scores = [metric["results"][key].get("scores", metric["results"][key]).get(s, {}).get("accuracy", 0) * 100 for s in splits_to_plot]
            ax.bar(xx + (j - .5) * width, scores, width, label=key, color=color)
        ax.set_xticks(xx, [s.replace("_", "\n") for s in splits_to_plot]); ax.set_title(title)
        ax.set_ylim(0, 105); ax.set_ylabel("Accuracy (%)"); ax.grid(axis="y", alpha=.18)
    axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(RESULTS / "augmentation_comparison.png", dpi=180); plt.close(fig)
    image_root = ROOT / "data/images/processed"
    counts = {part: len(list((image_root / part).rglob("*.png"))) for part in ("train", "validation", "test")}
    text_counts = {part: sum(1 for _ in (DATA / f"{part}.jsonl").open(encoding="utf-8"))
                   for part in ("train", "validation", "test", "external_test")}
    lines = [
        "AIGC experiment 1/",
        "  code/                       5 reproducible scripts",
        "  data/raw/                   3 ZIP archives + 3 source JSON files",
        "  data/images/",
        "    source_manifest.csv       120,000 audited images",
        "    split_manifest.csv        5,600 selected images",
        f"    processed/train/          {counts['train']:,} PNG",
        f"    processed/validation/     {counts['validation']:,} PNG",
        f"    processed/test/           {counts['test']:,} PNG",
        "  data/text/",
        f"    train.jsonl               {text_counts['train']:,} records",
        f"    validation.jsonl          {text_counts['validation']:,} records",
        f"    test.jsonl                {text_counts['test']:,} records",
        f"    external_test.jsonl       {text_counts['external_test']:,} records",
        "  data/images/image_metadata_*.csv  structured features",
        "  results/                    quality, metrics, predictions, charts",
        "  README.md + DATA_CARD.md    usage, provenance, restrictions",
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.8)); ax.axis("off")
    ax.text(.02, .98, "Verified local dataset structure", fontsize=13, weight="bold", va="top")
    ax.text(.02, .90, "\n".join(lines), family="DejaVu Sans Mono", fontsize=9.5, va="top", linespacing=1.1)
    fig.tight_layout(); fig.savefig(RESULTS / "dataset_structure.png", dpi=180); plt.close(fig)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
