"""Audit CIFAKE and text archives; build a small, reproducible experiment-one dataset."""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

SEED = 42
LABELS = {"REAL": 0, "FAKE": 1}
IMAGE_SIZES = {"train": 2000, "validation": 400, "test": 400}  # per class
REPLACEMENTS = {
    " important ": " significant ",
    " various ": " diverse ",
    " however ": " nevertheless ",
    " shows ": " demonstrates ",
}


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def image_audit(root: Path) -> dict:
    started = time.perf_counter()
    archive = root / "data/raw/cifake.zip"
    data_dir = root / "data/images"
    result_dir = root / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    records = []
    first_by_pixel = {}
    faulty = []
    with ZipFile(archive) as zf:
        members = sorted(zf.namelist(), key=lambda s: (0 if s.startswith("train/") else 1, s))
        for number, name in enumerate(members):
            pieces = name.split("/")
            if len(pieces) != 3 or pieces[0] not in ("train", "test") or pieces[1] not in LABELS:
                continue
            content = zf.read(name)
            try:
                with Image.open(io.BytesIO(content)) as source:
                    source.load()
                    width, height, image_format = source.width, source.height, source.format
                    rgb = source.convert("RGB")
                    array = np.asarray(rgb, dtype=np.uint8)
                pixel_hash = hashlib.sha256(array.tobytes()).hexdigest()
                brightness = float(array.mean())
                contrast = float(array.std())
                error = ""
            except Exception as exc:
                width = height = 0
                image_format = "unknown"
                pixel_hash = ""
                brightness = contrast = float("nan")
                error = f"{type(exc).__name__}: {exc}"
                faulty.append(name)
            duplicate_of = first_by_pixel.get(pixel_hash, "") if pixel_hash else ""
            if pixel_hash and not duplicate_of:
                first_by_pixel[pixel_hash] = name
            records.append({
                "sample_id": f"img_{number:06d}", "source_path": name,
                "source_split": pieces[0], "source_class": pieces[1],
                "label": LABELS[pieces[1]], "width": width, "height": height,
                "format": image_format, "file_bytes": len(content),
                "brightness": brightness, "contrast": contrast,
                "file_sha256": hashlib.sha256(content).hexdigest(),
                "pixel_sha256": pixel_hash, "duplicate_of": duplicate_of,
                "error": error,
            })
    manifest = pd.DataFrame(records)
    manifest.to_csv(data_dir / "source_manifest.csv", index=False, encoding="utf-8-sig")
    usable = manifest[(manifest.error == "") & (manifest.duplicate_of == "")].copy()
    # Preserve the public source's held-out test pool; validation is sampled from its train pool.
    random = np.random.default_rng(SEED)
    chosen = []
    for label in (0, 1):
        source_train = usable[(usable.source_split == "train") & (usable.label == label)]
        source_test = usable[(usable.source_split == "test") & (usable.label == label)]
        if len(source_train) < IMAGE_SIZES["train"] + IMAGE_SIZES["validation"] or len(source_test) < IMAGE_SIZES["test"]:
            raise RuntimeError("Not enough usable images after duplicate screening")
        indices = random.permutation(source_train.index.to_numpy())
        chosen.extend((index, "validation") for index in indices[:IMAGE_SIZES["validation"]])
        chosen.extend((index, "train") for index in indices[IMAGE_SIZES["validation"]:IMAGE_SIZES["validation"] + IMAGE_SIZES["train"]])
        indices = random.permutation(source_test.index.to_numpy())
        chosen.extend((index, "test") for index in indices[:IMAGE_SIZES["test"]])
    selected = manifest.loc[[index for index, _ in chosen]].copy()
    selected["split"] = [split for _, split in chosen]
    selected = selected.sort_values(["split", "label", "source_path"])
    selected["relative_path"] = selected.apply(
        lambda r: f"data/images/processed/{r['split']}/{r['label']}_{r['source_class'].lower()}/{r['sample_id']}.png", axis=1
    )
    with ZipFile(archive) as zf:
        for row in selected.itertuples():
            target = root / row.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(io.BytesIO(zf.read(row.source_path))) as source:
                # Source is already 32 x 32; upscaling would not add information.
                source.convert("RGB").save(target, format="PNG", optimize=True)
    selected.to_csv(data_dir / "split_manifest.csv", index=False, encoding="utf-8-sig")
    # Structured image metadata is the third data form. Fit scaler only on training images.
    cols = ["brightness", "contrast", "file_bytes"]
    metadata = selected[["sample_id", "relative_path", "split", "label", "source_path", *cols]].copy()
    metadata["bytes_per_pixel"] = metadata.file_bytes / (selected.width * selected.height)
    scaler = MinMaxScaler(clip=True)
    train_mask = metadata.split == "train"
    scaled_cols = [*cols, "bytes_per_pixel"]
    scaler.fit(metadata.loc[train_mask, scaled_cols])
    for index, col in enumerate(scaled_cols):
        metadata[col + "_scaled"] = scaler.transform(metadata[scaled_cols])[:, index]
    metadata.to_csv(data_dir / "image_metadata.csv", index=False, encoding="utf-8-sig")
    for split in ("train", "validation", "test"):
        metadata[metadata.split == split].to_csv(data_dir / f"image_metadata_{split}.csv", index=False, encoding="utf-8-sig")
    save_json(result_dir / "image_scaler_parameters.json", {
        "features": scaled_cols, "fit_split": "train", "clip": True,
        "data_min": scaler.data_min_.tolist(), "data_max": scaler.data_max_.tolist(),
    })
    metric = {
        "source_records": len(manifest),
        "source_split_counts": {f"{split}_{label}": int(count) for (split, label), count in manifest.groupby(["source_split", "source_class"]).size().items()},
        "decode_failures": int((manifest.error != "").sum()),
        "exact_duplicate_pixels": int((manifest.duplicate_of != "").sum()),
        "cross_source_split_duplicates": int(((manifest.source_split == "test") & manifest.duplicate_of.str.startswith("train/")).sum()),
        "within_test_duplicate_pixels": int(((manifest.source_split == "test") & manifest.duplicate_of.str.startswith("test/")).sum()),
        "exact_duplicate_files": int(manifest.file_sha256.duplicated().sum()),
        "non_32x32_images": int(((manifest.width != 32) | (manifest.height != 32)).sum()),
        "source_formats": manifest["format"].value_counts().to_dict(),
        "usable_records": len(usable),
        "processed_records": len(selected),
        "processed_split_counts": {f"{split}_{label}": int(count) for (split, label), count in selected.groupby(["split", "label"]).size().items()},
        "processed_image_size": "32x32 RGB PNG",
        "processed_missing_cells": int(metadata.isna().sum().sum()),
        "sampled_pixel_hash_overlap": int(selected.pixel_sha256.duplicated().sum()),
        "source_archive_sha256": json.loads((root / "data/raw/cifake_source.json").read_text(encoding="utf-8"))["archive_sha256"],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    save_json(result_dir / "image_quality.json", metric)
    return metric


def clean_text(text: object) -> str:
    if pd.isna(text):
        return ""
    value = html.unescape(str(text))
    value = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def match_human_length(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Take one contiguous human excerpt per source item, matching AI word counts."""
    prepared = frame.copy()
    prepared["original_length_chars"] = prepared.text.str.len()
    prepared["original_word_count"] = prepared.text.str.split().map(len)
    ai_word_counts = prepared.loc[prepared.label == 1, "original_word_count"].to_numpy()
    human_indices = prepared.index[prepared.label == 0]
    rng = np.random.default_rng(seed)
    targets = rng.choice(ai_word_counts, size=len(human_indices), replace=len(human_indices) > len(ai_word_counts))
    prepared["excerpt_start_word"] = 0
    for index, target in zip(human_indices, targets):
        words = prepared.at[index, "text"].split()
        take = min(len(words), int(target))
        start = int(rng.integers(0, len(words) - take + 1))
        prepared.at[index, "text"] = " ".join(words[start:start + take])
        prepared.at[index, "excerpt_start_word"] = start
    prepared["text_sha256"] = prepared.text.map(lambda x: hashlib.sha256(x.casefold().encode("utf-8")).hexdigest())
    prepared["length_chars"] = prepared.text.str.len()
    prepared["word_count"] = prepared.text.str.split().map(len)
    return prepared


def export_jsonl(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in frame.itertuples():
            stream.write(json.dumps({
                "sample_id": row.sample_id, "text": row.text, "label": int(row.label),
                "source_dataset": row.source_dataset,
            }, ensure_ascii=False) + "\n")


def text_audit(root: Path) -> dict:
    started = time.perf_counter()
    raw = root / "data/raw"
    data = root / "data/text"
    data.mkdir(parents=True, exist_ok=True)
    with ZipFile(raw / "text_primary.zip") as zf:
        source = pd.read_csv(io.BytesIO(zf.read(zf.namelist()[0])))
    actual_columns = list(source.columns)
    work = pd.DataFrame({
        "source_index": np.arange(len(source)),
        "text": source["Text"].map(clean_text),
        "label": source["Author"].map({"Human": 0, "AI": 1}),
    })
    work["source_dataset"] = "hasanyiitakbulut/ai-and-human-text-dataset"
    work["sample_id"] = work.source_index.map(lambda x: f"txt_{x:05d}")
    work["text_sha256"] = work.text.map(lambda x: hashlib.sha256(x.casefold().encode("utf-8")).hexdigest())
    work["length_chars"] = work.text.str.len()
    blank = int((work.text == "").sum())
    invalid_label = int(work.label.isna().sum())
    duplicates = int(work.text_sha256.duplicated().sum())
    too_short = int((work.length_chars < 40).sum())
    # Quarantine exact text duplicates; contradictory labels cannot be trusted.
    clean_source = work[(work.text != "") & work.label.notna() & (work.length_chars >= 40)].drop_duplicates("text_sha256").copy()
    clean_source["label"] = clean_source.label.astype(int)
    clean_source.to_csv(data / "primary_clean_manifest.csv", index=False, encoding="utf-8-sig")
    prepared = match_human_length(clean_source, SEED)
    new_duplicates = int(prepared.text_sha256.duplicated().sum())
    prepared = prepared.drop_duplicates("text_sha256").copy()
    prepared.to_csv(data / "primary_model_manifest.csv", index=False, encoding="utf-8-sig")
    train_val, test = train_test_split(prepared, test_size=0.1, stratify=prepared.label, random_state=SEED)
    train, validation = train_test_split(train_val, test_size=1/9, stratify=train_val.label, random_state=SEED)
    partitions = {"train": train, "validation": validation, "test": test}
    for name, frame in partitions.items():
        export_jsonl(frame, data / f"{name}.jsonl")
    assert len(set(train.text_sha256) & set(validation.text_sha256)) == 0
    assert len(set(train.text_sha256) & set(test.text_sha256)) == 0
    assert len(set(validation.text_sha256) & set(test.text_sha256)) == 0
    with ZipFile(raw / "text_external.zip") as zf:
        external_source = pd.read_csv(io.BytesIO(zf.read(zf.namelist()[0])))
    external = pd.DataFrame({
        "source_index": np.arange(len(external_source)),
        "text": external_source["text"].map(clean_text),
        "label": pd.to_numeric(external_source["generated"], errors="coerce"),
    })
    external["source_dataset"] = "navjotkaushal/human-vs-ai-generated-essays"
    external["sample_id"] = external.source_index.map(lambda x: f"ext_{x:05d}")
    external["text_sha256"] = external.text.map(lambda x: hashlib.sha256(x.casefold().encode("utf-8")).hexdigest())
    external["length_chars"] = external.text.str.len()
    external_before = external.copy()
    external = external[(external.text != "") & external.label.isin([0, 1]) & (external.length_chars >= 40)].drop_duplicates("text_sha256")
    overlap = int(external.text_sha256.isin(set(clean_source.text_sha256)).sum())
    external = external[~external.text_sha256.isin(set(clean_source.text_sha256))].copy()
    external["label"] = external.label.astype(int)
    external.to_csv(data / "external_original_manifest.csv", index=False, encoding="utf-8-sig")
    external = match_human_length(external, SEED + 1).drop_duplicates("text_sha256").copy()
    external.to_csv(data / "external_clean_manifest.csv", index=False, encoding="utf-8-sig")
    export_jsonl(external, data / "external_test.jsonl")
    metric = {
        "source_rows": len(source), "actual_source_columns": actual_columns,
        "source_description_claimed_columns": ["text", "label", "source"],
        "per_record_source_available": False,
        "source_class_counts": source.Author.value_counts().to_dict(),
        "blank_text_before": blank, "invalid_label_before": invalid_label,
        "exact_duplicate_text_before": duplicates, "under_40_characters_before": too_short,
        "clean_rows": len(clean_source),
        "prepared_rows": len(prepared),
        "prepared_duplicate_text_removed": new_duplicates,
        "split_counts": {name: {"rows": len(frame), "human": int((frame.label == 0).sum()), "ai": int((frame.label == 1).sum())} for name, frame in partitions.items()},
        "source_median_length_chars_by_label": {str(k): float(v) for k,v in clean_source.groupby("label").length_chars.median().items()},
        "prepared_median_length_chars_by_label": {str(k): float(v) for k,v in prepared.groupby("label").length_chars.median().items()},
        "source_median_word_count_by_label": {str(k): float(v) for k,v in clean_source.assign(words=clean_source.text.str.split().map(len)).groupby("label").words.median().items()},
        "prepared_median_word_count_by_label": {str(k): float(v) for k,v in prepared.groupby("label").word_count.median().items()},
        "prepared_length_chars": {"min": int(prepared.length_chars.min()), "median": float(prepared.length_chars.median()), "max": int(prepared.length_chars.max())},
        "human_excerpts": int((prepared.label == 0).sum()),
        "external_source_rows": len(external_source),
        "external_source_exact_duplicates": int(external_before.text_sha256.duplicated().sum()),
        "external_source_median_length_chars_by_label": {str(k): float(v) for k,v in external_before.groupby("label").length_chars.median().items()},
        "external_prepared_median_length_chars_by_label": {str(k): float(v) for k,v in external.groupby("label").length_chars.median().items()},
        "external_clean_rows": len(external),
        "external_overlap_removed": overlap,
        "external_class_counts": {"human": int((external.label == 0).sum()), "ai": int((external.label == 1).sum())},
        "external_per_record_source_available": False,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    save_json(root / "results/text_quality.json", metric)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.text_only:
        image = image_audit(root)
        print("image", image["source_records"], image["processed_records"], image["elapsed_seconds"], flush=True)
    text = text_audit(root)
    print("text", text["source_rows"], text["clean_rows"], text["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
