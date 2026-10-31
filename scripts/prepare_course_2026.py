"""Experiment 1: fresh public 2026 image sample and grouped Chinese text splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageOps
from fetch_ntire_2026_sample import RemoteZip, REVISION, URL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/models"))
from course_preprocessing import canonical_text as canonical
DATA = ROOT / "data/course_20261027"
RESULTS = ROOT / "results/course_20261027"
SEED = 20261027


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_images(proxy):
    manifest = DATA / "images.json"
    if manifest.exists():
        print("Existing fresh image split retained", flush=True)
        return json.loads(manifest.read_text(encoding="utf-8"))
    previous = json.loads((ROOT / "data/benchmarks_2026/ntire/manifest.json").read_text(encoding="utf-8"))
    excluded = {row["id"] for row in previous["records"]}
    with httpx.Client(proxy=proxy, timeout=120, follow_redirects=True, trust_env=False) as client:
        remote = RemoteZip(client)
        with zipfile.ZipFile(remote) as archive:
            entries = archive.infolist()
        label_info = next(row for row in entries if row.filename.endswith("labels.csv"))
        labels = list(csv.DictReader(io.StringIO(remote.member(label_info).decode("utf-8-sig"))))
        name_key = next(key for key in labels[0] if "name" in key.lower() or "file" in key.lower())
        label_key = next(key for key in labels[0] if "label" in key.lower())
        images = {Path(row.filename).name: row for row in entries if row.filename.lower().endswith(".jpg")}
        selected = []
        rng = random.Random(SEED)
        for label in (0, 1):
            candidates = sorted([row for row in labels if int(row[label_key]) == label and Path(row[name_key]).name not in excluded], key=lambda row: row[name_key])
            selected.extend(rng.sample(candidates, 210))

        def fetch(row):
            name = Path(row[name_key]).name
            info = images[name]
            target = DATA / "raw/images" / name
            if target.exists():
                value = target.read_bytes()
            else:
                value = remote.member(info)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(value)
            with Image.open(io.BytesIO(value)) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                size = list(rgb.size)
                thumb = rgb.resize((64, 64), Image.Resampling.LANCZOS)
                pixels = thumb.tobytes()
                processed = DATA / "processed/images" / (Path(name).stem + ".png")
                processed.parent.mkdir(parents=True, exist_ok=True)
                thumb.save(processed)
            return {"id": name, "label": int(row[label_key]), "path": processed.relative_to(ROOT).as_posix(),
                    "raw_path": target.relative_to(ROOT).as_posix(), "original_size": size,
                    "sha256": hashlib.sha256(value).hexdigest(), "pixel_sha256": hashlib.sha256(pixels).hexdigest(),
                    "source_member": info.filename, "bytes": len(value)}
        with ThreadPoolExecutor(max_workers=5) as pool:
            downloaded = list(pool.map(fetch, selected))
        transferred = remote.transferred
    seen, unique = set(), []
    for row in downloaded:
        if row["pixel_sha256"] not in seen:
            unique.append(row); seen.add(row["pixel_sha256"])
    records = []
    for label in (0, 1):
        group = sorted([row for row in unique if row["label"] == label], key=lambda row: row["id"])
        if len(group) < 200:
            raise RuntimeError("Not enough unique image records")
        rng.shuffle(group)
        for index, row in enumerate(group[:200]):
            row["split"] = "train" if index < 120 else "validation" if index < 160 else "test"
            records.append(row)
    arrays = [np.asarray(Image.open(ROOT / row["path"]), dtype=np.float64) / 255 for row in records if row["split"] == "train"]
    pixels = np.stack(arrays)
    result = {"dataset": "NTIRE 2026 official public training shard", "revision": REVISION, "source": URL,
              "seed": SEED, "excluded_prior_audit_ids": len(excluded), "records": records,
              "mean": pixels.mean(axis=(0, 1, 2)).tolist(), "std": pixels.std(axis=(0, 1, 2)).clip(.05).tolist(),
              "split_counts": {split: sum(row["split"] == split for row in records) for split in ("train", "validation", "test")},
              "processing": "CRC/SHA validation, exact 64x64 pixel deduplication, RGB/Lanczos 64x64, train-only channel mean/std; train-only random horizontal flips",
              "downloaded_bytes": transferred, "duplicate_candidates_removed": len(downloaded)-len(unique)}
    save(manifest, result)
    print("Prepared image splits", result["split_counts"], flush=True)
    return result


def prepare_text():
    source = ROOT / "data/benchmarks_2026/nlpcc/data/train_data.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = list(rows.values())
    rng = random.Random(SEED)
    rng.shuffle(rows)
    groups, seen = [], set()
    for row in rows:
        human, ai = canonical(row.get("HWT", "")), canonical(row.get("LGT", ""))
        if min(len(human), len(ai)) < 80 or max(len(human), len(ai)) > 10000:
            continue
        hashes = [hashlib.sha256(text.encode()).hexdigest() for text in (human, ai)]
        if hashes[0] == hashes[1] or any(value in seen for value in hashes):
            continue
        groups.append((row.get("ID", len(groups)), human, ai, hashes))
        seen.update(hashes)
        if len(groups) == 2000:
            break
    if len(groups) != 2000:
        raise RuntimeError("Insufficient grouped text examples")
    output = []
    for index, (group, human, ai, hashes) in enumerate(groups):
        split = "train" if index < 1400 else "validation" if index < 1700 else "test"
        for label, text, digest in zip((0, 1), (human, ai), hashes):
            output.append({"id": str(group)+"-"+str(label), "group": str(group), "label": label,
                           "text": text, "sha256": digest, "split": split})
    for split in ("train", "validation", "test"):
        path = DATA / "processed/text" / (split + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output if row["split"] == split), encoding="utf-8")
    result = {"dataset": "NLPCC 2026 Task 6 official train", "source": source.relative_to(ROOT).as_posix(),
              "source_sha256": hashlib.file_digest(source.open("rb"), "sha256").hexdigest(), "seed": SEED,
              "grouping": "HWT/LGT paired by original ID; same subject stays in one split; normalized text hashes globally unique",
              "records": [{key: value for key, value in row.items() if key != "text"} for row in output],
              "counts": {split: sum(row["split"] == split for row in output) for split in ("train", "validation", "test")}}
    save(DATA / "text.json", result)
    print("Prepared text splits", result["counts"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    text = prepare_text()
    images = prepare_images(args.proxy)
    save(RESULTS / "experiment1.json", {"status": "completed", "text": {key: value for key, value in text.items() if key != "records"},
         "images": {key: value for key, value in images.items() if key != "records"}})


if __name__ == "__main__":
    main()
