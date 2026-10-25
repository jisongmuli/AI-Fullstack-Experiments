"""Evaluate the frozen normal-resolution detector on external public image sets.

The sampled ZIP members are fixed by seed and retained in the CSV. No training,
threshold tuning, or test-set filtering is performed after prediction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from evaluate_acceptance import audit_file
from highres_inference import HighResolutionDetector


ZIP_FILES = {
    "shoes": "data/raw/shoes_real_ai.zip",
    "art": "data/raw/art_real_ai.zip",
    "mixed512": "data/raw/mixed512_real_ai.zip",
    "rhythmghai512": "data/raw/rhythmghai_real_ai_512.zip",
}


def member_label(dataset: str, member: str) -> int | None:
    normalized = member.replace("\\", "/").lower()
    if not normalized.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return None
    parts = normalized.split("/")
    if dataset == "shoes":
        return 1 if parts[0] == "ai-midjourney" else 0 if parts[0] == "real" else None
    if dataset == "rhythmghai512":
        return 1 if parts[0] == "ai_generated_dataset" else 0 if parts[0] == "real_dataset" else None
    if dataset == "art" and len(parts) >= 3 and parts[0] == "data":
        return 1 if parts[1] == "fake" else 0 if parts[1] == "real" else None
    if dataset == "mixed512":
        if len(parts) >= 3 and parts[0] == "hybridforensics_dataset_high":
            if parts[1] in {"fake_diffusion", "fake_gan"}:
                return 1
            if parts[1] == "real":
                return 0
        for part in parts[:-1]:
            if part in {"fake", "ai", "generated", "synthetic"}:
                return 1
            if part in {"real", "human", "authentic"}:
                return 0
    return None


def select_members(archive: zipfile.ZipFile, dataset: str, per_class: int, seed: int) -> list[tuple[str, int]]:
    groups: dict[int, list[str]] = defaultdict(list)
    for member in archive.namelist():
        label = member_label(dataset, member)
        if label is not None:
            groups[label].append(member)
    if not groups[0] or not groups[1]:
        raise ValueError(f"Cannot find both classes in {dataset}: {[name for name in archive.namelist()[:15]]}")
    rng = random.Random(seed)
    selected = []
    seen_digests: dict[str, int] = {}
    for label in (0, 1):
        members = sorted(groups[label])
        for name in rng.sample(members, len(members)):
            digest = hashlib.sha256(archive.read(name)).hexdigest()
            if digest in seen_digests:
                if seen_digests[digest] != label:
                    raise ValueError(f"Identical image has conflicting labels: {name}")
                continue
            seen_digests[digest] = label
            selected.append((name, label))
            if sum(selected_label == label for _, selected_label in selected) == per_class:
                break
    rng.shuffle(selected)
    return selected


def evaluate(root: Path, dataset: str, per_class: int, seed: int, model: HighResolutionDetector,
             model_key: str = "ateeqq") -> dict:
    archive_path = root / ZIP_FILES[dataset]
    prefix = "" if model_key == "ateeqq" else f"{model_key}_"
    output = root / f"results/highres/{prefix}{dataset}_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    observations = []
    seen_digests: dict[str, int] = {}
    with zipfile.ZipFile(archive_path) as archive, output.open("w", encoding="utf-8", newline="") as stream:
        members = select_members(archive, dataset, per_class, seed)
        writer = csv.DictWriter(stream, fieldnames=(
            "sample_id", "actual", "risk_score", "decision_at_0.8", "member", "width", "height", "latency_ms"
        ))
        writer.writeheader()
        start = time.perf_counter()
        for index, (member, actual) in enumerate(members, 1):
            raw = archive.read(member)
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen_digests and seen_digests[digest] != actual:
                raise ValueError(f"Identical image has conflicting labels: {member}")
            seen_digests[digest] = actual
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    size = image.size
                    tick = time.perf_counter()
                    prediction = model.predict(image)
                    latency = (time.perf_counter() - tick) * 1000
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise ValueError(f"Unreadable selected image: {member}") from exc
            score = prediction["risk_score"]
            writer.writerow({
                "sample_id": f"{dataset}_{digest[:16]}", "actual": actual,
                "risk_score": score, "decision_at_0.8": prediction["decision"],
                "member": member, "width": size[0], "height": size[1],
                "latency_ms": round(latency, 2),
            })
            observations.append((actual, score, latency, size))
            if index % 100 == 0 or index == len(members):
                print(f"{dataset}: {index}/{len(members)} in {time.perf_counter()-start:.1f}s", flush=True)
    actual = [item[0] for item in observations]
    scores = [item[1] for item in observations]
    predictions = [int(score >= 0.5) for score in scores]
    audit = audit_file(output, 0.2, 0.8)
    widths = [size[0] for _, _, _, size in observations]
    heights = [size[1] for _, _, _, size in observations]
    latency = sorted(item[2] for item in observations)
    return {
        "model_id": model.model_id,
        "label_quality": "invalid_proxy_labels_exclude_from_acceptance" if dataset == "mixed512" else "source_claims_unverified",
        "source_zip": str(archive_path),
        "source_zip_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "predictions_csv": str(output), "seed": seed,
        "sampled_per_class": per_class,
        "n": len(observations), "image_width_range": [min(widths), max(widths)],
        "image_height_range": [min(heights), max(heights)],
        "threshold_0_5_accuracy": accuracy_score(actual, predictions),
        "threshold_0_5_macro_f1": f1_score(actual, predictions, average="macro"),
        "threshold_0_5_confusion": confusion_matrix(actual, predictions, labels=[0, 1]).tolist(),
        "roc_auc": roc_auc_score(actual, scores),
        "latency_ms_p50": latency[len(latency) // 2],
        "latency_ms_p95": latency[int(0.95 * (len(latency) - 1))],
        "abstaining_audit": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataset", choices=list(ZIP_FILES), action="append")
    parser.add_argument("--model", choices=list(HighResolutionDetector.MODELS), default="ateeqq")
    parser.add_argument("--per-class", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("per-class must be positive")
    root = args.root.resolve()
    model = HighResolutionDetector(root, args.model)
    datasets = args.dataset or ["shoes", "art", "rhythmghai512"]
    results = {(name if args.model == "ateeqq" else f"{args.model}_{name}"):
               evaluate(root, name, args.per_class, args.seed, model, args.model) for name in datasets}
    destination = root / "results/highres/evaluation.json"
    old = json.loads(destination.read_text(encoding="utf-8")) if destination.is_file() else {}
    old.update(results)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(old, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
