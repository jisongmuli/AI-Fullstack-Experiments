"""Frozen-policy external audit on SemEval-2024 Task 8, Subtask C.

The official task locates one human-to-machine transition in English text. This
script never tunes the existing detector, and its test result is not a platform
accuracy estimate. Download IDs come from the organizers' Google Drive links.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import requests

from inference import DetectorService


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "text" / "semeval2024_subtask_c"
RESULTS_DIR = ROOT / "results"
OFFICIAL_REPO = "https://github.com/mbzuai-nlp/SemEval2024-task8"
OFFICIAL_TRAIN_DEV = "https://drive.google.com/drive/folders/16bRUuoeb_LxnCkcKM-ed6X6K5t_1C6mL"
OFFICIAL_TEST = "https://drive.google.com/drive/folders/10DKtClzkwIIAatzHBWXZXuQNID-DNGSG"
OFFICIAL_GOLD = "https://drive.google.com/drive/folders/13aFJK4UyY3Gxg_2ceEAWfJvzopB1vkPc"
FILES = {
    "dev": ("subtaskC_dev.jsonl", "1AmLZx58alwmvaHG3Lc96m9umO910MlQB", 505),
    "test": ("subtaskC_test.jsonl", "1IgnKxuqL7bZyxp5q3xiDcv6s_45FGWwB", 11123),
    "gold": ("subtaskC_test_gold.jsonl", "1qxBXf_CrVhZvyCUG1EOIrVwInoSEgHjP", 11123),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def download_official() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    details = {}
    for split, (name, file_id, expected_count) in FILES.items():
        path = DATA_DIR / name
        if not path.is_file():
            response = requests.get(
                "https://drive.google.com/uc",
                params={"export": "download", "id": file_id},
                timeout=90,
            )
            response.raise_for_status()
            if response.headers.get("Content-Type", "").startswith("text/html"):
                raise RuntimeError(f"Google Drive returned HTML rather than {name}")
            if len(response.content) > 35_000_000:
                raise RuntimeError(f"Unexpectedly large download for {name}")
            path.write_bytes(response.content)
        rows = read_jsonl(path)
        if len(rows) != expected_count:
            raise RuntimeError(f"{name}: expected {expected_count} rows, found {len(rows)}")
        required = {"id", "text"} | ({"label"} if split != "test" else set())
        if any(not required <= row.keys() for row in rows):
            raise RuntimeError(f"{name} has records without required keys")
        details[split] = {
            "path": str(path.relative_to(ROOT)),
            "source_url": f"https://drive.google.com/file/d/{file_id}/view",
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "rows": len(rows),
        }
    gold = read_jsonl(DATA_DIR / FILES["gold"][0])
    test = read_jsonl(DATA_DIR / FILES["test"][0])
    if any(a["id"] != b["id"] or a["text"] != b["text"] for a, b in zip(test, gold)):
        raise RuntimeError("Official test inputs and released gold differ in ID or text")
    details["test_gold_texts_identical"] = True
    (DATA_DIR / "source_manifest.json").write_text(
        json.dumps({
            "organizer_repo": OFFICIAL_REPO,
            "train_dev_folder": OFFICIAL_TRAIN_DEV,
            "test_folder": OFFICIAL_TEST,
            "test_gold_folder": OFFICIAL_GOLD,
            "files": details,
            "license_note": "Repository is Apache-2.0; underlying text-source reuse rights need separate review.",
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return details


def boundary_word_index(text: str, char_offset: int) -> int:
    """Map a character boundary to the task's literal-space word convention."""
    if char_offset <= 0:
        return 0
    prefix = text[:char_offset]
    if prefix.endswith(" "):
        prefix = prefix[:-1]
    return len(prefix.split(" ")) if prefix else 0


def summarize(rows: list[dict]) -> dict:
    absolute_errors = [abs(row["predicted_boundary"] - row["gold_boundary"])
                       for row in rows if row["predicted_boundary"] is not None]
    fallback_errors = [abs((row["predicted_boundary"] if row["predicted_boundary"] is not None
                            else row["word_count"]) - row["gold_boundary"]) for row in rows]
    return {
        "samples": len(rows),
        "with_any_scored_segment": sum(row["any_scored_segment"] for row in rows),
        "with_any_scored_segment_fraction": round(sum(row["any_scored_segment"] for row in rows) / len(rows), 6),
        "with_highlighted_boundary": len(absolute_errors),
        "boundary_coverage_fraction": round(len(absolute_errors) / len(rows), 6),
        "selective_mae_words": round(sum(absolute_errors) / len(absolute_errors), 4) if absolute_errors else None,
        "mae_words_with_end_of_text_fallback": round(sum(fallback_errors) / len(rows), 4),
        "within_10_words_among_covered": round(sum(error <= 10 for error in absolute_errors) / len(absolute_errors), 6)
        if absolute_errors else None,
        "mean_scored_character_percent": round(sum(row["scored_character_percent"] for row in rows) / len(rows), 2),
        "pure_ai_examples_gold_boundary_zero": sum(row["gold_boundary"] == 0 for row in rows),
        "pure_human_examples": 0,
        "pure_human_false_highlight_rate": None,
    }


def evaluate(split: str, service: DetectorService, limit: int | None) -> dict:
    name = FILES["dev" if split == "dev" else "gold"][0]
    source_rows = read_jsonl(DATA_DIR / name)
    if limit is not None:
        source_rows = source_rows[:limit]
    predictions = []
    start_time = time.perf_counter()
    for index, row in enumerate(source_rows):
        text = row["text"]
        gold_boundary = int(row["label"])
        result = service.predict_text_segments(text)
        highlighted = [segment for segment in result["segments"] if segment["highlighted"]]
        first = min((segment["start"] for segment in highlighted), default=None)
        predictions.append({
            "id": row["id"],
            "gold_boundary": gold_boundary,
            "predicted_boundary": boundary_word_index(text, first) if first is not None else None,
            "word_count": len(text.split(" ")),
            "any_scored_segment": int(any(segment["risk_score"] is not None for segment in result["segments"])),
            "scored_character_percent": result["segment_summary"]["scored_character_percent"],
            "highlighted_character_percent": result["segment_summary"]["highlighted_character_percent"],
        })
        if (index + 1) % 1000 == 0:
            print(f"{split}: {index + 1}/{len(source_rows)}", flush=True)
    suffix = f"_preview_{limit}" if limit is not None else ""
    csv_path = RESULTS_DIR / f"semeval_c_{split}_predictions{suffix}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)
    summary = summarize(predictions)
    summary["elapsed_seconds"] = round(time.perf_counter() - start_time, 2)
    summary["predictions_csv"] = str(csv_path.relative_to(ROOT))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test", "both"), default="both")
    parser.add_argument("--limit", type=int, help="Preview only; results are not an evaluation")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    file_details = download_official()
    if args.download_only:
        print(json.dumps(file_details, ensure_ascii=False, indent=2))
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    service = DetectorService(ROOT)
    split_names = ("dev", "test") if args.split == "both" else (args.split,)
    summaries = {split: evaluate(split, service, args.limit) for split in split_names}
    report = {
        "benchmark": "SemEval-2024 Task 8, Subtask C: English human-prefix/machine-suffix boundary",
        "source": {"repository": OFFICIAL_REPO, "files": file_details},
        "model_policy": "Unmodified DetectorService.predict_text_segments, including 0.99 highlight threshold; first highlighted segment start is predicted boundary.",
        "mapping": "Character offset mapped to literal-space-delimited word index; no highlight is an abstention. Full-coverage MAE uses document-end fallback solely for comparison.",
        "model_sha256": sha256(ROOT / "models" / "text_detector.joblib"),
        "config_sha256": sha256(ROOT / "models" / "model_config.json"),
        "test_status": "preview_not_valid_evaluation" if args.limit else "one_pass_frozen_external_test",
        "splits": summaries,
        "limitations": [
            "Selective MAE is conditional on a highlighted boundary and cannot be compared with unconditional benchmark MAE.",
            "No pure-human documents exist in this boundary benchmark, so its pure-human false-highlight rate is unmeasured.",
            "The benchmark has one human-to-machine boundary in English; it does not validate multiple switches, Chinese, or real platform traffic.",
            "Published organizer baseline of 3.53 MAE is on development data; paper reports 21.54 on held-out test. Compare only on the same split.",
        ],
    }
    suffix = f"_preview_{args.limit}" if args.limit is not None else ""
    output = RESULTS_DIR / f"semeval_c_frozen_baseline{suffix}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "splits": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
