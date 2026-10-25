"""Retrospective audit of the fixed 0.99 paragraph-highlighting policy.

This is not a mixed-authorship or target-platform validation study.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from inference import DetectorService
from train_chinese_text import read_anx, read_hc3, remove_leakage


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "text_segment_policy_audit.json"


def summarize(items: list[dict], service: DetectorService) -> dict:
    counts = {"human": 0, "ai": 0, "false_positive": 0, "true_positive": 0,
              "scored": 0, "segments": 0, "highlighted_segments": 0}
    for row in items:
        result = service.predict_text_segments(row["text"])
        positive = any(segment["highlighted"] for segment in result["segments"])
        label = int(row["label"])
        counts["ai" if label else "human"] += 1
        counts["scored"] += int(any(segment["risk_score"] is not None for segment in result["segments"]))
        counts["segments"] += len(result["segments"])
        counts["highlighted_segments"] += sum(segment["highlighted"] for segment in result["segments"])
        if positive and label:
            counts["true_positive"] += 1
        if positive and not label:
            counts["false_positive"] += 1
    counts["false_positive_rate"] = counts["false_positive"] / counts["human"] if counts["human"] else None
    counts["true_positive_rate"] = counts["true_positive"] / counts["ai"] if counts["ai"] else None
    counts["sample_scored_rate"] = counts["scored"] / len(items) if items else None
    return counts


def main() -> None:
    service = DetectorService(ROOT)
    with (ROOT / "data" / "text" / "external_test.jsonl").open(encoding="utf-8") as stream:
        english = [json.loads(line) for line in stream]
    instruction_starts = {"draft", "produce", "compile", "develop", "provide", "write",
                          "formulate", "create", "compose", "generate"}
    mislabeled_prompt_count = sum(
        row["label"] == 1 and row["text"].split()[0].lower() in instruction_starts
        for row in english if row["text"].split()
    )
    chinese_test = remove_leakage(read_hc3() + read_anx())[0]["test"]
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in chinese_test:
        groups[(row["corpus"], row["label"])].append(row)
    rng = random.Random(42)
    chinese_sample = [item for group in groups.values() for item in rng.sample(group, min(500, len(group)))]
    audits = {"chinese_test_sample": summarize(chinese_sample, service)}
    for corpus in sorted({row["corpus"] for row in chinese_sample}):
        audits[f"chinese_{corpus}"] = summarize(
            [row for row in chinese_sample if row["corpus"] == corpus], service)
    report = {
        "policy": "A document is flagged for review if any evaluated segment has a model score >= 0.99.",
        "input": "Previously reserved Chinese test samples, reused only for retrospective audit; no retraining or claimed independent confirmation after policy selection.",
        "english_benchmark_status": "invalid_for_AI_output_detection",
        "english_invalid_label_evidence": {
            "ai_labeled_total": sum(row["label"] == 1 for row in english),
            "ai_labeled_instruction_starts": mislabeled_prompt_count,
            "scoring_status": "excluded_from_accuracy_reporting",
            "example": "Draft a technical report analyzing internet of things in the context of technological advancement.",
            "reason": "Most AI-labeled records are writing prompts, not generated essays. Human-labeled records are often isolated essay fragments. English diagnostic rates must not be reported as detector accuracy.",
        },
        "limitations": "English benchmark labels are invalid for this task. Chinese examples have whole-sample labels only and sources are confounded. Mixed-authorship segment labels and real platform traffic are unavailable. This audit cannot estimate real platform error rates or the proportion written by AI.",
        "sample_seed": 42,
        "audits": audits,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audits, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
