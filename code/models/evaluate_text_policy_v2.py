"""Audit context-aware text annotations on previously reserved public samples.

This retrospective audit is not a platform validation study. Do not use its
error rates as a deployment claim or treat score/coverage as authorship share.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from inference import DetectorService
from text_segments import noncontent_reason
from train_chinese_text import read_anx, read_hc3, remove_leakage


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "text_policy_v2_audit.json"


def wilson_interval(positive: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.96
    p = positive / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    delta = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** .5) / denominator
    return [round(max(0, center - delta), 5), round(min(1, center + delta), 5)]


def evaluate(rows: list[dict], service: DetectorService) -> dict:
    counts = {
        "human": 0, "ai": 0, "human_any_highlight": 0,
        "ai_any_highlight": 0, "document_abstentions": 0,
        "segments": 0, "unscored_segments": 0,
    }
    for row in rows:
        result = service.predict_text_segments(row["text"])
        label = int(row["label"])
        counts["ai" if label else "human"] += 1
        counts["document_abstentions"] += int(result["risk_score"] is None)
        counts["segments"] += len(result["segments"])
        counts["unscored_segments"] += sum(segment["risk_score"] is None for segment in result["segments"])
        if any(segment["highlighted"] for segment in result["segments"]):
            counts["ai_any_highlight" if label else "human_any_highlight"] += 1
    counts["human_highlight_rate"] = counts["human_any_highlight"] / counts["human"] if counts["human"] else None
    counts["human_highlight_rate_wilson_95"] = wilson_interval(counts["human_any_highlight"], counts["human"])
    counts["ai_highlight_rate"] = counts["ai_any_highlight"] / counts["ai"] if counts["ai"] else None
    return counts


def main() -> None:
    service = DetectorService(ROOT)
    with (ROOT / "data" / "text" / "external_test.jsonl").open(encoding="utf-8") as stream:
        english = [json.loads(line) for line in stream]
    prompt_starts = {"draft", "produce", "compile", "develop", "provide", "write",
                     "formulate", "create", "compose", "generate"}
    known_prompts = [row for row in english if row["label"] == 1 and row["text"].split()
                     and row["text"].split()[0].lower() in prompt_starts]

    chinese_test = remove_leakage(read_hc3() + read_anx())[0]["test"]
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in chinese_test:
        groups[(row["corpus"], row["label"])].append(row)
    rng = random.Random(42)
    sample = [row for group in groups.values() for row in rng.sample(group, min(500, len(group)))]
    result = {
        "scope": "Retrospective checks on previously reserved corpus test records; policy and model were already developed with access to these sources.",
        "policy": "A segment is highlighted only when it is scored and risk_score >= 0.99. Explicit prompts, Markdown quotes and fenced code are unscored. A document score is withheld for conflicting segments or <80% scored non-whitespace characters.",
        "english_external_labels": {
            "ai_labeled_total": sum(row["label"] == 1 for row in english),
            "known_writing_prompts_mislabeled_ai": len(known_prompts),
            "known_prompts_abstained": sum(noncontent_reason(row["text"]) is not None for row in known_prompts),
            "human_labeled_abstained_as_instruction": sum(
                noncontent_reason(row["text"]) is not None for row in english if row["label"] == 0
            ),
            "accuracy_status": "invalid benchmark for generated prose detection; do not report detection accuracy",
        },
        "chinese_audit": {
            "overall": evaluate(sample, service),
            **{corpus: evaluate([row for row in sample if row["corpus"] == corpus], service)
               for corpus in sorted({row["corpus"] for row in sample})},
        },
        "limitations": [
            "Same public sources were already used for model and threshold design, so this is not independent confirmation.",
            "Human and AI source/genre are confounded, especially THUCNews versus generated long-form in AnxForever.",
            "Whole-record labels cannot validate paragraph-level highlighting, and mixed-authorship ground truth is absent.",
            "No blind, consented target-platform data are available; platform false-positive and false-negative rates remain unknown.",
            "Highlit character coverage is not a proportion of text written by AI.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
