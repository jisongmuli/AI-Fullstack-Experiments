"""Audit abstaining binary detectors on independent, labeled predictions.

This computes evidence for an acceptance review; it does not establish that a
public test set represents a platform's future traffic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    proportion = successes / total
    z2 = Z_95 * Z_95
    denominator = 1 + z2 / total
    center = (proportion + z2 / (2 * total)) / denominator
    radius = Z_95 * math.sqrt(
        proportion * (1 - proportion) / total + z2 / (4 * total * total)
    ) / denominator
    lower = 0.0 if successes == 0 else max(0.0, center - radius)
    upper = 1.0 if successes == total else min(1.0, center + radius)
    return lower, upper


def rate(successes: int, total: int) -> dict[str, int | float | None]:
    lower, upper = wilson_interval(successes, total)
    return {
        "count": successes,
        "denominator": total,
        "value": successes / total if total else None,
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
    }


def audit_file(path: Path, low: float, high: float) -> dict:
    counts = {
        "human": {"human": 0, "ai": 0, "uncertain": 0},
        "ai": {"human": 0, "ai": 0, "uncertain": 0},
    }
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"sample_id", "actual", "risk_score"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected sample_id, actual, risk_score columns")
        for line_number, row in enumerate(reader, 2):
            sample_id = (row["sample_id"] or "").strip()
            if not sample_id or sample_id in seen:
                raise ValueError(f"{path}:{line_number}: blank or repeated sample_id")
            seen.add(sample_id)
            if row["actual"] not in {"0", "1"}:
                raise ValueError(f"{path}:{line_number}: actual must be 0 or 1")
            actual = "ai" if row["actual"] == "1" else "human"
            raw_score = (row["risk_score"] or "").strip()
            if not raw_score:
                decision = "uncertain"
            else:
                try:
                    score = float(raw_score)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid risk_score") from exc
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"{path}:{line_number}: risk_score outside [0, 1]")
                decision = "human" if score <= low else "ai" if score >= high else "uncertain"
            counts[actual][decision] += 1

    negatives = sum(counts["human"].values())
    positives = sum(counts["ai"].values())
    if negatives == 0 or positives == 0:
        raise ValueError(f"{path}: both labeled classes are required")
    covered_negatives = negatives - counts["human"]["uncertain"]
    covered_positives = positives - counts["ai"]["uncertain"]
    covered = covered_negatives + covered_positives
    false_positives = counts["human"]["ai"]
    false_negatives = counts["ai"]["human"]
    predicted_ai = counts["human"]["ai"] + counts["ai"]["ai"]
    return {
        "file": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_count": negatives + positives,
        "class_counts": {"human": negatives, "ai": positives},
        "confusion_with_abstention": counts,
        "coverage": rate(covered, negatives + positives),
        "false_positive_rate_all_human": rate(false_positives, negatives),
        "false_negative_rate_all_ai": rate(false_negatives, positives),
        "false_positive_rate_covered_human": rate(false_positives, covered_negatives),
        "false_negative_rate_covered_ai": rate(false_negatives, covered_positives),
        "positive_predictive_value": rate(counts["ai"]["ai"], predicted_ai),
    }


def gate_audit(
    audit: dict, max_fpr: float, max_fnr: float, min_coverage: float, min_per_class: int
) -> dict:
    checks = {
        "min_human_samples": audit["class_counts"]["human"] >= min_per_class,
        "min_ai_samples": audit["class_counts"]["ai"] >= min_per_class,
        "max_fpr_wilson_upper": audit["false_positive_rate_all_human"]["wilson_95_upper"] <= max_fpr,
        "max_fnr_wilson_upper": audit["false_negative_rate_all_ai"]["wilson_95_upper"] <= max_fnr,
        "min_coverage_wilson_lower": audit["coverage"]["wilson_95_lower"] >= min_coverage,
    }
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--low", type=float, default=0.2)
    parser.add_argument("--high", type=float, default=0.8)
    parser.add_argument("--max-fpr", type=float)
    parser.add_argument("--max-fnr", type=float)
    parser.add_argument("--min-coverage", type=float)
    parser.add_argument("--min-per-class", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.low < args.high <= 1:
        parser.error("thresholds must satisfy 0 <= low < high <= 1")
    limits = (args.max_fpr, args.max_fnr, args.min_coverage)
    if any(value is not None for value in limits) and not all(value is not None for value in limits):
        parser.error("provide all three acceptance limits or none")
    if any(value is not None and not 0 <= value <= 1 for value in limits):
        parser.error("acceptance limits must be within [0, 1]")
    if args.min_per_class < 1:
        parser.error("min-per-class must be positive")

    datasets = {}
    for entry in args.dataset:
        name, separator, filename = entry.partition("=")
        if not separator or not name or not filename or name in datasets:
            parser.error("each dataset must be a unique NAME=CSV pair")
        audit = audit_file(Path(filename), args.low, args.high)
        if all(value is not None for value in limits):
            audit["acceptance_gate"] = gate_audit(
                audit, args.max_fpr, args.max_fnr, args.min_coverage, args.min_per_class
            )
        else:
            audit["acceptance_gate"] = {"status": "not_evaluated", "reason": "No predeclared limits"}
        datasets[name] = audit

    result = {
        "thresholds": {"human_at_or_below": args.low, "ai_at_or_above": args.high},
        "interval": "two-sided Wilson 95%; gates use conservative interval bounds",
        "interpretation": "Public-set results do not establish target-platform readiness.",
        "datasets": datasets,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
