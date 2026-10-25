"""Audit the fixed mean of two frozen image candidates on matched public samples.

The public sources have already influenced model selection. These results are
exploratory and must not be used as a platform acceptance test.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, roc_auc_score

from evaluate_acceptance import audit_file


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("shoes", "art", "rhythmghai512")


def load_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keyed = {row["sample_id"]: row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate sample IDs: {path}")
    return keyed


def main() -> None:
    output_dir = ROOT / "results/highres"
    summary = {"method": "fixed arithmetic mean of Ateeqq SigLIP and NYUAD ViT raw softmax scores",
               "status": "exploratory_public_sets_not_platform_acceptance", "datasets": {}}
    for dataset in DATASETS:
        first = load_rows(output_dir / f"{dataset}_predictions.csv")
        second = load_rows(output_dir / f"nyuad_{dataset}_predictions.csv")
        if first.keys() != second.keys():
            raise ValueError(f"Candidate sample sets differ for {dataset}; rerun both at the same sample count")
        result_file = output_dir / f"crosscheck_{dataset}_predictions.csv"
        actual: list[int] = []
        scores: list[float] = []
        with result_file.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "sample_id", "actual", "risk_score", "siglip_score", "vit_score", "disagreement"
            ))
            writer.writeheader()
            for sample_id in sorted(first):
                left, right = first[sample_id], second[sample_id]
                if left["actual"] != right["actual"]:
                    raise ValueError(f"Conflicting label for {sample_id}")
                values = [float(left["risk_score"]), float(right["risk_score"])]
                score = sum(values) / 2
                label = int(left["actual"])
                writer.writerow({"sample_id": sample_id, "actual": label,
                                 "risk_score": round(score, 6), "siglip_score": values[0],
                                 "vit_score": values[1], "disagreement": round(abs(values[0] - values[1]), 6)})
                actual.append(label)
                scores.append(score)
        summary["datasets"][dataset] = {
            "n": len(actual), "per_class": {str(label): actual.count(label) for label in (0, 1)},
            "accuracy_at_0_5": accuracy_score(actual, [score >= 0.5 for score in scores]),
            "roc_auc": roc_auc_score(actual, scores),
            "abstaining_audit": audit_file(result_file, 0.2, 0.8),
            "csv_sha256": hashlib.sha256(result_file.read_bytes()).hexdigest(),
        }
    destination = output_dir / "crosscheck_evaluation.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
