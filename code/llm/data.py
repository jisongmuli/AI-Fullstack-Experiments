"""Deterministic, narrow CoEdIT GEC sample selection for a CPU LoRA exercise."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEED = 42
NUMBER = re.compile(r"\b\d+(?:\.\d+)?%?\b")
NEGATION = re.compile(r"\b(?:no|not|never|without|cannot|can't|won't|doesn't|don't|isn't|aren't)\b", re.I)


def protected_tokens(text: str) -> tuple[Counter[str], Counter[str]]:
    return Counter(NUMBER.findall(text)), Counter(token.lower() for token in NEGATION.findall(text))


def _read_candidates(path: Path) -> list[dict]:
    selected = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("task") != "gec" or ":" not in row.get("src", ""):
                continue
            source = row["src"].split(":", 1)[1].strip()
            target = row["tgt"].strip()
            if not (25 <= len(source) <= 130 and 1 <= len(target) <= 130):
                continue
            similarity = SequenceMatcher(None, source.casefold(), target.casefold()).ratio()
            if not (0.75 <= similarity <= 0.98) or protected_tokens(source) != protected_tokens(target):
                continue
            key = source.casefold()
            if key in seen:
                continue
            seen.add(key)
            selected.append({
                "id": row["_id"], "source": source, "target": target,
                "source_similarity": round(similarity, 4),
            })
    selected.sort(key=lambda row: hashlib.sha256(
        f"{SEED}:{row['id']}:{row['source']}".encode("utf-8")
    ).hexdigest())
    return selected


def prepare(train_count: int = 96, validation_count: int = 12, test_count: int = 12) -> dict[str, list[dict]]:
    train = _read_candidates(ROOT / "results" / "exp5" / "source" / "train.jsonl")
    heldout = _read_candidates(ROOT / "results" / "exp5" / "source" / "validation.jsonl")
    train_sources = {row["source"].casefold() for row in train}
    heldout = [row for row in heldout if row["source"].casefold() not in train_sources]
    if len(train) < train_count or len(heldout) < validation_count + test_count:
        raise ValueError("Not enough eligible public examples after filtering.")
    splits = {
        "train": train[:train_count],
        "validation": heldout[:validation_count],
        "test": heldout[validation_count:validation_count + test_count],
    }
    directory = ROOT / "results" / "exp5" / "r4_coedit" / "splits"
    directory.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with (directory / f"{split}.jsonl").open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "seed": SEED,
        "candidate_counts": {"train": len(train), "original_validation_after_train_overlap": len(heldout)},
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "selection": (
            "CoEdIT task=gec, strip instruction before first colon, 25-130 source characters, "
            "1-130 target characters, SequenceMatcher ratio 0.75-0.98, identical number and "
            "negation tokens, exact source deduplication; SHA-256(seed:id:source) ordering."
        ),
        "provenance": "grammarly/coedit, pinned revision in results/exp5/source_manifest.json",
    }
    (directory / "selection.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return splits


if __name__ == "__main__":
    result = prepare()
    print({split: len(rows) for split, rows in result.items()})
