"""Reload the saved adapter and evaluate held-out number/negation cases."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, LlamaForCausalLM

from data import ROOT, _read_candidates, protected_tokens
from train import aggregate, chrf, generate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    output = ROOT / "results" / "exp5" / "r4_coedit"
    base_path = ROOT / "models" / "llm" / "base"
    adapter_path = ROOT / "models" / "llm" / "adapter_r4_coedit"
    config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    if config["r"] != comparison["training"]["lora_rank"]:
        raise RuntimeError("Saved LoRA rank does not match the training record.")
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    base = LlamaForCausalLM.from_pretrained(base_path, local_files_only=True, dtype=torch.float32)
    base.eval()
    heldout = _read_candidates(ROOT / "results" / "exp5" / "source" / "validation.jsonl")
    used = {
        json.loads(line)["source"].casefold()
        for name in ("train", "validation", "test")
        for line in (output / "splits" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    }
    challenge = [
        row for row in heldout
        if row["source"].casefold() not in used and any(protected_tokens(row["source"]))
    ]
    if not challenge:
        raise RuntimeError("No protected-token challenge cases available.")
    base_predictions = [generate(base, tokenizer, row["source"]) for row in challenge]
    adapted = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    adapted.eval()
    saved_after = json.loads((output / "after_predictions.json").read_text(encoding="utf-8"))
    reproduced = generate(adapted, tokenizer, saved_after[0]["source"])
    if reproduced != saved_after[0]["prediction"]:
        raise RuntimeError("Reloaded adapter does not reproduce the saved held-out prediction.")
    adapted_predictions = [generate(adapted, tokenizer, row["source"]) for row in challenge]
    records = []
    for row, baseline, tuned in zip(challenge, base_predictions, adapted_predictions):
        protected = protected_tokens(row["source"])
        records.append({
            "id": row["id"], "source": row["source"], "reference": row["target"],
            "base_prediction": baseline, "adapted_prediction": tuned,
            "has_number": bool(protected[0]), "has_negation": bool(protected[1]),
            "base_preserves_protected": protected_tokens(baseline) == protected,
            "adapted_preserves_protected": protected_tokens(tuned) == protected,
            "source_character_f2": round(chrf(row["target"], row["source"]), 4),
            "base_character_f2": round(chrf(row["target"], baseline), 4),
            "adapted_character_f2": round(chrf(row["target"], tuned), 4),
        })
    def protected_rate(key: str, rows: list[dict]) -> float | None:
        return sum(item[key] for item in rows) / len(rows) if rows else None
    results = {
        "adapter_sha256": sha256_file(adapter_path / "adapter_model.safetensors"),
        "adapter_rank": config["r"],
        "reloaded_prediction_matches_saved": True,
        "challenge_count": len(challenge),
        "with_number": sum(item["has_number"] for item in records),
        "with_negation": sum(item["has_negation"] for item in records),
        "all_cases": {
            "copy_source": aggregate(challenge, [row["source"] for row in challenge]),
            "base": aggregate(challenge, base_predictions),
            "adapted": aggregate(challenge, adapted_predictions),
        },
        "conditional_preservation": {
            "number_base": protected_rate("base_preserves_protected", [x for x in records if x["has_number"]]),
            "number_adapted": protected_rate("adapted_preserves_protected", [x for x in records if x["has_number"]]),
            "negation_base": protected_rate("base_preserves_protected", [x for x in records if x["has_negation"]]),
            "negation_adapted": protected_rate("adapted_preserves_protected", [x for x in records if x["has_negation"]]),
        },
        "limitation": "Token retention does not establish semantic or factual equivalence.",
    }
    (output / "challenge_predictions.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "artifact_verification.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
