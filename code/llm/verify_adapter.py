"""Verify the saved 160-step-run adapter and audit its held-out outputs."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter

import torch
from peft import PeftModel
from transformers import AutoTokenizer, LlamaForCausalLM

from data import ROOT, protected_tokens
from train import generate
from train_adapter import EXPLANATION, quality_metrics


IDENTIFIER = re.compile(r"\b(?:[A-Z]{2,}(?:-[A-Za-z0-9]+)?|[A-Za-z]+(?:-\d+|\d+))\b")


def identifiers(text: str) -> Counter[str]:
    return Counter(match.casefold() for match in IDENTIFIER.findall(text))


def sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    output = ROOT / "results/exp5"
    base_path = ROOT / "models/llm/base"
    adapter_path = ROOT / "models/llm/adapter"
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    if comparison["training"]["steps_completed"] != 160 or config["r"] != comparison["training"]["lora_rank"]:
        raise RuntimeError("Saved adapter does not match the 160-step training record.")
    manifest = json.loads((output / "source_manifest.json").read_text(encoding="utf-8"))
    checked_sources = {}
    for asset in manifest["assets"]:
        if asset["path"].replace("\\", "/").startswith("results/exp5/source/"):
            path = ROOT / asset["path"]
            actual = sha256(path)
            if actual != asset["sha256"]:
                raise RuntimeError(f"Source dataset hash mismatch: {path}")
            checked_sources[path.name] = actual
    split_rows = {
        name: [json.loads(line) for line in (output / "splits" / f"{name}.jsonl")
               .read_text(encoding="utf-8").splitlines()]
        for name in ("train", "validation", "test")
    }
    seen = set()
    for name, rows in split_rows.items():
        keys = {row["source"].casefold() for row in rows}
        if len(keys) != len(rows) or keys & seen:
            raise RuntimeError(f"Duplicate source in {name} or across splits.")
        seen |= keys
    saved = [json.loads(line) for line in (output / "test_predictions.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    if len(saved) != len(split_rows["test"]):
        raise RuntimeError("Saved test prediction count differs from test split.")
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = LlamaForCausalLM.from_pretrained(base_path, local_files_only=True, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    reproduced = [generate(model, tokenizer, row["source"]) for row in saved]
    mismatches = [row["id"] for row, prediction in zip(saved, reproduced)
                  if prediction != row["adapter_prediction"]]
    if mismatches:
        raise RuntimeError(f"Reloaded adapter output mismatch: {mismatches}")
    test_rows = split_rows["test"]
    before = [row["base_prediction"] for row in saved]
    after = [row["adapter_prediction"] for row in saved]
    protected_failures = [row["id"] for row in saved
                          if protected_tokens(row["source"]) != protected_tokens(row["adapter_prediction"])]
    identifier_failures = [
        {"id": row["id"], "source_identifiers": dict(identifiers(row["source"])),
         "prediction_identifiers": dict(identifiers(row["adapter_prediction"]))}
        for row in saved
        if identifiers(row["source"]) != identifiers(row["adapter_prediction"])
    ]
    report = {
        "adapter_sha256": sha256(adapter_path / "adapter_model.safetensors"),
        "source_sha256_verified": checked_sources,
        "split_counts": {name: len(rows) for name, rows in split_rows.items()},
        "split_sources_disjoint": True,
        "reloaded_predictions_match_saved": len(reproduced),
        "selected_checkpoint_step": comparison["training"]["selected_checkpoint_step"],
        "test_quality": {
            "copy_source": quality_metrics(test_rows, [row["source"] for row in test_rows]),
            "base": quality_metrics(test_rows, before),
            "adapter": quality_metrics(test_rows, after),
        },
        "adapter_protected_token_failures": protected_failures,
        "adapter_identifier_failures": identifier_failures,
        "adapter_unchanged_ids": [row["id"] for row in saved
                                   if row["source"].strip() == row["adapter_prediction"].strip()],
        "adapter_explanation_ids": [row["id"] for row in saved
                                     if EXPLANATION.search(row["adapter_prediction"])],
        "interpretation": (
            "Exact replay proves the artifact is loadable, not that editing is reliable. "
            "Identifiers, numbers, and negations are narrow surface checks; all model suggestions require human review."
        ),
    }
    (output / "artifact_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
