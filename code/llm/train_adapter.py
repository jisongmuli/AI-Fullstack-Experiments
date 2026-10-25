"""Fixed 160-step LoRA experiment with held-out editing evaluation."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import peft
import torch
import transformers
from peft import (LoraConfig, TaskType, get_peft_model,
                  get_peft_model_state_dict, set_peft_model_state_dict)
from transformers import AutoTokenizer, LlamaForCausalLM

from data import ROOT, SEED, _read_candidates, protected_tokens
from train import (aggregate, batchify, chrf, encode_example, evaluate_generation,
                   evaluate_loss)


TRAIN_COUNT = 200
VALIDATION_COUNT = 16
TEST_COUNT = 16
BATCH_SIZE = 2
MAX_STEPS = 160
CHECKPOINT_INTERVAL = 40
LEARNING_RATE = 2e-4
EXPLANATION = re.compile(r"^(?:here(?:'s| is)|the corrected sentence|i (?:made|changed))\b|\n\s*[-*] ", re.I)


def select_splits(root: Path) -> dict[str, list[dict]]:
    train_candidates = _read_candidates(root / "results/exp5/source/train.jsonl")
    heldout = _read_candidates(root / "results/exp5/source/validation.jsonl")
    train_sources = {row["source"].casefold() for row in train_candidates}
    heldout = [row for row in heldout if row["source"].casefold() not in train_sources]
    protected = [row for row in heldout if any(protected_tokens(row["source"]))]
    ordinary = [row for row in heldout if not any(protected_tokens(row["source"]))]
    if len(train_candidates) < TRAIN_COUNT or len(protected) < 8 or len(ordinary) < 24:
        raise ValueError("Not enough unique examples for the fixed experiment splits.")
    test = protected[:8] + ordinary[:8]
    used = {row["source"].casefold() for row in test}
    validation = [row for row in heldout if row["source"].casefold() not in used][:VALIDATION_COUNT]
    splits = {"train": train_candidates[:TRAIN_COUNT], "validation": validation, "test": test}
    if len(validation) != VALIDATION_COUNT or len(test) != TEST_COUNT:
        raise ValueError("Incomplete held-out split.")
    destination = root / "results/exp5/splits"
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (destination / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    selection = {
        "seed": SEED,
        "source": "grammarly/coedit GEC subset; pinned source in results/exp5/source_manifest.json",
        "counts": {name: len(rows) for name, rows in splits.items()},
        "method": "Deterministic SHA-256 order and filters from data.py; exact source deduplication across original source splits.",
        "test_sampling": "8 number/negation cases plus 8 ordinary cases, deliberately enriched for safeguard auditing; not a representative estimate.",
        "validation_sampling": "First 16 remaining eligible original-validation rows after test reservation.",
    }
    (destination / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    return splits


def quality_metrics(rows: list[dict], predictions: list[str]) -> dict:
    basic = aggregate(rows, predictions)
    basic.update({
        "unchanged_source_rate": sum(row["source"].strip() == guess.strip()
                                     for row, guess in zip(rows, predictions)) / len(rows),
        "empty_output_rate": sum(not guess.strip() for guess in predictions) / len(rows),
        "explanation_contamination_rate": sum(bool(EXPLANATION.search(guess))
                                              for guess in predictions) / len(rows),
        "mean_character_f2_against_reference": sum(chrf(row["target"], guess)
                                                  for row, guess in zip(rows, predictions)) / len(rows),
    })
    return basic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.perf_counter()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)

    splits = select_splits(root)
    base_path = root / "models/llm/base"
    adapter_path = root / "models/llm/adapter"
    output = root / "results/exp5"
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Transformers 5.x AutoModel misidentifies this pinned Llama checkpoint.
    base = LlamaForCausalLM.from_pretrained(base_path, local_files_only=True, dtype=torch.float32)
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM,
    ))
    encoded = {name: [encode_example(tokenizer, row) for row in rows]
               for name, rows in splits.items()}
    before_loss = {name: evaluate_loss(model, encoded[name], tokenizer.pad_token_id)
                   for name in ("validation", "test")}
    _, before_records = evaluate_generation(model, tokenizer, splits["test"])
    print(f"baseline losses: {before_loss}", flush=True)

    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=LEARNING_RATE)
    history = []
    checkpoints = []
    best_loss = float("inf")
    best_step = 0
    best_state = None
    indices = []
    epoch = 0
    history_path = output / "training_history.jsonl"
    with history_path.open("w", encoding="utf-8") as stream:
        for step in range(1, MAX_STEPS + 1):
            if len(indices) < BATCH_SIZE:
                indices = list(range(len(encoded["train"])))
                random.Random(SEED + epoch).shuffle(indices)
                epoch += 1
            batch = batchify([encoded["train"][indices.pop()] for _ in range(BATCH_SIZE)],
                             tokenizer.pad_token_id)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters()
                                            if parameter.requires_grad), 1.0)
            optimizer.step()
            record = {"step": step, "train_loss": round(float(loss.detach()), 6),
                      "elapsed_seconds": round(time.perf_counter() - started, 2)}
            history.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if step % CHECKPOINT_INTERVAL == 0:
                val_loss = evaluate_loss(model, encoded["validation"], tokenizer.pad_token_id)
                checkpoints.append({"step": step, "validation_loss": val_loss})
                print(f"step {step}/{MAX_STEPS}: train={record['train_loss']:.4f}, validation={val_loss:.4f}", flush=True)
                if val_loss < best_loss:
                    best_loss, best_step = val_loss, step
                    best_state = {key: value.detach().cpu().clone()
                                  for key, value in get_peft_model_state_dict(model).items()}

    if best_state is None:
        raise RuntimeError("No adapter checkpoint selected.")
    set_peft_model_state_dict(model, best_state)
    after_loss = {name: evaluate_loss(model, encoded[name], tokenizer.pad_token_id)
                  for name in ("validation", "test")}
    _, after_records = evaluate_generation(model, tokenizer, splits["test"])
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path, safe_serialization=True)
    tokenizer.save_pretrained(adapter_path / "tokenizer")

    before_predictions = [row["prediction"] for row in before_records]
    after_predictions = [row["prediction"] for row in after_records]
    before_metrics = quality_metrics(splits["test"], before_predictions)
    after_metrics = quality_metrics(splits["test"], after_predictions)
    copy_metrics = quality_metrics(splits["test"], [row["source"] for row in splits["test"]])
    comparison = {
        "task": "English grammar and clarity editing LoRA exercise",
        "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "dataset": "grammarly/coedit GEC subset",
        "seed": SEED,
        "hardware": {"device": "cpu", "cuda_available": torch.cuda.is_available(),
                     "cpu_threads": torch.get_num_threads()},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__},
        "parameters": {"total": sum(p.numel() for p in model.parameters()),
                       "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        "training": {"examples_selected": TRAIN_COUNT, "batch_size": BATCH_SIZE,
                     "steps_completed": MAX_STEPS, "learning_rate": LEARNING_RATE,
                     "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05,
                     "target_modules": ["q_proj", "v_proj"],
                     "selected_checkpoint_step": best_step,
                     "checkpoint_validation_losses": checkpoints},
        "evaluation": {"validation_examples": VALIDATION_COUNT, "test_examples": TEST_COUNT,
                       "test_set_enriched_with_protected_tokens": True,
                       "loss_before": before_loss, "loss_after": after_loss,
                       "copy_source": copy_metrics, "base": before_metrics, "adapter": after_metrics},
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "limitations": [
            "Small, selected English-only grammar data and 16 test examples cannot establish general editing quality.",
            "The test set over-samples number/negation examples, so aggregate rates are not population estimates.",
            "Character F2 rewards overlap and copying; report the source-copy baseline alongside model results.",
            "Surface token preservation cannot prove that meaning or factual claims are unchanged.",
            "The adapted model is an optional suggestion generator, not a verified automatic editor or authorship detector.",
        ],
    }
    for name in ("finetune_metrics.json", "comparison.json"):
        (output / name).write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "train_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "before_predictions.json").write_text(json.dumps(before_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "after_predictions.json").write_text(json.dumps(after_records, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row, before, after in zip(splits["test"], before_predictions, after_predictions):
            stream.write(json.dumps({
                "id": row["id"], "source": row["source"], "reference": row["target"],
                "base_prediction": before, "adapter_prediction": after,
                "base_protected_preserved": protected_tokens(row["source"]) == protected_tokens(before),
                "adapter_protected_preserved": protected_tokens(row["source"]) == protected_tokens(after),
            }, ensure_ascii=False) + "\n")
    print(json.dumps({"selected_step": best_step, "test_before": before_metrics,
                      "test_after": after_metrics, "loss_after": after_loss}, indent=2), flush=True)


if __name__ == "__main__":
    main()
