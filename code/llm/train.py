"""Actual small LoRA adaptation and held-out before/after evaluation."""
from __future__ import annotations

import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, LlamaForCausalLM

from data import ROOT, SEED, prepare, protected_tokens


SYSTEM = (
    "Correct the English grammar and clarity of the sentence while preserving its meaning, "
    "numbers, and negations. Return only the edited sentence."
)
MAX_SEQUENCE_TOKENS = 192
BATCH_SIZE = 2
EPOCHS = 2
LEARNING_RATE = 5e-4
MAX_NEW_TOKENS = 64


def prompt_for(tokenizer, source: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": source}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def encode_example(tokenizer, row: dict) -> dict:
    prompt = prompt_for(tokenizer, row["source"])
    target = tokenizer.encode(row["target"], add_special_tokens=False)
    end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    tokens = prompt + target + end
    if len(tokens) > MAX_SEQUENCE_TOKENS:
        raise ValueError(f"Example {row['id']} has {len(tokens)} tokens; limit {MAX_SEQUENCE_TOKENS}.")
    return {"input_ids": tokens, "labels": [-100] * len(prompt) + target + end}


def batchify(examples: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in examples)
    ids, labels, mask = [], [], []
    for row in examples:
        extra = width - len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_id] * extra)
        labels.append(row["labels"] + [-100] * extra)
        mask.append([1] * len(row["input_ids"]) + [0] * extra)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


def evaluate_loss(model, encoded: list[dict], pad_id: int) -> float:
    model.eval()
    weighted_loss = 0.0
    tokens = 0
    with torch.inference_mode():
        for offset in range(0, len(encoded), BATCH_SIZE):
            batch = batchify(encoded[offset:offset + BATCH_SIZE], pad_id)
            count = int((batch["labels"] != -100).sum())
            weighted_loss += float(model(**batch).loss) * count
            tokens += count
    return weighted_loss / tokens


def generate(model, tokenizer, source: str) -> str:
    model.eval()
    ids = torch.tensor([prompt_for(tokenizer, source)], dtype=torch.long)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    with torch.inference_mode():
        output = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=im_end,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
    return generated.strip()


def chrf(reference: str, hypothesis: str) -> float:
    if not reference or not hypothesis:
        return 0.0
    scores = []
    for size in range(1, 7):
        ref = Counter(reference[i:i + size] for i in range(len(reference) - size + 1))
        hyp = Counter(hypothesis[i:i + size] for i in range(len(hypothesis) - size + 1))
        matched = sum((ref & hyp).values())
        precision = matched / max(sum(hyp.values()), 1)
        recall = matched / max(sum(ref.values()), 1)
        scores.append(5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def aggregate(rows: list[dict], predictions: list[str]) -> dict:
    exact = sum(re.sub(r"\s+", " ", guess.strip().casefold()) ==
                re.sub(r"\s+", " ", row["target"].strip().casefold())
                for row, guess in zip(rows, predictions))
    chrfs = [chrf(row["target"], guess) for row, guess in zip(rows, predictions)]
    preserved = [protected_tokens(row["source"]) == protected_tokens(guess)
                 for row, guess in zip(rows, predictions)]
    return {
        "n": len(rows),
        "normalized_exact_match": exact / len(rows),
        "mean_character_f2": sum(chrfs) / len(chrfs),
        "number_and_negation_preservation": sum(preserved) / len(preserved),
        "preservation_support": {
            "with_number": sum(bool(protected_tokens(row["source"])[0]) for row in rows),
            "with_negation": sum(bool(protected_tokens(row["source"])[1]) for row in rows),
        },
    }


def evaluate_generation(model, tokenizer, rows: list[dict]) -> tuple[dict, list[dict]]:
    records = []
    for index, row in enumerate(rows, start=1):
        start = time.perf_counter()
        guess = generate(model, tokenizer, row["source"])
        records.append({
            "id": row["id"], "source": row["source"], "reference": row["target"],
            "prediction": guess, "character_f2": round(chrf(row["target"], guess), 4),
            "protected_tokens_preserved": protected_tokens(row["source"]) == protected_tokens(guess),
            "generation_seconds": round(time.perf_counter() - start, 3),
        })
        print(f"generated {index}/{len(rows)}", flush=True)
    return aggregate(rows, [item["prediction"] for item in records]), records


def main() -> None:
    started = time.perf_counter()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    paths = prepare()
    base = ROOT / "models" / "llm" / "base"
    adapter = ROOT / "models" / "llm" / "adapter_r4_coedit"
    output = ROOT / "results" / "exp5" / "r4_coedit"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Transformers 5.x AutoModel currently misidentifies this pinned Llama checkpoint.
    base_model = LlamaForCausalLM.from_pretrained(base, local_files_only=True, dtype=torch.float32)
    model = get_peft_model(base_model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM,
    ))
    parameter_counts = {
        "total": sum(t.numel() for t in model.parameters()),
        "trainable": sum(t.numel() for t in model.parameters() if t.requires_grad),
    }
    encoded = {name: [encode_example(tokenizer, row) for row in rows] for name, rows in paths.items()}
    pad_id = tokenizer.pad_token_id
    before_loss = {
        "validation": evaluate_loss(model, encoded["validation"], pad_id),
        "test": evaluate_loss(model, encoded["test"], pad_id),
    }
    print(f"before loss: {before_loss}", flush=True)
    before_metrics, before_predictions = evaluate_generation(model, tokenizer, paths["test"])

    optimizer = torch.optim.AdamW((t for t in model.parameters() if t.requires_grad), lr=LEARNING_RATE)
    history_path = output / "training_history.jsonl"
    step = 0
    epoch_validation = []
    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(EPOCHS):
            indices = list(range(len(encoded["train"])))
            random.Random(SEED + epoch).shuffle(indices)
            model.train()
            for offset in range(0, len(indices), BATCH_SIZE):
                batch = batchify([encoded["train"][i] for i in indices[offset:offset + BATCH_SIZE]], pad_id)
                optimizer.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_((t for t in model.parameters() if t.requires_grad), 1.0)
                optimizer.step()
                step += 1
                record = {"epoch": epoch + 1, "step": step, "train_loss": float(loss.detach()),
                          "elapsed_seconds": round(time.perf_counter() - started, 2)}
                history.write(json.dumps(record) + "\n")
                history.flush()
                if step % 12 == 0:
                    print(f"epoch {epoch + 1} step {step} loss {record['train_loss']:.4f}", flush=True)
            score = evaluate_loss(model, encoded["validation"], pad_id)
            epoch_validation.append(score)
            print(f"epoch {epoch + 1} validation_loss {score:.4f}", flush=True)

    after_loss = {
        "validation": evaluate_loss(model, encoded["validation"], pad_id),
        "test": evaluate_loss(model, encoded["test"], pad_id),
    }
    after_metrics, after_predictions = evaluate_generation(model, tokenizer, paths["test"])
    adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter / "tokenizer")
    comparison = {
        "task": "English grammar and clarity adaptation with meaning-safeguard screening",
        "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "dataset": "grammarly/coedit, GEC subset",
        "license": "Apache-2.0 for both model and dataset",
        "seed": SEED,
        "hardware": {"device": "cpu", "cuda_available": torch.cuda.is_available(),
                     "cpu_threads": torch.get_num_threads()},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__},
        "parameters": parameter_counts,
        "training": {"examples": len(paths["train"]), "epochs": EPOCHS, "steps": step,
                     "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                     "lora_rank": 4, "lora_alpha": 8, "lora_dropout": 0.05,
                     "target_modules": ["q_proj", "v_proj"],
                     "max_sequence_tokens": MAX_SEQUENCE_TOKENS},
        "loss": {"before": before_loss, "validation_by_epoch": epoch_validation, "after": after_loss},
        "generation": {
            "copy_source": aggregate(paths["test"], [row["source"] for row in paths["test"]]),
            "before": before_metrics,
            "after": after_metrics,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "limitations": [
            "Only a tiny selected subset of one public English grammar-editing corpus was used.",
            "Protected-token checks and character overlap do not prove semantic equivalence.",
            "The held-out generation sample is too small for a general performance claim.",
            "No image/text detection model or general-purpose Agent was fine-tuned.",
        ],
    }
    (output / "before_predictions.json").write_text(
        json.dumps(before_predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "after_predictions.json").write_text(
        json.dumps(after_predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"before": before_metrics, "after": after_metrics,
                      "loss": comparison["loss"], "elapsed_seconds": comparison["elapsed_seconds"]},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
