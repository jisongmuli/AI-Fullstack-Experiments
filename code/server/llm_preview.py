"""Opt-in, guarded preview of the experiment-five LoRA editor."""
from __future__ import annotations

import os
import re
import threading
from collections import Counter
from pathlib import Path

from editing import meaning_check


SYSTEM = (
    "Correct the English grammar and clarity of the sentence while preserving its meaning, "
    "numbers, and negations. Return only the edited sentence."
)
EXPLANATION = re.compile(
    r"\b(?:here(?:'s| is)|i (?:made|changed|corrected)|revised version|explanation|changes made)\b|\n\s*[-*]",
    re.I,
)
IDENTIFIER = re.compile(r"\b(?=[A-Za-z0-9_-]*\d)[A-Za-z][A-Za-z0-9_-]{2,}\b")


class LLMPreviewUnavailable(RuntimeError):
    pass


def screen_candidate(original: str, candidate: str) -> dict:
    check = meaning_check(original, candidate)
    reasons = []
    if not candidate or candidate == original:
        reasons.append("The model did not produce a distinct edit.")
    if EXPLANATION.search(candidate) or "\n" in candidate:
        reasons.append("The model returned commentary instead of only an edit.")
    if len(candidate) > min(480, len(original) * 1.5):
        reasons.append("The edit is unexpectedly long.")
    if not all(check["checks"].values()):
        reasons.append("A surface-level meaning safeguard failed.")
    if Counter(IDENTIFIER.findall(original)) != Counter(IDENTIFIER.findall(candidate)):
        reasons.append("An alphanumeric identifier changed or disappeared.")
    return {
        "status": "rejected" if reasons else "review_candidate",
        "candidate": None if reasons else candidate,
        "reason": " ".join(reasons) if reasons else "A writer must verify meaning and facts before use.",
        "meaning_check": check,
        "human_review_required": True,
        "model": "SmolLM2-135M-Instruct with CoEdIT LoRA adapter",
        "limitations": "English-only experimental adapter; no semantic-equivalence guarantee.",
    }


class LLMPreviewService:
    def __init__(self, root: Path):
        self.root = root
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self.adapter_dir = root / os.environ.get("AIGC_LLM_ADAPTER", "models/llm/adapter")

    def available(self) -> bool:
        return (self.root / "models/llm/base/model.safetensors").is_file() and (
            self.adapter_dir / "adapter_model.safetensors"
        ).is_file()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if not self.available():
                raise LLMPreviewUnavailable("The local LLM adapter is not available.")
            try:
                import torch
                from peft import PeftModel
                from transformers import AutoTokenizer, LlamaForCausalLM

                base = self.root / "models/llm/base"
                torch.set_num_threads(min(8, os.cpu_count() or 1))
                tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
                model = LlamaForCausalLM.from_pretrained(base, local_files_only=True, dtype=torch.float32)
                model = PeftModel.from_pretrained(model, self.adapter_dir, is_trainable=False)
                model.eval()
            except Exception as exc:
                raise LLMPreviewUnavailable(f"Could not load the local LLM: {type(exc).__name__}.") from exc
            self._tokenizer = tokenizer
            self._model = model

    def suggest(self, text: str) -> dict:
        value = text.strip()
        if not 20 <= len(value) <= 320 or not value.isascii() or len(re.findall(r"[A-Za-z]", value)) < 12:
            return {
                "status": "outside_scope", "candidate": None,
                "reason": "Preview accepts English passages of 20-320 characters only.",
                "human_review_required": True,
            }
        self._load()
        import torch

        tokenizer = self._tokenizer
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": value}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long)
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        with self._generation_lock, torch.inference_mode():
            result = self._model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=96,
                do_sample=False, eos_token_id=end, pad_token_id=tokenizer.pad_token_id,
            )
        candidate = tokenizer.decode(result[0, ids.shape[1]:], skip_special_tokens=True).strip()
        return screen_candidate(value, candidate)
