---
library_name: peft
pipeline_tag: text-generation
base_model: HuggingFaceTB/SmolLM2-135M-Instruct
license: apache-2.0
tags:
  - lora
  - grammar-correction
  - educational-experiment
---

# Experiment 5 English editing adapter

This LoRA adapter is a small classroom experiment for English grammar and clarity suggestions. It is **not** an automatic trustworthy editor, semantic-equivalence verifier, authorship detector, or detector-evasion tool. Its outputs need human review.

The base model is SmolLM2-135M-Instruct, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. Training data is the CoEdIT GEC subset, revision `e9a255c33ef910bc33a9d2b522653fa87521583e`. Both source cards state Apache-2.0. Dataset and model hashes are listed in `results/exp5/source_manifest.json`.

The script selected 200 training sentences and used 160 optimizer steps, batch size 2, LoRA rank 8 on `q_proj` and `v_proj`, and learning rate 2e-4. Checkpoints were compared on 16 validation sentences; the lowest validation loss was at step 120, whose weights are saved here. Another 16 original-validation sentences were reserved for the test set and intentionally include eight number/negation cases.

On this small, selected test set, the base model's character F2 against one reference edit was 0.586, the adapter's was 0.787, and simply copying the source scored 0.771. Exact match was 0/16 for all three. The adapter copied 7/16 sources unchanged and dropped the `DOTA2` identifier in one case. The number/negation surface check passed 16/16, but this cannot establish meaning preservation. See `results/exp5/finetune_metrics.json`, `test_predictions.jsonl`, and `artifact_verification.json` for full results.

Reload the local base model explicitly, then load this adapter through PEFT. The recorded `base_model_name_or_path` is specific to the machine used for training; supplying the base model object avoids relying on that absolute path. Reproduce training with `python code/llm/train_adapter.py` and verify with `python code/llm/verify_adapter.py` from the project root.
