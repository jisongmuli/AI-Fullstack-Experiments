# Experiment 5: current course Agent and historical editing adaptation

The current 2026-10-27 Agent uses the official Qwen3.5-0.8B text backbone with a genuinely trained LoRA tool router. Its 80 training, 20 validation and 30 heldout bilingual cases, 64-step loss log, checkpoint comparisons and actual API traces are in `results/course_20261027/agent`. Entry points: `agent_model.py`, `scripts/train_agent_router.py`, and the server's `/api/agent/run` multipart endpoint. Full course instructions are in `docs/课程要求对照与验收_20261027.md`.

The following sections describe the earlier SmolLM2 editing experiments. Their original result directories were not included in the transferred runtime bundle; the new course Agent has a complete new record set.

This is a real, limited LoRA training exercise for the trustworthy editing module. It does not fine-tune the image detector, text detector, or the deterministic Agent. The base model is [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`; the data is [CoEdIT](https://huggingface.co/datasets/grammarly/coedit), revision `e9a255c33ef910bc33a9d2b522653fa87521583e`. Both source cards state Apache-2.0. The download script uses the public `hf-mirror.com` mirror because direct Hugging Face access was unavailable on this machine. Original files, SHA-256 hashes, source URLs and revisions are recorded in `results/exp5/source_manifest.json`.

From the project root:

```powershell
python -m pip install -r code/llm/requirements.txt
python code/llm/download.py
python code/llm/train.py
python code/llm/verify.py
```

Training selects 96 short CoEdIT grammar examples from the original training split and reserves 12 validation plus 12 test examples from the original validation split. It requires a nontrivial reference edit, high source/reference overlap, unchanged numbers and negations, and exact source deduplication. Selection order is the SHA-256 hash of seed 42, ID and source; IDs and rules are in `results/exp5/r4_coedit/splits`. The fixed experiment uses two epochs, batch size 2, 96 AdamW steps, learning rate 5e-4, LoRA rank 4 on `q_proj` and `v_proj`. The saved adapter is `models/llm/adapter_r4_coedit/adapter_model.safetensors`, with its configuration alongside. The 269 MB base model remains in `models/llm/base`.

`results/exp5/r4_coedit/comparison.json` contains the baseline and adapted test losses, generated-output metrics, parameter counts, versions and elapsed time. The same 12 held-out test examples are evaluated before and after training. `training_history.jsonl` has every optimization step; `before_predictions.json` and `after_predictions.json` show each generated edit. `verify.py` reloads the saved adapter, checks that it reproduces a saved prediction, and tests separate held-out cases containing numbers or negations in `artifact_verification.json`.

The completed CPU run used 96 training examples and 96 optimizer steps in 89.21 seconds. It saved 230,400 trainable LoRA parameters and produced a 937,056-byte adapter. On 12 held-out test examples, reference-conditioned loss fell from 1.178 to 0.973 and generated character F2 rose from 0.501 to 0.782. **Copying the unedited source scored 0.810**, which is higher than the adapted model on this metric. Exact reference match remained 0/12. Validation loss improved after epoch one (1.280 to 1.105), then rose to 1.156 after epoch two; this suggests possible overfitting and limits claims about improvement.

The saved adapter was reloaded, its first held-out prediction reproduced exactly, and its SHA-256 was `f6c4fdaa326999cb160d1aa1fa4c651437caf03f58f98c9198808799c11dc63d`. On a separate 12-example challenge set from the original validation split (8 with numbers, 4 with negations), protected-token retention improved from 8/12 to 12/12. The character F2 rose from 0.599 to 0.833, while source copying scored 0.830. All per-example outputs are saved for inspection.

These are small, selected English grammar samples. Character F2 measures similarity to one reference edit and can reward copying; the source-copy baseline must be considered. Number/negation retention is a narrow surface check, not proof that the revised text keeps its meaning. Therefore this adapter is kept as an offline experiment and is not automatically substituted into the production editing endpoint. Human review remains necessary for any suggested change.

## Extended 160-step experiment

The separate `train_adapter.py` run completed 160 optimizer steps on 200 selected CoEdIT GEC training examples, with batch size 2 and LoRA rank 8. It evaluates validation loss at steps 40, 80, 120 and 160, and saves the best validation checkpoint to `models/llm/adapter`. The saved weights came from step 120, not step 160; the full 160-step log remains in `results/exp5/training_history.jsonl`. Reproduce and verify from the project root:

```powershell
python code/llm/train_adapter.py
python code/llm/verify_adapter.py
```

The 16-example test set was reserved from the original CoEdIT validation split before training and deliberately contains eight examples with numbers or negations. The test character F2 scores are 0.586 for the base model, 0.787 for the saved adapter, and 0.771 for simply copying the source. Exact match is 0/16. The adapter returns the source unchanged for 7/16 examples, and drops the `DOTA2` identifier in one example even though the number/negation check passes 16/16. These results support only a narrow observation about the selected English samples. They do not show general reliability or semantic preservation.

`results/exp5/finetune_metrics.json` contains all losses, checkpoint values and metrics. `test_predictions.jsonl` contains paired outputs, `artifact_verification.json` records source hashes, disjoint splits and exact replay of all 16 saved predictions after reloading the adapter, and `quality_review.md` identifies concrete output errors. Earlier 12-step pilot files in `results/exp5`, such as `validation_predictions.jsonl`, are legacy exploratory output and should not be cited as the final experiment.

The rule-based editor remains the default path. The saved 160-step adapter is also available through the opt-in `POST /api/text/edit/llm-preview` endpoint and the Agent's `llm_preview` tool in `code/server`. This guarded preview rejects out-of-scope or suspicious output and never silently substitutes the model text. The observed output failures rule out treating it as an automatic editor or a production-quality semantic editor.

The installed Transformers 5.5.4 `AutoModelForCausalLM` incorrectly maps this pinned model to a different architecture in this environment. The scripts explicitly instantiate `LlamaForCausalLM` and require matching checkpoint dimensions; do not set `ignore_mismatched_sizes=True`. The run used CPU PyTorch because the installed wheel has no CUDA support, despite the computer having an NVIDIA GPU.
