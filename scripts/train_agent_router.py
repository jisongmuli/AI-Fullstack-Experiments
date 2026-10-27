"""Experiment 5: genuine LoRA training of a causal-LM tool-plan decision token."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/llm"))
from agent_model import PLANS, eligible, load_base, prompt

INTENTS = {
 "A": ["介绍检测模型", "系统支持什么模型", "查看模型版本", "告诉我检测能力", "Explain model capabilities", "List detector versions", "查看支持范围", "What models are installed", "模型有哪些限制", "Show model information", "我想知道当前采用哪些检测算法", "Tell me the available detector capabilities", "介绍一下模型与适用范围"],
 "B": ["查看五次实验", "介绍课程实验", "实验一到五的结果", "展示课程成果", "Show the course experiments", "List the five experiment stages", "查看训练与压缩记录", "课程实验总览", "实验链路是什么", "Explain the course pipeline", "我要看从数据到部署的五阶段成果", "Where are the course training and optimization records", "展示课程各阶段的可复查材料"],
 "C": ["检测这段文本", "只检查文字", "判断粘贴文本", "分析文本的生成特征", "Detect the pasted text", "Analyze text only", "请检测正文", "不要看图片，只看文字", "评估这段文字", "Check this passage", "请只分析我粘贴的文章，图片不用管", "Screen the supplied passage and skip the image", "看看这篇正文的生成信号"],
 "D": ["检测这张图片", "只分析图片", "判断上传照片", "查看图像生成特征", "Detect the image", "Analyze image only", "请检查照片", "不检测文字，只看图", "评估图片", "Check this picture", "请只检查附件中的照片，正文不需要分析", "Inspect the attached photograph and ignore the text", "看看上传的图像有什么生成特征"],
 "E": ["检测这份文档", "分析附件文档", "给文档生成报告", "查看Word检测结果", "Analyze the document", "Screen the uploaded document", "请检查PDF", "全文文档检测", "评估文档段落", "Check the uploaded file", "请提取附件文章并给出分段检测报告", "Create a paragraph screening report for the uploaded document", "检查这份材料并保留原文标注"],
 "F": ["改进英文表达", "只修改语法", "润色这段英文", "给我英文编辑建议", "Edit the English passage", "Improve grammar only", "优化英文清晰度", "不要检测，只做编辑", "修正英文语法", "Correct the wording", "请只改进所给英文的语法和清晰度", "Suggest a clearer English revision without detection", "保留原意，给英文句子提修改建议"],
 "G": ["图片和文字都检测", "同时检查图文", "分别分析图片和正文", "检测两种内容", "Detect image and text", "Check both picture and passage", "图文分别给结果", "同时筛查照片与文字", "图像文字都看", "Analyze image plus pasted text", "请把图片与粘贴正文分别检测并汇总", "Screen both the photograph and pasted article separately", "文字和照片都需要检测结果"],
 "H": ["图片和文档都检测", "同时检查照片与PDF", "分别分析图像和文件", "检测文档与图片", "Detect image and document", "Check picture and uploaded document", "文档图像分别给结果", "同时筛查照片和Word", "图片文件一起分析", "Analyze image plus document", "请对照片和附件文章分别生成检测结果", "Screen both the image and the attached document", "检查图像并给文档出一份报告"],
 "I": ["检测文本后改进英文", "检查文字并给编辑建议", "检测并修改语法", "分析文本再润色", "Detect text then edit it", "Screen and improve the English passage", "文本检测与编辑都要", "先检测再优化表达", "检查英文并提改法", "Analyze and edit this text", "请检测粘贴内容，同时建议更清晰的英文表达", "Run text screening and also suggest grammar improvements", "给这段英文做检测及清晰度修改"],
 "J": ["请检测图片但我没上传", "检测文字但未提供内容", "检查PDF但没有附件", "请润色但没有正文", "Detect my image but no image is attached", "Check a document without an upload", "我还没有提交要检测的内容", "检测不存在的文件", "我没贴正文但想检测", "Edit without any supplied text", "还未上传任何材料，请先检查照片", "Please analyze the missing attachment", "正文尚未提供，先帮我检测文章"],
}


def dataset():
    rows = []
    for letter, messages in INTENTS.items():
        for index, message in enumerate(messages):
            available = {"text": letter in {"C", "F", "G", "I"}, "image": letter in {"D", "G", "H"}, "document": letter in {"E", "H"}}
            if letter in {"C", "D"} and index % 2:
                available.update(text=True, image=True)
            rows.append({"id": letter+str(index), "message": message, "available": available,
                         "letter": letter, "tools": PLANS[letter], "split": "train" if index < 8 else "validation" if index < 10 else "test",
                         "sha256": hashlib.sha256(message.encode()).hexdigest()})
    assert len({row["sha256"] for row in rows}) == len(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()
    torch.manual_seed(20261027); random.seed(20261027); torch.set_num_threads(4)
    output = ROOT / "results/course_20261027/agent"
    output.mkdir(parents=True, exist_ok=True)
    rows = dataset()
    (output / "routing-data.json").write_text(json.dumps({"source": "Author-written bilingual tool-intent teaching cases, created 2026-10-27", "records": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    model, tokenizer, codes = load_base(ROOT)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=.05, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM))
    letters = list(PLANS)
    encoded = {}
    for row in rows:
        encoded[row["id"]] = tokenizer(prompt(tokenizer, row["message"], row["available"]), return_tensors="pt", truncation=True, max_length=256)
    def evaluate(split):
        model.eval(); predictions = []
        with torch.inference_mode():
            for row in [r for r in rows if r["split"] == split]:
                logits = model(**encoded[row["id"]], use_cache=False, logits_to_keep=1).logits[0, -1].float()
                choice = max(eligible(row["available"]), key=lambda letter: float(logits[codes[letters.index(letter)]]))
                predictions.append({"id": row["id"], "expected": row["letter"], "predicted": choice, "correct": choice == row["letter"]})
        return {"accuracy": sum(r["correct"] for r in predictions)/len(predictions), "samples": len(predictions), "predictions": predictions}
    before = evaluate("test")
    print("Base routing test", before["accuracy"], flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    training = [r for r in rows if r["split"] == "train"]
    best, best_state, checkpoints = -1., None, []
    started = time.perf_counter()
    with (output / "training-history.jsonl").open("w", encoding="utf-8") as stream:
        for step in range(args.steps):
            row = training[(step*37) % len(training)]
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits = model(**encoded[row["id"]], use_cache=False, logits_to_keep=1).logits[0, -1, codes].float().unsqueeze(0)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([letters.index(row["letter"])]))
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.); optimizer.step()
            record = {"step": step+1, "sample_id": row["id"], "loss": float(loss.detach()), "elapsed_seconds": round(time.perf_counter()-started, 2)}
            stream.write(json.dumps(record)+"\n"); stream.flush()
            if (step+1) % 16 == 0 or step+1 == args.steps:
                validation = evaluate("validation")
                checkpoints.append({"step": step+1, "validation_accuracy": validation["accuracy"]})
                print("Checkpoint", checkpoints[-1], flush=True)
                if validation["accuracy"] > best:
                    best = validation["accuracy"]
                    best_state = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}
    set_peft_model_state_dict(model, best_state)
    after = evaluate("test")
    directory = ROOT / "models/agent_router"
    model.save_pretrained(directory, safe_serialization=True)
    config_path = directory / "adapter_config.json"
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    adapter_config.update(base_model_name_or_path="Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17")
    config_path.write_text(json.dumps(adapter_config, indent=2), encoding="utf-8")
    result = {"model": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17", "seed": 20261027,
              "method": "LoRA causal-LM next-token cross-entropy over ten tool-plan letter tokens", "rank": 4,
              "steps": args.steps, "training_cases": 80, "validation_cases": 20, "test_cases": 30,
              "checkpoint_selection": "validation only, earliest highest accuracy", "checkpoints": checkpoints,
              "before": before, "after": after, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "elapsed_seconds": round(time.perf_counter()-started, 2), "scope": "Small authored bilingual routing set; actual API execution is tested separately"}
    (output / "finetune-metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    from write_course_model_cards import write_agent_card
    write_agent_card(directory, result)
    print("Saved fine-tuned router", after["accuracy"], flush=True)


if __name__ == "__main__": main()
