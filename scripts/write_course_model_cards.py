"""Generate portable course model cards from the recorded measurements."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_agent_card(directory, result):
    (directory / "README.md").write_text(f'''---
base_model: Qwen/Qwen3.5-0.8B
library_name: peft
license: apache-2.0
tags: [lora, tool-routing, course-experiment]
---

# 本地大模型工具路由适配器

基于 Qwen 官方 `Qwen3.5-0.8B` 的文本骨干，固定 revision `{result['revision']}`。
使用 LoRA rank 4，对 q_proj / v_proj 的 {result['trainable_parameters']} 个参数实际训练 {result['steps']} 步。
任务是在允许的十种工具计划中，由因果语言模型的下一 token 分数选择计划。服务器验证输入与计划，再通过真实 API 执行工具。

数据为 2026-10-27 编写的 130 条中英文课程工具指令，训练 80、验证 20、测试 30。
按验证集选择检查点；测试指令准确率为微调前 {result['before']['accuracy']:.2%}、微调后 {result['after']['accuracy']:.2%}。
完整逐项错误保存在 `results/course_20261027/agent/finetune-metrics.json`，训练损失和样本 ID 在同目录 `training-history.jsonl`。

本记录统计工具选择；图像、文本检测效果分别由各自测试集统计。
本地载入入口：`code/llm/agent_model.py`；实际工具 API：`POST /api/agent/run`。
权重重现：`python scripts/train_agent_router.py --steps 64`。
''', encoding="utf-8")


def main():
    folder = ROOT / "results/course_20261027"
    agent = json.loads((folder / "agent/finetune-metrics.json").read_text(encoding="utf-8"))
    write_agent_card(ROOT / "models/agent_router", agent)
    image = json.loads((folder / "experiment2-image.json").read_text(encoding="utf-8"))
    text = json.loads((folder / "experiment2-text.json").read_text(encoding="utf-8"))
    (ROOT / "models/course_20261027/README.md").write_text(f'''# 本组实验二、三模型

训练数据：2026 年 NTIRE 公开训练分片新抽样、NLPCC 2026 官方训练数据。
数据划分及预处理在 `data/course_20261027/`；脚本为 `scripts/prepare_course_2026.py` 与 `scripts/train_course_2026.py`。

图像 CNN 参数由 {image['trained']['parameters']} 减少为 {image['optimized']['parameters']}，使用真实通道裁剪和恢复训练。
80 张保留测试图的准确率由 {image['trained']['accuracy']:.2%} 变为 {image['optimized']['accuracy']:.2%}；模型计算中位耗时由 {image['trained']['latency']['median_ms']:.3f} 变为 {image['optimized']['latency']['median_ms']:.3f} ms。
中文 TF-IDF 词表由 {text['trained']['features']} 压缩为 {text['optimized']['features']}，600 条保留测试文本准确率由 {text['trained']['accuracy']:.2%} 变为 {text['optimized']['accuracy']:.2%}。

课程 API：`/api/course/image/detect?stage=trained|optimized`、`/api/course/text/detect`。
图像按 64×64 RGB 和训练集通道均值、标准差处理；中文模型使用正文字符 2–4 元特征。
常规检测默认继续采用增强方案；课程页用于完整展示本组训练、压缩、部署链路与取舍。
''', encoding="utf-8")


if __name__ == "__main__": main()
