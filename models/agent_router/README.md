---
base_model: Qwen/Qwen3.5-0.8B
library_name: peft
license: apache-2.0
tags: [lora, tool-routing, course-experiment]
---

# 本地大模型工具路由适配器

基于 Qwen 官方 `Qwen3.5-0.8B` 的文本骨干，固定 revision `2fc06364715b967f1860aea9cf38778875588b17`。
使用 LoRA rank 4，对 q_proj / v_proj 的 159744 个参数实际训练 64 步。
任务是在允许的十种工具计划中，由因果语言模型的下一 token 分数选择计划。服务器验证输入与计划，再通过真实 API 执行工具。

数据为 2026-10-27 编写的 130 条中英文课程工具指令，训练 80、验证 20、测试 30。
按验证集选择检查点；测试指令准确率为微调前 36.67%、微调后 96.67%。
完整逐项错误保存在 `results/course_20261027/agent/finetune-metrics.json`，训练损失和样本 ID 在同目录 `training-history.jsonl`。

本记录统计工具选择；图像、文本检测效果分别由各自测试集统计。
本地载入入口：`code/llm/agent_model.py`；实际工具 API：`POST /api/agent/run`。
权重重现：`python scripts/train_agent_router.py --steps 64`。
