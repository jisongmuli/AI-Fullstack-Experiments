# 本组实验二、三模型

训练数据：2026 年 NTIRE 公开训练分片新抽样、NLPCC 2026 官方训练数据。
数据划分及预处理在 `data/course_20261027/`；脚本为 `scripts/prepare_course_2026.py` 与 `scripts/train_course_2026.py`。

图像 CNN 参数由 23938 减少为 13634，使用真实通道裁剪和恢复训练。
80 张保留测试图的准确率由 68.75% 变为 66.25%；模型计算中位耗时由 0.447 变为 0.293 ms。
中文 TF-IDF 词表由 12000 压缩为 2000，600 条保留测试文本准确率由 98.33% 变为 98.50%。

课程 API：`/api/course/image/detect?stage=trained|optimized`、`/api/course/text/detect`。
图像按 64×64 RGB 和训练集通道均值、标准差处理；中文模型使用正文字符 2–4 元特征。
常规检测默认继续采用增强方案；课程页用于完整展示本组训练、压缩、部署链路与取舍。
