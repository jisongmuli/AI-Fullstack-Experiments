# 运行模型来源与许可依据

核查日期：2026 年 10 月 11 日。本文件由交接包整理者编写，不是上游原始模型卡或作者签署的许可证明。

## 固定版本

下表与本包 RUNTIME_MODELS.json 一致。完整包包含权重，轻量包由 download_runtime_weights.py 恢复同一版本，并核对大小和 SHA-256。

| 本地目录 | 上游仓库 | 固定 revision | 声明依据 |
|---|---|---|---|
| models/highres_image | Ateeqq/ai-vs-human-image-detector | 60e82406916921b823616bee33397baab38af3f0 | 固定版本模型卡声明 Apache-2.0 |
| models/highres_nyuad | NYUAD-ComNets/NYUAD_AI-generated_images_detector | b0a2e665f6e40164436cae6c0c95e7e4386f7684 | 官方模型卡声明 Apache-2.0；main 当前提交对应此版本 |
| models/llm/base | HuggingFaceTB/SmolLM2-135M-Instruct | 12fd25f77366fa6b3b4b768ec3050bf629380bac | 固定版本模型卡声明 Apache-2.0 |

标准许可全文随包提供于 [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt)，来源为 [Apache Software Foundation](https://www.apache.org/licenses/LICENSE-2.0.txt)。标准文本没有替模型作者添加未经核实的版权姓名。

## 两个图像候选模型

[Ateeqq 固定版本模型卡](https://huggingface.co/Ateeqq/ai-vs-human-image-detector/blob/60e82406916921b823616bee33397baab38af3f0/README.md) 明确声明 Apache-2.0。[该版本文件列表](https://huggingface.co/Ateeqq/ai-vs-human-image-detector/tree/60e82406916921b823616bee33397baab38af3f0) 未显示独立 LICENSE 或 NOTICE。上游卡提醒有用户反映过拟合，不能把上游自报成绩当成本系统跨平台准确率。

[NYUAD 官方模型卡](https://huggingface.co/NYUAD-ComNets/NYUAD_AI-generated_images_detector/blob/main/README.md) 明确声明 Apache-2.0，并保留 Nouar AlDahoul、Yasir Zaki 的研究和仓库引用信息。[文件列表](https://huggingface.co/NYUAD-ComNets/NYUAD_AI-generated_images_detector/tree/main) 最新提交对应本包的 [固定 revision](https://huggingface.co/NYUAD-ComNets/NYUAD_AI-generated_images_detector/commit/b0a2e665f6e40164436cae6c0c95e7e4386f7684)；未显示独立 LICENSE 或 NOTICE。补取原卡时应保留其作者引用。

原始模型卡固定版本下载地址：

- [Ateeqq README](https://huggingface.co/Ateeqq/ai-vs-human-image-detector/resolve/60e82406916921b823616bee33397baab38af3f0/README.md)
- [NYUAD README](https://huggingface.co/NYUAD-ComNets/NYUAD_AI-generated_images_detector/resolve/b0a2e665f6e40164436cae6c0c95e7e4386f7684/README.md)

浏览工具可读取官方模型卡；本机直接请求官方原始文件及镜像时出现连接重置或 SSL EOF，未取得可验证的原始 README 字节。因此本包两个图像目录的 README 均明确注明“本项目来源说明”，不是上游原文。接收方可按上述固定地址补取原卡，并在改包后重新生成完整性清单。

## 英文编辑底座与本地适配器

实际底座为 HuggingFaceTB/SmolLM2-135M-Instruct，不是 Qwen2.5。[固定版本官方模型卡](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md) 声明 Apache-2.0；可从 [原始 README 地址](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md) 补取原卡。本地 base README 已更正为本项目来源说明。应使用统一标准 Apache-2.0 文本，不能把旧 Qwen 的 Alibaba Cloud 版权声明解释为 SmolLM2 的版权归属。

models/llm/adapter 是本地课程训练的英文语法与清晰度 LoRA 适配器。其 README 保留训练配置与局限，并记录训练数据为 [grammarly/coedit](https://huggingface.co/datasets/grammarly/coedit) GEC 子集、固定 revision e9a255c33ef910bc33a9d2b522653fa87521583e。部分 README 引用的历史训练结果不在运行包中，不能推定已随包提供。服务显式加载本地底座对象后加载适配器，不依赖训练机器绝对路径。编辑输出没有事实正确性或语义等价保证。

## 本地分类器的数据来源

- image_detector.ts / image_detector.pt：本地 CIFAKE 教学模型；数据发布者许可声明及原作者引用见 [data/raw/cifake_source.json](data/raw/cifake_source.json)。
- text_detector.joblib / text_detector_full.joblib：本地英文分类器；主数据发布者的 CC BY 4.0 声明及来源见 [data/raw/text_primary_source.json](data/raw/text_primary_source.json)。
- text_detector_zh.joblib：本地中文统计分类器，使用 HC3-Chinese 和 AnxForever 的筛选子集；固定数据版本、来源及许可限制见 [results/chinese_text/source_manifest.json](results/chinese_text/source_manifest.json)。其中 HC3 记录 CC BY-SA 4.0，Anx 发布者声明 MIT，底层转载来源仍需核查。这些数据声明不能自动解释成所有训练产物的一致许可。

原始训练语料未打包。来源记录没有宣称已完成全部商用权利链核查。模型风险分数、字符覆盖率及图像遮挡敏感度不能证明内容来源，运行包没有独立平台上线验收结论。
