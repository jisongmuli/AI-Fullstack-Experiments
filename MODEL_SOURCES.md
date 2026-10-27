# 运行模型来源与许可依据

## 2026-10-27 课程链路与本地大模型 Agent

课程检测使用本组在 2026 年 NTIRE / NLPCC 样本上新训练的 CNN 与中文 TF-IDF 模型，权重位于 `models/course_20261027`，对应数据划分、训练与压缩记录在 `results/course_20261027`。增强检测方案继续使用下述 DEAR / MPU。

Agent 使用 [Qwen 官方 Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)，固定 revision `2fc06364715b967f1860aea9cf38778875588b17`。完整官方权重为 1746942600 B，SHA-256 为 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`；Apache-2.0 许可原文保存在 `models/agent_qwen35/LICENSE`，逐文件来源与哈希见 `results/course_20261027/agent-assets.json`。

运行时载入官方文本骨干；图像内容由图像检测 API 分析。LoRA 工具路由由本组课程脚本实际训练 64 步、rank 4、159744 个参数，适配器为 642112 B，SHA-256 `372a5b49d5c45a112156ffc78bd8d064f3a9ed2a845ca12e7caa419c91b0f527`。80 / 20 / 30 条教学指令的划分、微调前后预测和逐步损失均保留在 `results/course_20261027/agent`。

完整五实验对应关系见 `docs/课程要求对照与验收_20261027.md`。

## 2026-10-26 开发版补充

当前运行策略为 `models/algorithm_policy.json`，中文和英文使用 MPU 作者 v3 Short，图片按分辨率选择 DEAR-r / DEAR-c。原有模型保留为历史基线。

| 模型目录 | 固定上游来源 | revision / 指纹依据 | 许可说明 |
| --- | --- | --- | --- |
| `models/mpu_zh_v3` | [yuchuantian/AIGC_detector_zhv3short](https://huggingface.co/yuchuantian/AIGC_detector_zhv3short) | `7bf6e1d6496353a88365f1dcdb3a6ace35f46b39`，权重 SHA-256 `f64faae413291a78be2cb36dd9889a916a6202a03b6f6f904d5579089b31f2a8` | 作者模型卡与 [MPU 代码](https://github.com/YuchuanTian/AIGC_text_detector) 声明 Apache-2.0 |
| `models/mpu_en_v3` | [yuchuantian/AIGC_detector_env3short](https://huggingface.co/yuchuantian/AIGC_detector_env3short) | `32a89cd8ace09f3eccdf930bd0a3f80f77b55d27`，权重 SHA-256 `f2d08394f64b0481565ad77c1ead347b62f546cb360751ca82f93a026d32cd4f` | 同上 |
| `models/dear_r` | [k-aisi-anti-deepfake/dear-checkpoints](https://huggingface.co/k-aisi-anti-deepfake/dear-checkpoints) | `5b57350b0ee75553109b3844f3c8a9341fc7e707`，权重 SHA-256 `430fde11debe1850ab24af43945b4d68f2bb3ee52a837d84cf525eb603ccde97` | CC BY-NC 4.0，附 CreativeML OpenRAIL-M 使用条件；原文保存于 `licenses/dear/` |
| `models/dear_c` | 同上 | 同 revision，权重 SHA-256 `c8fa321d0cac86d36def6508f787f65bf22420d4648722956ad00375a347c7bc` | 同上 |
| `models/taste` | [TASTE 作者实现与公开权重链接](https://github.com/Liyuuuu111/MGT-Eval/tree/master/src/detectors/finetuned/TASTE) | 官方链接权重包 SHA-256 `e9cd44fa5ab134ce0c2061438368036109729429fdaa534f8f54d2753d05a80e`；提取权重 SHA-256 `f79328c19ec0f1044dd24dd6eb9c5d5c96aa01b7036765251be9a426b9a46cbf` | 本次作为研究对照保留；公开包未附单独权重许可证，按上游说明记录来源 |

MPU 主仓库截至本次核查列出的最新模型系列为 2025 年 6 月 v3；方法论文为 ICLR 2024。DEAR 论文为 ICML 2026，权重于 2026 年 7 月发布。中文生产校准参数来自 NLPCC 2026 Phase 1；小图模型选择和参数来自单独 NTIRE 验证切片。

模型下载与数据获取脚本使用固定 revision、大小与 SHA-256。逐文件记录位于 `results/upgrade_20261026/asset-receipts.json`、`mpu-assets.json`、`phase1-assets.json`，数据许可与出处保留在原数据目录说明中。

以下保留原始 2026-10-11 交接包的来源记录。

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
