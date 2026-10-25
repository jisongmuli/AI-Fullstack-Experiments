# 多模态AIGC内容检测与可信编辑系统 五次实验

本目录包含实验一至五的代码、数据、模型、结果和报告。每次实验有独立成果文件夹；实验二至五的文件夹内各有报告 DOCX、详细讲解 DOCX、对应代码快照和关键结果。根目录 `code/`、`models/`、`results/`、`web/` 是运行时的权威版本；各实验文件夹中的代码用于按阶段查阅与提交。

| 阶段 | 已完成内容 | 成果入口 |
| --- | --- | --- |
| 实验一 | 来源核验、清洗增强、标准化、基线与数据卡 | `实验1_数据采集与预处理/`；根目录原件及 `DATA_CARD.md` |
| 实验二 | 图像 CNN 与英文文本 TF-IDF 分类器训练和测试 | `实验2_模型训练/` |
| 实验三 | TorchScript 导出、词表压缩、计时与弃判 | `实验3_算法优化/` |
| 实验四 | FastAPI、浏览器界面、可信编辑与前后端联调 | `实验4_前后端开发/` |
| 实验五 | Agent 工具编排、真实 LoRA 训练、受控大模型编辑预览 | `实验5_Agent与大模型适配/` |

## 启动系统

Windows 可直接双击根目录的 [启动系统.cmd](启动系统.cmd)。脚本检查 127.0.0.1:8002：已有健康的本项目服务就复用；端口被其他进程占用则提示，不会重启或结束该进程。成功后自动打开界面。双击 [检查服务.cmd](检查服务.cmd) 可查看状态；[停止本项目服务.cmd](停止本项目服务.cmd) 只会停止由启动脚本记录且通过 PID、启动时间、命令行、路径和端口校验的进程，手动启动的服务不会被它停止。启动输出和错误日志保存在 logs 文件夹。脚本需要 Python 3.12 及服务依赖，缺失时会显示安装命令。

使用 Python 3.12，在项目根目录安装 `code/server/requirements.txt`；需要大模型预览时再安装 `code/llm/requirements.txt`。运行：

```powershell
python -m uvicorn app:app --app-dir code/server --host 127.0.0.1 --port 8002
python -m unittest discover -s code/server/tests -v
```

界面在 `http://127.0.0.1:8002/`，交互式 API 文档在 `/docs`。32×32 CIFAKE 图像使用原教学模型；其他 JPG/PNG 图片在短边至少 200 像素时使用 SigLIP 与 ViT 两个候选模型交叉检查，显示双模型分数、均分、分歧及可用时的九宫格遮挡敏感度。289×202 这类小图现在会返回实验性模型风险指数；短边低于 200 像素的缩略图仍弃判。热图只表示遮住某格对模型输出的影响，**不是该区域由 AI 生成的概率或位置真值**；整图分数也不是“AI 像素比例”，普通图的来源结论未经过平台校准。详见 [小图与局部敏感度说明](docs/2026-10-11_小图候选分数与局部敏感度.md)。系统另对 JPG/PNG 上传原始字节进行**本地 C2PA 内容凭证核验**，单独显示凭证状态与经过核验的有限声明；没有凭证不表示 AI 生成，签名有效但签发者未受信任也不能作为来源证明。核验禁用远程清单抓取和在线撤销查询，不返回完整清单、签发者身份或私人元数据。12 GB 显存机器已实测双模型与九宫格 4K 单请求推理；显存、延迟及限制见 [普通图片与显存验证](docs/普通图片与12GB显存验证.md)。

文本检测支持最多 10,000 字符的中文或英文输入，返回逐段原文位置、风险分数和疑似段落高亮。明显的写作指令、Markdown 引用块、代码围栏及过短或超范围段落不评分；整篇超出已评估长度、可评分非空白字符不足 80%，或段落分数同时出现 ≤0.2 与 ≥0.8 的冲突信号时，不给单一整篇分数，只展示可评分段落的实验性结果。高亮阈值为 0.99，页面显示的“高亮字符占已评分字符比例”**不是 AI 撰写比例**。Agent 页面可上传一张图片、导入 UTF-8 `.txt/.md` 文件或直接粘贴文本，分别显示图像、文本与图像内容凭证结果；接口为 `POST /api/agent/upload`，原 `POST /api/agent/run` 仍可用于 JSON 调用。图文没有经验证的融合模型，不生成统一“AI 率”。检测结果仅用于人工筛查，不能认定作者身份；大模型编辑预览需要作者确认。**当前未达到正式应用验收门槛**：普通图像跨来源误报仍高，真实平台文本和混合作者段落没有独立盲测，编辑模型仅有 16 条独立测试且发生过专名丢失。进一步要求见 [上线验收清单](上线验收清单.md)。

## 文档检测与下划线报告

新增“文档检测”页面：DOCX、带文本层的 PDF、TXT 和 Markdown 可直接上传，最大 20 MiB、提取后最多 100,000 字符，PDF 最多 200 页；旧版 DOC 请转为 DOCX，扫描 PDF 不执行 OCR。结果显示 **AI 特征覆盖率**、模型评分覆盖率、片段分数分布及原文下划线，支持来源位置导航、未评分提示和 Word / HTML / JSON 导出。HTML 可通过浏览器打印为 PDF。百分比是高风险片段字符覆盖率，**不是真实 AI 撰写比例**；原文标注依据是整段模型输出，不是逐字来源真值。

Agent 现在可上传文档，与图片分别检测；已完成的报告可直接交给 Agent 汇总而不重新调用模型。报告仅暂存在本地内存中，最多 30 分钟、20 份、32 MiB，并进行后台过期清理；浏览器记录默认不保存，主动勾选时只保存摘要，原文不进入本地历史。报告包含文件指纹、时间、模型版本及片段依据。详细步骤、百分比分母与格式限制见 [文档检测使用说明](docs/2026-10-11_文档检测使用说明.md)。

新增文档测试覆盖提取边界、异常文件、长文、emoji 定位、分母、导出、缓存与 PDF 段落间距；当前服务测试共 71 项通过。真实浏览器已跑通三种文件格式、下载、Agent 图文流程及 390px 手机布局。公开混合样本中 TXT / DOCX 原文和覆盖指标一致，PDF 恢复段落后只标注同一高风险片段；PDF 的不同字符量导致全文分母不同。99,736 字符长文保留完整原文、364 个片段，暖机后一次实测约 0.63 秒。记录见 `results/document_qa/real-workflow-checks.json`、`ui-live-checks.json` 和 `ui-positive-checks.json`。这些样本按分数范围挑选用于流程验证，**不能视为独立准确率评估、并发压测或通用性能保证**。

## 当前补充验证与适用边界

`results/provenance_evaluation.json` 记录 4 张官方 C2PA JPEG 测试文件和 1 张重新编码图片的凭证状态检查，5/5 与预期一致；这只验证凭证状态处理，**不是图像 AI 检测准确率**。`results/text_policy_v2_audit.json` 记录文本上下文弃判与高亮策略的回顾性审计；其中原英文外部作文集 1,045 条 AI 标签记录有 961 条实际上以写作指令开头，均被该策略弃判。

另从 [OwensLab CommunityForensics-Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval) 固定抽取 Firefly Image2 生成图和 LAION 真实图各 50 张，统一处理为 512×512 JPEG 做跨来源研究审计（`results/highres/communityforensics_firefly_holdout.json`）。双模型均分 AUC 为 0.752；在 0.8 离线界线下，真实图误报 5/50（10%），生成图命中 30/50（60%）；要求**两个模型都 ≥0.8**时仍误报 5/50，命中 29/50（58%）。该数据集为 CC BY-NC-SA 4.0、仅限研究教育用途；原全集约 206 GB，本项目只抽取小样本，没有下载全集。这 100 张样本此前未用于本地训练、选模或调阈值，但上游快照未固定、预训练重叠不能排除，来源、内容及裁剪也可能混杂；这些数字**不能证明模型已提升或可供商业平台验收**。早期公开集已参与模型与策略开发，仍缺少目标平台独立盲测和混合作者文本的逐段真值。待完成事项见 [上线验收清单](上线验收清单.md)。

## 实验一 数据与基线

实验一输入包含公开的 CIFAKE 小图像、两个英文文本集及从图像计算的结构化元数据。标签统一为 `0=真实图像/人类文本`、`1=生成图像/AI文本`。图像与文本彼此**没有配对关系**，不构造伪造图文对。

## 公开来源与下载

| 数据 | 页面及直接下载 | 本次用途 | 平台许可字段和核验限制 |
| --- | --- | --- | --- |
| CIFAKE | [页面](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) / [ZIP](https://www.kaggle.com/api/v1/datasets/download/birdy654/cifake-real-and-ai-generated-synthetic-images) | 图像训练、验证、测试 | 平台字段为 Other；页面正文称 MIT 条件，并要求引用 CIFAR-10 与 CIFAKE。原始 CIFAR-10 的再利用条件须独立核对。 |
| AI and Human Text Dataset | [页面](https://www.kaggle.com/datasets/hasanyiitakbulut/ai-and-human-text-dataset) / [ZIP](https://www.kaggle.com/api/v1/datasets/download/hasanyiitakbulut/ai-and-human-text-dataset) | 文本训练、验证、测试 | 平台字段 CC BY 4.0；实际 CSV 缺少页面声称的逐条 `source` 字段，原始人类文本权利链不完整。 |
| Human vs AI Generated Essays | [页面](https://www.kaggle.com/datasets/navjotkaushal/human-vs-ai-generated-essays) / [ZIP](https://www.kaggle.com/api/v1/datasets/download/navjotkaushal/human-vs-ai-generated-essays) | 原拟作英文外部测试；标签审计后停用该准确率 | 平台字段 CC0；原始作者、生成模型和逐条来源缺失，页面总量写为 N，实际 ZIP 为 2750 行；1,045 条 AI 标签记录中有 961 条以写作指令开头。 |

原始 ZIP、下载时间、说明、SHA-256 保存在 `data/raw/*_source.json`。本地处理后数据集在 `data/images/processed/`、`data/text/*.jsonl` 和 `data/images/image_metadata_*.csv`。**没有将处理后数据上传公网，因此不存在处理后数据集的公网下载链接。**需再次获取时运行下载脚本，并遵守源站条件。

## 目录与字段

```text
AIGC实验1/
  code/                  download_sources.py, prepare_dataset.py,
                         evaluate_augmentation.py, audit_confounds.py, audit_privacy.py
  data/raw/              三个原始 ZIP 与各自来源说明 JSON
  data/images/           原始审计清单、划分清单、processed/{train,validation,test}/{0_real,1_fake}/*.png
  data/text/             原始清洗/长度匹配清单、train/validation/test/external_test.jsonl
  data/images/image_metadata*.csv  亮度、对比度、文件大小及归一化值
  results/               质量审计、基线指标、逐样本预测、图表和图像模型权重
  DATA_CARD.md           数据集卡、适用范围和隐私核查
```

`split_manifest.csv` 记录 `sample_id,source_path,source_split,source_class,label,width,height,format,file_bytes,brightness,contrast,file_sha256,pixel_sha256,duplicate_of,error,split,relative_path`。文本 JSONL 含 `sample_id,text,label,source_dataset`；完整文本、摘录位置及长度保存在相应 CSV 清单，便于回溯。图像均为 32×32 RGB PNG；主文本处理后字符长度 23 至 587，中位数 331。英文文本太短或非英语文本不属于本次可靠适用范围。

## 处理链与运行

使用 Python 3.12。依赖包括 `numpy,pandas,requests,beautifulsoup4,Pillow,scikit-learn,torch,torchvision,nltk,matplotlib`。按顺序运行：

```powershell
python code/download_sources.py
python code/prepare_dataset.py
python code/evaluate_augmentation.py
python code/audit_confounds.py
python code/audit_privacy.py
```

下载脚本公开获取并校验 ZIP。准备脚本审计所有 12 万张图像的解码、尺寸和像素 SHA-256，排除重复后从原训练池取训练/验证，从原测试池取测试；转换 5600 张 PNG。文本执行 HTML 实体解码、标签映射、空白归一化、低质量筛查、文本去重、按 AI 类词数随机采样人类原文的连续摘录，再按 8:1:1 分层划分。外部集曾单独去重、长度匹配并用于早期外测，后来因标签问题撤销了该准确率证据。所有随机步骤固定种子 42。结构化元数据在图像训练集拟合 Min-Max 缩放器，然后只变换验证/测试集，避免统计量泄漏。隐私扫描脚本只保存模式命中数和样本 ID，不保存疑似个人信息原文。

增强仅作用训练集：图像随机水平翻转与 ±8° 旋转；文本随机挑选约 25% 的训练样本，各替换一个保守同义词。验证与测试集保持不变。图像三层小 CNN、文本 TF-IDF + 逻辑回归只用于实验一受控比较。`results/*.json` 有所有精确指标、训练参数和版本；`results/*predictions.csv` 有逐样本预测。另用 `log1p(字符数)` 单特征回归做数据捷径审计。

## 关键结果与解释

图像原包 120000 张，像素级重复 1336 张，解码失败 0；处理后 5600 张，训练/验证/测试 4000/800/800，所抽样本无像素哈希交叉。文本主集 6069 条，去重后仍为 6069；原人类/AI 文本字符长度中位数 1871/332，匹配后 330/332。外部集 2750 行，精确重复 330，剩余 2420。只看长度的原始主集测试准确率 99.84%，匹配后 50.74%，证明原数据的明显捷径被削弱，但不证明所有混杂都消失。

图像小 CNN 原始/增强测试准确率为 81.25%/81.63%；文本同源测试为 95.88%/96.21%。早期报告曾将外部集的 55.21%/60.12% 写作“外部测试准确率”；后续逐条审计发现其中多数 AI 标签实际对应写作指令，而人类标签多为作文片段，因此**该数值不能作为 AI 成稿检测准确率**。这些是一次固定划分的小样本教学比较，增强收益未做置信区间和显著性检验。更多质量、许可、隐私和使用限制见 [数据集卡](DATA_CARD.md)；段落审计见 `results/text_segment_policy_audit.json`。
