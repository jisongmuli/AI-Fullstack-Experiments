# 最新版系统交接给另一台电脑的 Codex

这份包是2026年10月11日运行版本，包含图像、文本、文档、可信编辑和Agent页面。原始训练数据不是运行检测服务的前提。

## 选哪个压缩包

完整运行包包含实际使用的两个普通图像模型、中文与英文分类器、CIFAKE教学模型、SmolLM2底座及本地编辑适配器。模型无需再下载，但首次安装Python依赖需要网络。轻量代码包只省去三份大型公开模型权重，保留本地训练的文本分类器、教学模型及适配器；需要先运行download_runtime_weights.py恢复权重。不要把轻量包误称为离线完整包。

## Windows 快速运行

1. 安装Python 3.12，解压到可写目录，例如D:\AIGC。不要在ZIP预览窗口里启动。
2. 轻量包先执行 `py -3.12 download_runtime_weights.py`；网络不通可指定 `--mirror https://hf-mirror.com`。完整包跳过此步。
3. 执行 `py -3.12 verify_package.py`，检查文件完整性。
4. 双击安装环境.cmd，或执行 `py -3.12 setup_environment.py --with-llm`。默认安装CPU版，创建本项目.venv，不修改全局Python。
5. 双击运行系统.cmd。浏览器打开 http://127.0.0.1:8002/?v=handoff#document 。窗口保留运行，Ctrl+C停止；端口冲突时执行 `.venv\Scripts\python.exe run.py --port 8003`，不会结束别人的服务。
6. 若希望隐藏窗口运行，可以使用原有启动系统.cmd；已改为优先使用本项目.venv。检查服务.cmd、停止本项目服务.cmd保留项目PID验证。

CPU能够运行分类检测，普通图片和编辑模型会慢一些。NVIDIA CUDA可选，需先核对接收电脑显卡型号和驱动；源机器已验证的构建是torch 2.14.0+cu132，可在兼容机器执行 `py -3.12 setup_environment.py --with-llm --torch-index https://download.pytorch.org/whl/cu132`。不能把源机器12 GB测试当作所有显卡的兼容保证。其他平台的可用wheel范围为Linux x86_64/aarch64（glibc至少2.28）、macOS 14及以上Apple Silicon；使用Python3.12执行同一安装脚本，再执行 `.venv/bin/python run.py`。Intel Mac、Windows ARM、Alpine不在当前固定依赖的保证范围内。跨系统实际安装尚未实测。

## 给接收方 Codex 的任务

先阅读本文、AGENTS.md、README.md、RUNTIME_MODELS.json、environment_reference.json。核对Python3.12、系统、显卡/驱动、磁盘及网络，再检查verify_package.py。使用项目.venv，不直接修改全局Python。运行服务测试和真实上传；有错误应提供日志和具体原因，不能虚构通过。默认保持本机127.0.0.1，不擅自公开发布。

代码目录为code，界面为web，运行模型为models，启动辅助为scripts。所有路径以项目位置计算，没有固定原电脑用户名。原始实验数据和历史实验快照没有打包；训练/复现实验需使用code/download_sources.py及相关模型脚本重新获取对应数据。英文外部作文集大量标签是写作指令，不再用于验收准确率。requirements-research.txt仅用于补充数据处理/绘图实验，torchvision要匹配所选torch构建；实验模型训练另按各脚本依赖与数据步骤执行。

## 接收后的验证

`.venv\Scripts\python.exe -m unittest discover -s code/server/tests -q`

`curl http://127.0.0.1:8002/health`

真实页面需检查普通JPG/PNG、中文/英文文本、DOCX/带文本层PDF、下划线、Word/HTML/JSON导出、Agent文档与图片。web/assets中提供两个32×32教学样本。下载报告仅是提取文本标注，不保留上传文件原版式。PDF扫描件不执行OCR，旧DOC需转换DOCX。数据最多100000字符/2000片段，上传文档20MiB/200PDF页。

## 版本与限制

源机器的中文joblib是在scikit-learn1.9.0训练，服务使用1.9.1，可能出现版本警告；现有源机器加载与推理通过，但迁移到新版本必须复测，不要随意升级或掩盖警告。joblib/pickle只加载可信交接包，先验证文件指纹。

模型风险分数和AI特征覆盖率不是实际AI撰写比例。图像热图是遮挡敏感度，不是区域生成真值。正式平台盲测、概率校准和自动审核验收未完成；功能通过不能写成已达到商用准确率。

包内不含Python、虚拟环境或安装轮子，安装依赖仍需网络。离线部署需要另行准备对应操作系统、Python和CPU/CUDA版本的wheelhouse。公开模型与数据许可见来源清单，正式再分发/商用前核对全部权利链；本交接包用于课程协作。

## 补充依赖与测试说明

普通图像处理同时需要torch2.14.0与torchvision0.29.0；安装脚本从相同CPU或CUDA源安装二者并执行pip check。官方索引：https://download.pytorch.org/whl/cpu/torchvision/ 。

包内71项服务测试有1项因未附带官方C2PA测试图片而跳过，不能说每项都已执行。scripts/verify_document_workflow.py是源项目额外QA工具，需要原中文测试数据、ReportLab和Windows宋体，未包含这些数据时不能直接运行；迁移验收使用真实页面上传或PACKAGE_VERIFICATION.json里的步骤。README_SOURCE.md保留原项目阶段说明，其中历史实验文件和部分数据/结果不在交接包中。

## 本次独立环境实测

2026年10月11日，在新建目录和不读取全局包的Python3.12.10虚拟环境中，CPU版torch2.14.0与torchvision0.29.0安装成功，pip check通过。70项服务测试通过，1项跳过；真实HTTP跑通中文/英文文本、两个图像模型、DOCX/PDF/TXT上传、Word/HTML/JSON导出以及Agent文档与图片组合。编辑适配器能够执行，但测试句被质量守卫拒绝，不表示编辑质量已经验收。

安装脚本优先使用requirements-runtime-tested.txt和requirements-llm-tested.txt里的已验证版本。CPU/CUDA的torch、torchvision单独从同一个索引安装。跨操作系统仍需接收方复测。详细验证记录见PACKAGE_VERIFICATION.json；可用scripts/portable_smoke.py在新电脑重复检查。此脚本生成测试素材，不测检测准确率。

模型来源及固定版本见MODEL_SOURCES.md，Apache许可标准文本位于licenses/Apache-2.0.txt。原始数据只在来源清单中列出，检测运行不要求先下载数据。

独立CPU环境的桌面和390px手机页面均已浏览器验证：文档菜单、真实上传、三种报告下载和Agent文档流程通过，无页面异常和横向溢出。这里只证明功能流程，不证明分类准确率。数据的后续获取步骤另见数据获取与实验复现.md。
