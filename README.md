# AIGC最新版跨电脑运行包

这是2026年10月11日的运行代码和模型交接包，包含图像、文本、文档检测、可信编辑与Agent工具工作流。先读先读我_跨电脑运行与Codex交接.md。

Windows x64：Python3.12，双击安装环境.cmd，然后运行系统.cmd；浏览器打开 http://127.0.0.1:8002/?v=handoff#document 。默认CPU，CUDA另按交接说明配置。

完整包附运行模型，轻量包先执行 `py -3.12 download_runtime_weights.py`，网络受限可指定镜像。两者均需安装依赖。`verify_package.py`校验全部运行文件；模型来源和下载revision见RUNTIME_MODELS.json。

不要复制别人电脑的虚拟环境，使用本目录.venv。原始训练数据和历史实验文件夹不在此包，检测服务不依赖这些训练数据。

接收方Codex请先读AGENTS.md。目录code为最新源码，web为界面，models为实际运行模型；PACKAGE_VERIFICATION.json记录本机从独立目录实测的结果及未验证范围。README_SOURCE.md仅保留原项目历史背景。

AI特征覆盖率不是实际AI撰写比例，图像热图不是区域生成真值。本系统未完成真实平台审核上线验收。
