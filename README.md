# 内容可信工作台

当前开发版：2026-10-26。系统包含图像检测、中英文文本检测、文档检测和可信编辑四个模块。

- 图片：ICML 2026 DEAR，按分辨率使用 DEAR-r / DEAR-c 与各自校准参数；支持 4K 及以下任意正尺寸的静态 JPG/PNG。
- 文本：MPU 作者当前公开的中英文 v3 Short 权重；中文使用 NLPCC 2026 Phase 1 校准，长文本通过 512 token 滑窗覆盖。
- 文档：DOCX、带文字层的 PDF、TXT、Markdown；进度、取消、后台继续、刷新恢复；Word / HTML / JSON 报告与记录管理。
- 编辑：英文清晰度建议、数字等内容检查，以及可选的本地大模型编辑预览。

本机新版入口为 **http://127.0.0.1:8005/**。变更、来源与样本实测见 [本轮升级记录](docs/20261026_模型升级与Agent移除.md)，原始记录位于 `results/upgrade_20261026/`。

模型信息查询采用轻量文件检查，图像与文本模型按需加载；图片在成功解码后开放检测，损坏文件会清除旧预览。耗时与回归结果见 [启动与预览优化记录](docs/启动与预览优化_20261026.md)。

## 获取与运行

从公开仓库 `codex/system-stability` 分支获取代码，执行 `git lfs pull` 恢复模型与实验大文件。使用 Python 3.12 和项目 `.venv`：

```powershell
py -3.12 setup_environment.py --with-llm
.venv\Scripts\python.exe run.py
```

新环境默认地址为 `http://127.0.0.1:8002/`。端口被占用时，使用 `run.py --port 8005` 或其他空闲端口。服务默认仅监听本机，当前电脑使用 CPU 推理。

`verify_package.py` 校验运行文件；新样本与模型指纹用 `scripts/verify_2026_assets.py` 核查。检测运行无需训练数据，数据与中间特征用于实验复现。

模型来源与许可见 `MODEL_SOURCES.md`。原始 2026-10-11 运行包与交接说明保留在仓库中，`README_SOURCE.md` 保留更早的实验背景。报告高亮统计表示字符覆盖率，图片热图表示模型敏感度。

接收项目的 Codex 先阅读 `AGENTS.md`。
