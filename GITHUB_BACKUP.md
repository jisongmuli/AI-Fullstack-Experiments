# GitHub 私有备份说明

本仓库保存 AI 全栈开发实验项目的运行代码、运行模型、实验资料和报告。

## 内容

- `code`、`web`、`scripts`：项目源码、页面与运行辅助。
- `models`：完整运行模型，通过 Git LFS 保存大型二进制文件。
- `data`、`results`、`docs`：原交接包附带的数据来源清单、质量记录和实验说明。
- `reports/实验1`：用户提供的实验一报告。
- `reports/实验3`：25 页实验三报告，以及测量脚本、原始统计、图表、模板参照和核验记录。
- `source-attachments`：原始完整运行包、交接说明、实验三任务书和原始报告模板。
- `logs`：保留本机搭建和功能验证资料；实时服务进程记录与服务输出日志不纳入版本控制。

## 获取完整文件

先安装 Git 与 Git LFS，然后克隆仓库并拉取大文件：

```powershell
git lfs install
git clone https://github.com/jisongmuli/AI-Fullstack-Experiments.git
cd AI-Fullstack-Experiments
git lfs pull
```

仓库访问需要所属 GitHub 账号或获得授权的账号。网页上的大型模型文件以 LFS 指针管理，完整文件由 Git LFS 拉取。

## 本机运行

项目需要 Python 3.12。依赖环境不随仓库上传，在仓库根目录执行：

```powershell
py -3.12 setup_environment.py --with-llm
.venv\Scripts\python.exe run.py
```

页面地址为 `http://127.0.0.1:8002/`。本机部署细节见 `本机搭建说明.md`，模型来源及许可见 `MODEL_SOURCES.md` 和 `licenses`。

## 报告资料

最终实验三报告为 `reports/实验3/实验3报告_算法优化实验.docx`。`work` 内保留正文源稿、测量记录、图表及构建脚本。反复生成的页面 PNG 和中间 PDF 可由渲染流程重新生成，因此未上传重复副本。

本次备份不改变模型、业务功能和既有实验结果。原始训练数据仍按交接包的来源清单获取。
