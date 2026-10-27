"""Course evidence and the actual trained/optimized detector deployment."""
import json
import threading
from pathlib import Path


class CourseService:
    def __init__(self, root):
        self.root = Path(root)
        self._models = None
        self.lock = threading.Lock()

    def models(self):
        with self.lock:
            if self._models is None:
                from course_models import CourseDetectors
                self._models = CourseDetectors(self.root)
            return self._models

    def overview(self):
        folder = self.root / "results/course_20261027"
        records = {}
        for name in ("experiment1.json", "experiment2-image.json", "experiment2-text.json", "experiment3.json", "agent/finetune-metrics.json"):
            path = folder / name
            if path.is_file():
                records[name] = json.loads(path.read_text(encoding="utf-8"))
        return {"version": "2026-10-27", "stages": [
            {"number": 1, "name": "数据采集与预处理", "ready": "experiment1.json" in records,
             "description": "2026 数据、分组划分、像素与文本去重、训练集标准化"},
            {"number": 2, "name": "模型训练", "ready": "experiment2-image.json" in records and "experiment2-text.json" in records,
             "description": "本组 CNN 与中文 TF-IDF 分类器，保留逐轮训练和测试记录"},
            {"number": 3, "name": "算法优化", "ready": "experiment3.json" in records,
             "description": "真实通道剪枝、词表压缩、TorchScript 导出与前后对照"},
            {"number": 4, "name": "前后端开发", "ready": (self.root / "models/course_20261027/config.json").is_file(),
             "description": "实验二基线与实验三优化权重直接接入课程检测 API"},
            {"number": 5, "name": "大模型 Agent 与微调", "ready": "agent/finetune-metrics.json" in records,
             "description": "本地 Qwen3.5 + LoRA 选择工具、调用 API 并解释结果"}],
            "records": records,
            "deployment": {"image": "/api/course/image/detect", "text": "/api/course/text/detect",
                           "stages": ["trained", "optimized"], "model_directory": "models/course_20261027"}}
