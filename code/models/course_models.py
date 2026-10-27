"""The group's trained and optimized course models, with fixed preprocessing."""
import io
import json
import math
import threading
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn


class CourseCNN(nn.Module):
    def __init__(self, channels=(16, 32, 64)):
        super().__init__()
        layers, previous = [], 3
        for width in channels:
            layers.append(nn.Sequential(nn.Conv2d(previous, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)))
            previous = width
        self.blocks = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(previous, 2)

    def forward(self, values):
        for block in self.blocks:
            values = block(values)
        return self.head(self.pool(values).flatten(1))


def image_tensor(image, config):
    rgb = ImageOps.exif_transpose(image).convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
    values = np.asarray(rgb, dtype=np.float32) / 255.
    values = (values - np.array(config["mean"], dtype=np.float32)) / np.array(config["std"], dtype=np.float32)
    return torch.from_numpy(values.copy()).permute(2, 0, 1)


def calibrated(value, calibration):
    z = max(-60., min(60., float(value) * calibration["coefficient"] + calibration["intercept"]))
    return 1 / (1 + math.exp(-z))


class CourseDetectors:
    def __init__(self, root):
        self.directory = Path(root) / "models/course_20261027"
        self.config = json.loads((self.directory / "config.json").read_text(encoding="utf-8"))
        torch.set_num_threads(4)
        self.models, self.lock = {}, threading.Lock()

    def model(self, modality, stage):
        if stage not in {"trained", "optimized"}:
            raise ValueError("Unknown course model stage")
        key = modality, stage
        with self.lock:
            if key not in self.models:
                if modality == "image":
                    if stage == "optimized":
                        value = torch.jit.load(str(self.directory / "image_optimized.ts"), map_location="cpu").eval()
                    else:
                        value = CourseCNN(self.config["image"]["trained_channels"])
                        value.load_state_dict(torch.load(self.directory / "image_trained.pt", weights_only=True), strict=True)
                        value.eval()
                else:
                    value = joblib.load(self.directory / ("text_compact.joblib" if stage == "optimized" else "text_full.joblib"))
                self.models[key] = value
            return self.models[key]

    def predict_image(self, image, stage="optimized"):
        with torch.inference_mode():
            logits = self.model("image", stage)(image_tensor(image, self.config["image"]).unsqueeze(0))[0]
        score = calibrated(float(logits[1]-logits[0]), self.config["image"][stage + "_calibration"])
        result = self.result(score, "image", stage)
        result.update(input_size=list(image.size), model_input_size=[64, 64],
                      model_scores=[{"model": "课程 CNN · " + stage, "score": score}], metadata_hints=[])
        return result

    def predict_text(self, text, stage="optimized"):
        if len(text.strip()) < 80 or sum("\u4e00" <= c <= "\u9fff" for c in text) < 40:
            return {"modality": "text", "risk_score": None, "decision": "uncertain", "out_of_scope": True,
                    "scope_reason": "课程文本模型使用中文正文训练，请输入至少 80 字符、40 个汉字。",
                    "applicability": "course_chinese_scope", "model_version": "course-text-"+stage+"-20261027"}
        model = self.model("text", stage)
        features = model["vectorizer"].transform([text])
        if not features.nnz:
            return {"out_of_scope": True, "scope_reason": "正文未匹配课程模型词表特征。", "applicability": "insufficient_model_features"}
        score = calibrated(model["classifier"].decision_function(features)[0], self.config["text"][stage + "_calibration"])
        return self.result(score, "text", stage)

    @staticmethod
    def result(score, modality, stage):
        return {"modality": modality, "risk_score": round(score, 6), "decision": "ai" if score >= .8 else "human" if score <= .2 else "uncertain",
                "out_of_scope": False, "threshold": .8, "model_version": f"course-{modality}-{stage}-20261027",
                "applicability": "course_2026", "signals": ["本组使用 2026 年公开样本训练，训练、验证、测试记录可在课程实验页查看。",
                "实验三压缩模型已部署到课程检测 API。" if stage == "optimized" else "此处运行实验二训练基线。"],
                "limitations": "课程模型的样本范围和实测指标见实验记录；常规检测可选择增强方案。"}
