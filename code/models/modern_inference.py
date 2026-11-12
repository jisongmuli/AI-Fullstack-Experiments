"""Local 2026 inference adapters with recorded preprocessing and calibration."""
from __future__ import annotations

import json
import math
import re
import threading
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from research_detectors import DearDetector, TasteDetector, MpuDetector
from text_segments import noncontent_reason
from runtime_assets import text_policy


def sigmoid(value):
    return 1. / (1. + math.exp(-max(-60., min(60., float(value)))))


def unscored(applicability, reason):
    return {"modality": "text", "risk_score": None, "decision": "uncertain", "out_of_scope": True,
            "applicability": applicability, "scope_reason": reason, "signals": [reason],
            "limitations": "请提供完整的中英文自然语言正文。"}


class ModernTextAdapter:
    def __init__(self, root: Path, english_vocabulary, chinese_vocabulary):
        self.root = root
        self.pipeline = text_policy(root)
        needs_taste = any(self.pipeline.get(language + "_backend", "taste") == "taste"
                          for language in ("chinese", "english"))
        self.policy = json.loads((root / "models/taste/policy.json").read_text(encoding="utf-8")) if needs_taste else {}
        self.engine = None
        self._mpu_engines, self._load_lock = {}, threading.Lock()
        self.vocabularies = {"english": english_vocabulary, "chinese": chinese_vocabulary}
        if self.pipeline.get("chinese_backend", "taste") == "taste":
            with np.load(root / "models/taste/chinese_head.npz", allow_pickle=False) as head:
                self.coefficient, self.intercept = head["coefficient"], float(head["intercept"])
            if self.coefficient.shape != (768,) or not np.isfinite(self.coefficient).all() or not math.isfinite(self.intercept):
                raise ValueError("Invalid local Chinese classifier head")

    def _engine(self, language, backend):
        with self._load_lock:
            if backend == "taste":
                if self.engine is None:
                    self.engine = TasteDetector(self.root)
                return self.engine
            if language not in self._mpu_engines:
                self._mpu_engines[language] = MpuDetector(self.root, language)
            return self._mpu_engines[language]

    def predict_many(self, texts):
        predictions, languages, active = {}, {}, []
        for text in dict.fromkeys(texts):
            context = noncontent_reason(text)
            if context:
                predictions[text] = unscored(*context)
                continue
            han = sum("\u4e00" <= c <= "\u9fff" for c in text)
            alpha = sum(c.isalpha() for c in text)
            chinese = han >= 40 and han / max(alpha, 1) >= .45
            words = re.findall(r"[A-Za-z]+", text)
            valid = (len(text) >= 80 and han >= 40) if chinese else (
                len(text) >= 25 and len(words) >= 5 and sum(map(len, words)) / max(alpha, 1) >= .65)
            if not valid:
                predictions[text] = unscored("outside_training_scope", "该段正文较短或语言超出中英文模型范围。")
            else:
                languages[text] = "chinese" if chinese else "english"
        for language in ("english", "chinese"):
            group = [text for text, lang in languages.items() if lang == language]
            if not group:
                continue
            normalized = [re.sub(r"\s+", " ", re.sub(r"[*#`_]+", " ", unicodedata.normalize("NFKC", t))).strip()
                          if language == "chinese" else t for t in group]
            features = self.vocabularies[language].transform(normalized)
            for text, count in zip(group, features.getnnz(axis=1)):
                if count:
                    active.append(text)
                else:
                    predictions[text] = unscored("insufficient_model_features", "未匹配到有效语言特征，请补充完整正文。")
        if active:
            pipeline = getattr(self, "pipeline", {})
            for language in ("english", "chinese"):
                group = [text for text in active if languages[text] == language]
                if not group:
                    continue
                backend = pipeline.get(language + "_backend", "taste")
                chinese = language == "chinese"
                if backend == "mpu":
                    margins = self._engine(language, backend).logits(group)
                    calibration = pipeline[language + "_calibration"]
                    version = "MPU-v3-short-" + ("zh" if chinese else "en")
                    evidence = ["采用 ICLR MPU 项目作者发布的中英文专项 v3 Short 权重。",
                                "长文本按 512 token 滑窗覆盖，窗口按新增 token 数加权。"]
                else:
                    engine = self.engine if self.engine is not None else self._engine(language, backend)
                    logits, hidden = engine.representations(group)
                    selected_head = pipeline.get("chinese_taste_head", self.policy["chinese_head"])
                    adapted = chinese and selected_head == "adapted"
                    margins = (hidden @ self.coefficient + self.intercept).tolist() if adapted else logits
                    calibration = pipeline.get(language + "_calibration", self.policy[language + "_calibration"])
                    version = "TASTE-ICLR2026" + ("-zh-" + selected_head if chinese else "-en-official")
                    evidence = ["TASTE 多语言 Transformer 编码器，使用词典驱动对抗训练。",
                                "中文分类头使用 NLPCC 2026 训练样本适配。" if adapted else "使用作者发布的分类头。"]
                for text, margin in zip(group, margins):
                    score = sigmoid(margin * calibration["coefficient"] + calibration["intercept"])
                    predictions[text] = {"modality": "text", "risk_score": round(score, 6),
                        "decision": "ai" if score >= .8 else "human" if score <= .2 else "uncertain",
                        "out_of_scope": False, "threshold": .8, "model_version": version,
                        "applicability": "research_2026_" + language, "signals": list(evidence),
                        "limitations": "中英文分别评估；混合信号与短段请结合逐段结果复核。"}
        return [{**predictions[text], "signals": list(predictions[text]["signals"])} for text in texts]


class ModernImageAdapter:
    def __init__(self, root: Path):
        self.policy = json.loads((root / "models/dear_r/policy.json").read_text(encoding="utf-8"))
        self.engine = DearDetector(root, self.policy["variant"], tile_size=self.policy["tile_size"])
        self.root, self._secondary, self._load_lock = root, None, threading.Lock()

    def selection(self, image):
        policy = self.policy.get("low_resolution", {})
        if not policy or min(image.size) >= policy["short_side_below"]:
            return self.engine, self.policy["calibration"], self.policy["variant"]
        if policy["variant"] == self.policy["variant"]:
            engine = self.engine
        else:
            with self._load_lock:
                if self._secondary is None:
                    self._secondary = DearDetector(self.root, policy["variant"], tile_size=policy["tile_size"])
                engine = self._secondary
        return engine, policy["calibration"], policy["variant"]

    def score(self, image, selection=None):
        engine, calibration, _ = selection or self.selection(image)
        return sigmoid(engine.logit(image) * calibration["coefficient"] + calibration["intercept"])

    def predict(self, image: Image.Image, *, include_local_evidence=False):
        from highres_inference import blur_grid, metadata_hints
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        selection = self.selection(rgb)
        engine, _, variant = selection
        score = self.score(rgb, selection)
        result = {"modality": "image", "risk_score": round(score, 6),
            "decision": "uncertain" if min(rgb.size) < 16 else "ai" if score >= .8 else "human" if score <= .2 else "uncertain",
            "threshold": .8, "out_of_scope": False,
            "applicability": "low_resolution_experimental" if min(rgb.size) < 200 else "research_2026_image",
            "model_version": f"{variant}-ICML2026-native-crops",
            "model_scores": [{"model": variant.replace("_", "-").upper() + " · ICML 2026", "score": round(score, 6)}],
            "input_size": list(image.size), "model_input_size": [self.engine.tile_size] * 2,
            "signals": ["DEAR 通过通道筛选与分类头优化，增强对压缩、缩放后生成图的识别。",
                f"原图 {rgb.width}×{rgb.height} 像素；分析 {len(self.engine.tile_boxes(*rgb.size))} 个原生尺度图块。"],
            "metadata_hints": metadata_hints(image),
            "limitations": "分辨率、压缩和生成器会影响模型输出；本机评测记录随模型版本提供。"}
        result["signals"].extend(result["metadata_hints"])
        if include_local_evidence:
            analysis = rgb.copy()
            analysis.thumbnail((256, 256), Image.Resampling.LANCZOS)
            base = self.score(analysis, selection)
            raw = [base - self.score(probe, selection) for probe in blur_grid(analysis)]
            maximum = max((max(0., value) for value in raw), default=0.)
            values = [max(0., value) / maximum for value in raw] if maximum >= .02 else [0.] * 9
            result["local_evidence"] = {"grid_rows": 3, "grid_cols": 3,
                "values": [round(v, 6) for v in values], "raw_deltas": [round(v, 6) for v in raw],
                "method": "blur_occlusion_3x3", "analysis_size": list(analysis.size),
                "analysis_baseline_score": round(base, 6),
                "meaning": "九宫格颜色显示模糊该区域后模型分数的变化，供定位模型关注区域。"}
        return result
