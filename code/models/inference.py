"""Inference API shared by the experiment-four backend."""

from __future__ import annotations

import io
import json
import re
import unicodedata
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import torch
from PIL import Image

from text_segments import noncontent_reason, segment_ranges


def _calibrate(raw_score: float, coefficient: float, intercept: float) -> float:
    value = np.clip(coefficient * raw_score + intercept, -60, 60)
    return float(1.0 / (1.0 + np.exp(-value)))


class DetectorService:
    # Post-hoc resize stress tests found substantial changes already at 202px;
    # 200px is a display guardrail, not a calibrated image-authenticity limit.
    MIN_EXPERIMENTAL_IMAGE_SIDE = 200

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.config = json.loads((self.root / "models/model_config.json").read_text(encoding="utf-8"))
        self.image_model = torch.jit.load(io.BytesIO((self.root / "models/image_detector.ts").read_bytes()), map_location="cpu")
        self.image_model.eval()
        self.text_model = joblib.load(self.root / "models/text_detector.joblib")
        self._english_feature_names = self.text_model["vectorizer"].get_feature_names_out()
        zh_model_path = self.root / "models/text_detector_zh.joblib"
        zh_config_path = self.root / "models/text_zh_config.json"
        self.zh_text_model = joblib.load(zh_model_path) if zh_model_path.is_file() else None
        self.zh_config = json.loads(zh_config_path.read_text(encoding="utf-8")) if zh_config_path.is_file() else None
        self.highres_image_model = None
        self._highres_lock = Lock()

    def _result(self, modality: str, probability: float, signals: list[str], out_of_scope: bool) -> dict:
        threshold = float(self.config["abstain_probability"])
        if out_of_scope or 1 - threshold < probability < threshold:
            decision = "uncertain"
        else:
            decision = "ai" if probability >= threshold else "human"
        return {
            "modality": modality,
            "risk_score": round(probability, 6),
            "decision": decision,
            "threshold": threshold,
            "signals": signals,
            "out_of_scope": out_of_scope,
            "limitations": "这是来源数据上的统计风险分数，不能证明内容来源或作者身份。",
        }

    def predict_image(self, image: bytes | str | Path | Image.Image,
                      *, include_local_evidence: bool = False) -> dict:
        if isinstance(image, bytes):
            if len(image) > 25 * 1024 * 1024:
                raise ValueError("Image exceeds the 25 MB input limit")
            source = Image.open(io.BytesIO(image))
        elif isinstance(image, Image.Image):
            source = image.copy()
        else:
            source = Image.open(image)
        with source:
            original_size = source.size
            if original_size != (32, 32):
                if min(original_size) < self.MIN_EXPERIMENTAL_IMAGE_SIDE:
                    return {
                        "out_of_scope": True,
                        "scope_reason": "图像短边小于 200 像素，缩放实验发现双模型候选分数明显不稳定；不提供分数。",
                        "applicability": "insufficient_resolution",
                    }
                if self.highres_image_model is None:
                    with self._highres_lock:
                        if self.highres_image_model is None:
                            from highres_inference import HighResolutionCrossCheck
                            self.highres_image_model = HighResolutionCrossCheck(self.root)
                result = self.highres_image_model.predict(
                    source, include_local_evidence=include_local_evidence)
                result["decision"] = "uncertain"
                result["threshold"] = None
                if min(original_size) < 512:
                    result["applicability"] = "low_resolution_experimental"
                    result["signals"].append(
                        "短边低于 512 像素，候选分数对缩放和 JPEG 重编码可能明显变化。")
                    result["limitations"] = (
                        "短边 200–511 像素仅输出适用域外的实验性候选分数；"
                        "未在原生低分辨率平台样本上独立校准或验收。"
                        "该数值不是 AI 创作比例、后验概率或来源鉴定。"
                    )
                else:
                    result["applicability"] = "experimental_cross_source"
                    result["limitations"] = (
                        "这是未经目标平台校准的双模型候选输出。公开跨来源图集已参与模型选择，"
                        "其中艺术图误报仍高；不能据此判定图像来源。缩放到 224×224 会丢失局部细节。"
                    )
                return result
            pixels = np.asarray(source.convert("RGB").resize((32, 32)), dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
        with torch.inference_mode():
            logits = self.image_model(tensor)[0]
        raw_score = float(logits[1] - logits[0])
        calibration = self.config["image_calibration"]
        probability = _calibrate(raw_score, calibration["coefficient"], calibration["intercept"])
        out_of_scope = False
        signals = ["模型利用 32×32 图像的统计纹理；没有经过可验证的局部归因分析。"]
        return self._result("image", probability, signals, out_of_scope)

    def predict_text(self, text: str) -> dict:
        return self.predict_text_batch([text])[0]

    @staticmethod
    def _unscored_text(applicability: str, reason: str) -> dict:
        return {
            "modality": "text", "risk_score": None, "decision": "uncertain",
            "out_of_scope": True, "applicability": applicability,
            "scope_reason": reason, "signals": [reason],
            "limitations": "请提供模型适用范围内的自然语言正文进行分析。",
        }

    @staticmethod
    def _validate_text(text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("Text input must be a string")
        if len(text) > 100_000:
            raise ValueError("Text exceeds the 100,000-character input limit")

    def predict_text_batch(self, texts: list[str], *, batch_size: int = 32) -> list[dict]:
        """Group sparse inference by language; reuse identical inputs only in this call."""
        if not isinstance(texts, (list, tuple)):
            raise TypeError("Text batch must be a list or tuple of strings")
        if type(batch_size) is not int or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be an integer between 1 and 64")
        for text in texts:
            self._validate_text(text)
        predictions = {}
        groups = {"english": [], "chinese": []}
        for text in dict.fromkeys(texts):
            context = noncontent_reason(text)
            if context is not None:
                predictions[text] = self._unscored_text(*context)
                continue
            han = sum("\u4e00" <= char <= "\u9fff" for char in text)
            chinese = han >= 40 and han / max(sum(c.isalpha() for c in text), 1) >= .45
            groups["chinese" if chinese else "english"].append(text)
        for language, group in groups.items():
            predict = self._predict_chinese_batch if language == "chinese" else self._predict_english_batch
            for start in range(0, len(group), batch_size):
                batch = group[start:start + batch_size]
                predictions.update(zip(batch, predict(batch)))
        # Each caller receives its own mutable evidence list, including duplicates.
        return [{**predictions[text], "signals": list(predictions[text]["signals"])} for text in texts]

    def _predict_english_batch(self, texts: list[str]) -> list[dict]:
        vectorizer = self.text_model["vectorizer"]
        classifier = self.text_model["classifier"]
        features = vectorizer.transform(texts).tocsr()
        raw_scores = classifier.decision_function(features)
        calibration = self.config["text_calibration"]
        contributions = features.multiply(classifier.coef_[0]).tocsr()
        names = self._english_feature_names
        results = []
        for row, text in enumerate(texts):
            if features.indptr[row] == features.indptr[row + 1]:
                results.append(self._unscored_text("insufficient_model_features",
                    "未匹配到模型词表中的有效特征，请补充完整的自然语言正文。"))
                continue
            words = re.findall(r"[A-Za-z]+", text)
            alpha = sum(character.isalpha() for character in text)
            english_ratio = sum(len(word) for word in words) / max(alpha, 1)
            out_of_scope = not (25 <= len(text) <= 587 and len(words) >= 5 and english_ratio >= 0.65)
            probability = _calibrate(float(raw_scores[row]), calibration["coefficient"], calibration["intercept"])
            start, end = contributions.indptr[row:row + 2]
            values = list(zip(contributions.indices[start:end], contributions.data[start:end]))
            supporting_ai = sorted((item for item in values if item[1] > 0), key=lambda item: item[1], reverse=True)[:2]
            supporting_human = sorted((item for item in values if item[1] < 0), key=lambda item: item[1])[:2]
            signals = [f"字符片段 '{names[index].strip()}' 对 AI 类打分贡献 {value:+.3f}（仅为模型相关性）"
                       for index, value in supporting_ai + supporting_human]
            if not signals:
                signals = ["未匹配到有非零权重的模型特征。"]
            if out_of_scope:
                signals.append("文本长度或语言特征超出本次训练样本的适用范围。")
            result = self._result("text", probability, signals, out_of_scope)
            result["applicability"] = "english_no_valid_external_benchmark" if not out_of_scope else "outside_training_scope"
            result["limitations"] = (
                "英文模型仅在来源受限的学术摘要片段上训练；既有外测数据混入大量写作指令，"
                "不能用其准确率推断真实平台表现。分数不是 AI 撰写比例或身份判定。"
            )
            results.append(result)
        return results

    def predict_text_chinese(self, text: str) -> dict:
        self._validate_text(text)
        return self._predict_chinese_batch([text])[0]

    def _predict_chinese_batch(self, texts: list[str]) -> list[dict]:
        if self.zh_text_model is None or self.zh_config is None:
            raise FileNotFoundError("Chinese text model artifact is missing")
        cleaned = []
        for text in texts:
            clean = unicodedata.normalize("NFKC", text)
            clean = clean.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
            clean = re.sub(r"[*#`_]+", " ", clean)
            cleaned.append(re.sub(r"\s+", " ", clean).strip())
        model = self.zh_text_model
        features = model["vectorizer"].transform(cleaned).tocsr()
        raw_scores = model["classifier"].decision_function(features)
        calibration = self.zh_config["calibration"]
        results = []
        for row, clean in enumerate(cleaned):
            if features.indptr[row] == features.indptr[row + 1]:
                result = self._unscored_text("insufficient_model_features",
                    "未匹配到中文模型词表中的有效特征，请补充完整的自然语言正文。")
            else:
                chinese_characters = sum("\u4e00" <= char <= "\u9fff" for char in clean)
                in_scope = 80 <= len(clean) <= 4000 and chinese_characters >= 40
                probability = _calibrate(float(raw_scores[row]), calibration["coefficient"], calibration["intercept"])
                signals = ["中文字符 2-4 元片段的统计分类；训练集主要为问答、新闻和生成长文。"]
                signals.append("该分数不是内容来源证明；小说、平台内容与新模型尚缺独立盲测。" if in_scope else
                               "输入超出中文模型已评估的 80-4000 字范围。")
                result = self._result("text", probability, signals, not in_scope)
                result["applicability"] = "unverified_platform"
                result["limitations"] = "基于公开中文问答、新闻与生成文本的筛查分数；文学创作、真实平台内容和新模型尚无可靠外部验证，不能裁定作者身份。"
            result["model_version"] = self.zh_config["model_version"]
            results.append(result)
        return results

    def predict_text_segments(self, text: str) -> dict:
        """Screen a document and separately annotate evaluated text ranges."""
        if not isinstance(text, str):
            raise TypeError("Text input must be a string")
        if len(text) > 10_000:
            raise ValueError("Text exceeds the 10,000-character input limit")

        ranges = segment_ranges(text)
        segments = []
        scored_characters = 0
        highlighted_characters = 0
        highlight_threshold = 0.99
        total_characters = sum(not char.isspace() for char in text)
        excerpts = [text[start:end] for start, end in ranges]
        predictions = self.predict_text_batch([*excerpts, text])
        for (start, end), prediction in zip(ranges, predictions):
            excerpt = text[start:end]
            valid = not prediction["out_of_scope"]
            score = prediction["risk_score"] if valid else None
            highlighted = valid and score >= highlight_threshold
            visible_characters = sum(not char.isspace() for char in excerpt)
            if valid:
                scored_characters += visible_characters
            if highlighted:
                highlighted_characters += visible_characters
            segments.append({
                "start": start,
                "end": end,
                "text": excerpt,
                "risk_score": score,
                "decision": {"ai": "likely_ai", "human": "likely_human"}.get(
                    prediction["decision"], "uncertain") if valid else "uncertain",
                "highlighted": highlighted,
                "evidence": prediction["signals"] if valid else [prediction.get(
                    "scope_reason", "该段长度或语言超出已评估范围。")],
                "applicability": prediction.get("applicability", "limited_same_domain") if valid else prediction.get("applicability", "outside_training_scope"),
            })

        document = predictions[-1]
        valid_scores = [segment["risk_score"] for segment in segments if segment["risk_score"] is not None]
        mixed_signals = any(score <= 0.2 for score in valid_scores) and any(score >= 0.8 for score in valid_scores)
        scored_coverage = scored_characters / max(total_characters, 1)
        incomplete_coverage = scored_coverage < 0.8
        if document["out_of_scope"] or not scored_characters or mixed_signals or incomplete_coverage:
            document["out_of_scope"] = True
            document["risk_score"] = None
            document["decision"] = "uncertain"
            document["scope_reason"] = (
                "各段模型信号明显冲突；整篇不提供单一风险分数，请查看逐段结果。"
                if mixed_signals else
                "只有部分内容达到可评分要求；整篇不提供单一风险分数，请查看逐段结果。"
                if incomplete_coverage and scored_characters else
                "整篇文本超出已评估范围；仅提供各适用段落的实验性筛查结果。"
                if scored_characters else "没有达到最短可评分长度的段落；无法可靠判断。"
            )
            if mixed_signals:
                document["applicability"] = "mixed_segment_signals"
            elif incomplete_coverage and scored_characters:
                document["applicability"] = "partial_segment_coverage"
        document["segments"] = segments
        document["segment_summary"] = {
            "highlighted_character_percent": round(100 * highlighted_characters / scored_characters, 1) if scored_characters else None,
            "scored_character_percent": round(100 * scored_coverage, 1),
            "highlight_threshold": highlight_threshold,
            "mixed_signals": mixed_signals,
            "meaning": "高亮字符占已评分字符的比例；不是 AI 撰写比例，也不是作者身份判断。",
        }
        document["limitations"] = (
            document.get("limitations", "") + " 段落高亮阈值仅用于人工复核；混合作者文本和真实平台段落未经过独立标注验证。"
        ).strip()
        return document
