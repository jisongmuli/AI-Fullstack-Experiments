"""Local normal-resolution image detector with explicit model provenance."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageFilter, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTImageProcessor


GENERATOR_NAMES = ("stable diffusion", "comfyui", "automatic1111", "invokeai", "midjourney",
                   "dall-e", "dalle", "firefly", "fooocus", "flux")
GENERATION_KEYS = {"parameters", "workflow", "generation_data", "sd-metadata"}


def metadata_hints(image: Image.Image) -> list[str]:
    """Report metadata field names only; prompts and private EXIF are never returned."""
    info = {str(key).lower(): value for key, value in image.info.items()}
    hints = []
    found = sorted(GENERATION_KEYS.intersection(info))
    if found:
        hints.append(f"文件包含生成参数字段：{', '.join(found)}。元数据可被修改，不能单独证明来源。")
    software = info.get("software") or image.getexif().get(305)
    if isinstance(software, (str, bytes)):
        value = software.decode("utf-8", errors="ignore") if isinstance(software, bytes) else software
        match = next((name for name in GENERATOR_NAMES if name in value.lower()[:500]), None)
        if match:
            hints.append(f"软件字段提到 {match}；这是可修改的来源线索，不是数字签名。")
    return hints


def screening_consensus(scores: list[float] | tuple[float, ...]) -> str:
    """Conservative model-agreement signal; never an authenticity decision.

    The 0.2/0.8 boundaries predate this policy. We deliberately require both
    candidate scores to cross the same boundary, rather than letting their mean
    conceal disagreement. This is not a calibrated posterior or a moderation
    threshold.
    """
    if len(scores) < 2 or any(not 0 <= value <= 1 for value in scores):
        raise ValueError("Expected at least two finite unit-interval scores")
    if all(value >= 0.8 for value in scores):
        return "joint_high"
    if all(value <= 0.2 for value in scores):
        return "joint_low"
    return "mixed"


class HighResolutionDetector:
    MODELS = {
        "ateeqq": ("Ateeqq/ai-vs-human-image-detector", "60e82406916921b823616bee33397baab38af3f0", "highres_image"),
        "siglip2": ("prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2", "4b0c6081b9c9d890cf0251d7a5e736adc10f2d49", "highres_siglip2"),
        "nyuad": ("NYUAD-ComNets/NYUAD_AI-generated_images_detector", "b0a2e665f6e40164436cae6c0c95e7e4386f7684", "highres_nyuad"),
    }

    def __init__(self, root: Path, model_key: str = "ateeqq"):
        self.model_id, self.revision, directory_name = self.MODELS[model_key]
        directory = root / "models" / directory_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor_class = ViTImageProcessor if model_key == "nyuad" else AutoImageProcessor
        self.processor = processor_class.from_pretrained(directory, local_files_only=True)
        self.model = AutoModelForImageClassification.from_pretrained(
            directory, local_files_only=True, use_safetensors=True
        ).to(self.device).eval()
        labels = {int(index): label.lower() for index, label in self.model.config.id2label.items()}
        self.generated_indices = [index for index, label in labels.items()
                                  if label in {"ai", "deepfake", "dalle", "sd"}]
        if not self.generated_indices or "real" not in labels.values() and "hum" not in labels.values():
            raise ValueError(f"Unexpected high-resolution model labels: {labels}")

    def predict(self, image: Image.Image) -> dict:
        original_size = image.size
        score = self.predict_batch([image])[0]
        return {
            "modality": "image",
            "risk_score": round(score, 6),
            "decision": "ai" if score >= 0.8 else "human" if score <= 0.2 else "uncertain",
            "threshold": 0.8,
            "signals": [
                "图像经模型处理器缩放到 224×224 后，与训练模式比较；这是统计分类结果，不是可定位的伪造证据。"
            ],
            "out_of_scope": False,
            "model_version": f"{self.model_id}@{self.revision[:12]}",
            "input_size": list(original_size),
            "model_input_size": [224, 224],
            "limitations": (
                "高分辨率输入可分析，但缩放会丢失细节；模型训练来源未完全可核验，"
                "跨平台、生成器和编辑方式可能失准。分数未经目标平台概率校准，不证明来源。"
            ),
        }

    def predict_batch(self, images: list[Image.Image], *, batch_size: int = 3) -> list[float]:
        """Bound GPU use when running several occlusion probes on one upload."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        scores = []
        for start in range(0, len(images), batch_size):
            rgb = [ImageOps.exif_transpose(image).convert("RGB")
                   for image in images[start:start + batch_size]]
            inputs = self.processor(images=rgb, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs).logits.float()
                probabilities = torch.softmax(logits, dim=-1)
                generated = probabilities[:, self.generated_indices].sum(dim=-1)
                scores.extend(float(value) for value in generated.cpu().tolist())
        return scores


def blur_grid(image: Image.Image, rows: int = 3, cols: int = 3) -> list[Image.Image]:
    """Return one controlled blur probe per grid cell (row-major)."""
    if rows < 1 or cols < 1:
        raise ValueError("Grid dimensions must be positive")
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    radius = max(3, min(rgb.size) // 32)
    blurred = rgb.filter(ImageFilter.GaussianBlur(radius=radius))
    probes = []
    for row in range(rows):
        for col in range(cols):
            box = (round(col * rgb.width / cols), round(row * rgb.height / rows),
                   round((col + 1) * rgb.width / cols), round((row + 1) * rgb.height / rows))
            probe = rgb.copy()
            probe.paste(blurred.crop(box), box)
            probes.append(probe)
    return probes


class HighResolutionCrossCheck:
    """Expose two independently trained candidates and their disagreement."""

    def __init__(self, root: Path):
        self.detectors = (
            ("SigLIP", HighResolutionDetector(root, "ateeqq")),
            ("ViT", HighResolutionDetector(root, "nyuad")),
        )

    def predict(self, image: Image.Image, *, include_local_evidence: bool = False) -> dict:
        # Each model ultimately uses a 224-pixel input. Bound the probe canvas
        # to 512 pixels to avoid keeping nine 25 MP copies for large uploads.
        analysis = ImageOps.exif_transpose(image).convert("RGB") if include_local_evidence else image
        if include_local_evidence and max(analysis.size) > 512:
            analysis.thumbnail((512, 512), Image.Resampling.LANCZOS)
        probes = blur_grid(analysis) if include_local_evidence else []
        # The bounded 512px canvas has its own baseline for grid deltas. The
        # published whole-image score stays on exactly the no-overlay path.
        probe_images = [analysis, *probes] if include_local_evidence else []
        model_outputs = []
        estimates = []
        for label, detector in self.detectors:
            candidate = detector.predict(image)["risk_score"]
            if include_local_evidence:
                if hasattr(detector, "predict_batch"):
                    output = detector.predict_batch(probe_images, batch_size=3)
                else:
                    output = [detector.predict(probe)["risk_score"] for probe in probe_images]
                model_outputs.append(output)
            estimates.append({"model": label, "score": round(candidate, 6)})
        scores = [item["score"] for item in estimates]
        spread = abs(scores[0] - scores[1])
        score = sum(scores) / len(scores)
        consensus = screening_consensus(scores)
        hints = metadata_hints(image)
        signals = [
            "两种独立架构的候选模型分别计算风险分数；显示的总分是算术平均，未经平台概率校准。",
            {
                "joint_high": "两个候选模型均达到预设高分界线；这仍只是模型一致性线索，不能判定图像由 AI 生成。",
                "joint_low": "两个候选模型均低于预设低分界线；不能据此证明图像由真人制作。",
                "mixed": "两个候选模型未共同达到高分或低分界线；这一分数缺乏一致的模型支持。",
            }[consensus],
            "候选模型分歧较大，整图均分尤其不稳定。" if spread >= 0.4 else
            "候选模型分歧未超过 0.4，但这不等于来源证明。",
        ] + hints
        result = {
            "modality": "image",
            "risk_score": round(score, 6),
            "decision": "uncertain",
            "threshold": None,
            "signals": signals,
            "model_scores": estimates,
            "model_disagreement": round(spread, 6),
            "metadata_hints": hints,
            "out_of_scope": False,
            "model_version": "Ateeqq-SigLIP + NYUAD-ViT cross-check v2",
            "applicability": "experimental_cross_source",
            "input_size": list(image.size),
            "model_input_size": [224, 224],
            "limitations": (
                "两个候选模型在公开跨来源图集上的误报和漏报仍不稳定；平均分不是 AI 来源概率，"
                "更不能证明图片由 AI 创作。短边不足 512 像素时上采样风险更高。"
            ),
        }
        if include_local_evidence:
            # Positive loss of average model score under one blur is only a
            # sensitivity measurement. Negative deltas are retained for audit.
            means = [sum(pair) / len(pair) for pair in zip(*model_outputs)]
            raw = [means[0] - perturbed for perturbed in means[1:]]
            maximum = max((max(0.0, value) for value in raw), default=0.0)
            # Suppress changes below two score points, so numerical noise is
            # not amplified into a saturated overlay. This is a display
            # heuristic, not a validated localization threshold.
            display_floor = 0.02
            values = ([max(0.0, value) / maximum for value in raw]
                      if maximum >= display_floor else [0.0] * len(raw))
            result["local_evidence"] = {
                "grid_rows": 3,
                "grid_cols": 3,
                "values": [round(value, 6) for value in values],
                "raw_deltas": [round(value, 6) for value in raw],
                "per_model_raw_deltas": {
                    label: [round(output[0] - value, 6) for value in output[1:]]
                    for (label, _), output in zip(self.detectors, model_outputs)
                },
                "method": "blur_occlusion_3x3",
                "analysis_size": list(analysis.size),
                "analysis_baseline_score": round(means[0], 6),
                "meaning": "遮挡一个网格后双模型平均分的下降幅度；最大下降不足 0.02 时不着色。颜色只显示模型对该区域的敏感度，不是区域由 AI 生成的概率或局部伪造真值。",
            }
            result["signals"].append("九宫格为模糊遮挡敏感度；即使某格颜色较深，也不能判定该区域由 AI 生成。")
        return result
