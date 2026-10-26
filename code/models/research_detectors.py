"""Local-only adapters for released TASTE (ICLR 2026) and DEAR (ICML 2026) weights.

DEAR's fixed channel gate and stride-1 ResNet-50 follow the authors' inference
architecture. Native-resolution tiling bounds memory for large uploads. Sources,
checkpoint fingerprints and licenses are recorded with the installed assets.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import resnet50
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


class _ChannelGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("gate", torch.ones(2048))

    def forward(self, value):
        return value * self.gate.view(1, -1, 1, 1)


class _GatedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=None, num_classes=1)
        self.backbone.conv1.stride = (1, 1)
        self.backbone.maxpool.stride = 1
        self.gate = _ChannelGate()

    def forward(self, value):
        model = self.backbone
        value = model.maxpool(model.relu(model.bn1(model.conv1(value))))
        value = model.layer4(model.layer3(model.layer2(model.layer1(value))))
        value = model.avgpool(self.gate(value)).flatten(1)
        return model.fc(value)


class DearDetector:
    def __init__(self, root: Path, variant="dear_r", *, tile_size=256):
        if variant not in {"dear_c", "dear_r"} or not 32 <= tile_size <= 512:
            raise ValueError("Invalid DEAR configuration")
        self.variant, self.tile_size = variant, tile_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _GatedResNet()
        checkpoint = torch.load(root / "models" / variant / "model.pth", map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(memory_format=torch.channels_last)
        self.model.to(self.device).eval()
        self._lock = threading.Lock()
        self.mean = torch.tensor((.485, .456, .406)).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor((.229, .224, .225)).view(1, 3, 1, 1).to(self.device)

    def tile_boxes(self, width: int, height: int):
        if min(width, height) < 1:
            raise ValueError("Image dimensions must be positive")
        cols, rows = math.ceil(width / self.tile_size), math.ceil(height / self.tile_size)
        if cols * rows > 9:
            cols = min(cols, 9, max(1, round(math.sqrt(9 * width / height))))
            rows = min(rows, max(1, 9 // cols))
        xs = np.linspace(0, max(0, width - self.tile_size), cols).round().astype(int)
        ys = np.linspace(0, max(0, height - self.tile_size), rows).round().astype(int)
        return [(int(x), int(y), min(int(x) + self.tile_size, width), min(int(y) + self.tile_size, height))
                for y in ys for x in xs]

    def logit(self, image: Image.Image) -> float:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        total, weighted = 0, 0.
        with self._lock, torch.inference_mode():
            for box in self.tile_boxes(*rgb.size):
                # Bounded, evenly spaced native-scale crops retain fine artifacts.
                tile = rgb.crop(box)
                array = np.array(tile, dtype=np.float32, copy=True) / 255.
                value = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
                logit = float(self.model((value - self.mean) / self.std).item())
                if not math.isfinite(logit):
                    raise RuntimeError("DEAR returned a nonfinite logit")
                area = tile.width * tile.height
                weighted += logit * area
                total += area
        return weighted / total

    def predict(self, image: Image.Image) -> dict:
        logit = self.logit(image)
        return {"risk_score": 1. / (1. + math.exp(-max(-60., min(60., logit)))), "logit": logit}

    def predict_batch(self, images, *, batch_size=1):
        # Each native image has its own shape; keep one bounded tile in flight.
        return [self.predict(image)["risk_score"] for image in images]


class TasteDetector:
    def __init__(self, root: Path, *, batch_size=4):
        if not 1 <= batch_size <= 8:
            raise ValueError("TASTE batch_size must be between 1 and 8")
        directory = root / "models/taste"
        config = AutoConfig.from_pretrained(directory, num_labels=2, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_config(config)
        state = torch.load(directory / "detector_ep3.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        # Preserve the released BERT WordPiece tokenizer; Mistral's regex migration
        # is unrelated to this checkpoint's older config.
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, use_fast=True,
                                                      fix_mistral_regex=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        self._lock = threading.Lock()

    def logits(self, texts: list[str]) -> list[float]:
        return self.representations(texts)[0]

    def representations(self, texts: list[str]) -> tuple[list[float], np.ndarray]:
        if not texts:
            return [], np.empty((0, 768), dtype=np.float32)
        encoded = self.tokenizer(texts, max_length=512, truncation=True, stride=64,
                                 return_overflowing_tokens=True, padding=False)
        mapping = encoded.pop("overflow_to_sample_mapping")
        sums, weights = np.zeros(len(texts)), np.zeros(len(texts))
        embeddings = np.zeros((len(texts), 768), dtype=np.float64)
        seen = set()
        with self._lock, torch.inference_mode():
            for start in range(0, len(mapping), self.batch_size):
                windows = [{key: value[index] for key, value in encoded.items()}
                           for index in range(start, min(start + self.batch_size, len(mapping)))]
                batch = self.tokenizer.pad(windows, padding=True, return_tensors="pt")
                batch = {key: value.to(self.device) for key, value in batch.items()}
                # The authors' classifier consumes the BERT pooled output.
                encoded_output = self.model.bert(**batch).pooler_output
                output = self.model.classifier(self.model.dropout(encoded_output)).float().cpu().numpy()
                hidden = encoded_output.float().cpu().numpy()
                for offset, pair in enumerate(output):
                    index = start + offset
                    owner = mapping[index]
                    weight = max(1, len(encoded["input_ids"][index]) - 2 - (64 if owner in seen else 0))
                    seen.add(owner)
                    sums[owner] += float(pair[1] - pair[0]) * weight
                    embeddings[owner] += hidden[offset] * weight
                    weights[owner] += weight
        if not np.isfinite(sums).all():
            raise RuntimeError("TASTE returned nonfinite logits")
        return (sums / weights).tolist(), (embeddings / weights[:, None]).astype(np.float32)

    def predict_scores(self, texts: list[str]) -> list[float]:
        return [1. / (1. + math.exp(-max(-60., min(60., value)))) for value in self.logits(texts)]


class MpuDetector:
    """Author-released MPU v3 short checkpoint; label 1 is AI in the official demo."""
    def __init__(self, root: Path, language: str, *, batch_size=4):
        if language not in {"english", "chinese"} or not 1 <= batch_size <= 8:
            raise ValueError("Invalid MPU configuration")
        directory = root / "models" / ("mpu_zh_v3" if language == "chinese" else "mpu_en_v3")
        self.model, loaded = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True, use_safetensors=False, output_loading_info=True)
        allowed_buffers = {"bert.embeddings.position_ids", "roberta.embeddings.position_ids"}
        if (loaded.get("missing_keys") or loaded.get("mismatched_keys") or
                set(loaded.get("unexpected_keys", [])) - allowed_buffers or loaded.get("error_msgs")):
            raise RuntimeError("The released MPU checkpoint does not match its classifier architecture")
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, use_fast=True,
                                                      fix_mistral_regex=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.batch_size, self._lock = batch_size, threading.Lock()

    def logits(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        encoded = self.tokenizer(texts, max_length=512, truncation=True, stride=64,
                                 return_overflowing_tokens=True, padding=False)
        mapping = encoded.pop("overflow_to_sample_mapping")
        sums, weights, seen = np.zeros(len(texts)), np.zeros(len(texts)), set()
        with self._lock, torch.inference_mode():
            for start in range(0, len(mapping), self.batch_size):
                windows = [{key: value[index] for key, value in encoded.items()}
                           for index in range(start, min(start + self.batch_size, len(mapping)))]
                batch = self.tokenizer.pad(windows, padding=True, return_tensors="pt")
                output = self.model(**{key: value.to(self.device) for key, value in batch.items()}).logits.float().cpu().numpy()
                for offset, pair in enumerate(output):
                    index, owner = start + offset, mapping[start + offset]
                    weight = max(1, len(encoded["input_ids"][index]) - 2 - (64 if owner in seen else 0))
                    seen.add(owner)
                    sums[owner] += float(pair[1] - pair[0]) * weight
                    weights[owner] += weight
        if not np.isfinite(sums).all():
            raise RuntimeError("MPU returned nonfinite logits")
        return (sums / weights).tolist()
