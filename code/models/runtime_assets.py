"""Inspect active detector assets without importing ML libraries or loading weights."""
from __future__ import annotations

import json
from pathlib import Path


def algorithm_policy(root: Path) -> dict:
    path = root / "models/algorithm_policy.json"
    value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(value, dict):
        raise ValueError("Detector policy must be an object")
    return value


def text_policy(root: Path) -> dict:
    value = {}
    for name in ("text_policy_2026.json", "text_calibration_phase1.json"):
        path = root / "models" / name
        if path.is_file():
            incoming = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(incoming, dict):
                raise ValueError(f"Invalid text policy: {name}")
            value.update(incoming)
    return value


def required_assets(root: Path, policy: dict | None = None) -> list[Path]:
    policy = algorithm_policy(root) if policy is None else policy
    names = ["model_config.json", "text_detector.joblib"]
    backend = policy.get("text_backend")
    if backend in {"taste", "transformer_2026"}:
        names += ["text_detector_zh.joblib", "text_zh_config.json"]
        pipeline = text_policy(root)
        if backend == "transformer_2026":
            names += ["text_policy_2026.json", "text_calibration_phase1.json"]
        for language in ("english", "chinese"):
            selected = pipeline.get(language + "_backend", "taste")
            if selected == "mpu":
                folder = "mpu_zh_v3" if language == "chinese" else "mpu_en_v3"
                names += [f"{folder}/{name}" for name in ("pytorch_model.bin", "config.json", "tokenizer_config.json")]
                names += [f"{folder}/vocab.txt"] if language == "chinese" else [f"{folder}/vocab.json", f"{folder}/merges.txt"]
            elif selected == "taste":
                names += ["taste/detector_ep3.pt", "taste/config.json", "taste/tokenizer.json", "taste/policy.json"]
                if language == "chinese":
                    names += ["taste/chinese_head.npz"]
            else:
                raise ValueError(f"Unsupported {language} detector: {selected}")
    elif backend is not None:
        raise ValueError(f"Unsupported text detector: {backend}")
    image_backend = policy.get("image_backend")
    if image_backend == "dear":
        names += ["dear_r/model.pth", "dear_c/model.pth", "dear_r/policy.json"]
    elif image_backend is None:
        names += ["image_detector.ts"]
    else:
        raise ValueError(f"Unsupported image detector: {image_backend}")
    return [root / "models" / name for name in dict.fromkeys(names)]


def asset_problems(root: Path, policy: dict | None = None) -> list[str]:
    problems = []
    for path in required_assets(root, policy):
        name = path.relative_to(root).as_posix()
        if not path.is_file() or path.stat().st_size < 16:
            problems.append(f"Missing detector asset: {name}")
        elif path.suffix in {".pt", ".pth", ".bin"} and path.stat().st_size < 1024 * 1024:
            problems.append(f"Restore the full LFS checkpoint: {name}")
        elif path.suffix in {".joblib", ".ts"} and path.stat().st_size < 512:
            with path.open("rb") as stream:
                if stream.read(42).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    problems.append(f"Restore the full LFS checkpoint: {name}")
    return problems
