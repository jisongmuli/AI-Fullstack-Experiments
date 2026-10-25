"""Download the normal-resolution image model and independent public test sets.

Run from the project root: python code/models/download_highres.py
All downloads are streamed, size-capped, SHA-256 hashed, and recorded locally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests


MODEL_ID = "Ateeqq/ai-vs-human-image-detector"
MODEL_REVISION = "60e82406916921b823616bee33397baab38af3f0"
MIRROR = f"https://hf-mirror.com/{MODEL_ID}/resolve/{MODEL_REVISION}"
SIGLIP2_ID = "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2"
SIGLIP2_REVISION = "4b0c6081b9c9d890cf0251d7a5e736adc10f2d49"
SIGLIP2_MIRROR = f"https://hf-mirror.com/{SIGLIP2_ID}/resolve/{SIGLIP2_REVISION}"
NYUAD_ID = "NYUAD-ComNets/NYUAD_AI-generated_images_detector"
NYUAD_REVISION = "b0a2e665f6e40164436cae6c0c95e7e4386f7684"
NYUAD_MIRROR = f"https://hf-mirror.com/{NYUAD_ID}/resolve/{NYUAD_REVISION}"
SOURCES = {
    "model_config": (f"{MIRROR}/config.json", "models/highres_image/config.json", 100_000),
    "model_processor": (f"{MIRROR}/preprocessor_config.json", "models/highres_image/preprocessor_config.json", 100_000),
    "model_weights": (f"{MIRROR}/model.safetensors", "models/highres_image/model.safetensors", 600_000_000),
    "siglip2_config": (f"{SIGLIP2_MIRROR}/config.json", "models/highres_siglip2/config.json", 100_000),
    "siglip2_processor": (f"{SIGLIP2_MIRROR}/preprocessor_config.json", "models/highres_siglip2/preprocessor_config.json", 100_000),
    "siglip2_weights": (f"{SIGLIP2_MIRROR}/model.safetensors", "models/highres_siglip2/model.safetensors", 600_000_000),
    "nyuad_config": (f"{NYUAD_MIRROR}/config.json", "models/highres_nyuad/config.json", 100_000),
    "nyuad_processor": (f"{NYUAD_MIRROR}/preprocessor_config.json", "models/highres_nyuad/preprocessor_config.json", 100_000),
    "nyuad_weights": (f"{NYUAD_MIRROR}/model.safetensors", "models/highres_nyuad/model.safetensors", 600_000_000),
    "shoes_eval": (
        "https://www.kaggle.com/api/v1/datasets/download/sunnykakar/shoes-dataset-real-and-ai-generated-images",
        "data/raw/shoes_real_ai.zip", 100_000_000,
    ),
    "art_eval": (
        "https://www.kaggle.com/api/v1/datasets/download/doctorstrange420/real-and-fake-ai-generated-art-images-dataset",
        "data/raw/art_real_ai.zip", 500_000_000,
    ),
    "rhythmghai_eval": (
        "https://www.kaggle.com/api/v1/datasets/download/rhythmghai/ai-vs-real-images-dataset",
        "data/raw/rhythmghai_real_ai_512.zip", 400_000_000,
    ),
    "mixed512_eval": (
        "https://www.kaggle.com/api/v1/datasets/download/shanmuk4622/real-and-fake-ai-generated-512px-dataset",
        "data/raw/mixed512_real_ai.zip", 1_200_000_000,
    ),
}


def download(url: str, destination: Path, max_bytes: int) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": digest, "reused": True}
    temporary = destination.with_suffix(destination.suffix + ".part")
    size = 0
    digest = hashlib.sha256()
    try:
        with requests.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            declared = int(response.headers.get("Content-Length", 0))
            if declared > max_bytes:
                raise ValueError(f"Source exceeds expected size: {declared} > {max_bytes}")
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError(f"Source exceeded limit while downloading: {url}")
                    output.write(chunk)
                    digest.update(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"path": str(destination), "bytes": size, "sha256": digest.hexdigest(), "reused": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--only", choices=list(SOURCES), action="append")
    args = parser.parse_args()
    root = args.root.resolve()
    names = args.only or [name for name in SOURCES if name != "mixed512_eval"]
    records = {}
    for name in names:
        url, relative, limit = SOURCES[name]
        print(f"Downloading {name}...", flush=True)
        record = download(url, root / relative, limit)
        records[name] = {"source": url.split("/resolve/")[0] if "resolve" in url else url,
                         "revision": (MODEL_REVISION if name.startswith("model_") else
                                      SIGLIP2_REVISION if name.startswith("siglip2_") else
                                      NYUAD_REVISION if name.startswith("nyuad_") else None),
                         "license_on_host": ("Apache-2.0" if name.startswith(("model_", "siglip2_", "nyuad_")) else
                                             "CC BY-NC-SA 4.0" if name == "rhythmghai_eval" else
                                             "CC0" if name == "mixed512_eval" else "MIT"),
                         **record}
        print(f"{name}: {record['bytes']:,} bytes, sha256 {record['sha256']}", flush=True)
    manifest = root / "data/raw/highres_source_manifest.json"
    prior = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    prior.update(records)
    manifest.write_text(json.dumps(prior, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
