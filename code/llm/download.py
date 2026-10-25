"""Download pinned Apache-2.0 model/data assets with local checksums."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
MIRROR = "https://hf-mirror.com"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
DATA_ID = "grammarly/coedit"
DATA_REVISION = "e9a255c33ef910bc33a9d2b522653fa87521583e"
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
)
DATA_FILES = ("train.jsonl", "validation.jsonl")
EXPECTED_MODEL_SHA256 = "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, expected_sha256: str | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        digest = sha256_file(path)
        if expected_sha256 is None or digest == expected_sha256:
            return {"url": url, "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest}
    temporary = path.with_name(path.name + ".part")
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    output.write(block)
    digest = sha256_file(temporary)
    if expected_sha256 is not None and digest != expected_sha256:
        temporary.unlink()
        raise ValueError(f"SHA-256 mismatch for {path.name}: {digest}")
    temporary.replace(path)
    return {"url": url, "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest}


def main() -> None:
    assets = []
    for name in MODEL_FILES:
        url = f"{MIRROR}/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
        path = ROOT / "models" / "llm" / "base" / name
        expected = EXPECTED_MODEL_SHA256 if name == "model.safetensors" else None
        item = download(url, path, expected)
        assets.append(item)
        print(f"model {name}: {item['bytes']} bytes", flush=True)
    for name in DATA_FILES:
        url = f"{MIRROR}/datasets/{DATA_ID}/resolve/{DATA_REVISION}/{name}"
        path = ROOT / "results" / "exp5" / "source" / name
        item = download(url, path)
        assets.append(item)
        print(f"dataset {name}: {item['bytes']} bytes", flush=True)
    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "license": "Apache-2.0"},
        "dataset": {"id": DATA_ID, "revision": DATA_REVISION, "license": "Apache-2.0"},
        "assets": assets,
        "note": "The model is 135M parameters; CoEdIT is used only for a small English grammar-edit adaptation.",
    }
    path = ROOT / "results" / "exp5" / "source_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest {path}", flush=True)


if __name__ == "__main__":
    main()
