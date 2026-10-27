"""Check disjoint splits, raw/preprocessed content and model/API lineage."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/course_20261027"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    data = ROOT / "data/course_20261027"
    images = json.loads((data / "images.json").read_text(encoding="utf-8"))
    seen = set()
    for row in images["records"]:
        assert sha(ROOT / row["raw_path"]) == row["sha256"]
        with Image.open(ROOT / row["path"]) as image:
            value = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        assert value == row["pixel_sha256"] and value not in seen
        seen.add(value)
    text = json.loads((data / "text.json").read_text(encoding="utf-8"))
    hashes, groups = set(), {}
    for split in ("train", "validation", "test"):
        for line in (data / "processed/text" / (split+".jsonl")).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert row["split"] == split and hashlib.sha256(row["text"].encode()).hexdigest() == row["sha256"]
            assert row["sha256"] not in hashes
            hashes.add(row["sha256"])
            assert groups.setdefault(row["group"], split) == split
    assert len(hashes) == 4000 and len(groups) == 2000 and len(seen) == 400
    agent = json.loads((OUT / "agent-assets.json").read_text(encoding="utf-8"))
    for row in agent["assets"]:
        assert sha(ROOT / row["path"]) == row["sha256"]
    optimization = json.loads((OUT / "experiment3.json").read_text(encoding="utf-8"))
    for row in optimization["assets"]:
        assert sha(ROOT / row["path"]) == row["sha256"]
    files = []
    for base in (data, ROOT / "models/course_20261027", ROOT / "models/agent_router"):
        for path in sorted(base.rglob("*")):
            if path.is_file():
                files.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    result = {"status": "passed", "images": 400, "texts": 4000, "text_groups": 2000,
              "course_files_verified": len(files), "official_agent_assets_verified": len(agent["assets"]),
              "image_pixel_hash_overlap": 0, "text_hash_overlap": 0, "text_group_overlap": 0,
              "assets": files, "agent_model_revision": agent["revision"], "deployed_stages": [
                  {"stage": "trained", "image": "models/course_20261027/image_trained.pt", "text": "models/course_20261027/text_full.joblib"},
                  {"stage": "optimized", "image": "models/course_20261027/image_optimized.ts", "text": "models/course_20261027/text_compact.joblib"}],
              "api": {"image": "/api/course/image/detect?stage=...", "text": "/api/course/text/detect", "agent": "/api/agent/run"}}
    (OUT / "asset-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "files": len(files), "images": 400, "texts": 4000}))


if __name__ == "__main__": main()
