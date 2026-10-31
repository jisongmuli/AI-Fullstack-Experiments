"""Refresh runtime inventories and fingerprints for the checked-out release."""
from __future__ import annotations

import hashlib
import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results/upgrade_20261026"


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fingerprint(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=lambda value: date.fromisoformat(value).isoformat())
    args = parser.parse_args()
    receipts_path = AUDIT / "asset-receipts.json"
    receipts = {row["path"]: row for row in json.loads(receipts_path.read_text(encoding="utf-8"))["assets"]}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "detector_ep3.pt"):
        row = fingerprint(ROOT / "models/taste" / name)
        if name == "detector_ep3.pt":
            assert row["sha256"] == "f79328c19ec0f1044dd24dd6eb9c5d5c96aa01b7036765251be9a426b9a46cbf"
        row.update(url="https://drive.usercontent.google.com/download?id=1w1hbdiZMS_JzPntVMWM3qrTQ4KxJf-t6&export=download&confirm=t",
                   archive_member="TASTE/detector/" + name)
        receipts[row["path"]] = row
    write(receipts_path, {"assets": list(receipts.values())})

    path = ROOT / "RUNTIME_MODELS.json"
    runtime = json.loads(path.read_text(encoding="utf-8"))
    downloads = {row["path"]: row for row in runtime["downloads"]}
    for variant in ("dear_r", "dear_c"):
        row = dict(receipts[f"models/{variant}/model.pth"])
        row.pop("url")
        row.update(repo="k-aisi-anti-deepfake/dear-checkpoints", revision="5b57350b0ee75553109b3844f3c8a9341fc7e707",
                   filename=f"{variant}/model_best.pth", role="active image detector")
        downloads[row["path"]] = row
    for group in json.loads((AUDIT / "mpu-assets.json").read_text(encoding="utf-8")):
        row = dict(next(row for row in group["files"] if row["path"].endswith("pytorch_model.bin")))
        row.pop("url")
        row.update(repo=group["repo"], revision=group["revision"], filename="pytorch_model.bin", role="active text detector")
        downloads[row["path"]] = row
    course_assets = ROOT / "results/course_20261027/agent-assets.json"
    if course_assets.exists():
        agent_assets = json.loads(course_assets.read_text(encoding="utf-8"))
        for item in agent_assets["assets"]:
            if item["path"].endswith(".safetensors"):
                row = {key: item[key] for key in ("path", "bytes", "sha256")}
                row.update(repo=agent_assets["repo"], revision=agent_assets["revision"],
                           filename=Path(item["path"]).name, role="local course Agent text backbone")
                downloads[row["path"]] = row
    release_version = args.version or max(runtime.get("version", "2026-10-26"), "2026-10-27" if course_assets.exists() else "2026-10-26")
    runtime.update(version=release_version, downloads=list(downloads.values()),
                   active_policy="models/algorithm_policy.json",
                   research_candidates=[{"path": "models/taste/detector_ep3.pt", "role": "evaluated comparison candidate",
                                         "restore": "scripts/fetch_2026_assets.py"}],
                   notes="DEAR and MPU are the advanced detectors. The course model chain and fine-tuned Qwen3.5 Agent are available separately. Clone with Git LFS to restore all bundled assets.")
    if course_assets.exists():
        runtime.update(course_models="models/course_20261027", agent_adapter="models/agent_router",
                       agent_backbone="models/agent_qwen35")
    write(path, runtime)

    manifest_path = ROOT / "FILE_MANIFEST.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = {row["path"] for row in old["files"] if (ROOT / row["path"]).is_file()}
    output = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    for raw in output.split(b"\0"):
        if not raw:
            continue
        name = raw.decode("utf-8")
        if name.startswith(("code/", "web/", "models/", "scripts/", "licenses/", "docs/")) or (
                "/" not in name and Path(name).suffix in {".md", ".py", ".cmd", ".ps1", ".txt"}):
            if (ROOT / name).is_file():
                names.add(name)
    names.add("RUNTIME_MODELS.json")
    names.discard("FILE_MANIFEST.json")
    write(manifest_path, {"version": release_version, "files": [fingerprint(ROOT / name) for name in sorted(names)]})
    print(json.dumps({"runtime_downloads": len(downloads), "fingerprints": len(names)}))


if __name__ == "__main__":
    main()
