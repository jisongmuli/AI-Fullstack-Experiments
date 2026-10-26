"""Fetch pinned research checkpoints and public 2026 evaluation records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEAR_REV = "5b57350b0ee75553109b3844f3c8a9341fc7e707"
SIGHT_REV = "423b1768f0827675295fd6dc3cd255ad2f411183"
NLPCC_REV = "961a485fa72f9821d6315836d46c9d94818b18b1"
TASTE_SHA = "e9cd44fa5ab134ce0c2061438368036109729429fdaa534f8f54d2753d05a80e"


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(client, url, target, expected_sha=None, expected_size=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and expected_sha and file_hash(target) == expected_sha:
        print("Verified existing", target.relative_to(ROOT), flush=True)
    else:
        temporary = target.with_name(target.name + ".part")
        size, digest = 0, hashlib.sha256()
        with client.stream("GET", url, timeout=180) as response:
            response.raise_for_status()
            if "text/html" in response.headers.get("content-type", ""):
                raise RuntimeError("Received an HTML page instead of the requested asset")
            with temporary.open("wb") as stream:
                for chunk in response.iter_bytes(1024 * 1024):
                    stream.write(chunk); digest.update(chunk); size += len(chunk)
        if expected_size is not None and size != expected_size:
            raise RuntimeError(f"Size mismatch: {target.name}")
        if expected_sha is not None and digest.hexdigest() != expected_sha:
            raise RuntimeError(f"SHA-256 mismatch: {target.name}")
        temporary.replace(target)
        print("Downloaded", target.relative_to(ROOT), size, flush=True)
    return {"path": str(target.relative_to(ROOT)).replace("\\", "/"), "url": url,
            "bytes": target.stat().st_size, "sha256": file_hash(target)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy")
    parser.add_argument("--skip-taste", action="store_true")
    args = parser.parse_args()
    receipts = []
    with httpx.Client(proxy=args.proxy, follow_redirects=True, timeout=45, trust_env=False) as client:
        for variant, digest in [
            ("dear_c", "c8fa321d0cac86d36def6508f787f65bf22420d4648722956ad00375a347c7bc"),
            ("dear_r", "430fde11debe1850ab24af43945b4d68f2bb3ee52a837d84cf525eb603ccde97"),
        ]:
            url = f"https://huggingface.co/k-aisi-anti-deepfake/dear-checkpoints/resolve/{DEAR_REV}/{variant}/model_best.pth"
            receipts.append(download(client, url, ROOT / "models" / variant / "model.pth", digest, 94372114))
        for name in ["LICENSE", "NOTICE", "README.md"]:
            url = f"https://huggingface.co/k-aisi-anti-deepfake/dear-checkpoints/resolve/{DEAR_REV}/{name}"
            receipts.append(download(client, url, ROOT / "licenses/dear" / name))
        for name in ["README.md", "data/ai_generated_2026-09-29.jsonl", "data/human_pmc_ccby_nonanglophone.jsonl", "data/human_wikipedia_pre2022.jsonl"]:
            url = f"https://huggingface.co/datasets/textsightai/ai-text-detection-benchmark-2026/resolve/{SIGHT_REV}/{name}"
            receipts.append(download(client, url, ROOT / "data/benchmarks_2026/textsight" / name))
        for name in ["README.md", "data/README.md", "data/testp1.json", "data/testp1_testing_label.json", "data/testp2_testing_label.json", "data/train_data.json"]:
            url = f"https://raw.githubusercontent.com/NLP2CT/NLPCC-2026-Task6-Detection/{NLPCC_REV}/{name}"
            target = ROOT / "data/benchmarks_2026/nlpcc" / name
            if name == "data/train_data.json" and target.exists() and target.stat().st_size > 1000:
                url = f"https://media.githubusercontent.com/media/NLP2CT/NLPCC-2026-Task6-Detection/{NLPCC_REV}/{name}"
                receipts.append(download(client, url, target, "d07059d8af0c7ba86629a8ceba3f09669d79f13356f09d0853861b53a8d5c01f", 212700850))
                continue
            receipts.append(download(client, url, target))
            if target.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
                pointer = target.read_text(encoding="utf-8")
                digest = next(line.split(":", 1)[1] for line in pointer.splitlines() if line.startswith("oid sha256:"))
                size = int(next(line.split()[1] for line in pointer.splitlines() if line.startswith("size ")))
                url = f"https://media.githubusercontent.com/media/NLP2CT/NLPCC-2026-Task6-Detection/{NLPCC_REV}/{name}"
                receipts.append(download(client, url, target, digest, size))
        if not args.skip_taste:
            url = "https://drive.usercontent.google.com/download?id=1w1hbdiZMS_JzPntVMWM3qrTQ4KxJf-t6&export=download&confirm=t"
            target = ROOT / ".cache/research_2026/TASTE.zip"
            receipts.append(download(client, url, target, expected_sha=TASTE_SHA, expected_size=712225547))
    output = ROOT / "results/upgrade_20261026/asset-receipts.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(output.read_text(encoding="utf-8"))["assets"] if output.exists() else []
    by_path = {r["path"]: r for r in existing + receipts}
    receipts = list(by_path.values())
    output.write_text(json.dumps({"assets": receipts}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
