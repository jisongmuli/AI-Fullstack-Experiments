"""Fetch the latest author-released MPU v3 short-text checkpoints at pinned revisions."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import httpx

from fetch_2026_assets import ROOT, download


CATALOG = [
    ("yuchuantian/AIGC_detector_zhv3short", "7bf6e1d6496353a88365f1dcdb3a6ace35f46b39", "mpu_zh_v3"),
    ("yuchuantian/AIGC_detector_env3short", "32a89cd8ace09f3eccdf930bd0a3f80f77b55d27", "mpu_en_v3"),
]


def fetch(item, proxy):
    repo, revision, directory = item
    receipts = []
    with httpx.Client(proxy=proxy, follow_redirects=True, timeout=180, trust_env=False) as client:
        response = client.get(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true")
        response.raise_for_status()
        metadata = response.json()
        for file in metadata["siblings"]:
            name = file["rfilename"]
            if name == ".gitattributes":
                continue
            receipts.append(download(client, f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
                                     ROOT / "models" / directory / name,
                                     file.get("lfs", {}).get("sha256"), file.get("size")))
    return {"repo": repo, "revision": revision, "last_modified": metadata.get("lastModified"), "files": receipts}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda item: fetch(item, args.proxy), CATALOG))
    (ROOT / "results/upgrade_20261026/mpu-assets.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
