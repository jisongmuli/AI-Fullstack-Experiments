"""Download public dataset archives and record checksums and source metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import requests

SOURCES = {
    "cifake": "birdy654/cifake-real-and-ai-generated-synthetic-images",
    "text_primary": "hasanyiitakbulut/ai-and-human-text-dataset",
    "text_external": "navjotkaushal/human-vs-ai-generated-essays",
}


def download(root: Path, name: str, ref: str) -> dict:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / f"{name}.zip"
    metadata_path = raw / f"{name}_source.json"
    metadata_url = f"https://www.kaggle.com/api/v1/datasets/view/{ref}"
    download_url = f"https://www.kaggle.com/api/v1/datasets/download/{ref}"
    session = requests.Session()
    response = session.get(metadata_url, timeout=40)
    response.raise_for_status()
    metadata = response.json()
    if not archive.exists() or not ZipFile(archive).testzip() is None:
        temp = raw / f"{name}.download"
        with session.get(download_url, timeout=120, stream=True) as result:
            result.raise_for_status()
            with temp.open("wb") as stream:
                for chunk in result.iter_content(1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        temp.replace(archive)
    with ZipFile(archive) as zf:
        broken = zf.testzip()
        if broken:
            raise RuntimeError(f"Damaged ZIP member: {broken}")
        members = zf.namelist()
    h = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    record = {
        "name": name,
        "reference": ref,
        "dataset_page": f"https://www.kaggle.com/datasets/{ref}",
        "metadata_url": metadata_url,
        "download_url": download_url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "license_field": metadata.get("licenseName"),
        "description": metadata.get("description", ""),
        "archive_name": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": h.hexdigest(),
        "zip_member_count": len(members),
        "zip_members_first_20": members[:20],
        "download_requires_credentials": False,
    }
    metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    for name, ref in SOURCES.items():
        record = download(args.root, name, ref)
        print(name, record["archive_bytes"], record["zip_member_count"], record["archive_sha256"][:12], flush=True)


if __name__ == "__main__":
    main()
