"""Fetch a deterministic labeled NTIRE 2026 sample using bounded HTTP ZIP ranges."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import struct
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
REVISION = "b85af61d93cd4fd5e558b3ee69f6d93639fce83f"
URL = f"https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train/resolve/{REVISION}/shard_5.zip"
SIZE = 11370161676


class RemoteZip(io.RawIOBase):
    def __init__(self, client):
        self.client, self.position, self.transferred = client, 0, 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else SIZE + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = SIZE - self.position if size < 0 else min(size, SIZE - self.position)
        value = self.range(self.position, size)
        self.position += len(value)
        return value

    def range(self, start, size):
        if size == 0:
            return b""
        if size < 0 or size > 32 * 1024 * 1024:
            raise ValueError("Refusing an unbounded remote ZIP read")
        end = start + size - 1
        # The query keeps range responses distinct in intermediary caches.
        with self.client.stream("GET", URL + f"?range_start={start}&range_end={end}",
                                headers={"Range": f"bytes={start}-{end}"}) as response:
            response.raise_for_status()
            if response.status_code != 206 or response.headers.get("content-range") != f"bytes {start}-{end}/{SIZE}":
                raise RuntimeError("Server did not honor the bounded byte range")
            data = response.read()
        if len(data) != size:
            raise RuntimeError("Incomplete byte range")
        self.transferred += size
        return data

    def member(self, info):
        header = self.range(info.header_offset, 30)
        fields = struct.unpack("<4s5H3I2H", header)
        if fields[0] != b"PK\x03\x04" or info.file_size > 20 * 1024 * 1024:
            raise RuntimeError("Unexpected ZIP member")
        offset = info.header_offset + 30 + fields[-2] + fields[-1]
        raw = self.range(offset, info.compress_size)
        if info.compress_type == zipfile.ZIP_DEFLATED:
            value = zlib.decompress(raw, -15)
        elif info.compress_type == zipfile.ZIP_STORED:
            value = raw
        else:
            raise RuntimeError("Unsupported compression")
        if len(value) != info.file_size or zlib.crc32(value) != info.CRC:
            raise RuntimeError("ZIP member length/CRC mismatch")
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy")
    parser.add_argument("--per-class", type=int, default=132)
    args = parser.parse_args()
    output = ROOT / "data/benchmarks_2026/ntire"
    output.mkdir(parents=True, exist_ok=True)
    with httpx.Client(proxy=args.proxy, timeout=90, follow_redirects=True, trust_env=False) as client:
        remote = RemoteZip(client)
        with zipfile.ZipFile(remote) as archive:
            entries = archive.infolist()
        labels = next(item for item in entries if item.filename.endswith("labels.csv"))
        records = list(csv.DictReader(io.StringIO(remote.member(labels).decode("utf-8-sig"))))
        print("Label columns:", list(records[0]), "rows:", len(records), flush=True)
        images = {Path(item.filename).name: item for item in entries if item.filename.lower().endswith(".jpg")}
        name_key = next(key for key in records[0] if "name" in key.lower() or "file" in key.lower())
        label_key = next(key for key in records[0] if "label" in key.lower())
        rng = random.Random(20261026)
        selected = []
        for label in (0, 1):
            group = sorted([row for row in records if int(row[label_key]) == label], key=lambda row: row[name_key])
            selected.extend(rng.sample(group, args.per_class))

        def fetch(row):
            from PIL import Image
            name = Path(row[name_key]).name
            info = images[name]
            value = remote.member(info)
            with Image.open(io.BytesIO(value)) as image:
                image.verify()
                width, height = image.size
            target = output / "images" / name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(value)
            return {"id": name, "path": "images/" + name, "label": int(row[label_key]),
                    "width": width, "height": height, "bytes": len(value),
                    "sha256": hashlib.sha256(value).hexdigest(), "zip_member": info.filename,
                    "zip_crc32": f"{info.CRC:08x}"}

        with ThreadPoolExecutor(max_workers=6) as executor:
            rows = list(executor.map(fetch, selected))
        for label in (0, 1):
            group = sorted([row for row in rows if row["label"] == label], key=lambda row: row["id"])
            for index, row in enumerate(group):
                row["split"] = "validation" if index < 32 else "test"
        result = {"dataset": "deepfakesMSU/NTIRE-RobustAIGenDetection-train", "revision": REVISION,
                  "source": URL, "shard_size": SIZE, "seed": 20261026,
                  "selection": "132 random records per class before model scoring; first 32 sorted IDs per class for validation",
                  "evaluation_role": "External audit sample from official public training shard; not an official challenge test score",
                  "transfer_integrity": "Bounded HTTP ranges, each member checked against central-directory CRC32 and length; per-file SHA-256 recorded",
                  "downloaded_bytes": remote.transferred, "records": rows}
        (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Saved", len(rows), "images; transferred", remote.transferred, "bytes", flush=True)


if __name__ == "__main__":
    main()
