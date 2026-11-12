"""Stream-check pinned checkpoints and downloaded public audit samples."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    expected = {}
    for name in ("asset-receipts.json", "phase1-assets.json"):
        path = ROOT / "results/upgrade_20261026" / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data.get("assets", []) if isinstance(data, dict) else data
            for row in rows:
                if not row["path"].startswith(".cache/"):
                    expected[row["path"]] = row
    for group in json.loads((ROOT / "results/upgrade_20261026/mpu-assets.json").read_text(encoding="utf-8")):
        for row in group["files"]:
            expected[row["path"]] = row
    manifest = json.loads((ROOT / "data/benchmarks_2026/ntire/manifest.json").read_text(encoding="utf-8"))
    for row in manifest["records"]:
        expected["data/benchmarks_2026/ntire/" + row["path"]] = row
    for name, row in expected.items():
        path = ROOT / name
        assert path.is_file(), f"Missing asset: {name}"
        with path.open("rb") as stream:
            value = hashlib.file_digest(stream, "sha256").hexdigest()
        assert path.stat().st_size == row["bytes"] and value == row["sha256"], f"Fingerprint mismatch: {name}"
    result = {"status": "passed", "files_verified": len(expected),
              "bytes_verified": sum(row["bytes"] for row in expected.values())}
    (ROOT / "results/upgrade_20261026/asset-verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
