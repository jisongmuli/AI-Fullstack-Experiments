"""Heuristic privacy screening without copying matched personal data into results."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EMAIL = re.compile(r"(?<![\w.-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d(). -]{8,}\d)(?!\d)")


def scan_text(path: Path) -> dict:
    frame = pd.read_csv(path, usecols=["sample_id", "text"])
    emails = frame.text.fillna("").map(lambda value: bool(EMAIL.search(value)))
    phones = frame.text.fillna("").map(lambda value: bool(PHONE.search(value)))
    return {
        "rows": len(frame), "email_pattern_rows": int(emails.sum()),
        "possible_phone_pattern_rows": int(phones.sum()),
        "email_sample_ids": frame.loc[emails, "sample_id"].head(20).tolist(),
        "possible_phone_sample_ids": frame.loc[phones, "sample_id"].head(20).tolist(),
    }


def main() -> None:
    images = list((ROOT / "data/images/processed").rglob("*.png"))
    embedded_metadata = 0
    for path in images:
        with Image.open(path) as image:
            embedded_metadata += bool(image.getexif() or image.info.get("exif"))
    result = {
        "scope": "heuristic pattern scan; sample IDs only, no matched values retained",
        "primary_text": scan_text(ROOT / "data/text/primary_clean_manifest.csv"),
        "external_text": scan_text(ROOT / "data/text/external_original_manifest.csv"),
        "processed_images_scanned": len(images),
        "processed_images_with_exif": embedded_metadata,
        "limitations": [
            "Phone-like patterns may be dates or other numbers; no content-level manual verification.",
            "Names, addresses, quotations and image subjects are not detected by this script.",
            "Source archive rights and opt-out claims require separate human review.",
        ],
    }
    (ROOT / "results/privacy_screen.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
