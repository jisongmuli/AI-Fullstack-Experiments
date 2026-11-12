"""Shared text normalization for course data preparation and inference."""
import unicodedata


def canonical_text(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).split())
