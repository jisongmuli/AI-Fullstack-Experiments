"""Deterministic slow inference for browser progress/cancellation verification."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/server"))
from app import create_app


class SlowModel:
    def get_bundle(self):
        return self

    def predict_text_segments(self, text):
        time.sleep(.08)
        return {"risk_score": .2, "decision": "likely_human", "model_version": "slow-browser-fixture",
                "segments": [{"start": 0, "end": len(text), "text": text, "risk_score": .2,
                              "decision": "likely_human", "evidence": []}], "segment_summary": {}}


app = create_app(SlowModel())
