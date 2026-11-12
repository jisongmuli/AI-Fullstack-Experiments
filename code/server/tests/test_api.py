"""Focused integration tests with deterministic stand-in model outputs."""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import MAX_REQUEST_BYTES, ModelUnavailable, create_app  # noqa: E402
from llm_preview import screen_candidate  # noqa: E402


TEXT = "In order to evaluate the result, the team compared 20 samples and found no clear difference."


class FakeBundle:
    def __init__(self):
        self.image_calls = 0
        self.text_calls = 0

    def get_bundle(self):
        return self

    def predict_image(self, image):
        self.image_calls += 1
        if image.size != (32, 32):
            if min(image.size) < 512:
                return {"out_of_scope": True, "scope_reason": "Thumbnail resolution is insufficient.",
                        "applicability": "insufficient_resolution"}
            return {"risk_score": 0.65, "decision": "uncertain",
                    "evidence": ["Normal-resolution candidate requires external review"],
                    "applicability": "experimental_cross_source", "model_version": "test-image-highres",
                    "model_scores": [{"model": "SigLIP", "score": 0.7}, {"model": "ViT", "score": 0.6}],
                    "model_disagreement": 0.1, "metadata_hints": ["Software field mentioned a generator."],
                    "input_size": list(image.size), "model_input_size": [224, 224]}
        return {"risk_score": 0.82, "decision": "likely_ai",
                "evidence": ["CIFAKE-domain image pattern"], "model_version": "test-image-v1"}

    def predict_text(self, text):
        self.text_calls += 1
        return {"risk_score": 0.24, "decision": "likely_human",
                "evidence": ["Short English essay pattern"], "model_version": "test-text-v1"}

    def predict_text_chinese(self, text):
        self.text_calls += 1
        return {"risk_score": 0.77, "decision": "uncertain",
                "evidence": ["Chinese text pattern"], "model_version": "test-text-zh-v1",
                "applicability": "unverified_platform"}


class MissingBundle:
    def get_bundle(self):
        raise ModelUnavailable("No local model artifacts.")


class FakeLLM:
    def available(self):
        return True

    def suggest(self, text):
        return {"status": "review_candidate", "candidate": text.replace("In order to", "To"),
                "human_review_required": True, "reason": "Writer review required."}


class BrokenLLM(FakeLLM):
    def suggest(self, text):
        raise RuntimeError("Inference failed.")


class CurrentContractBundle(FakeBundle):
    def predict_text(self, text):
        self.text_calls += 1
        return {"risk_score": 0.91, "decision": "ai", "signals": ["TF-IDF pattern"],
                "threshold": 0.8, "out_of_scope": False,
                "limitations": "Same-source validation calibration only."}


class SegmentedBundle(FakeBundle):
    def predict_text_segments(self, text):
        self.text_calls += 1
        return {
            "out_of_scope": True, "scope_reason": "Whole document is outside evaluated length.",
            "segments": [{"start": 0, "end": len(text), "text": text,
                          "risk_score": 0.995, "decision": "likely_ai", "highlighted": True,
                          "evidence": ["Statistical pattern only"],
                          "applicability": "unverified_platform"}],
            "segment_summary": {"highlighted_character_percent": 100.0,
                                "scored_character_percent": 100.0,
                                "highlight_threshold": 0.99,
                                "meaning": "High-risk coverage, not AI authorship percentage."},
        }


class LocalizedImageBundle(FakeBundle):
    def predict_image(self, image, *, include_local_evidence=False):
        self.image_calls += 1
        self.last_localize = include_local_evidence
        return {
            "risk_score": 0.62, "decision": "uncertain", "out_of_scope": False,
            "applicability": "low_resolution_experimental", "signals": ["Candidate only"],
            "model_scores": [{"model": "SigLIP", "score": 0.64}, {"model": "ViT", "score": 0.60}],
            "model_disagreement": 0.04, "metadata_hints": [],
            "local_evidence": {
                "grid_rows": 3, "grid_cols": 3,
                "values": [0, 0.1, 0.2, 0, 0.9, 1, 0, 0.3, 0.4],
                "raw_deltas": [-0.1, 0.01, 0.02, 0, 0.09, 0.1, 0, 0.03, 0.04],
                "method": "blur_occlusion_3x3",
                "meaning": "Model sensitivity, not local AI probability.",
            },
        }


def png(width=32, height=32, mode="RGB"):
    image = Image.new(mode, (width, height), (80, 120, 160) if mode == "RGB" else 120)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.bundle = FakeBundle()
        self.client = TestClient(create_app(self.bundle), base_url="http://localhost")

    def test_health_and_model_status(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["available"])

    def test_image_detection_and_normal_resolution_input(self):
        response = self.client.post("/api/image/detect", files={"file": ("test.png", png(), "image/png")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["decision"], "likely_ai")
        self.assertEqual(response.json()["provenance"]["status"], "credential_absent")
        self.assertEqual(self.bundle.image_calls, 1)
        response = self.client.post("/api/image/detect", files={"file": ("large.png", png(800, 600), "image/png")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["decision"], "uncertain")
        self.assertEqual(response.json()["applicability"], "experimental_cross_source")
        self.assertEqual(response.json()["model_scores"][0]["model"], "SigLIP")
        self.assertEqual(response.json()["provenance"]["status"], "credential_absent")
        self.assertEqual(response.json()["model_disagreement"], 0.1)
        self.assertEqual(response.json()["metadata_hints"], ["Software field mentioned a generator."])
        self.assertEqual(self.bundle.image_calls, 2)
        response = self.client.post("/api/image/detect", files={"file": ("4k.png", png(3840, 2160), "image/png")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["applicability"], "experimental_cross_source")
        self.assertEqual(self.bundle.image_calls, 3)

    def test_small_image_returns_candidate_score_and_local_evidence(self):
        bundle = LocalizedImageBundle()
        client = TestClient(create_app(bundle), base_url="http://localhost")
        response = client.post(
            "/api/image/detect", files={"file": ("small.png", png(289, 202), "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(bundle.last_localize)
        self.assertEqual(result["applicability"], "low_resolution_experimental")
        self.assertEqual(result["decision"], "uncertain")
        self.assertEqual(result["risk_score"], 0.62)
        self.assertEqual(result["local_evidence"]["values"][5], 1.0)
        self.assertEqual(result["local_evidence"]["method"], "blur_occlusion_3x3")

    def test_thumbnail_abstains_without_score(self):
        response = self.client.post("/api/image/detect", files={"file": ("thumbnail.png", png(200, 200), "image/png")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["decision"], "uncertain")
        self.assertIsNone(response.json()["risk_score"])
        self.assertEqual(response.json()["applicability"], "insufficient_resolution")

    def test_bad_image_rejected(self):
        response = self.client.post("/api/image/detect", files={"file": ("bad.png", b"not an image", "image/png")})
        self.assertEqual(response.status_code, 400)

    def test_non_rgb_image_is_supported(self):
        response = self.client.post("/api/image/detect", files={"file": ("gray.png", png(mode="L"), "image/png")})
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.json()["risk_score"])
        self.assertEqual(self.bundle.image_calls, 1)


    def test_request_size_limit_before_endpoint_parsing(self):
        payload = b"x" * (MAX_REQUEST_BYTES + 1)
        response = self.client.post("/api/text/detect", content=payload, headers={"content-type": "application/json"})
        self.assertEqual(response.status_code, 413)
        self.assertIn("35 MB", response.json()["detail"])

        def chunks():
            yield b"x" * (MAX_REQUEST_BYTES // 2)
            yield b"x" * (MAX_REQUEST_BYTES // 2 + 1)

        response = self.client.post(
            "/api/text/detect", content=chunks(),
            headers={"content-type": "application/json", "transfer-encoding": "chunked"},
        )
        self.assertEqual(response.status_code, 413)

    def test_nonlocal_host_is_rejected(self):
        response = self.client.get("/health", headers={"host": "evil.example"})
        self.assertEqual(response.status_code, 400)

    def test_text_detection_and_language_abstention(self):
        response = self.client.post("/api/text/detect", json={"text": TEXT})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["decision"], "likely_human")
        self.assertEqual(self.bundle.text_calls, 1)

    def test_chinese_text_routes_to_chinese_model(self):
        text = "研究团队比较了公开数据中的真实图片与生成图片，并记录标签质量、来源差异和可能的误报。" * 3
        response = self.client.post("/api/text/detect", json={"text": text})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_version"], "test-text-zh-v1")
        self.assertEqual(response.json()["applicability"], "unverified_platform")
        self.assertEqual(self.bundle.text_calls, 1)
        response = self.client.post("/api/text/detect", json={"text": "这个中文段落用于检验超出训练语言范围时系统能否拒绝给出确定结论。"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["decision"], "uncertain")
        self.assertIsNone(response.json()["risk_score"])
        self.assertEqual(self.bundle.text_calls, 1)

    def test_current_model_contract_is_normalized(self):
        client = TestClient(create_app(CurrentContractBundle()), base_url="http://localhost")
        response = client.post("/api/text/detect", json={"text": TEXT})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["decision"], "likely_ai")
        self.assertEqual(result["evidence"], ["TF-IDF pattern"])
        self.assertEqual(result["threshold"], 0.8)
        self.assertIn("Same-source", result["limitations"])

    def test_long_text_segment_details_survive_api(self):
        bundle = SegmentedBundle()
        client = TestClient(create_app(bundle), base_url="http://localhost")
        long_text = "研究团队核对了真实平台内容的来源、标签和误报。" * 180
        response = client.post("/api/text/detect", json={"text": long_text})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIsNone(result["risk_score"])
        self.assertEqual(result["segments"][0]["end"], len(long_text))
        self.assertTrue(result["segments"][0]["highlighted"])


    def test_editing_retains_numbers_negation_and_flags_vagueness(self):
        response = self.client.post("/api/text/edit", json={
            "text": "In order to improve 20 samples, the the team found no clear difference. This is very important."
        })
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("20", result["edited_text"])
        self.assertIn("no", result["edited_text"])
        self.assertIn("to improve", result["edited_text"].lower())
        self.assertIn("specificity", [item["category"] for item in result["suggestions"]])
        self.assertEqual(result["meaning_check"]["status"], "surface_checks_passed_human_review_required")

    def test_editor_reports_rejected_short_rewrite(self):
        response = self.client.post("/api/text/edit", json={"text": "In order to"})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["edited_text"], "In order to")
        self.assertEqual(result["meaning_check"]["status"], "blocked")
        self.assertEqual(len(result["rejected_edits"]), 1)


    def test_llm_preview_is_opt_in(self):
        client = TestClient(create_app(FakeBundle(), FakeLLM()), base_url="http://localhost")
        preview = client.post("/api/text/edit/llm-preview", json={"text": TEXT})
        self.assertEqual(preview.status_code, 200)
        self.assertTrue(preview.json()["human_review_required"])


    def test_llm_preview_blocks_identifier_loss_and_explanation(self):
        original = "The DOTA2 team reviewed 20 samples and found no error."
        lost = screen_candidate(original, "The team reviewed 20 samples and found no error.")
        self.assertEqual(lost["status"], "rejected")
        self.assertIsNone(lost["candidate"])
        self.assertIn("identifier", lost["reason"])
        commentary = screen_candidate(original, "Here's a revised version: The DOTA2 team reviewed 20 samples and found no error.")
        self.assertEqual(commentary["status"], "rejected")

    def test_llm_preview_failure_is_service_unavailable(self):
        client = TestClient(create_app(FakeBundle(), BrokenLLM()), base_url="http://localhost")
        response = client.post("/api/text/edit/llm-preview", json={"text": TEXT})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(TEXT, response.text)

    def test_missing_model_is_explicit(self):
        client = TestClient(create_app(MissingBundle()), base_url="http://localhost")
        response = client.post("/api/text/detect", json={"text": TEXT})
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
