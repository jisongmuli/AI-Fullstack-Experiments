"""Document upload, no-loss analysis, report downloads and local retention."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import ModelUnavailable, create_app  # noqa: E402
from documents import MAX_DOCUMENT_BYTES, MAX_DOCUMENT_CHARS  # noqa: E402


HIGH = "HIGH 研究团队审核公开内容，并记录来源、文字、图像和可能的误报。🙂这段文字用于验证完整原文与模型标注的对应关系。"
LOW = "LOW 同学在学校记录了自己完成实验的过程，注明数据取得的位置和实验运行时的观察结果。😀"
UNSCORED = "UNSCORED 引用、公式或短句不参与本次模型评分。"


class DocumentBundle:
    def __init__(self):
        self.text_inputs = []
        self.image_calls = 0

    def get_bundle(self):
        return self

    def predict_text_segments(self, text):
        self.text_inputs.append(text)
        segments, cursor = [], 0
        for line in text.splitlines(keepends=True):
            body = line.rstrip("\r\n")
            if body.strip():
                score = None if body.startswith("UNSCORED") else (.1 if body.startswith("LOW") else .995)
                segments.append({
                    "start": cursor, "end": cursor + len(body), "text": body,
                    "risk_score": score,
                    "highlighted": score is not None and score >= .99,
                    "decision": "uncertain" if score is None else ("likely_ai" if score >= .8 else "likely_human"),
                    "evidence": ["Deterministic test output; no real accuracy claim"],
                    "applicability": "test_fixture",
                })
            cursor += len(line)
        return {"risk_score": .995, "decision": "likely_ai", "model_version": "test-doc-model",
                "segments": segments,
                "segment_summary": {"highlighted_character_percent": 100.0,
                                    "scored_character_percent": 100.0,
                                    "highlight_threshold": .99,
                                    "mixed_signals": False, "meaning": "Test fixture"}}

    def predict_image(self, image):
        self.image_calls += 1
        return {"risk_score": .4, "decision": "uncertain", "model_version": "test-image-model",
                "evidence": ["Test image score"], "applicability": "test_fixture"}


class MissingDocumentBundle:
    def get_bundle(self):
        raise ModelUnavailable("Test model unavailable.")


def docx_bytes(*paragraphs):
    from docx import Document
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def pdf_bytes(text="Document audit content for an integration test."):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=595, height=842)
    if text is not None:
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        content = DecodedStreamObject()
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content.set_data(f"BT /F1 12 Tf 40 780 Td ({escaped}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(content)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), "gray").save(buffer, "PNG")
    return buffer.getvalue()


class DocumentApiTests(unittest.TestCase):
    def setUp(self):
        self.bundle = DocumentBundle()
        self.app = create_app(self.bundle)
        self.client = TestClient(self.app, base_url="http://localhost")

    def detect(self, *, filename="sample.txt", content=None, mime="text/plain"):
        data = HIGH.encode("utf-8") if content is None else content
        return self.client.post("/api/document/detect", files={"file": (filename, data, mime)})

    def test_report_list_survives_browser_clients_and_includes_agent_reports(self):
        report = self.detect(filename="first.txt").json()
        agent = self.client.post("/api/agent/upload", files={
            "document": ("agent.txt", LOW.encode("utf-8"), "text/plain")}).json()
        agent_id = agent["findings"]["document"]["report_id"]
        browser = TestClient(self.app, base_url="http://localhost")
        listing = browser.get("/api/document/reports")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.headers["cache-control"], "no-store")
        self.assertEqual([item["report_id"] for item in listing.json()["reports"]],
                         [agent_id, report["report_id"]])
        for item in listing.json()["reports"]:
            self.assertNotIn("extracted_text", item)
            self.assertNotIn("detection", item)
        self.assertEqual(browser.get(f"/api/document/reports/{agent_id}").status_code, 200)
        self.assertEqual(browser.delete(f"/api/document/reports/{agent_id}").status_code, 200)
        self.assertEqual([item["report_id"] for item in browser.get("/api/document/reports").json()["reports"]],
                         [report["report_id"]])
        self.app.state.report_store.close()

    def test_report_list_starts_empty_and_prunes_expired_reports(self):
        self.assertEqual(self.client.get("/api/document/reports").json(), {"reports": []})
        now = [1000.0]
        self.app.state.report_store.clock = lambda: now[0]
        self.detect()
        self.assertEqual(len(self.client.get("/api/document/reports").json()["reports"]), 1)
        now[0] += self.app.state.report_store.ttl_seconds
        self.assertEqual(self.client.get("/api/document/reports").json(), {"reports": []})
        self.app.state.report_store.close()

    def test_word_upload_returns_sha_sections_and_exact_original(self):
        content = docx_bytes(HIGH, LOW)
        response = self.detect(filename="实验.docx", content=content,
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertEqual(report["extracted_text"], HIGH + "\n\n" + LOW)
        self.assertEqual(report["source_sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(report["format"], "docx")
        self.assertEqual(len(report["sections"]), 2)
        self.assertTrue(report["warnings"])
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertTrue(report["report_id"])

    def test_text_layer_pdf_and_image_only_pdf_are_distinguished(self):
        response = self.detect(filename="paper.pdf", content=pdf_bytes(), mime="application/pdf")
        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertIn("Document audit content", report["extracted_text"])
        self.assertEqual(report["sections"][0]["page"], 1)
        self.assertEqual(report["format"], "pdf")
        blank = self.detect(filename="scan.pdf", content=pdf_bytes(None), mime="application/pdf")
        self.assertEqual(blank.status_code, 422, blank.text)
        self.assertIn("OCR", blank.json()["detail"])

    def test_bad_empty_binary_and_unsupported_documents_rejected(self):
        examples = [
            ("bad.docx", b"not a ZIP", 422),
            ("bad.pdf", b"not a PDF", 422),
            ("old.doc", b"old binary document", 415),
            ("script.exe", b"executable", 415),
            ("empty.txt", b"\n \t", 422),
            ("binary.txt", b"\0\1\2", 422),
        ]
        for filename, content, expected in examples:
            with self.subTest(filename=filename):
                response = self.detect(filename=filename, content=content)
                self.assertEqual(response.status_code, expected, response.text)
        self.assertEqual(len(self.bundle.text_inputs), 0)
        self.assertEqual(len(self.app.state.report_store._items), 0)

    def test_file_and_character_bounds_reject_without_truncation(self):
        large = self.detect(content=b"x" * (MAX_DOCUMENT_BYTES + 1))
        self.assertEqual(large.status_code, 413)
        over = self.client.post("/api/document/detect", data={"text": "中" * (MAX_DOCUMENT_CHARS + 1)})
        self.assertEqual(over.status_code, 413)
        self.assertEqual(len(self.bundle.text_inputs), 0)

    def test_pasted_text_or_file_is_exclusive_and_empty_rejected(self):
        both = self.client.post("/api/document/detect", data={"text": HIGH},
                                files={"file": ("a.txt", HIGH.encode(), "text/plain")})
        self.assertEqual(both.status_code, 400)
        empty = self.client.post("/api/document/detect", data={"text": "   "})
        self.assertEqual(empty.status_code, 422)
        valid = self.client.post("/api/document/detect", data={"text": HIGH})
        self.assertEqual(valid.status_code, 200, valid.text)
        self.assertEqual(valid.json()["extracted_text"], HIGH)

    def test_long_document_over_text_limit_preserves_every_character(self):
        text = ("研究团队在每一轮实验中核对公开数据和模型输出，并记录结果。🙂" * 500)[:10_001]
        response = self.client.post("/api/document/detect", data={"text": text})
        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertEqual(report["extracted_text"], text)
        self.assertEqual(len(text), 10_001)
        self.assertIsNone(report["detection"]["risk_score"])
        self.assertGreater(len(report["detection"]["segments"]), 1)
        self.assertTrue(all(len(part) <= 10_000 for part in self.bundle.text_inputs))
        for segment in report["detection"]["segments"]:
            self.assertEqual(segment["text"], text[segment["start"]:segment["end"]])
        self.assertEqual(report["document_summary"]["scored_character_percent"], 100.0)

    def test_mixed_scores_and_unscored_content_have_separate_denominators(self):
        text = HIGH + "\n\n" + LOW + "\n\n" + UNSCORED
        response = self.client.post("/api/document/detect", data={"text": text})
        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        summary = report["document_summary"]
        high_count = sum(not char.isspace() for char in HIGH)
        scored_count = high_count + sum(not char.isspace() for char in LOW)
        total_count = sum(not char.isspace() for char in text)
        self.assertAlmostEqual(summary["highlighted_total_percent"], 100 * high_count / total_count, places=2)
        self.assertAlmostEqual(summary["highlighted_character_percent"], 100 * high_count / scored_count, places=2)
        self.assertLess(summary["highlighted_total_percent"], summary["highlighted_character_percent"])
        self.assertEqual(summary["unscored_segments"], 1)
        self.assertTrue(summary["mixed_signals"])
        self.assertIsNone(report["detection"]["risk_score"])
        self.assertIn("不是 AI 实际撰写比例", summary["metric_note"])

    def test_report_json_html_and_docx_downloads_are_safe_and_annotated(self):
        filename = '作文"<script>.txt'
        report = self.detect(filename=filename).json()
        report_id = report["report_id"]
        retrieved = self.client.get(f"/api/document/reports/{report_id}")
        self.assertEqual(retrieved.status_code, 200)
        self.assertEqual(retrieved.json()["extracted_text"], HIGH)
        self.assertEqual(retrieved.headers["cache-control"], "no-store")
        for format_ in ("docx", "html", "json"):
            with self.subTest(format=format_):
                exported = self.client.get(f"/api/document/reports/{report_id}/{format_}")
                self.assertEqual(exported.status_code, 200, exported.text[:100])
                self.assertEqual(exported.headers["cache-control"], "no-store")
                self.assertEqual(exported.headers["x-content-type-options"], "nosniff")
                disposition = exported.headers["content-disposition"]
                self.assertIn("attachment", disposition)
                self.assertIn("filename*=UTF-8''", disposition)
                self.assertNotIn("<script>", disposition)
                self.assertNotIn("\r", disposition)
                self.assertNotIn("\n", disposition)
                if format_ == "json":
                    self.assertEqual(json.loads(exported.content)["extracted_text"], HIGH)
                elif format_ == "html":
                    self.assertNotIn("<script>", exported.text)
                    self.assertIn("class='flag'", exported.text)
                else:
                    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    with ZipFile(io.BytesIO(exported.content)) as archive:
                        root = ET.fromstring(archive.read("word/document.xml"))
                    marked = []
                    for run in root.findall(".//w:r", ns):
                        if run.find("w:rPr/w:u", ns) is not None:
                            marked.append("".join(node.text or "" for node in run.findall("w:t", ns)))
                    self.assertEqual("".join(marked), HIGH)
        invalid = self.client.get(f"/api/document/reports/{report_id}/pdf")
        self.assertEqual(invalid.status_code, 400)

    def test_delete_expire_restart_and_unknown_report(self):
        now = [1_000.0]
        self.app.state.report_store.clock = lambda: now[0]
        report = self.detect().json()
        report_id = report["report_id"]
        now[0] += 1_800
        self.assertEqual(self.client.get(f"/api/document/reports/{report_id}").status_code, 404)
        self.assertEqual(self.client.get(f"/api/document/reports/{report_id}/json").status_code, 404)
        other = self.detect().json()["report_id"]
        deleted = self.client.delete(f"/api/document/reports/{other}")
        self.assertTrue(deleted.json()["deleted"])
        self.assertEqual(deleted.headers["cache-control"], "no-store")
        self.assertFalse(self.client.delete(f"/api/document/reports/{other}").json()["deleted"])
        restarted = TestClient(create_app(DocumentBundle()), base_url="http://localhost")
        self.assertEqual(restarted.get(f"/api/document/reports/{other}").status_code, 404)
        self.assertEqual(self.client.get("/api/document/reports/unknown").status_code, 404)

    def test_agent_document_alone_and_with_image_without_score_fusion(self):
        for with_image in (False, True):
            with self.subTest(with_image=with_image):
                files = {"document": ("作业.txt", HIGH.encode(), "text/plain")}
                if with_image:
                    files["file"] = ("picture.png", png_bytes(), "image/png")
                response = self.client.post("/api/agent/upload", files=files)
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                expected = {"document", "image"} if with_image else {"document"}
                self.assertEqual(set(result["findings"]), expected)
                self.assertEqual(result["findings"]["document"]["extracted_text"], HIGH)
                self.assertEqual(result["tool_calls"][-1]["tool"], "document_detection")
                self.assertNotIn("risk_score", result)
                self.assertNotIn(HIGH, str(result["tool_calls"]))

    def test_agent_rejects_document_plus_text_and_bad_documents(self):
        both = self.client.post("/api/agent/upload", data={"text": HIGH},
                                files={"document": ("作业.txt", HIGH.encode(), "text/plain")})
        self.assertEqual(both.status_code, 400)
        bad = self.client.post("/api/agent/upload", files={"document": ("fake.docx", b"bad", "application/octet-stream")})
        self.assertEqual(bad.status_code, 422)

    def test_missing_model_returns_service_error_without_persisting_report(self):
        app = create_app(MissingDocumentBundle())
        client = TestClient(app, base_url="http://localhost")
        response = client.post("/api/document/detect", data={"text": HIGH})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(HIGH, response.text)
        self.assertEqual(len(app.state.report_store._items), 0)


if __name__ == "__main__":
    unittest.main()
