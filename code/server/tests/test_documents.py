"""Meaningful ingestion and metric checks without loading model weights."""
from __future__ import annotations

import io
import hashlib
import math
import sys
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import documents  # noqa: E402
from documents import DocumentError, analyze_document, extract_document, extract_pasted_text  # noqa: E402


NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def docx_bytes(body: str, extra: dict[str, bytes] | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", b"<Types />")
        archive.writestr("word/document.xml", f'<w:document xmlns:w="{NS}"><w:body>{body}</w:body></w:document>')
        for name, value in (extra or {}).items():
            archive.writestr(name, value)
    return buffer.getvalue()


def pdf_bytes(texts: list[str], encrypted=False, compressed=False) -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=595, height=842)
        if text:
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                     NameObject("/Subtype"): NameObject("/Type1"),
                                     NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
            content = DecodedStreamObject()
            safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            content.set_data(f"BT /F1 12 Tf 40 750 Td ({safe}) Tj ET".encode("ascii"))
            if compressed:
                content = content.flate_encode()
            page[NameObject("/Contents")] = writer._add_object(content)
    if encrypted:
        writer.encrypt("secret")
    result = io.BytesIO()
    writer.write(result)
    return result.getvalue()


def segment_detector(text: str) -> dict:
    segments = []
    for start, end in documents.segment_ranges(text):
        excerpt = text[start:end]
        context = documents.noncontent_reason(excerpt)
        score = None if context else .995 if "HIGH" in excerpt else .1
        segments.append({"start": start, "end": end, "text": excerpt,
                         "risk_score": score, "decision": "uncertain", "highlighted": False,
                         "evidence": ["stand-in statistical score"], "applicability": "test-only"})
    return {"risk_score": .5, "decision": "uncertain", "model_version": "test-v1", "segments": segments}


class ExtractionTests(unittest.TestCase):
    def test_maximum_pasted_text_canonicalizes_browser_newlines_before_limit(self):
        canonical = "😀\n" * 50_000
        browser_form = canonical.replace("\n", "\r\n")
        self.assertEqual(len(canonical), documents.MAX_DOCUMENT_CHARS)
        self.assertGreater(len(browser_form), documents.MAX_DOCUMENT_CHARS)
        source = extract_pasted_text(browser_form)
        self.assertEqual(source["text"], canonical)
        self.assertEqual(source["sections"][0]["end"], 100_000)
        self.assertEqual(source["text"][2:3], "😀")
        self.assertEqual(source["source_sha256"], hashlib.sha256(canonical.encode("utf-8")).hexdigest())
        self.assertEqual(source["source_sha256_kind"], "normalized_pasted_text_utf8")
        self.assertTrue(any("LF" in warning for warning in source["warnings"]))

    def test_docx_body_table_nested_table_and_emoji_reading_order(self):
        body = ('<w:p><w:r><w:t>首段😀</w:t><w:tab/><w:t>后半</w:t></w:r></w:p>'
                '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>左格</w:t></w:r></w:p>'
                '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>嵌套格</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:tc>'
                '<w:tc><w:p><w:r><w:t>右格</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
                '<w:p><w:del><w:r><w:delText>旧词</w:delText></w:r></w:del>'
                '<w:ins><w:r><w:t>尾段</w:t></w:r></w:ins></w:p>')
        source = extract_document(docx_bytes(body, {"word/header1.xml": b"<unused/>"}), "../../unsafe.docx")
        self.assertEqual(source["filename"], "unsafe.docx")
        self.assertEqual(source["text"], "首段😀\t后半\n\n左格\n嵌套格\t右格\n\n尾段")
        self.assertEqual(source["text"][source["sections"][1]["start"]:source["sections"][1]["end"]], "左格\n嵌套格\t右格")
        self.assertTrue(any("页眉" in warning for warning in source["warnings"]))
        self.assertTrue(any("修订" in warning for warning in source["warnings"]))

    def test_docx_unsafe_package_paths_and_entities_rejected(self):
        for name in ["../outside", "/absolute", "C:/outside", "word\\bad"]:
            with self.subTest(name=name), self.assertRaises(DocumentError):
                extract_document(docx_bytes("", {name: b"x"}), "bad.docx")
        xml = b'<!DOCTYPE x [<!ENTITY x "boom">]><x/>'
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("word/document.xml", xml)
        with self.assertRaises(DocumentError):
            extract_document(data.getvalue(), "entity.docx")

    def test_docx_xml_and_output_limits_reject_without_truncation(self):
        with patch.object(documents, "MAX_DOCUMENT_XML", 100), self.assertRaises(DocumentError) as caught:
            extract_document(docx_bytes("<w:p>" + "x" * 200 + "</w:p>"), "oversize.docx")
        self.assertEqual(caught.exception.status_code, 413)
        with patch.object(documents, "MAX_DOCUMENT_CHARS", 10), self.assertRaises(DocumentError) as caught:
            extract_document(docx_bytes("<w:p><w:r><w:t>" + "x" * 20 + "</w:t></w:r></w:p>"), "long.docx")
        self.assertEqual(caught.exception.status_code, 413)

    def test_text_encodings_and_binary_rejection(self):
        for value in ["中文😀".encode("utf-8"), "中文😀".encode("utf-16")]:
            self.assertEqual(extract_document(value, "text.txt")["text"], "中文😀")
        self.assertEqual(extract_document("中文测试".encode("gb18030"), "text.md")["text"], "中文测试")
        with self.assertRaises(DocumentError):
            extract_document(b"hello\x00binary", "binary.txt")
        with self.assertRaises(DocumentError) as caught:
            extract_document(b"legacy", "old.doc")
        self.assertEqual(caught.exception.status_code, 415)

    def test_file_limits_corrupt_and_spoofed_format(self):
        with patch.object(documents, "MAX_DOCUMENT_BYTES", 5), self.assertRaises(DocumentError) as caught:
            extract_document(b"123456", "large.txt")
        self.assertEqual(caught.exception.status_code, 413)
        for filename in ["fake.pdf", "fake.docx"]:
            with self.subTest(filename=filename), self.assertRaises(DocumentError):
                extract_document(b"hello", filename)
        with self.assertRaises(DocumentError):
            extract_document(b"", "empty.txt")

    def test_pdf_pages_offsets_and_compressed_content(self):
        source = extract_document(pdf_bytes(["First page", "Second page"], compressed=True), "report.pdf")
        self.assertIn("First page", source["text"])
        self.assertIn("Second page", source["text"])
        self.assertEqual([section["page"] for section in source["sections"]], [1, 2])
        second = source["sections"][1]
        self.assertIn("Second page", source["text"][second["start"]:second["end"]])

    def test_pdf_scanned_encrypted_and_page_limit_rejected(self):
        with self.assertRaises(DocumentError) as caught:
            extract_document(pdf_bytes([""]), "scan.pdf")
        self.assertIn("OCR", str(caught.exception))
        with self.assertRaises(DocumentError):
            extract_document(pdf_bytes(["Secret"], encrypted=True), "secret.pdf")
        with patch.object(documents, "MAX_PDF_PAGES", 1), self.assertRaises(DocumentError) as caught:
            extract_document(pdf_bytes(["A", "B"]), "many.pdf")
        self.assertEqual(caught.exception.status_code, 413)

    def test_pdf_stream_inflation_bound(self):
        from pypdf.generic import EncodedStreamObject, NameObject
        stream = EncodedStreamObject()
        stream[NameObject("/Filter")] = NameObject("/FlateDecode")
        import zlib
        stream._data = zlib.compress(b"x" * 1000)
        with patch.object(documents, "MAX_PDF_STREAM", 100), self.assertRaises(DocumentError) as caught:
            documents._bounded_pdf_stream(stream)
        self.assertEqual(caught.exception.status_code, 413)


class AnalysisTests(unittest.TestCase):
    def test_mixed_document_coverage_and_unicode_offsets(self):
        text = "😀 HIGH text\n\nlow text"
        result = analyze_document(extract_pasted_text(text), segment_detector)
        segments = result["detection"]["segments"]
        self.assertEqual(segments[1]["start"], text.index("low"))
        self.assertEqual(text[segments[0]["start"]:segments[0]["end"]], segments[0]["text"])
        summary = result["document_summary"]
        self.assertEqual(summary["scored_character_percent"], 100)
        self.assertTrue(summary["mixed_signals"])
        self.assertIsNone(result["detection"]["risk_score"])
        self.assertEqual(summary["high_risk_segments"], 1)
        self.assertAlmostEqual(summary["highlighted_total_percent"], 56.25, places=2)
        self.assertEqual(summary["highlighted_character_percent"], 56.25)
        self.assertTrue(segments[0]["highlighted"])

    def test_partial_and_no_score_never_report_false_zero_ai(self):
        result = analyze_document(extract_pasted_text("HIGH text\n\n> quoted content"), segment_detector)
        self.assertIsNone(result["detection"]["risk_score"])
        self.assertEqual(result["document_summary"]["highlighted_character_percent"], 100)
        self.assertLess(result["document_summary"]["highlighted_total_percent"], 100)
        none = analyze_document(extract_pasted_text("> quoted content"), segment_detector)
        self.assertEqual(none["document_summary"]["scored_character_percent"], 0)
        self.assertIsNone(none["document_summary"]["highlighted_character_percent"])
        self.assertIsNone(none["document_summary"]["highlighted_total_percent"])
        self.assertIsNone(none["document_summary"]["mean_segment_score"])

    def test_long_document_no_loss_no_whole_probability_and_bounded_callbacks(self):
        text = "\n\n".join(["HIGH " + "word " * 100 for _ in range(30)])
        calls = []
        def detector(value):
            calls.append(value)
            return segment_detector(value)
        result = analyze_document(extract_pasted_text(text), detector)
        self.assertEqual(result["extracted_text"], text)
        self.assertTrue(all(len(call) <= 10_000 for call in calls))
        self.assertLess(max(map(len, calls)), 1000)
        self.assertEqual(result["document_summary"]["scored_character_percent"], 100)
        self.assertIsNone(result["detection"]["risk_score"])
        self.assertAlmostEqual(result["document_summary"]["mean_segment_score"], .995)

    def test_long_fence_context_not_lost_at_chunk_boundary(self):
        text = "```python\n" + "print('HIGH')\n" * 1200 + "```\n\nHIGH authored prose."
        calls = []
        def detector(value):
            calls.append(value)
            return segment_detector(value)
        result = analyze_document(extract_pasted_text(text), detector)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], "HIGH authored prose.")
        self.assertIsNone(result["detection"]["segments"][0]["risk_score"])
        self.assertEqual(result["document_summary"]["high_risk_segments"], 1)

    def test_missing_text_overlap_nonfinite_scores_rejected(self):
        text = "whole text"
        malformed = [
            [{"start": 0, "end": 5, "text": "whole", "risk_score": .9}],
            [{"start": 0, "end": len(text), "text": text, "risk_score": math.nan}],
            [{"start": 0, "end": len(text), "text": text, "risk_score": True}],
            [{"start": 0, "end": len(text), "text": "wrong", "risk_score": .9}],
        ]
        for segments in malformed:
            with self.subTest(segments=segments), self.assertRaises(RuntimeError):
                analyze_document(extract_pasted_text(text), lambda _: {"risk_score": .5, "segments": segments})

    def test_highlight_threshold_cannot_be_forged_by_detector(self):
        text = "some prose"
        def detector(value):
            return {"risk_score": .2, "decision": "likely_human", "segments": [
                {"start": 0, "end": len(value), "text": value, "risk_score": .98,
                 "highlighted": True, "evidence": []}]}
        result = analyze_document(extract_pasted_text(text), detector)
        self.assertFalse(result["detection"]["segments"][0]["highlighted"])
        self.assertEqual(result["document_summary"]["high_risk_segments"], 0)

    def test_excessive_tiny_paragraphs_rejected_before_model_calls(self):
        source = extract_pasted_text("x\n\n" * 2001)
        with self.assertRaises(DocumentError) as caught:
            analyze_document(source, lambda _: self.fail("Should not invoke the model"))
        self.assertEqual(caught.exception.status_code, 413)


if __name__ == "__main__":
    unittest.main()
