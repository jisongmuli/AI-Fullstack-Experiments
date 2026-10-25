"""Integrity and injection checks for downloadable application reports."""
from __future__ import annotations

import io
import json
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from document_reports import REPORT_NOTE, build_report  # noqa: E402


def result_fixture():
    text = "第一段真实文字🙂\n\n高风险段落😀和英文 AI text。\n末段未评分内容。"
    left = text.index("高风险")
    right = text.index("\n末段")
    return {
        "filename": "../作文<script>.docx",
        "format": "docx",
        "source_sha256": "a" * 64,
        "created_at": "2026-10-11T08:00:00Z",
        "extracted_text": text,
        "warnings": ["图片和复杂公式不纳入文字评分。"],
            "sections": [{"label": "第一段", "start": 0, "end": left},
                         {"label": "检测段", "start": left, "end": right, "page": 2}],
        "detection": {
            "risk_score": None,
            "decision": "uncertain",
            "model_version": "fixture-model",
            "segments": [
                {"start": 0, "end": left, "text": text[:left], "risk_score": 0.08,
                 "highlighted": False, "evidence": ["低模型分数"]},
                {"start": left, "end": right, "text": text[left:right], "risk_score": 0.999,
                 "highlighted": True, "evidence": ["<script>alert('evidence')</script>"]},
                {"start": right, "end": len(text), "text": text[right:], "risk_score": None,
                 "highlighted": False, "evidence": ["长度不足"]},
            ],
        },
        "document_summary": {
            "highlighted_total_percent": 42.1,
            "highlighted_character_percent": 62.5,
            "scored_character_percent": 67.4,
            "mean_segment_score": 0.501,
            "high_risk_segments": 1,
            "total_segments": 3,
            "unscored_segments": 1,
        },
    }


class ReportTests(unittest.TestCase):
    def test_docx_preserves_text_and_codepoint_underlines(self):
        result = result_fixture()
        data, mime, filename = build_report(result, "docx")
        self.assertIn("wordprocessingml.document", mime)
        self.assertNotIn("/", filename)
        self.assertNotIn("<", filename)
        with ZipFile(io.BytesIO(data)) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
            self.assertEqual(archive.namelist().count("word/document.xml"), 1)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = root.findall("w:body/w:p", ns)
        texts = ["".join(p.itertext()) for p in paragraphs]
        origin_index = texts.index("带标注的提取原文") + 2
        count = len(result["extracted_text"].split("\n"))
        original = "\n".join(texts[origin_index:origin_index + count])
        self.assertEqual(original, result["extracted_text"])
        marked = []
        for paragraph in paragraphs[origin_index:origin_index + count]:
            for run in paragraph.findall("w:r", ns):
                if run.find("w:rPr/w:u", ns) is not None:
                    marked.append("".join(run.findall("w:t", ns)[0].itertext()))
        self.assertEqual("".join(marked), result["detection"]["segments"][1]["text"])
        self.assertIn("😀", "".join(marked))
        self.assertIn(REPORT_NOTE, texts)

    def test_html_escapes_source_evidence_and_active_content(self):
        result = result_fixture()
        result["filename"] = "x\" autofocus onfocus=alert(1) <script>.txt"
        text = result["extracted_text"] + "\n<script>alert('source')</script>"
        result["extracted_text"] = text
        data, mime, filename = build_report(result, "html")
        html = data.decode("utf-8")
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("Content-Security-Policy", html)
        self.assertIn("62.5%", html)
        self.assertIn("42.1%", html)
        self.assertIn("未评分", html)
        self.assertIn("第 2 页 检测段", html)
        self.assertIn("text/html", mime)
        self.assertTrue(filename.endswith(".html"))

    def test_json_is_unicode_lossless_and_records_metric_semantics(self):
        result = result_fixture()
        data, mime, filename = build_report(result, "json")
        payload = json.loads(data)
        self.assertEqual(payload["extracted_text"], result["extracted_text"])
        self.assertIn("不是实际 AI 撰写比例", payload["report_export"]["metric_note"])
        self.assertIn("Unicode codepoints", payload["report_export"]["offset_unit"])
        self.assertIn("application/json", mime)
        self.assertTrue(filename.endswith(".json"))

    def test_invalid_offsets_or_modified_text_fail_closed(self):
        for mutation in ("overlap", "text", "utf16"):
            with self.subTest(mutation=mutation):
                result = result_fixture()
                segment = result["detection"]["segments"][1]
                if mutation == "overlap":
                    segment["start"] = 0
                elif mutation == "text":
                    segment["text"] = "不一致的原文"
                else:
                    segment["end"] += 1
                with self.assertRaises(ValueError):
                    build_report(result, "docx")

    def test_json_rejects_nonfinite_and_unknown_format(self):
        result = result_fixture()
        result["document_summary"]["mean_segment_score"] = float("nan")
        with self.assertRaises(ValueError):
            build_report(result, "json")
        with self.assertRaises(ValueError):
            build_report(result_fixture(), "pdf")

    def test_unscored_segment_cannot_be_underlined_as_high_risk(self):
        result = result_fixture()
        result["detection"]["segments"][2]["highlighted"] = True
        with self.assertRaises(ValueError):
            build_report(result, "html")

    def test_empty_text_is_exportable_without_inventing_scores(self):
        result = {"extracted_text": "", "filename": "", "detection": {"segments": []}}
        html = build_report(result, "html")[0].decode()
        self.assertIn("未提供", html)
        self.assertNotIn("0.0%", html)


if __name__ == "__main__":
    unittest.main()
