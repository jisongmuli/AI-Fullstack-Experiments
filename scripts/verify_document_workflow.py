"""Create public-data QA inputs and verify the real local document API.

This is a workflow/format test, not an independent accuracy evaluation. The
examples are selected for a visible range of scores and must not be used to
claim detector performance. Run with the bundled document QA Python runtime.
"""
from __future__ import annotations

import csv
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "document_qa"
BASE = "http://127.0.0.1:8002"


def post_json(path, payload):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(BASE + path, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def post_document(data: bytes, filename: str):
    boundary = "aigc-document-qa-fixed-boundary"
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode("utf-8")
    body += data + f"\r\n--{boundary}--\r\n".encode("ascii")
    request = urllib.request.Request(BASE + "/api/document/detect", body,
                                     {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    return result, round((time.perf_counter() - started) * 1000, 1)


def main():
    from docx import Document
    from docx.shared import Pt
    from docx.oxml.ns import qn
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from xml.sax.saxutils import escape

    OUT.mkdir(parents=True, exist_ok=True)
    candidates = []
    counts = {"0": 0, "1": 0}
    with (ROOT / "data/text_chinese/raw/anx_test.csv").open(encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            label = row["label"]
            if label not in counts or counts[label] >= 8 or not 120 <= len(row["text"]) <= 700:
                continue
            counts[label] += 1
            prediction = post_json("/api/text/detect", {"text": row["text"]})
            valid = [s["risk_score"] for s in prediction.get("segments", []) if s.get("risk_score") is not None]
            if valid:
                candidates.append({"row": index, "text": row["text"], "dataset_label": label,
                                   "source": row["source"], "score": max(valid)})
            if min(counts.values()) >= 8:
                break
    if not candidates:
        raise RuntimeError("No public-data text candidates could be scored")
    selected = [min(candidates, key=lambda x: x["score"]), max(candidates, key=lambda x: x["score"])]
    paragraphs = ["公开混合样本文档检测流程测试",
                  "本文件仅用于验证上传、文本提取与下划线报告。样本从本地公开测试数据中选择，覆盖不同分数；这不是独立准确率测试。",
                  selected[0]["text"], selected[1]["text"],
                  "> 这是明确标记的引用材料。" * 8,
                  "```python\nprint('代码不会记为人类或AI写作比例')\n```",
                  "字符定位测试：🙂🚀 <script>不是可执行网页内容</script>"]
    text = "\n\n".join(paragraphs)
    (OUT / "公开混合样本.txt").write_text(text, encoding="utf-8", newline="")
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.size = Pt(11)
    normal.font.name = "Arial"
    normal._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "微软雅黑")
    for paragraph in paragraphs:
        doc.add_paragraph(paragraph)
    doc.save(OUT / "公开混合样本.docx")
    # PDF input intentionally omits emoji: SimSun contains the Chinese glyphs
    # needed for extraction but is not an emoji font.
    pdfmetrics.registerFont(TTFont("QASimSun", r"C:\Windows\Fonts\simsun.ttc", subfontIndex=0))
    style = ParagraphStyle("body", fontName="QASimSun", fontSize=11, leading=19,
                           alignment=TA_LEFT, wordWrap="CJK", spaceAfter=12)
    story = []
    for paragraph in paragraphs[:-1]:
        story.append(Paragraph(escape(paragraph).replace("\n", "<br/>"), style))
        story.append(Spacer(1, 4))
    SimpleDocTemplate(str(OUT / "公开混合样本.pdf"), leftMargin=48, rightMargin=48,
                      topMargin=42, bottomMargin=42).build(story)

    checks = {"purpose": "real model workflow QA; score-selected examples, not accuracy",
              "selection": [{k: v for k, v in row.items() if k != "text"} for row in selected], "formats": {}}
    for suffix in ("txt", "docx", "pdf"):
        path = OUT / f"公开混合样本.{suffix}"
        result, latency = post_document(path.read_bytes(), path.name)
        if suffix in {"txt", "docx"}:
            assert result["extracted_text"] == text
        assert all(result["extracted_text"][s["start"]:s["end"]] == s["text"]
                   for s in result["detection"]["segments"])
        assert result["document_summary"]["high_risk_segments"] == 1, "Annotations spread across paragraph boundaries"
        assert result["document_summary"]["mixed_signals"] is True
        checks["formats"][suffix] = {"latency_ms": latency, "characters": len(result["extracted_text"]),
                                    "report_id": result["report_id"], "summary": result["document_summary"],
                                    "warnings": result["warnings"]}
        (OUT / f"real-{suffix}-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if suffix == "docx":
            for format_ in ("docx", "html", "json"):
                url = BASE + f"/api/document/reports/{result['report_id']}/{format_}"
                with urllib.request.urlopen(url) as response:
                    content = response.read()
                (OUT / f"真实模型标注文档报告.{format_}").write_bytes(content)
    long_text = (selected[1]["text"] + "\n\n") * (99_800 // (len(selected[1]["text"]) + 2))
    result, latency = post_document(long_text.encode("utf-8"), "long-public-qa.txt")
    assert result["extracted_text"] == long_text
    assert result["detection"]["risk_score"] is None
    checks["long_document"] = {"characters": len(long_text), "latency_ms": latency,
                               "total_segments": result["document_summary"]["total_segments"],
                               "whole_score": result["detection"]["risk_score"], "text_preserved": True}
    (OUT / "real-workflow-checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    main()
