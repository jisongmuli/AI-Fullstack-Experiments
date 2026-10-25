"""Bounded, local document extraction and auditable paragraph screening.

Offsets are Python Unicode code-point offsets in ``extracted_text``. Extraction
does not execute macros, follow external relationships, or perform OCR.
"""
from __future__ import annotations

import base64
import hashlib
import io
import math
import re
import statistics
import sys
import unicodedata
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable
from xml.etree import ElementTree as ET

MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
MAX_DOCUMENT_CHARS = 100_000
MAX_DOCUMENT_SEGMENTS = 2_000
MAX_PDF_PAGES = 200
MAX_ZIP_UNCOMPRESSED = 64 * 1024 * 1024
MAX_DOCUMENT_XML = 16 * 1024 * 1024
MAX_PDF_STREAM = 8 * 1024 * 1024
MAX_PDF_DECODED_TOTAL = 64 * 1024 * 1024
HIGHLIGHT_THRESHOLD = .99
SCHEMA_VERSION = "document-screening-v1"

_MODELS_PATH = str(Path(__file__).resolve().parents[1] / "models")
if _MODELS_PATH not in sys.path:
    sys.path.insert(0, _MODELS_PATH)
from text_segments import noncontent_reason, segment_ranges  # noqa: E402

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class DocumentError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _safe_filename(filename: str) -> str:
    if not isinstance(filename, str) or not filename:
        return "document.txt"
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(char for char in name if unicodedata.category(char)[0] != "C")
    return name[:180] or "document.txt"


def _validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise DocumentError("文档内容必须是文本。")
    if len(text) > MAX_DOCUMENT_CHARS:
        raise DocumentError("文档超过 100,000 字符上限；请拆分后检测，系统不会截断原文。", 413)
    if not text.strip():
        raise DocumentError("文档中没有可提取的文字。", 422)
    if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        raise DocumentError("文档包含无效的 Unicode 字符，请重新导出。", 422)


def _source(filename: str, format_: str, data: bytes, text: str,
            sections: list[dict], warnings: list[str]) -> dict:
    _validate_text(text)
    return {"filename": _safe_filename(filename), "format": format_, "text": text,
            "sections": sections, "warnings": list(dict.fromkeys(warnings)),
            "source_sha256": hashlib.sha256(data).hexdigest()}


def extract_pasted_text(text: str) -> dict:
    if not isinstance(text, str):
        raise DocumentError("文档内容必须是文本。")
    original = text
    # Browsers serialize textarea FormData line breaks as CRLF. Canonical LF
    # keeps the displayed textarea count and codepoint offsets consistent.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    _validate_text(text)
    source = _source("粘贴文本.txt", "text", text.encode("utf-8"), text,
                     [{"label": "原文", "start": 0, "end": len(text)}],
                     ["粘贴文本的换行符已统一为 LF；位置和指纹以规范化文本为准。"] if original != text else [])
    source["source_sha256_kind"] = "normalized_pasted_text_utf8"
    return source


def _extract_plain(data: bytes, filename: str, format_: str) -> dict:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ["utf-16"]
    elif data.startswith(b"\xef\xbb\xbf"):
        encodings = ["utf-8-sig"]
    else:
        encodings = ["utf-8", "gb18030"]
    text = None
    encoding = None
    for candidate in encodings:
        try:
            text = data.decode(candidate)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise DocumentError("无法识别文本编码；请保存为 UTF-8、带 BOM 的 UTF-16 或 GB18030。", 422)
    # A text extension is insufficient: do not turn binary data into a result.
    invalid = [char for char in text if unicodedata.category(char) in {"Cc", "Cs"}
               and char not in "\t\n\r\f"]
    if invalid:
        raise DocumentError("文件包含二进制或无效控制字符；请上传纯文本文件。", 422)
    warnings = ["文本按 GB18030 解码；请核对提取结果。"] if encoding == "gb18030" else []
    return _source(filename, format_, data, text,
                   [{"label": "原文", "start": 0, "end": len(text)}], warnings)


def _docx_package(data: bytes) -> tuple[bytes, list[str]]:
    if not data.startswith(b"PK\x03\x04"):
        raise DocumentError("文件不是有效的 DOCX 文档。", 422)
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        entries = archive.infolist()
        if len(entries) > 4096 or sum(item.file_size for item in entries) > MAX_ZIP_UNCOMPRESSED:
            raise DocumentError("DOCX 解压后的内容过大，超过安全处理上限。", 413)
        names = set()
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if (entry.filename in names or "\\" in entry.filename or path.is_absolute()
                    or ".." in path.parts or re.match(r"^[A-Za-z]:", entry.filename)):
                raise DocumentError("DOCX 包含异常路径或重复文件，无法处理。", 422)
            names.add(entry.filename)
            if entry.flag_bits & 1:
                raise DocumentError("不支持加密的 DOCX，请先解除密码保护。", 422)
            if entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise DocumentError("DOCX 使用了不支持的压缩格式。", 422)
            if entry.file_size > max(entry.compress_size, 1) * 1000:
                raise DocumentError("DOCX 压缩比例异常，超过安全处理上限。", 413)
        if "[Content_Types].xml" not in names or "word/document.xml" not in names:
            raise DocumentError("DOCX 缺少正文，不能当作 Word 文档读取。", 422)
        if archive.getinfo("word/document.xml").file_size > MAX_DOCUMENT_XML:
            raise DocumentError("DOCX 正文 XML 超过安全处理上限。", 413)
        xml = archive.read("word/document.xml")
        archive.close()
        if b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
            raise DocumentError("DOCX 包含不支持的 XML 实体声明。", 422)
        return xml, list(names)
    except DocumentError:
        raise
    except (zipfile.BadZipFile, RuntimeError, OSError, ValueError, EOFError) as exc:
        raise DocumentError("DOCX 已损坏，无法读取。", 422) from exc


def _extract_docx(data: bytes, filename: str) -> dict:
    xml, names = _docx_package(data)
    try:
        document = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise DocumentError("DOCX 正文 XML 损坏，无法读取。", 422) from exc
    body = document.find(f"{_W}body")
    if body is None:
        raise DocumentError("DOCX 没有正文。", 422)
    warnings = ["检测基于提取后的正文；原 Word 排版、图片和页码不保留。"]
    if any(re.match(r"word/(?:header|footer|footnotes|endnotes|comments)", name) for name in names):
        warnings.append("页眉、页脚、脚注、尾注和批注不参与本次检测。")
    if document.find(f".//{_W}drawing") is not None or document.find(f".//{_W}pict") is not None:
        warnings.append("图片、图表和文本框中的内容不参与检测；未执行 OCR。")
    if any(document.find(f".//{_W}{tag}") is not None for tag in ("altChunk", "object", "subDoc")):
        warnings.append("嵌入对象、外部子文档和替代正文不参与检测；未读取外部文件或链接。")
    if document.find(f".//{_W}del") is not None or document.find(f".//{_W}ins") is not None:
        warnings.append("正文按当前可见文本提取：保留插入内容，忽略删除修订。")
    if document.find(".//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath") is not None:
        warnings.append("公式对象未转换为自然语言，不参与检测。")

    def paragraph(node: ET.Element) -> str:
        pieces: list[str] = []

        def visit(child: ET.Element) -> None:
            if child.tag in {f"{_W}del", f"{_W}drawing", f"{_W}pict", f"{_W}txbxContent"}:
                return
            if child.tag == f"{_W}t":
                pieces.append(child.text or "")
            elif child.tag == f"{_W}tab":
                pieces.append("\t")
            elif child.tag in {f"{_W}br", f"{_W}cr"}:
                pieces.append("\n")
            elif child.tag == f"{_W}noBreakHyphen":
                pieces.append("‑")
            else:
                for descendant in child:
                    visit(descendant)
        visit(node)
        return "".join(pieces)

    def block_text(node: ET.Element) -> list[tuple[str, str]]:
        if node.tag == f"{_W}p":
            return [("段落", paragraph(node))]
        if node.tag == f"{_W}tbl":
            rows = []
            for row in node.findall(f"{_W}tr"):
                cells = []
                for cell in row.findall(f"{_W}tc"):
                    content = []
                    for child in cell:
                        content.extend(value for _, value in block_text(child))
                    cells.append("\n".join(content))
                rows.append("\t".join(cells))
            return [("表格", "\n".join(rows))]
        # Structured document tags and custom XML may wrap real body blocks.
        if node.tag in {f"{_W}sdt", f"{_W}sdtContent", f"{_W}customXml", f"{_W}ins"}:
            blocks = []
            for child in node:
                blocks.extend(block_text(child))
            return blocks
        return []

    blocks = []
    for child in body:
        blocks.extend(block_text(child))
    parts, sections = [], []
    length = 0
    for index, (kind, value) in enumerate(blocks):
        separator = "\n\n" if index else ""
        length += len(separator)
        parts.append(separator)
        start = length
        parts.append(value)
        length += len(value)
        if length > MAX_DOCUMENT_CHARS:
            raise DocumentError("文档超过 100,000 字符上限；请拆分后检测，系统不会截断原文。", 413)
        if value:
            sections.append({"label": f"{kind} {index + 1}", "start": start, "end": length})
    return _source(filename, "docx", data, "".join(parts), sections, warnings)


def _bounded_pdf_stream(stream) -> bytes:
    """Decode the common text-stream filters without unbounded inflation."""
    raw = getattr(stream, "_data", b"")
    if not isinstance(raw, bytes) or len(raw) > MAX_PDF_STREAM:
        raise DocumentError("PDF 内容流过大，超过安全处理上限。", 413)
    filters = stream.get("/Filter", [])
    if isinstance(filters, str):
        filters = [filters]
    params = stream.get("/DecodeParms")
    if params and not isinstance(params, list):
        params = [params]
    for index, filter_ in enumerate(filters):
        parameter = params[index] if params and index < len(params) else None
        if parameter and parameter.get("/Predictor", 1) != 1:
            raise DocumentError("PDF 文字流采用不支持的预测编码；请重新导出 PDF。", 422)
        try:
            if filter_ in {"/FlateDecode", "/Fl"}:
                inflater = zlib.decompressobj()
                raw = inflater.decompress(raw, MAX_PDF_STREAM + 1)
                if len(raw) > MAX_PDF_STREAM or inflater.unconsumed_tail:
                    raise DocumentError("PDF 解压后的内容流过大。", 413)
                if not inflater.eof:
                    raise DocumentError("PDF 压缩内容已损坏。", 422)
            elif filter_ in {"/ASCIIHexDecode", "/AHx"}:
                value = re.sub(rb"\s+", b"", raw).split(b">", 1)[0]
                if len(value) % 2:
                    value += b"0"
                raw = bytes.fromhex(value.decode("ascii"))
            elif filter_ in {"/ASCII85Decode", "/A85"}:
                value = raw.strip()
                if value.startswith(b"<~"):
                    value = value[2:]
                if value.endswith(b"~>"):
                    value = value[:-2]
                raw = base64.a85decode(value)
            else:
                raise DocumentError("PDF 文字流采用不支持的编码；请重新导出 PDF。", 422)
        except DocumentError:
            raise
        except (zlib.error, ValueError, TypeError) as exc:
            raise DocumentError("PDF 文字内容流损坏，无法读取。", 422) from exc
        if len(raw) > MAX_PDF_STREAM:
            raise DocumentError("PDF 解压后的内容流过大。", 413)
    return raw


def _pdf_page_text(page) -> str:
    """Recover visible paragraph spacing without allocating rows for coordinates.

    PDF has no universal paragraph structure. Nearby text blocks are grouped
    using observed line gaps and column alignment; complex layouts still need
    the extraction warning. Image blocks are never treated as text.
    """
    import pymupdf
    blocks = [block for block in page.get_text("blocks", sort=True, flags=pymupdf.TEXTFLAGS_TEXT)
              if len(block) >= 7 and block[6] == 0 and str(block[4]).strip()]
    if sum(len(str(block[4])) for block in blocks) > MAX_DOCUMENT_CHARS:
        raise DocumentError("PDF 页面文字超过 100,000 字符上限。", 413)
    gaps = [right[1] - left[3] for left, right in zip(blocks, blocks[1:])
            if abs(right[0] - left[0]) < 8 and 0 < right[1] - left[3] < 40]
    typical_gap = statistics.median(gaps) if gaps else 4
    groups: list[list[str]] = []
    previous = None
    for block in blocks:
        if (previous is None or abs(block[0] - previous[0]) > 8
                or block[1] < previous[3] - 2
                or block[1] - previous[3] > max(6, typical_gap * 1.7)):
            groups.append([])
        groups[-1].append(str(block[4]).strip("\r\n"))
        previous = block
    paragraphs = []
    for group in groups:
        value = "\n".join(group)
        if value.lstrip().startswith(("```", "~~~")):
            paragraphs.append(value)
            continue
        lines = value.splitlines()
        # PDF line wrapping is visual, not an authored paragraph boundary.
        # Keep lists explicit; join wrapped prose and marked quotations.
        if not value.lstrip().startswith(">") and any(re.match(r"\s*(?:[-*•]|\d+[.)])\s", line) for line in lines):
            paragraphs.append(value)
            continue
        joined = lines[0] if lines else ""
        han = sum("\u4e00" <= char <= "\u9fff" for char in value)
        chinese = han / max(sum(not char.isspace() for char in value), 1) >= .25
        for line in lines[1:]:
            boundary = " "
            if chinese and not (joined and line and joined[-1].isascii() and joined[-1].isalnum()
                                and line[0].isascii() and line[0].isalnum()):
                boundary = ""
            joined += boundary + line
        paragraphs.append(joined)
    return "\n\n".join(paragraphs)


def _extract_pdf(data: bytes, filename: str) -> dict:
    if not data.lstrip().startswith(b"%PDF-"):
        raise DocumentError("文件不是有效的 PDF。", 422)
    try:
        from pypdf import PdfReader
        from pypdf.generic import ArrayObject, DecodedStreamObject, NameObject
        reader = PdfReader(io.BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise DocumentError("不支持加密的 PDF，请先解除密码保护。", 422)
        if len(reader.pages) > MAX_PDF_PAGES:
            raise DocumentError("PDF 超过 200 页上限，请拆分后检测。", 413)
        decoded_total = 0
        seen_resources = set()

        def decoded(stream):
            nonlocal decoded_total
            output = _bounded_pdf_stream(stream)
            decoded_total += len(output)
            if decoded_total > MAX_PDF_DECODED_TOTAL:
                raise DocumentError("PDF 解压后的文字内容总量过大。", 413)
            replacement = DecodedStreamObject()
            replacement.set_data(output)
            # Forms need their transform and local resources after replacement.
            for key, value in stream.items():
                if key not in {"/Filter", "/DecodeParms", "/Length"}:
                    replacement[key] = value
            return replacement

        def sanitize_resources(resources, depth=0):
            if not resources:
                return
            resources = resources.get_object()
            if id(resources) in seen_resources:
                return
            seen_resources.add(id(resources))
            if depth > 20 or len(seen_resources) > 2048:
                raise DocumentError("PDF 资源嵌套过深或数量过多。", 413)
            for font_ref in resources.get("/Font", {}).get_object().values() if resources.get("/Font") else []:
                font = font_ref.get_object()
                if font.get("/ToUnicode"):
                    font[NameObject("/ToUnicode")] = decoded(font["/ToUnicode"].get_object())
            xobjects = resources.get("/XObject")
            if xobjects:
                xobjects = xobjects.get_object()
                for key, reference in list(xobjects.items()):
                    form = reference.get_object()
                    if form.get("/Subtype") == "/Form":
                        replacement = decoded(form)
                        xobjects[key] = replacement
                        sanitize_resources(replacement.get("/Resources"), depth + 1)

        for page in reader.pages:
            contents = page.get("/Contents")
            if contents:
                contents = contents.get_object()
                streams = list(contents) if isinstance(contents, ArrayObject) else [contents]
                replacements = ArrayObject([decoded(stream.get_object()) for stream in streams])
                page[NameObject("/Contents")] = replacements
            sanitize_resources(page.get("/Resources"))
        import pymupdf
        parts, sections, length = [], [], 0
        empty_pages = []
        with pymupdf.open(stream=data, filetype="pdf") as layout:
            for number, page in enumerate(layout, 1):
                value = _pdf_page_text(page)
                separator = "\n\n" if number > 1 else ""
                parts.append(separator)
                length += len(separator)
                start = length
                parts.append(value)
                length += len(value)
                if length > MAX_DOCUMENT_CHARS:
                    raise DocumentError("PDF 文字超过 100,000 字符上限；请拆分后检测。", 413)
                sections.append({"label": f"第 {number} 页", "page": number,
                                 "start": start, "end": length})
                if not value.strip():
                    empty_pages.append(number)
        text = "".join(parts)
        if not text.strip():
            raise DocumentError("PDF 未提取到文字，可能是扫描件；当前不支持 OCR，请先转换为可复制文字的 PDF 或 DOCX。", 422)
        warnings = ["PDF 按文本位置恢复段落，并合并段内自动换行；分栏、表格、公式及段落边界可能有偏差，请核对提取原文。",
                    "图片不参与文档文本检测；未执行 OCR。"]
        if empty_pages:
            preview = "、".join(str(page) for page in empty_pages[:20])
            warnings.append(f"第 {preview}{' 等' if len(empty_pages) > 20 else ''} 页未提取到文字，未对这些页面评分。")
        return _source(filename, "pdf", data, text, sections, warnings)
    except DocumentError:
        raise
    except Exception as exc:
        # Parser details may include document text; expose only a fixed message.
        raise DocumentError("PDF 已损坏或文字结构不受支持；请重新导出文件。", 422) from exc


def extract_document(data: bytes, filename: str) -> dict:
    if not isinstance(data, bytes) or not data:
        raise DocumentError("上传文件为空。")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise DocumentError("文档超过 20 MiB 文件上限。", 413)
    filename = _safe_filename(filename)
    extension = Path(filename).suffix.lower()
    if extension == ".doc":
        raise DocumentError("暂不支持旧版 .doc；请在 Word 中另存为 .docx 后上传。", 415)
    if extension == ".docx":
        return _extract_docx(data, filename)
    if extension == ".pdf":
        return _extract_pdf(data, filename)
    if extension in {".txt", ".md"}:
        return _extract_plain(data, filename, extension[1:])
    raise DocumentError("支持 DOCX、可复制文字的 PDF、TXT 和 Markdown 文件。", 415)


def _valid_score(score) -> bool:
    return (isinstance(score, (int, float)) and not isinstance(score, bool)
            and math.isfinite(score) and 0 <= score <= 1)


def _validate_segments(raw, text: str) -> list[dict]:
    if not isinstance(raw, list) or len(raw) > MAX_DOCUMENT_CHARS:
        raise RuntimeError("Detector returned invalid document segments.")
    output, previous_end = [], 0
    for segment in raw:
        if not isinstance(segment, dict):
            raise RuntimeError("Detector returned an invalid segment.")
        start, end, score = segment.get("start"), segment.get("end"), segment.get("risk_score")
        if (not isinstance(start, int) or isinstance(start, bool)
                or not isinstance(end, int) or isinstance(end, bool)
                or not 0 <= previous_end <= start < end <= len(text)
                or segment.get("text") != text[start:end]
                or (score is not None and not _valid_score(score))):
            raise RuntimeError("Detector returned invalid document offsets or scores.")
        evidence = segment.get("evidence", [])
        if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
            raise RuntimeError("Detector returned invalid segment evidence.")
        result = dict(segment)
        result["highlighted"] = bool(score is not None and score >= HIGHLIGHT_THRESHOLD)
        result["decision"] = segment.get("decision") if segment.get("decision") in {
            "likely_ai", "likely_human", "uncertain"} and score is not None else "uncertain"
        output.append(result)
        previous_end = end
    # A malformed detector must not silently omit text while showing 100% coverage.
    covered = sum(sum(not char.isspace() for char in segment["text"]) for segment in output)
    expected = sum(not char.isspace() for char in text)
    if covered != expected:
        raise RuntimeError("Detector segments omitted document content.")
    return output


def _abstain_segment(text: str, start: int, end: int, reason: str, applicability: str) -> dict:
    return {"start": start, "end": end, "text": text[start:end], "risk_score": None,
            "highlighted": False, "decision": "uncertain", "evidence": [reason],
            "applicability": applicability}


def _detect_ranges(text: str, detect_text: Callable[[str], dict]) -> tuple[list[dict], list[str]]:
    segments, versions = [], []
    for start, end in segment_ranges(text):
        excerpt = text[start:end]
        context = noncontent_reason(excerpt)
        if context:
            segments.append(_abstain_segment(text, start, end, context[1], context[0]))
            continue
        if len(excerpt) > 10_000:
            raise RuntimeError("Document segmentation exceeded the detector input limit.")
        prediction = detect_text(excerpt)
        if not isinstance(prediction, dict):
            raise RuntimeError("Detector returned a non-object result.")
        if prediction.get("model_version"):
            versions.append(str(prediction["model_version"]))
        if "segments" in prediction:
            local = _validate_segments(prediction["segments"], excerpt)
            for segment in local:
                segments.append({**segment, "start": start + segment["start"],
                                 "end": start + segment["end"]})
        else:
            score = prediction.get("risk_score") if not prediction.get("out_of_scope") else None
            if score is not None and not _valid_score(score):
                raise RuntimeError("Detector returned an invalid document score.")
            evidence = prediction.get("evidence", prediction.get("signals", []))
            segments.append({"start": start, "end": end, "text": excerpt,
                             "risk_score": score, "decision": prediction.get("decision", "uncertain"),
                             "highlighted": score is not None and score >= HIGHLIGHT_THRESHOLD,
                             "evidence": evidence, "applicability": prediction.get("applicability", "unverified_platform")})
    return _validate_segments(segments, text), list(dict.fromkeys(versions))


def analyze_document(source: dict, detect_text: Callable[[str], dict]) -> dict:
    """Preserve all extracted text and clearly separate coverage from model risk."""
    text = source.get("text")
    _validate_text(text)
    if len(segment_ranges(text)) > MAX_DOCUMENT_SEGMENTS:
        raise DocumentError("文档超过 2,000 个评分片段上限；请按章节拆分后检测，不会截断原文。", 413)
    sections = source.get("sections", [])
    if not isinstance(sections, list):
        raise RuntimeError("Invalid extracted document sections.")
    for section in sections:
        if (not isinstance(section, dict) or not isinstance(section.get("label"), str)
                or type(section.get("start")) is not int or type(section.get("end")) is not int
                or not 0 <= section["start"] <= section["end"] <= len(text)):
            raise RuntimeError("Invalid extracted document section offsets.")
    if len(text) <= 10_000:
        detection = detect_text(text)
        if not isinstance(detection, dict):
            raise RuntimeError("Detector returned a non-object result.")
        detection = dict(detection)
        if "segments" in detection:
            segments = _validate_segments(detection["segments"], text)
        else:
            segments, _ = _detect_ranges(text, detect_text)
    else:
        segments, versions = _detect_ranges(text, detect_text)
        detection = {"risk_score": None, "decision": "uncertain", "out_of_scope": True,
                     "applicability": "long_document_segments", "modality": "text",
                     "model_version": " / ".join(versions) or "local-text-model",
                     "evidence": ["长文按段检测；整篇不生成未经验证的来源概率。"],
                     "limitations": "可评分段落模型均分是描述性统计，不是整篇 AI 撰写比例。"}
    score = detection.get("risk_score")
    if score is not None and not _valid_score(score):
        raise RuntimeError("Detector returned an invalid whole-document score.")
    scored = [segment for segment in segments if segment["risk_score"] is not None]
    highlighted = [segment for segment in scored if segment["highlighted"]]
    visible = sum(not char.isspace() for char in text)
    visible_counts = [sum(not char.isspace() for char in segment["text"]) for segment in scored]
    scored_count = sum(visible_counts)
    highlighted_count = sum(sum(not char.isspace() for char in segment["text"]) for segment in highlighted)
    values = [segment["risk_score"] for segment in scored]
    mixed = any(value <= .2 for value in values) and any(value >= .8 for value in values)
    coverage = round(100 * scored_count / visible, 2)
    highlighted_percent = round(100 * highlighted_count / scored_count, 2) if scored_count else None
    mean_score = sum(segment["risk_score"] * count for segment, count in zip(scored, visible_counts)) / scored_count if scored_count else None
    # Re-check abstention even if a stand-in detector accidentally provides a score.
    if mixed or coverage < 80 or not scored_count or detection.get("out_of_scope"):
        detection["risk_score"] = None
        detection["decision"] = "uncertain"
    elif detection.get("decision") not in {"likely_ai", "likely_human", "uncertain"}:
        detection["decision"] = "uncertain"
    note = "AI 特征占比按整段标注：所在段落模型风险分数达到 0.99 的非空白字符占比；不是 AI 实际撰写比例，也不是逐字来源定位。模型均分不是整篇来源概率。"
    detection["segments"] = segments
    detection["segment_summary"] = {"highlighted_character_percent": highlighted_percent,
                                    "scored_character_percent": coverage,
                                    "highlight_threshold": HIGHLIGHT_THRESHOLD,
                                    "mixed_signals": mixed, "meaning": note}
    summary = {"total_characters": len(text), "visible_characters": visible,
               "scored_character_percent": coverage,
               "highlighted_character_percent": highlighted_percent,
               "highlighted_total_percent": round(100 * highlighted_count / visible, 2) if scored_count else None,
               "mean_segment_score": round(mean_score, 6) if mean_score is not None else None,
               "high_risk_segments": len(highlighted), "unscored_segments": len(segments) - len(scored),
               "scored_segments": len(scored), "total_segments": len(segments),
               "mixed_signals": mixed, "metric_note": note}
    return {"schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "filename": _safe_filename(source.get("filename", "document.txt")),
            "format": source.get("format", "text"), "source_sha256": source.get("source_sha256"),
            "source_sha256_kind": source.get("source_sha256_kind", "uploaded_file_bytes"),
            "extracted_text": text, "sections": sections, "warnings": list(source.get("warnings", [])),
            "detection": detection, "document_summary": summary}
