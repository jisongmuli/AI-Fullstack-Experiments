"""Split text into auditable, non-overlapping ranges for screening."""

from __future__ import annotations

import re


_ENGLISH_INSTRUCTION = re.compile(
    r"^\s*(?:please\s+|could\s+you\s+|can\s+you\s+)?"
    r"(?:write|draft|generate|compose|create|produce|provide|compile|develop|"
    r"formulate|rewrite|paraphrase|summari[sz]e|translate|outline)\b.{0,80}?"
    r"\b(?:essay|article|report|paragraph|story|summary|review|outline|analysis|"
    r"exploration|argument|guide|list|text|content|draft|piece)\b",
    re.IGNORECASE | re.DOTALL,
)
_CHINESE_INSTRUCTION = re.compile(
    r"^\s*(?:(?:请|请你|麻烦|帮我|给我|你能|你可以).{0,80}?"
    r"(?:写|撰写|生成|创作|改写|润色|总结|概括|翻译|拟定|起草)"
    r"|(?:根据|围绕|以.{1,40}?为题).{0,80}?"
    r"(?:写一篇|写出|撰写|生成|创作|拟定|起草))"
)


def noncontent_reason(text: str) -> tuple[str, str] | None:
    """Identify obvious non-authorial inputs; do not infer provenance from them."""
    stripped = text.strip()
    if not stripped:
        return None
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines and all(line.startswith(">") for line in lines):
        return ("quoted_material", "这是明确标记的引用块；不能据此判断上传者的写作来源。")
    if stripped.startswith(("```", "~~~")):
        return ("code_or_fenced_material", "这是代码或围栏材料，不在自然语言写作检测范围。")
    if len(stripped) <= 500 and (_ENGLISH_INSTRUCTION.search(stripped) or _CHINESE_INSTRUCTION.search(stripped)):
        return ("instruction_not_content", "这段文字更像写作指令，而非待检测的成稿。")
    return None


def _meaningful_range(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _paragraph_ranges(text: str) -> list[tuple[int, int]]:
    lines = list(re.finditer(r"[^\r\n]+", text))
    if not lines:
        return []
    ranges: list[tuple[int, int]] = []
    first, last = lines[0].start(), lines[0].end()
    previous_length = len(lines[0].group().strip())
    in_fence = False

    def line_kind(value: str) -> str:
        nonlocal in_fence
        stripped = value.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            return "fence"
        if in_fence:
            return "fence"
        if stripped.startswith(">"):
            return "quote"
        context = noncontent_reason(value)
        return "instruction" if context and context[0] == "instruction_not_content" else "prose"

    previous_kind = line_kind(lines[0].group())
    for line in lines[1:]:
        separator = text[last:line.start()]
        current_length = len(line.group().strip())
        current_kind = line_kind(line.group())
        blank_line = bool(re.search(r"\r?\n[ \t]*\r?\n", separator)) and current_kind != "fence"
        substantial_lines = previous_length >= 80 and current_length >= 80 and current_kind == previous_kind == "prose"
        if blank_line or substantial_lines or current_kind != previous_kind:
            span = _meaningful_range(text, first, last)
            if span:
                ranges.append(span)
            first = line.start()
        last = line.end()
        previous_length = current_length
        previous_kind = current_kind
    span = _meaningful_range(text, first, last)
    if span:
        ranges.append(span)
    return ranges


def _chunk_range(text: str, start: int, end: int) -> list[tuple[int, int]]:
    han = sum("\u4e00" <= char <= "\u9fff" for char in text[start:end])
    target, minimum = (900, 80) if han >= (end - start) * .25 else (520, 25)
    chunks: list[tuple[int, int]] = []
    while end - start > target + minimum:
        search_start = start + target // 2
        search_end = start + target
        boundary = max((text.rfind(mark, search_start, search_end) for mark in "。！？；.!?;\n"), default=-1)
        if boundary >= search_start:
            boundary += 1
        else:
            boundary = max(text.rfind(" ", search_start, search_end), text.rfind("\t", search_start, search_end))
            if boundary < search_start:
                boundary = search_end
        span = _meaningful_range(text, start, boundary)
        if span:
            chunks.append(span)
        start = boundary
        while start < end and text[start].isspace():
            start += 1
    span = _meaningful_range(text, start, end)
    if span:
        chunks.append(span)
    return chunks


def segment_ranges(text: str) -> list[tuple[int, int]]:
    """Return half-open character offsets in the unmodified input string."""
    ranges: list[tuple[int, int]] = []
    for start, end in _paragraph_ranges(text):
        if noncontent_reason(text[start:end]) is not None:
            ranges.append((start, end))
        else:
            ranges.extend(_chunk_range(text, start, end))
    return ranges
