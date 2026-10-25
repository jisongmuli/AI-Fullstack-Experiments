"""Offset and abstention checks for document-level text annotation."""

from __future__ import annotations

import unittest
from pathlib import Path

from inference import DetectorService
from text_segments import segment_ranges


ROOT = Path(__file__).resolve().parents[2]


class TextSegmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = DetectorService(ROOT)

    def test_offsets_preserve_original_text(self) -> None:
        text = "  第一段讨论数据来源与审核误报。" * 7 + "\n\n" + "第二段说明模型的局限和人工复核。" * 8
        ranges = segment_ranges(text)
        self.assertEqual(len(ranges), 2)
        self.assertTrue(all(text[start:end].strip() == text[start:end] for start, end in ranges))
        self.assertTrue(all(left_end < right_start for (_, left_end), (right_start, _) in zip(ranges, ranges[1:])))
        result = self.service.predict_text_segments(text)
        self.assertEqual([(part["start"], part["end"]) for part in result["segments"]], ranges)
        self.assertTrue(all(text[part["start"]:part["end"]] == part["text"] for part in result["segments"]))
        self.assertIn("不是 AI 撰写比例", result["segment_summary"]["meaning"])

    def test_short_paragraph_has_no_score_or_highlight(self) -> None:
        text = "短句。\n\n这一段也太短。"
        result = self.service.predict_text_segments(text)
        self.assertIsNone(result["risk_score"])
        self.assertTrue(all(part["risk_score"] is None and not part["highlighted"] for part in result["segments"]))
        self.assertIsNone(result["segment_summary"]["highlighted_character_percent"])

    def test_many_short_paragraphs_do_not_gain_document_score(self) -> None:
        text = "研究样本不足。\n\n模型无法判断。\n\n仍需人工复核。\n\n来源尚未核验。\n\n不应直接定性。\n\n请补充分布。"
        result = self.service.predict_text_segments(text)
        self.assertIsNone(result["risk_score"])
        self.assertTrue(result["out_of_scope"])
        self.assertEqual(result["segment_summary"]["scored_character_percent"], 0.0)

    def test_long_chinese_document_has_segment_scores_but_no_document_score(self) -> None:
        text = "团队记录了数据来源、标签质量、评估方法和误报风险。" * 190
        self.assertGreater(len(text), 4000)
        result = self.service.predict_text_segments(text)
        self.assertIsNone(result["risk_score"])
        self.assertEqual(result["decision"], "uncertain")
        self.assertGreater(len(result["segments"]), 1)
        self.assertTrue(all(part["risk_score"] is not None for part in result["segments"]))
        self.assertEqual(result["segment_summary"]["scored_character_percent"], 100.0)

    def test_english_document_is_chunked_into_evaluable_ranges(self) -> None:
        text = "The team compared multiple sources and reviewed each result before publication. " * 32
        result = self.service.predict_text_segments(text)
        self.assertIsNone(result["risk_score"])
        self.assertGreater(len(result["segments"]), 1)
        self.assertTrue(all(part["risk_score"] is not None for part in result["segments"]))
        self.assertTrue(all(part["end"] - part["start"] <= 587 for part in result["segments"]))

    def test_english_above_training_length_gets_only_segment_scores(self) -> None:
        text = "The team compared multiple sources and reviewed each result before publication. " * 9
        self.assertGreater(len(text), 587)
        result = self.service.predict_text_segments(text)
        self.assertIsNone(result["risk_score"])
        self.assertTrue(all(part["risk_score"] is not None for part in result["segments"]))

    def test_writing_instruction_is_not_scored_as_ai_output(self) -> None:
        text = "Draft a technical report analyzing internet of things in the context of technological advancement."
        result = self.service.predict_text_segments(text)
        self.assertIsNone(result["risk_score"])
        self.assertIsNone(result["segments"][0]["risk_score"])
        self.assertEqual(result["segments"][0]["applicability"], "instruction_not_content")

    def test_chinese_writing_instruction_is_not_scored(self) -> None:
        text = "请根据这段资料写一篇关于人工智能教育的报告，要求包含背景、问题和建议。"
        result = self.service.predict_text_segments(text)
        self.assertEqual(result["segments"][0]["applicability"], "instruction_not_content")
        self.assertIsNone(result["segments"][0]["risk_score"])
        self.assertFalse(result["segments"][0]["highlighted"])

    def test_prompt_line_does_not_hide_following_prose(self) -> None:
        prompt = "请根据这些资料写一篇关于公共服务的报告。"
        prose = "公共服务质量需要公开的数据来源、清楚的评估方法和可追溯的复核记录。" * 5
        result = self.service.predict_text_segments(prompt + "\n" + prose)
        self.assertEqual(len(result["segments"]), 2)
        self.assertIsNone(result["segments"][0]["risk_score"])
        self.assertIsNotNone(result["segments"][1]["risk_score"])

    def test_quote_and_fenced_code_are_excluded_with_original_offsets(self) -> None:
        prose = "这是一段需要评估的正文，讨论数据来源、数据清洗、交叉验证、误报率及审查流程。" * 3
        quote = "> " + "这段内容来自外部引用，不能代表上传者本人写作来源。" * 5
        code = "```python\n" + "print('generated text')\n" * 30 + "```"
        text = prose + "\n" + quote + "\n" + code
        result = self.service.predict_text_segments(text)
        kinds = {part["applicability"]: part for part in result["segments"]}
        self.assertIn("quoted_material", kinds)
        self.assertIn("code_or_fenced_material", kinds)
        for key in ("quoted_material", "code_or_fenced_material"):
            self.assertIsNone(kinds[key]["risk_score"])
            self.assertFalse(kinds[key]["highlighted"])
            self.assertEqual(text[kinds[key]["start"]:kinds[key]["end"]], kinds[key]["text"])

    def test_partial_coverage_abstains_from_document_score(self) -> None:
        prose = "这是一段完整的实验正文，记录数据清洗、模型训练、交叉验证和人工复核。" * 4
        quote = "> " + "这是一段从别处引用的材料，并非上传者原创的文本。" * 10
        result = self.service.predict_text_segments(prose + "\n" + quote)
        self.assertIsNone(result["risk_score"])
        self.assertEqual(result["applicability"], "partial_segment_coverage")
        self.assertLess(result["segment_summary"]["scored_character_percent"], 80)

    def test_english_benchmark_prompts_are_excluded_without_censoring_prose(self) -> None:
        from text_segments import noncontent_reason
        self.assertIsNotNone(noncontent_reason("Develop a persuasive argument about open source software."))
        self.assertIsNone(noncontent_reason(
            "Scientists develop a persuasive argument based on evidence and data collected over many years."
        ))


if __name__ == "__main__":
    unittest.main()
